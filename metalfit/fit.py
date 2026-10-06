"""Will this GGUF fit this Mac's GPU, and at what context?

On Apple Silicon the CPU and GPU share one memory, and what decides a model's speed is not how many of its
layers are on the GPU but whether *all* of them are: once nearly every tensor in the file belongs to the GPU's
buffer, llama.cpp maps the file into it instead of copying the GPU's weights out, the pages stay file-backed and
nothing is read from the SSD while it answers.  Below that line every GPU layer costs a copy and takes memory
from the page cache the CPU-side experts are read through.  Measured on a 48 GB M5 Pro with Qwen3.8-Flash-Next
Q2_0 at `-c 32768`: 44 layers 23.8 tok/s with a 32.2 GiB footprint and 31.7 s to load, 47 layers 29.2 tok/s at
0.9 GiB and 11.8 s, 49 (all) 33.4 tok/s.  Kolibri-1 went from 32 tok/s as a split to 59 whole.

So the only question worth answering before starting a model is: does it fit, and if not, what would make it -
a smaller quantization, a shorter context, or a raised `iogpu.wired_limit_mb`.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import gguf

GIB = 1024 ** 3
MIB = 1024 ** 2

# llama.cpp reads a tensor from the file row by row instead of loading it when the architecture marks it and it
# is larger than this (LLAMA_LAZY_MODE_AUTO, llama-model-loader.cpp: `auto_min_size`).  Only two architectures
# mark anything: qwen4exp and gemma4, both their per-layer embedding table.  Such a tensor is on disk, not in
# memory, so it must not be counted - Qwen3.8-Flash-Next's is 28.8 GB of its 66.4.
LAZY_MIN = 4 * GIB
LAZY_NAMES = ("per_layer_token_embd",)

# bytes per element of a KV cache entry: a quantized block plus its scale, divided by the block's elements
KV_BPE = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 34 / 32, "q5_1": 24 / 32, "q5_0": 22 / 32,
          "q4_1": 20 / 32, "q4_0": 18 / 32}

# Compute buffers and llama.cpp's own overhead, on top of the weights and the KV cache.  Derived, not guessed:
# on a 48 GB M5 Pro at the stock 37.44 GiB working set, Qwen Q2_0 (35.0 GiB of weights, so 2.44 GiB left) runs
# at -c 65536 and fails at -c 131072, whose KV caches are 0.80 and 1.60 GiB.  That puts the rest between 0.84
# and 1.64 GiB, and 1.2 also leaves Kolibri Q3_K_M (34.9 GiB, 0.69 GiB of KV at -c 65536) fitting, as measured.
COMPUTE_RESERVE = int(1.2 * GIB)

# What macOS and its apps have wired before any model loads.  Measured on an idle 48 GB M5 Pro with a browser
# and an editor open: 4.2 - 4.7 GiB.  It is subtracted when working out how many layers leave a wanted amount
# of the machine free, because that memory was never available to begin with.
SYSTEM_WIRED = int(4.3 * GIB)


@dataclass(frozen=True)
class Model:
    path: Path                 # the file given; for a split model its first shard
    shards: tuple[Path, ...]
    arch: str
    n_layers: int              # transformer blocks; llama.cpp's -ngl takes one more for the output head
    file_bytes: int            # every shard on disk
    lazy_bytes: int            # stays in the file while it answers
    must_hold: int             # what has to be in memory: file_bytes - lazy_bytes
    kv_layers: int             # layers with a context-sized KV cache
    swa_layers: int            # layers whose KV cache is only as long as their sliding window
    swa_window: int
    n_head_kv: int
    k_len: int
    v_len: int
    ssm_bytes: int             # the recurrent state of a hybrid model's non-attention layers, per sequence
    block_bytes: tuple[int, ...] = ()   # each transformer block's weights, in file order
    head_bytes: int = 0                 # the output head, which -ngl counts as one more layer

    @property
    def n_gpu_layers_all(self) -> int:
        """What to pass as -ngl to put the whole model on the GPU: the blocks plus the output head."""
        return self.n_layers + 1

    @property
    def runnable(self) -> bool:
        """A GGUF one can start a server with.  A vision encoder (`mmproj-*.gguf`, architecture `clip`) is a
        companion file loaded through `--mmproj`, not a model, and has no transformer blocks of its own."""
        return self.n_layers > 0 and self.arch not in ("clip",)


def _int(v, default=0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def inspect(path: str | Path) -> Model:
    """Read a model's headers and work out what it needs.  Only metadata is read, never the weights."""
    shards = gguf.shards(path)
    first = gguf.read(shards[0])

    file_bytes = lazy = head = 0
    blocks: dict[int, int] = {}
    for sh in shards:
        g = first if sh == shards[0] else gguf.read(sh)
        for t in g.tensors:
            file_bytes += t.nbytes
            if t.nbytes > LAZY_MIN and t.name.startswith(LAZY_NAMES):
                lazy += t.nbytes
                continue
            b = re.match(r"blk\.(\d+)\.", t.name)
            if b:
                blocks[int(b.group(1))] = blocks.get(int(b.group(1)), 0) + t.nbytes
            elif t.name.startswith("output"):
                head += t.nbytes

    n_layers = _int(first.key("block_count"))
    n_head_kv = _int(first.key("attention.head_count_kv"))
    k_len = _int(first.key("attention.key_length"))
    v_len = _int(first.key("attention.value_length"), k_len)
    window = _int(first.key("attention.sliding_window"))

    # Which layers hold a KV cache, and how long.  Three shapes, in the order they can be told apart:
    #   - a per-layer sliding-window pattern (Kolibri-1: 40 of 50 layers sliding, true means sliding)
    #   - a hybrid model marking its attention layers (Qwen3.8-Flash-Next: compress_ratios is non-zero on every
    #     full_attention_interval-th layer, the other 36 are recurrent and have no KV cache at all)
    #   - everything else: every layer, context-sized
    pattern = first.key("attention.sliding_window_pattern")
    ratios = first.key("attention.compress_ratios")
    if isinstance(pattern, list) and pattern:
        swa_layers = sum(1 for p in pattern if p)
        kv_layers = len(pattern) - swa_layers
    elif isinstance(ratios, list) and ratios:
        kv_layers, swa_layers = sum(1 for r in ratios if r), 0
    else:
        kv_layers, swa_layers = n_layers, 0

    # a hybrid model's recurrent layers keep a fixed state instead, which does not grow with the context
    ssm_inner = _int(first.key("ssm.inner_size"))
    ssm_state = _int(first.key("ssm.state_size"))
    ssm_conv = _int(first.key("ssm.conv_kernel"))
    recurrent = max(n_layers - kv_layers - swa_layers, 0)
    ssm_bytes = recurrent * (ssm_inner * ssm_state + ssm_inner * max(ssm_conv - 1, 0)) * 4

    return Model(path=Path(shards[0]), shards=tuple(shards), arch=first.arch, n_layers=n_layers,
                 file_bytes=file_bytes, lazy_bytes=lazy, must_hold=file_bytes - lazy,
                 kv_layers=kv_layers, swa_layers=swa_layers, swa_window=window,
                 n_head_kv=n_head_kv, k_len=k_len, v_len=v_len, ssm_bytes=ssm_bytes,
                 block_bytes=tuple(blocks[i] for i in sorted(blocks)), head_bytes=head)


def kv_bytes(m: Model, n_ctx: int, ctk: str = "q8_0", ctv: str = "q8_0") -> int:
    """The KV cache for `n_ctx` tokens.  A sliding-window layer only caches its window."""
    per_tok = m.n_head_kv * (m.k_len * KV_BPE.get(ctk, 2.0) + m.v_len * KV_BPE.get(ctv, 2.0))
    swa_ctx = min(n_ctx, m.swa_window) if m.swa_window else n_ctx
    return int(per_tok * (m.kv_layers * n_ctx + m.swa_layers * swa_ctx)) + m.ssm_bytes


def physical_memory() -> int:
    """The Mac's memory in bytes (hw.memsize), 0 when it cannot be read."""
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def working_set(llama_server: str | Path | None = None) -> int:
    """What Metal will let a process hold: `recommendedMaxWorkingSetSize`, or the `iogpu.wired_limit_mb` a user
    set.  Read from llama-server when it is around, because that is the number it plans against; else ~75% of
    the machine's memory, which is what macOS defaults to."""
    if llama_server:
        try:
            out = subprocess.run([str(llama_server), "--list-devices"], capture_output=True, text=True,
                                 timeout=60)
            m = re.search(r"MTL\d+:.*?\((\d+)\s*MiB", out.stdout + out.stderr)
            if m:
                return int(m.group(1)) * MIB
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        mb = int(subprocess.check_output(["sysctl", "-n", "iogpu.wired_limit_mb"], text=True).strip())
        if mb > 0:
            return mb * MIB
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    try:
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        return int(ram * 0.75)
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0


def layers_for_free(m: Model, n_ctx: int, want_free: int, ram: int, ws: int,
                    ctk: str = "q8_0", ctv: str = "q8_0") -> int:
    """The most GPU layers that still leave `want_free` bytes of the machine for everything else.

    Measured on a 48 GB M5 Pro with Kolibri-1 Q3_K_M (34.9 GiB): 51 layers wire 40.2 GiB and leave 7.8 GB at
    62.0 tok/s, 36 wire 29.5 and leave 18.5 at 39.1, 25 wire 21.7 and leave 26.3 at 32.2, 16 wire 15.5 and
    leave 32.5 at 24.3.  Layers left off the GPU are mapped file pages macOS can drop under pressure, which is
    why this trade is worth making on a Mac you also work on."""
    budget = min(ws, max(ram - want_free - SYSTEM_WIRED, 0)) - kv_bytes(m, n_ctx, ctk, ctv) - COMPUTE_RESERVE
    return layers_for_budget(m, max(budget, 0))


def layers_for_budget(m: Model, weight_budget: int) -> int:
    """How many layers fit `weight_budget` of wired GPU memory.  llama.cpp's -ngl N puts the output head and
    the last N-1 blocks on the GPU, so this counts from the end."""
    if not m.block_bytes or m.head_bytes > weight_budget:
        return 0
    used, n = m.head_bytes, 1
    for b in reversed(m.block_bytes):
        if used + b > weight_budget:
            break
        used, n = used + b, n + 1
    return n


CONTEXTS = (2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 393216, 524288)


@dataclass(frozen=True)
class Plan:
    model: Model
    working_set: int
    n_ctx: int
    kv: int
    needed: int                # weights + KV + the compute reserve
    fits_whole: bool
    max_ctx: int               # the largest context from CONTEXTS that still fits whole; 0 when none does
    n_gpu_layers: int          # what to pass as -ngl
    headroom: int              # working_set - needed, negative when it does not fit

    @property
    def advice(self) -> str:
        if self.fits_whole:
            return (f"all {self.n_gpu_layers} layers on the GPU at -c {self.n_ctx}, "
                    f"{self.headroom / GIB:.1f} GiB spare")
        if self.max_ctx:
            return (f"-c {self.n_ctx} does not fit; all {self.model.n_gpu_layers_all} layers fit at "
                    f"-c {self.max_ctx}")
        short = (self.model.must_hold + COMPUTE_RESERVE - self.working_set) / GIB
        return (f"does not fit whole at any context: the weights alone need "
                f"{self.model.must_hold / GIB:.1f} GiB of {self.working_set / GIB:.1f} GiB"
                + (f", {short:.1f} GiB too much" if short > 0 else ""))


def plan(m: Model, n_ctx: int, ws: int | None = None, ctk: str = "q8_0", ctv: str = "q8_0",
         gpu_layers: int | None = None) -> Plan:
    """Whether `m` fits the GPU whole at `n_ctx`, and the largest context at which it would.

    `gpu_layers` forces a count instead.  That is worth doing on a Mac you also work on: the GPU's share is
    wired and macOS cannot page it out, so a model held whole pushes everything else into swap, while the
    layers left to the CPU are mapped file pages the system can simply drop and read again."""
    ws = working_set() if ws is None else ws
    kv = kv_bytes(m, n_ctx, ctk, ctv)
    needed = m.must_hold + kv + COMPUTE_RESERVE
    fits = needed <= ws
    best = 0
    for c in CONTEXTS:
        if m.must_hold + kv_bytes(m, c, ctk, ctv) + COMPUTE_RESERVE <= ws:
            best = c
    if gpu_layers is not None:
        n = max(0, min(gpu_layers, m.n_gpu_layers_all))
        return Plan(model=m, working_set=ws, n_ctx=n_ctx, kv=kv, needed=needed,
                    fits_whole=(n >= m.n_gpu_layers_all and fits), max_ctx=best, n_gpu_layers=n,
                    headroom=ws - needed)
    return Plan(model=m, working_set=ws, n_ctx=n_ctx, kv=kv, needed=needed, fits_whole=fits, max_ctx=best,
                n_gpu_layers=m.n_gpu_layers_all if fits else 0, headroom=ws - needed)
