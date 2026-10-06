"""Starting and stopping one llama-server, with the flags a model needs on this Mac."""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from . import fit

# Where llama-server usually is: next to us, in a sibling checkout's engine/, or on PATH.
CANDIDATES = ("llama-server", "engine/llama-server", "build/bin/llama-server", "build-llama/bin/llama-server")


def find_llama_server(extra: str | Path | None = None) -> Path | None:
    for c in ([extra] if extra else []) + [os.environ.get("METALFIT_LLAMA_SERVER")]:
        if c and Path(c).is_file():
            return Path(c)
    here = Path.cwd()
    for base in (here, *here.parents[:2]):
        for c in CANDIDATES:
            p = base / c
            if p.is_file() and os.access(p, os.X_OK):
                return p
    found = shutil.which("llama-server")
    return Path(found) if found else None


def performance_cores() -> int:
    """Threads for llama-server: the cores of the fastest performance level only.  Every step waits for its
    slowest thread, so including the slower cores costs speed - on an M5 Pro (5 "Super" + 10 "Performance")
    Kolibri-1 wrote ~15 tok/s with 15 threads and ~27 with 5 at the same GPU split."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.perflevel0.physicalcpu"], text=True).strip()
        if out.isdigit() and int(out) > 0:
            return int(out)
    except (OSError, subprocess.SubprocessError):
        pass
    return max(1, (os.cpu_count() or 4) // 2)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def model_name(path: Path) -> str:
    """What to call the model over the API: the file's stem without the shard suffix."""
    stem = Path(path).name
    for suffix in (".gguf",):
        stem = stem[: -len(suffix)] if stem.endswith(suffix) else stem
    return re.sub(r"-\d{5}-of-\d{5}$", "", stem)


def command(llama_server: Path, plan: fit.Plan, port: int, threads: int | None = None,
            ctk: str = "q8_0", ctv: str = "q8_0") -> list[str]:
    """llama-server's command line for a plan.

    A model that fits goes on the GPU whole with an explicit -ngl, which also switches off llama.cpp's --fit:
    its own planner looks at the GPU's working set and takes the CPU side as unlimited, which on a shared
    memory means it can plan a split that then has nowhere to live.  A model that does not fit is left to
    --fit, because choosing a split well needs measurement, not arithmetic (see the README)."""
    cmd = [str(llama_server), "-m", str(plan.model.path), "-c", str(plan.n_ctx),
           # without an alias llama-server reports the model by its full path, which is what a chat app then
           # shows in its model picker
           "--alias", model_name(plan.model.path),
           "--port", str(port), "--host", "127.0.0.1", "--parallel", "1",
           "--threads", str(threads or performance_cores()),
           "-ctk", ctk, "-ctv", ctv]
    if plan.fits_whole:
        cmd += ["--fit", "off", "-ngl", str(plan.n_gpu_layers)]
    else:
        # the weights do not fit, so the CPU side reads experts from the file: repacking would copy them into
        # memory macOS can only swap, where the mapping lets it drop them and read them again
        cmd += ["--no-repack"]
    return cmd


@dataclass
class Engine:
    """One running llama-server.  Only one at a time: two big models do not fit a shared memory."""
    llama_server: Path
    plan: fit.Plan
    port: int = 0
    proc: subprocess.Popen | None = None
    started: float = 0.0
    log: list[str] = field(default_factory=list)

    def start(self, timeout: float = 600.0) -> None:
        self.port = free_port()
        cmd = command(self.llama_server, self.plan, self.port)
        sys.stderr.write(f"[metalfit] {self.plan.advice}\n[metalfit] {' '.join(cmd)}\n")
        sys.stderr.flush()
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.started = time.monotonic()
        deadline = self.started + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode} before answering")
            if self.healthy():
                return
            time.sleep(0.3)
        self.stop()
        raise TimeoutError(f"llama-server did not answer within {timeout:.0f} s")

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=1.5) as r:
                return r.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
