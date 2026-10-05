"""A GGUF header reader that does not care which quantization types exist.

Tensor sizes come from the gaps between consecutive data offsets, never from the type, because the
interesting files are the ones whose types are too new for the tooling: `GGML_TYPE_Q2_0` is 42 of 43 and
neither the `gguf` PyPI package nor Ollama's parser can size a tensor that uses it.  Only the header is
read - the weights are never touched.
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"GGUF"

# the value types a GGUF key can have, as (struct format, size); 8 = string and 9 = array are special
_SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
            6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


class GgufError(Exception):
    pass


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    type_id: int
    offset: int          # from the start of the file's tensor data
    nbytes: int          # from the next tensor's offset, or the end of the file


@dataclass(frozen=True)
class Gguf:
    path: Path
    meta: dict
    tensors: tuple[Tensor, ...]
    data_start: int

    @property
    def arch(self) -> str:
        return str(self.meta.get("general.architecture", ""))

    def key(self, suffix: str, default=None):
        """A metadata value under this file's own architecture prefix: key("block_count")."""
        return self.meta.get(f"{self.arch}.{suffix}", default)


class _Reader:
    def __init__(self, blob: bytes):
        self.b, self.i = blob, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise GgufError("header ends mid-value")
        out = self.b[self.i:self.i + n]
        self.i += n
        return out

    def scalar(self, fmt: str, n: int):
        return struct.unpack(fmt, self.take(n))[0]

    def string(self) -> str:
        return self.take(self.scalar("<Q", 8)).decode("utf-8", "replace")

    def value(self, vtype: int):
        if vtype in _SCALARS:
            return self.scalar(*_SCALARS[vtype])
        if vtype == 8:
            return self.string()
        if vtype == 9:
            elem = self.scalar("<I", 4)
            count = self.scalar("<Q", 8)
            return [self.value(elem) for _ in range(count)]
        raise GgufError(f"unknown metadata value type {vtype}")


def _parse(blob: bytes, path: Path, file_size: int):
    r = _Reader(blob)
    if r.take(4) != MAGIC:
        raise GgufError(f"{path.name} does not start with GGUF")
    version = r.scalar("<I", 4)
    if version < 2:
        raise GgufError(f"GGUF version {version} is older than this reader supports")
    n_tensors = r.scalar("<Q", 8)
    n_kv = r.scalar("<Q", 8)

    meta: dict = {}
    for _ in range(n_kv):
        k = r.string()
        meta[k] = r.value(r.scalar("<I", 4))

    raw = []
    for _ in range(n_tensors):
        name = r.string()
        dims = tuple(r.scalar("<Q", 8) for _ in range(r.scalar("<I", 4)))
        raw.append((name, dims, r.scalar("<I", 4), r.scalar("<Q", 8)))

    align = int(meta.get("general.alignment", 32)) or 32
    data_start = r.i + (-r.i % align)

    # sizes from the gaps between offsets: the last tensor runs to the end of the file
    by_offset = sorted(raw, key=lambda t: t[3])
    end = file_size - data_start
    sizes = {}
    for cur, nxt in zip(by_offset, list(by_offset[1:]) + [None]):
        sizes[cur[0]] = (nxt[3] if nxt else end) - cur[3]
    tensors = tuple(Tensor(n, d, t, o, sizes[n]) for n, d, t, o in raw)
    return Gguf(path=path, meta=meta, tensors=tensors, data_start=data_start)


def read(path: str | Path) -> Gguf:
    """One GGUF file's header.  The read grows until the tensor table fits, because the header's length
    depends on the tensor count and on the name lengths inside it."""
    p = Path(path)
    size = p.stat().st_size
    want = min(1 << 21, size)
    while True:
        with p.open("rb") as f:
            blob = f.read(want)
        try:
            return _parse(blob, p, size)
        except GgufError:
            if want >= size:
                raise
            want = min(want * 4, size)


SHARD = re.compile(r"(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


def shards(path: str | Path) -> list[Path]:
    """Every file of a model: all shards of a split GGUF, else the one file.  A split model is named
    `<stem>-00001-of-0000N.gguf` and its metadata shard is the first."""
    p = Path(path)
    m = SHARD.match(p.name)
    if not m:
        return [p]
    found = sorted(p.parent.glob(f"{m['stem']}-*-of-{m['total']}.gguf"))
    return found or [p]
