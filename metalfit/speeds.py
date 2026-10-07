"""How fast each model actually runs on this Mac, from llama-server's own timings of real replies.

Why measured and not predicted: on a 32 GB M1 Max, Qwen3.6-35B-A3B's speed follows the bytes it reads per
token (about 16 ms fixed plus 400 GB/s), but Qwen3.8-27B, a dense model, reached only 52-64 GB/s of
effective bandwidth - two to four times slower than the same arithmetic predicts - and single kernels timed
in isolation varied by up to 2x between runs.  A number taken from the replies people actually got is the
only one that holds, so metalfit keeps those: per Mac (chip and memory) and per file (name and size).
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import threading
from pathlib import Path

FILE = Path.home() / "Library" / "Application Support" / "metalfit" / "speeds.json"
KEEP = 30                 # the latest replies per model; older ones say less about the build running now
MIN_GEN = 64              # fewer generated tokens than this is mostly start-up, not speed
MIN_PROMPT = 256          # likewise for prompt processing
SHORT_CTX = 4096          # replies with less context than this are "short"; long context writes slower

_lock = threading.Lock()


def machine() -> str:
    """The chip and its memory, e.g. 'Apple M1 Max 32GB': the speed of a model belongs to one of these."""
    try:
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        mem = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        return f"{chip} {round(mem / 1024 ** 3)}GB"
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown"


MACHINE = machine()


def key(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    return f"{MACHINE}|{path.name}|{size}"


def _load() -> dict:
    try:
        return json.loads(FILE.read_text())
    except (OSError, ValueError):
        return {}


def _save(data: dict) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, FILE)


def extract_timings(tail: bytes) -> dict | None:
    """llama-server's `timings` object from the end of a reply: the JSON body, or the last event of a stream."""
    text = tail.decode("utf-8", "replace")
    at = text.rfind('"timings"')
    if at < 0:
        return None
    colon = text.find(":", at)
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[colon + 1:].lstrip())
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def record(path: Path, t: dict) -> None:
    """Keep one reply's speeds, if it was long enough to mean anything."""
    try:
        gen_n, gen_tps = int(t.get("predicted_n", 0)), float(t.get("predicted_per_second", 0))
        pp_n, pp_tps = int(t.get("prompt_n", 0)), float(t.get("prompt_per_second", 0))
        ctx = int(t.get("cache_n", 0)) + pp_n
    except (TypeError, ValueError):
        return
    if gen_n < MIN_GEN and pp_n < MIN_PROMPT:
        return
    with _lock:
        data = _load()
        e = data.setdefault(key(path), {"gen": [], "prompt": []})
        if gen_n >= MIN_GEN and gen_tps > 0:
            e["gen"] = (e["gen"] + [[ctx, round(gen_tps, 2)]])[-KEEP:]
        if pp_n >= MIN_PROMPT and pp_tps > 0:
            e["prompt"] = (e["prompt"] + [round(pp_tps, 1)])[-KEEP:]
        try:
            _save(data)
        except OSError:
            pass


def summary(path: Path) -> dict | None:
    """Medians of what was measured: writing at short and at long context, and reading the prompt."""
    e = _load().get(key(path))
    if not e:
        return None
    short = [v for c, v in e.get("gen", []) if c < SHORT_CTX]
    long_ = [v for c, v in e.get("gen", []) if c >= SHORT_CTX]
    med = lambda xs: round(statistics.median(xs), 1) if xs else None
    return {"gen": med(short), "gen_long": med(long_), "prompt": med(e.get("prompt", [])),
            "replies": len(e.get("gen", []))}


def describe(s: dict | None) -> str:
    """One line for `metalfit list` and the page."""
    if not s or not (s["gen"] or s["gen_long"] or s["prompt"]):
        return "not measured on this Mac yet"
    parts = []
    if s["gen"]:
        parts.append(f"{s['gen']:.1f} tok/s writing")
    if s["gen_long"]:
        parts.append(f"{s['gen_long']:.1f} at long context")
    if s["prompt"]:
        parts.append(f"{s['prompt']:.0f} tok/s reading prompts")
    return ", ".join(parts) + f" ({s['replies']} replies on this Mac)"
