"""metalfit's command line: look at a model, list a folder, or serve them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import engine as eng
from . import fit
from . import proxy

DEFAULT_PORT = 8099


def _models_dir(given: str | None) -> Path:
    if given:
        return Path(given).expanduser()
    for c in ("~/Documents/Strata-data/models", "~/models", "~/.cache/llama.cpp"):
        p = Path(c).expanduser()
        if p.is_dir():
            return p
    return Path.cwd()


def _one(path: Path, ws: int, n_ctx: int) -> None:
    m = fit.inspect(path)
    p = fit.plan(m, n_ctx, ws)
    print(f"{m.path.name}")
    print(f"  architecture   {m.arch}, {m.n_layers} blocks (-ngl {m.n_gpu_layers_all} is all of it)")
    print(f"  on disk        {m.file_bytes / 1e9:.1f} GB" + (f" over {len(m.shards)} shards" if len(m.shards) > 1 else ""))
    if m.lazy_bytes:
        print(f"  read from file {m.lazy_bytes / 1e9:.1f} GB (lazy, so it needs no memory)")
    print(f"  in memory      {m.must_hold / fit.GIB:.1f} GiB of weights")
    print(f"  KV cache       {p.kv / fit.GIB:.2f} GiB at -c {n_ctx} "
          f"({m.kv_layers} layers context-sized" + (f", {m.swa_layers} capped at {m.swa_window}" if m.swa_layers else "") + ")")
    print(f"  Metal          {ws / fit.GIB:.2f} GiB working set")
    print(f"  -> {p.advice}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="metalfit", description=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    f = sub.add_parser("fit", help="what one model needs and whether it fits")
    f.add_argument("model", help="a .gguf file (the first shard of a split model)")
    f.add_argument("-c", "--ctx", type=int, default=131072)

    ls = sub.add_parser("list", help="every model in a folder, with its verdict")
    ls.add_argument("--models", help="where the .gguf files are")
    ls.add_argument("-c", "--ctx", type=int, default=131072)

    sv = sub.add_parser("serve", help="one address in front of the models, with a page to switch them")
    sv.add_argument("--models", help="where the .gguf files are")
    sv.add_argument("--port", type=int, default=DEFAULT_PORT)
    sv.add_argument("-c", "--ctx", type=int, default=0, help="context to prefer (default: the largest that fits)")
    sv.add_argument("--keep-alive", type=float, default=600.0, metavar="SECONDS",
                    help="give the memory back after this long without a request (0 = never; default 600). "
                         "A model on the GPU whole wires nearly all of it, and wired memory cannot be paged "
                         "out, so everything else on the Mac swaps while it sits there unused")

    for p in (f, ls, sv):
        p.add_argument("--llama-server", help="path to llama-server (it reports the Metal working set)")
        p.add_argument("--working-set-gib", type=float, help="override what Metal will let a process hold")

    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 2

    server = eng.find_llama_server(getattr(a, "llama_server", None))
    ws = int(a.working_set_gib * fit.GIB) if a.working_set_gib else fit.working_set(server)
    if not ws:
        print("could not work out the Metal working set; pass --working-set-gib", file=sys.stderr)
        return 1

    if a.cmd == "fit":
        _one(Path(a.model).expanduser(), ws, a.ctx)
        return 0

    if a.cmd == "list":
        d = _models_dir(a.models)
        found = 0
        for p in sorted(d.rglob("*.gguf")):
            shards = fit.gguf.shards(p)
            if shards and shards[0] != p:
                continue
            try:
                if not fit.inspect(p).runnable:
                    continue
                _one(p, ws, a.ctx)
            except Exception as exc:
                print(f"{p.name}\n  cannot read: {exc}")
            print()
            found += 1
        if not found:
            print(f"no .gguf files under {d}")
        return 0

    if not server:
        print("llama-server not found; pass --llama-server <path>", file=sys.stderr)
        return 1
    proxy.serve(_models_dir(a.models), server, a.port, a.ctx, a.keep_alive)
    return 0


if __name__ == "__main__":
    sys.exit(main())
