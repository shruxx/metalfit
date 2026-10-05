"""A stable address in front of a model that changes underneath it.

llama-server is one model per process on one port; point a chat app or an editor at it and the address dies
whenever you switch models.  This keeps one port open, starts and stops llama-server behind it, and passes
`/v1/*` through unchanged - including streamed replies - so Open WebUI, an editor or anything else
OpenAI-compatible can be configured once.
"""
from __future__ import annotations

import json
import signal
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import engine as eng
from . import fit

WEB = Path(__file__).parent / "web"
HOP_BY_HOP = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade",
              "proxy-authorization", "proxy-authenticate", "content-length", "host"}


class State:
    """What is loaded, and the models we could load.  Guarded by a lock because a switch takes a minute and
    requests keep arriving during it."""

    def __init__(self, models_dir: Path, llama_server: Path, default_ctx: int = 0):
        self.models_dir = models_dir
        self.llama_server = llama_server
        self.default_ctx = default_ctx
        self.working_set = fit.working_set(llama_server)
        self.engine: eng.Engine | None = None
        self.lock = threading.Lock()
        self.busy = ""                      # a human-readable "what is happening" while switching
        self._models: list[fit.Model] | None = None

    def models(self, rescan: bool = False) -> list[fit.Model]:
        if self._models is None or rescan:
            found = []
            for p in sorted(self.models_dir.rglob("*.gguf")):
                shards = fit.gguf.shards(p)
                if shards and shards[0] != p:       # a later shard of a split model: the first one covers it
                    continue
                try:
                    m = fit.inspect(p)
                except Exception:                   # not a model we can read; skip rather than fail the list
                    continue
                if m.runnable:
                    found.append(m)
            self._models = found
        return self._models

    def best_ctx(self, m: fit.Model) -> int:
        p = fit.plan(m, self.default_ctx or 131072, self.working_set)
        return p.n_ctx if p.fits_whole else (p.max_ctx or 8192)

    def describe(self) -> dict:
        cur = self.engine
        return {
            "working_set_gib": round(self.working_set / fit.GIB, 2),
            "busy": self.busy,
            "loaded": None if not (cur and cur.alive()) else {
                "name": cur.plan.model.path.name,
                "path": str(cur.plan.model.path),
                "n_ctx": cur.plan.n_ctx,
                "n_gpu_layers": cur.plan.n_gpu_layers,
                "whole": cur.plan.fits_whole,
                "advice": cur.plan.advice,
            },
            "models": [{
                "name": m.path.name,
                "path": str(m.path),
                "arch": m.arch,
                "layers": m.n_layers,
                "file_gb": round(m.file_bytes / 1e9, 1),
                "hold_gib": round(m.must_hold / fit.GIB, 1),
                "lazy_gb": round(m.lazy_bytes / 1e9, 1),
                "shards": len(m.shards),
                "max_ctx": fit.plan(m, 131072, self.working_set).max_ctx,
                "suggest_ctx": self.best_ctx(m),
                "advice": fit.plan(m, self.best_ctx(m), self.working_set).advice,
            } for m in self.models()],
        }

    def load(self, path: str, n_ctx: int | None = None) -> dict:
        with self.lock:
            m = fit.inspect(path)
            ctx = n_ctx or self.best_ctx(m)
            plan = fit.plan(m, ctx, self.working_set)
            self.busy = f"stopping the model that is loaded"
            if self.engine:
                self.engine.stop()
                self.engine = None
            self.busy = f"loading {m.path.name} at -c {ctx}"
            e = eng.Engine(self.llama_server, plan)
            try:
                e.start()
            except Exception as exc:
                self.busy = ""
                raise RuntimeError(f"{m.path.name} did not start: {exc}") from exc
            self.engine, self.busy = e, ""
            return self.describe()

    def unload(self) -> dict:
        with self.lock:
            if self.engine:
                self.engine.stop()
                self.engine = None
            return self.describe()


class Handler(BaseHTTPRequestHandler):
    server_version = "metalfit"
    state: State

    def log_message(self, fmt, *args):            # one line per request, not three
        if not self.path.startswith("/api/status"):
            print(f"[metalfit] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    # ---- plumbing
    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj).encode())

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def _file(self, name: str) -> None:
        p = WEB / name
        if not p.is_file():
            self._send(404, b"not found", "text/plain")
            return
        kind = {".html": "text/html; charset=utf-8", ".css": "text/css", ".js": "text/javascript"}
        self._send(200, p.read_bytes(), kind.get(p.suffix, "application/octet-stream"))

    # ---- the pass-through
    def _upstream(self) -> None:
        e = self.state.engine
        if not (e and e.alive()):
            msg = self.state.busy or "no model is loaded - open the page and start one"
            self._json(503, {"error": {"message": msg, "type": "metalfit_no_model"}})
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
        headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP}
        req = urllib.request.Request(f"http://127.0.0.1:{e.port}{self.path}", data=body,
                                     headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=3600) as up:
                self.send_response(up.status)
                for k, v in up.headers.items():
                    if k.lower() not in HOP_BY_HOP:
                        self.send_header(k, v)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while chunk := up.read(8192):     # streamed replies have to go out as they arrive
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
        except urllib.error.HTTPError as err:
            self._send(err.code, err.read() or b"", err.headers.get("Content-Type", "application/json"))
        except (urllib.error.URLError, OSError) as err:
            self._json(502, {"error": {"message": f"the model stopped answering: {err}"}})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._file("index.html")
        if self.path.startswith("/api/status"):
            return self._json(200, self.state.describe())
        if self.path.startswith("/api/rescan"):
            self.state.models(rescan=True)
            return self._json(200, self.state.describe())
        if self.path.startswith("/v1/") or self.path in ("/health", "/props", "/slots"):
            return self._upstream()
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path == "/api/load":
            b = self._body()
            try:
                return self._json(200, self.state.load(b["path"], b.get("n_ctx")))
            except KeyError:
                return self._json(400, {"error": "which model? pass a path"})
            except Exception as exc:
                return self._json(500, {"error": str(exc)})
        if self.path == "/api/unload":
            return self._json(200, self.state.unload())
        if self.path.startswith("/v1/"):
            return self._upstream()
        self._send(404, b"not found", "text/plain")


def serve(models_dir: Path, llama_server: Path, port: int = 8099, default_ctx: int = 0) -> None:
    state = State(models_dir, llama_server, default_ctx)
    handler = type("BoundHandler", (Handler,), {"state": state})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    n = len(state.models())
    print(f"[metalfit] {n} model{'s' if n != 1 else ''} in {models_dir}")
    print(f"[metalfit] Metal working set {state.working_set / fit.GIB:.2f} GiB")
    print(f"[metalfit] open http://127.0.0.1:{port}/   API: http://127.0.0.1:{port}/v1")

    # Without this a `kill` or a closed terminal leaves llama-server running, and a 35 GiB model stays wired
    # with nothing in front of it.  Default SIGTERM ends Python without unwinding, so `finally` never runs.
    def bye(signum, _frame):
        print(f"\n[metalfit] signal {signum}, stopping the model")
        state.unload()
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, bye)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[metalfit] stopping")
    finally:
        state.unload()
