"""One HTTP shell and snapshot API shared by all registered dashboard views."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import threading
import time

from .config import DashboardConfig
from .store import SnapshotStore
from .views import get_view


def make_server(config: DashboardConfig, host: str = "127.0.0.1", port: int = 8791):
    assets, _ = get_view(config.kind)
    common = files("primebeaker.dashboards").joinpath("assets")
    store = SnapshotStore(config.state_dir)
    history_lock = threading.Lock()
    history_cache = {'at': 0.0, 'body': b''}
    static = {
        "/": (assets.joinpath("index.html"), "text/html"),
        "/assets/dashboard.js": (assets.joinpath("dashboard.js"), "text/javascript"),
        "/assets/core.js": (common.joinpath("core.js"), "text/javascript"),
        "/assets/styles.css": (common.joinpath("styles.css"), "text/css"),
    }
    snapshots = {"/api/snapshot": "latest.json", "/latest.json": "latest.json", "/api/curves": "curves.json", "/wandb.json": "curves.json"}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split("?", 1)[0]
            try:
                if route in static:
                    path, content_type = static[route]
                    body = path.read_bytes()
                elif route in snapshots:
                    content_type = "application/json"
                    body = json.dumps(store.read(snapshots[route]), allow_nan=False).encode()
                elif route == "/api/inference-history":
                    content_type = "application/json"
                    with history_lock:
                        if time.monotonic() - history_cache['at'] > 10:
                            history_cache['body'] = json.dumps(store.inference_history(), allow_nan=False).encode()
                            history_cache['at'] = time.monotonic()
                        body = history_cache['body']
                elif route == "/api/config":
                    content_type = "application/json"
                    # Only public display metadata; the manifest may contain
                    # private credential environment-variable names and paths.
                    body = json.dumps({"title": config.title, "kind": config.kind, "refresh_seconds": config.interval, "judge_label": config.judge.get("label", "Judge"),
                                       "runs": [{"id": r["id"], "label": r.get("label", r["id"])} for r in config.runs]}).encode()
                else:
                    self.send_error(404)
                    return
            except (OSError, ValueError):
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(b'{"error":"Waiting for the first successful collector snapshot"}')
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    return ThreadingHTTPServer((host, int(port)), Handler)
