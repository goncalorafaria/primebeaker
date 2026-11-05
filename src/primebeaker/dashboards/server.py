"""One HTTP shell and snapshot API shared by all registered dashboard views."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import threading
import time

from .config import DashboardConfig, read_json
from .store import SnapshotStore
from .views import get_view
from .wandb import curve_matches_run


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
                    data = store.read(snapshots[route])
                    if snapshots[route] == 'curves.json':
                        # Receipts can change while a collector still holds an
                        # older cache. Check identity again at the API boundary.
                        for spec in config.runs:
                            run = config.run(spec)
                            curve = data.get('runs', {}).get(run['id'], {})
                            if not curve_matches_run(curve, run.get('wandb_run')):
                                data.setdefault('runs', {})[run['id']] = {
                                    'label': run['label'], 'available': False,
                                    'error': 'No history for the current run yet',
                                    'wandb_run': run.get('wandb_run')}
                        for baseline in config.data.get('baselines', []):
                            try:
                                curve = read_json(config.resolve(baseline['curves_file']))
                                if not curve_matches_run(curve, baseline.get('wandb_run')):
                                    raise ValueError('Baseline history does not match its configured W&B run')
                                curve.update(label=baseline['label'], historical=True)
                            except (OSError, ValueError, KeyError):
                                curve = {'available': False, 'historical': True,
                                         'error': 'Saved baseline history unavailable or run identity mismatch'}
                            data.setdefault('runs', {})[baseline['id']] = curve
                    body = json.dumps(data, allow_nan=False).encode()
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
                                       "runs": [{"id": r["id"], "label": r.get("label", r["id"]), "color": r.get('color')} for r in config.runs],
                                       "baselines": [{k: b.get(k) for k in ('id', 'label', 'color')} for b in config.data.get('baselines', [])],
                                       "curve_max_step": config.data.get('curve_max_step'),
                                       "curve_caveat": config.data.get('curve_caveat', '')}).encode()
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
