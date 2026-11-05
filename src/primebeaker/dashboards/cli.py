"""Config-driven serving and independently deployable collection workers."""
from __future__ import annotations

import json
import signal
import threading

from .collector import Collector
from .config import DashboardConfig
from .server import make_server
from .views import get_view


def source_names(source: str) -> list[str]:
    if source not in ("all", "metrics", "wandb"):
        raise ValueError("source must be all, metrics, or wandb")
    return ["metrics", "wandb"] if source == "all" else [source]


class DashboardCLI:
    """Serve a live dashboard or collect its configured read-only sources."""

    def serve(self, config: str, host: str = "127.0.0.1", port: int = 8791, collect: bool = True, source: str = "all"):
        cfg = DashboardConfig(config)
        _, curves = get_view(cfg.kind)
        collector = Collector(cfg, curves)
        stop = threading.Event()
        server = make_server(cfg, host, port)
        threads = []
        if collect:
            for name in source_names(source):
                thread = threading.Thread(target=collector.loop, args=(name, stop), daemon=True, name=f"dashboard-{name}")
                thread.start()
                threads.append(thread)
        print(json.dumps({"url": f"http://{host}:{server.server_port}/", "state_dir": str(cfg.state_dir)}), flush=True)
        # Raise in the main thread so the HTTP socket and collection stop event
        # are cleaned up together on a Beaker SIGTERM.
        def terminate(signum, frame):
            raise KeyboardInterrupt
        previous = signal.signal(signal.SIGTERM, terminate)
        try:
            server.serve_forever(poll_interval=0.5)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            server.server_close()
            for thread in threads:
                thread.join(timeout=30)
            collector.close()
            signal.signal(signal.SIGTERM, previous)

    def collect(self, config: str, source: str = "all", once: bool = False):
        cfg = DashboardConfig(config)
        _, curves = get_view(cfg.kind)
        collector = Collector(cfg, curves)
        stop = threading.Event()
        names = source_names(source)
        threads = []
        previous = signal.signal(signal.SIGTERM, lambda *_: stop.set())
        try:
            if once:
                for name in names:
                    collector.loop(name, stop, once=True)
            else:
                for name in names:
                    thread = threading.Thread(target=collector.loop, args=(name, stop), daemon=True, name=f"dashboard-{name}")
                    thread.start()
                    threads.append(thread)
                while any(t.is_alive() for t in threads):
                    stop.wait(1)
        except KeyboardInterrupt:
            stop.set()
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=30)
            collector.close()
            signal.signal(signal.SIGTERM, previous)


def main():
    import fire
    fire.Fire(DashboardCLI())
