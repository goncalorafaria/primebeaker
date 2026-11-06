"""Successful web responses shared across jobs on the mounted filesystem."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path


def response_key(mode: str, endpoint: str, payload: dict, headers: dict) -> str:
    # Credentials authorize requests; they do not identify public web content.
    headers = {
        name.lower(): value for name, value in headers.items() if name.lower() not in {"authorization", "x-api-key"}
    }
    canonical = json.dumps(
        {"mode": mode, "endpoint": endpoint, "payload": payload, "headers": headers},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class WebResponseCache:
    """No expiry; atomic entries and per-key locks shared by all workers."""

    def __init__(self, directory: Path):
        self.directory = directory
        self._inflight: dict[str, asyncio.Task] = {}

    def _path(self, key: str) -> Path:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Web cache keys must be SHA-256 hex digests")
        return self.directory / key[:2] / key[2:4] / f"{key}.json"

    @staticmethod
    def _read(path: Path) -> dict | None:
        try:
            with path.open() as stream:
                record = json.load(stream)
        except FileNotFoundError:
            return None
        response = record["response"]
        if response.get("success") is not True:
            raise ValueError(f"Unsuccessful response in web cache: {path}")
        return {**response, "cache_hit": True}

    @staticmethod
    def _write(path: Path, response: dict) -> None:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump({"saved_at": time.time(), "response": response}, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _try_lock(path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(".lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            return None
        except BaseException:
            lock.close()
            raise
        return lock

    @asynccontextmanager
    async def _locked(self, path: Path):
        while True:
            acquiring = asyncio.create_task(asyncio.to_thread(self._try_lock, path))
            try:
                lock = await asyncio.shield(acquiring)
            except asyncio.CancelledError:
                # The thread can acquire a lock after its waiter was cancelled.
                lock = await acquiring
                if lock is not None:
                    await asyncio.to_thread(lock.close)
                raise
            if lock is not None:
                break
            await asyncio.sleep(0.1)
        try:
            yield
        finally:
            # Keep the lock file: unlinking lets another process lock a new inode.
            await asyncio.to_thread(lock.close)

    async def _load_or_fetch(self, key: str, fetch, accept_cached) -> dict:
        path = self._path(key)
        cached = await asyncio.to_thread(self._read, path)
        if cached is not None and (accept_cached is None or accept_cached(cached)):
            return cached
        async with self._locked(path):
            cached = await asyncio.to_thread(self._read, path)
            if cached is not None and (accept_cached is None or accept_cached(cached)):
                return cached
            response = await fetch()
            if response.get("success") is True and response.get("_cacheable", True):
                # Finish the atomic publication before releasing the lock, even
                # when shutdown cancels this request during filesystem I/O.
                writing = asyncio.create_task(asyncio.to_thread(self._write, path, response))
                try:
                    await asyncio.shield(writing)
                except asyncio.CancelledError:
                    await writing
                    raise
            return response

    async def get_or_fetch(self, key: str, fetch, *, accept_cached=None) -> dict:
        task = self._inflight.get(key)
        joined = task is not None
        if task is None:
            task = asyncio.create_task(self._load_or_fetch(key, fetch, accept_cached))
            self._inflight[key] = task

            def finished(done):
                self._inflight.pop(key, None)
                if not done.cancelled():
                    done.exception()  # Retrieve errors if every HTTP waiter disconnected.

            task.add_done_callback(finished)
        response = await asyncio.shield(task)
        return {**response, "cache_hit": response.get("cache_hit", False) or (joined and response["success"])}
