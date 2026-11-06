"""Resolve Serper's opaque Google result URLs before exposing search results."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
from urllib.parse import urlsplit

URL_FIELDS = {"link", "url", "website", "resolved_url", "pdfUrl"}
RESOLVE_TIMEOUT = 50
REQUEST_TIMEOUT = 60  # Stay within the search client's default 65-second timeout.
RESOLVE_CONCURRENCY = 64


def is_google_redirect(value):
    if not isinstance(value, str):
        return False
    parsed = urlsplit(value)
    host = (parsed.hostname or "").rstrip(".").lower()
    return (
        parsed.scheme in {"http", "https"}
        and (host == "google.com" or host.endswith(".google.com"))
        and parsed.path.rstrip("/") in {"/goto", "/url"}
    )


def redirect_paths(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in URL_FIELDS and is_google_redirect(item):
                yield (*path, key)
            else:
                yield from redirect_paths(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from redirect_paths(item, (*path, index))


def merge_resolved_links(original, resolved):
    """Keep ranking/text/metadata intact; accept only corresponding URL changes."""
    merged = copy.deepcopy(original)
    for path in redirect_paths(original):
        source, target = resolved, merged
        try:
            for key in path[:-1]:
                source, target = source[key], target[key]
            value = source[path[-1]]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("Serper resolver omitted a result URL") from exc
        if not isinstance(value, str):
            raise ValueError("Serper resolver returned a non-string URL")
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Serper resolver returned an invalid URL")
        target[path[-1]] = value
    return merged


def resolution_error(response, message, status=None):
    return {
        **response, "success": False, "data": None, "cache_hit": False,
        "upstream_status": status, "error": message,
    }


def partial_response(response, data, metrics):
    """Omit source sections still carrying opaque URLs; never persist a partial view."""
    filtered = {}
    omitted = 0
    for key, value in data.items():
        if key == "retryAfter":
            continue
        if isinstance(value, list):
            kept = []
            for item in value:
                paths = list(redirect_paths(item))
                if paths:
                    omitted += len(paths)
                else:
                    kept.append(item)
            filtered[key] = kept
        elif paths := list(redirect_paths(value)):
            omitted += len(paths)
        else:
            filtered[key] = value
    if any(redirect_paths(filtered)):
        metrics["failures"] += 1
        return resolution_error(response, "Unresolved source could not be omitted; search results unavailable.")
    metrics["partial_successes"] += 1
    metrics["unresolved_links_omitted"] += omitted
    return {**response, "data": filtered, "cache_hit": False, "_cacheable": False}


async def resolve_response(response, *, session, endpoint, headers, semaphore, metrics, timeout=RESOLVE_TIMEOUT):
    """Bound all resolver waits/retries and never return an unresolved success."""
    import aiohttp

    data = response["data"]
    received_partial = False
    started = asyncio.get_running_loop().time()
    try:
        async with asyncio.timeout(timeout):
            for attempt in range(3):
                async with semaphore:
                    metrics["requests"] += 1
                    async with session.post(
                        endpoint, json=data, headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout),
                    ) as result:
                        metrics[f"status_{result.status}"] += 1
                        if result.status != 200:
                            metrics["failures"] += 1
                            return resolution_error(response, f"Serper URL resolution failed: HTTP {result.status}", result.status)
                        resolved = await result.json(content_type=None)
                if not isinstance(resolved, dict):
                    raise ValueError("Serper resolver returned an invalid response")
                data = merge_resolved_links(data, resolved)
                if not any(redirect_paths(data)):
                    metrics["successes"] += 1
                    return {**response, "data": data, "cache_hit": False}
                received_partial = True
                delay = resolved.get("retryAfter")
                if (
                    isinstance(delay, bool) or not isinstance(delay, (int, float))
                    or not math.isfinite(delay) or delay < 0
                ):
                    metrics["invalid_retry_after"] += 1
                    return partial_response(response, data, metrics)
                if delay >= timeout - (asyncio.get_running_loop().time() - started):
                    return partial_response(response, data, metrics)
                if attempt < 2:
                    metrics["retries"] += 1
                    await asyncio.sleep(max(0.1, delay))
            return partial_response(response, data, metrics)
    except TimeoutError:
        metrics["timeouts"] += 1
        if received_partial:
            return partial_response(response, data, metrics)
        metrics["failures"] += 1
        return resolution_error(response, "Serper URL resolution timed out; search results unavailable.")
    except (aiohttp.ClientError, ValueError) as exc:
        metrics["failures"] += 1
        logging.getLogger(__name__).warning("Serper resolver rejected response: %s: %s", type(exc).__name__, exc)
        return resolution_error(response, f"Serper URL resolution failed ({type(exc).__name__}); search results unavailable.")
    finally:
        metrics["seconds"] += asyncio.get_running_loop().time() - started
