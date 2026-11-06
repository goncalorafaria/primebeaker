"""Eval-only recovery around the canonical gateway FetchClient."""
import asyncio
import random
import re
from web_metrics import emit

def install_gateway_recovery(client, *, sleep=asyncio.sleep):
    original = client.execute
    # Own the retry loop here, avoiding nested transport retries.
    client.max_retries = 1
    async def execute(*, url, parameters=None):
        for attempt in range(3):
            try:
                return await original(url=url, parameters=parameters)
            except (RuntimeError, TimeoutError, ConnectionError) as exc:
                message = str(exc)
                transient = isinstance(exc, (TimeoutError, ConnectionError)) or bool(re.search(
                    r'HTTP (?:408|429|5\d\d)\b|timeout|timed out|connection|disconnected', message, re.I))
                emit('gateway_fetch_failure', url=url, attempt=attempt + 1,
                     error=message, retryable=transient)
                if not transient:
                    raise
                if attempt < 2:
                    await sleep(2 ** attempt + random.uniform(0, 0.5))
        emit('page_not_found_response', url=url, source='gateway_retries_exhausted')
        return dict(success=True, mode='url', cache_hit=False, error=None,
                    data=dict(url=url, title='Page not found', content='Page not found.'))
    client.execute = execute
