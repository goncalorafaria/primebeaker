import asyncio,os
from literegistry.coop.endpoints import wait_for_endpoint
async def main():
 record=await wait_for_endpoint(os.environ['HEAD_REGISTRY'], 'redis', timeout=1800, poll_interval=2, healthcheck='redis')
 if not record.uri.startswith(('redis://','rediss://')):raise ValueError('Expected Redis endpoint')
 print(record.uri,flush=True)
asyncio.run(main())
