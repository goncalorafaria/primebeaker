import asyncio,json
from pathlib import Path
from literegistry.coop.endpoints import wait_for_endpoint
from literegistry import RegistryClient,get_kvstore
from literegistry_tool_client import SearchClient,TerminalExecutionClient,FetchClient
ROOT=Path('/weka/gfaria/search-agent-repr')
async def main():
 head='sqlite:///weka/gfaria/search-agent-repr/registry/control.sqlite3'
 redis=await wait_for_endpoint(head,'redis',timeout=30,poll_interval=2,healthcheck='redis')
 gateway=await wait_for_endpoint(head,'gateway',timeout=30,poll_interval=2,healthcheck='http')
 client=RegistryClient(get_kvstore(redis.uri,raise_on_error=True),cache_ttl=1)
 names=['terminal','localsearch:frontier-science','judge:search-agent-pengin9b-rl400']
 try:counts={n:len(await client.get_all(n,force=True)) for n in names}
 finally:await client.close()
 result={'registry':redis.uri,'gateway':gateway.uri,'registered':counts}
 print(json.dumps(result),flush=True)
 if counts['terminal'] and counts['localsearch:frontier-science']:
  search=SearchClient(gateway.uri+'/search',model_path='localsearch:frontier-science',timeout=30,max_retries=1)
  r=await search.execute(query='science research',num_results=10)
  assert r.get('success'),r
  hits=r['data'].get('organic') or r['data'].get('results');assert hits,r
  url=hits[0].get('link') or hits[0].get('url')
  # Exercise the same canonical /search URL mode used by FetchClient.
  import httpx
  async with httpx.AsyncClient(timeout=30) as h:
   fetched=await h.post(gateway.uri+'/search',json={'model_path':'localsearch:frontier-science','mode':'url','url':url});fetched.raise_for_status();assert fetched.json()['data']['content']
  terminal=TerminalExecutionClient(gateway.uri+'/terminal',timeout=30,max_retries=1)
  out=await terminal.execute(command='head -n 1',contents='search-agent-service-smoke\n')
  assert 'search-agent-service-smoke' in json.dumps(out),out
  result['smoke']={'search_hits':len(hits),'url_fetch':True,'terminal':True}
  print('Search, stored-page fetch, and terminal smoke passed',flush=True)
 (ROOT/'services/readiness.json').write_text(json.dumps(result,indent=2)+'\n')
asyncio.run(main())
