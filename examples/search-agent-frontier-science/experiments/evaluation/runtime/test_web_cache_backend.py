import asyncio, tempfile, os, json
from pathlib import Path
from unittest.mock import patch
from eval_search_server import EvalSearchServer, SearchServer, SearchResponse, WebResponseCache, response_key
from literegistry.services.search_server import SearchRequest, SearchServerConfig
async def main():
 with tempfile.TemporaryDirectory() as d:
  os.environ['WEB_RESPONSE_CACHE_DIR']=d+'/local'; os.environ['WEB_RESPONSE_SEED_CACHE_DIR']=d+'/seed'
  server=object.__new__(EvalSearchServer)
  server.config=SearchServerConfig(jina_api_key='test',upstream_headers={'X-API-KEY':'test'})
  calls=0
  async def upstream(self,request):
   nonlocal calls
   calls+=1
   return SearchResponse(success=True,mode='url',data={'url':request.url,'title':'Readable','content':'Useful scientific evidence.'},execution_time=0)
  with patch.object(SearchServer,'execute',upstream):
   req=SearchRequest(mode='url',url='https://example.org/paper')
   results=await asyncio.gather(*(server.execute(req) for _ in range(10)))
   assert calls==1 and all(r.success for r in results)
   assert (await server.execute(req)).cache_hit
   seedreq=SearchRequest(mode='url',url='https://example.org/seed')
   payload,ep=server._payload_and_endpoint(seedreq)
   key=response_key('url',ep,payload,server._headers_for_request(seedreq))
   seed=WebResponseCache(Path(d+'/seed'));p=seed._path(key);p.parent.mkdir(parents=True)
   seed._write(p,dict(success=True,mode='url',data={'url':seedreq.url,'title':'Seed','content':'Saved evidence'},execution_time=0))
   before=p.read_bytes();r=await server.execute(seedreq)
   assert r.cache_hit and calls==1 and p.read_bytes()==before
   assert server.web_response_cache._path(key).exists()
   assert not (await server.execute(SearchRequest(mode='url',url='https://huggingface.co/data'))).success
   assert calls==1
  async def challenge(self,request):
   return SearchResponse(success=True,mode='url',data={'url':request.url,'title':'Robot Challenge Screen','content':'Checking the site connection security'},execution_time=0)
  with patch.object(SearchServer,'execute',challenge):
   req=SearchRequest(mode='url',url='https://example.org/challenge')
   assert not (await server.execute(req)).success
   payload,ep=server._payload_and_endpoint(req)
   key=response_key('url',ep,payload,server._headers_for_request(req))
   assert not server.web_response_cache._path(key).exists()
  print('PASS: backend concurrent cache reuse; read-only seed import; blocked-host rejection; HTTP200 challenge exclusion')
asyncio.run(main())
