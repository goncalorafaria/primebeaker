"""Versioned live-web backend for canonical clients; original backend unchanged.
Redirect resolver and challenge checks adapted from Steven's web-search implementation.
"""
import os,asyncio,time
from collections import Counter
from pathlib import Path
from web_cache import WebResponseCache, response_key
from urllib.parse import urlsplit
import aiohttp,uvicorn
from literegistry.services.search_server import SearchServer,SearchServerConfig,SearchResponse
from eval_web_policy import DOMAINS,blocked_url,blocked_response,filter_response
from serper_resolve import is_google_redirect,redirect_paths,resolve_response
from web_quality import reject_challenge_page
from web_metrics import emit
from fetch_recovery import recover
class EvalSearchServer(SearchServer):
    def _headers_for_request(self, request):
        headers = super()._headers_for_request(request)
        if request.mode == 'url' and os.environ.get('JINA_FORMAT', 'markdown') == 'default':
            headers = {k: v for k, v in headers.items()
                       if k.lower() not in {'x-return-format', 'x-respond-with'}}
        return headers
    def _metadata(self):return {**super()._metadata(),'model_path':os.environ.get('EVAL_SEARCH_MODEL_PATH','search:frontier-web-v2')}
    async def _cache_get(self,key):return None
    async def _cache_set(self,key,value):pass
    async def execute(self,request):
        started = time.monotonic()
        try:
            result = await self._execute_observed(request)
        except Exception as exc:
            emit('request', mode=request.mode, success=False, exception=type(exc).__name__, seconds=time.monotonic()-started)
            if request.mode != 'url' or blocked_url(request.url, domains=DOMAINS):
                raise
            return self._page_not_found(request, time.monotonic()-started)
        emit('request', mode=request.mode, success=result.success, cache_hit=result.cache_hit,
             redirect_requested=request.mode == 'url' and is_google_redirect(request.url),
             error=result.error, quality_warnings=(result.data.get('quality_warnings', []) if isinstance(result.data, dict) else []), seconds=time.monotonic()-started)
        if request.mode == 'url' and not result.success and not blocked_url(request.url, domains=DOMAINS):
            return self._page_not_found(request, time.monotonic()-started)
        return result

    def _page_not_found(self, request, elapsed):
        # Presentation only: after retries, validation, cache and failure telemetry.
        # Never insert this placeholder into the successful web-response cache.
        emit('page_not_found_response', url=request.url)
        return SearchResponse(success=True, mode='url', cache_hit=False, error=None,
            data={'url': request.url, 'title': 'Page not found',
                  'content': 'Page not found.'}, execution_time=elapsed)

    async def _execute_observed(self,request):
        start=time.monotonic()
        if request.mode == 'url' and blocked_url(request.url, domains=DOMAINS):
            return SearchResponse(**blocked_response(), execution_time=0.0)
        if not hasattr(self, 'web_response_cache'):
            self.web_response_cache = WebResponseCache(Path(os.environ.get(
                'WEB_RESPONSE_CACHE_DIR',
                '/weka/gfaria/search-agent-repr/evals/web-response-cache-v2')))
        payload, endpoint = self._payload_and_endpoint(request)
        key = response_key(request.mode, endpoint, payload, self._headers_for_request(request))
        def accept_cached(response):
            checked = reject_challenge_page(filter_response(dict(response)), reject_google_redirects=True)
            return checked.get('success') is True and not any(redirect_paths(checked.get('data'))) if request.mode == 'query' else checked.get('success') is True
        async def fetch_or_import():
            # Read-only fallback: never lock, write, or modify Steven's cache.
            source = WebResponseCache(Path(os.environ.get('WEB_RESPONSE_SEED_CACHE_DIR', '/weka/stevenc/cache/prime-rl/web/v1')))
            try:
                cached = await asyncio.to_thread(source._read, source._path(key))
            except (OSError, ValueError, KeyError):
                cached = None
            if cached is not None and accept_cached(cached):
                emit('seed_cache_hit', mode=request.mode)
                return reject_challenge_page(filter_response(dict(cached)), reject_google_redirects=True)
            return await self._execute_uncached(request)
        result = await self.web_response_cache.get_or_fetch(
            key, fetch_or_import, accept_cached=accept_cached)
        result = reject_challenge_page(filter_response(result), reject_google_redirects=True)
        if result.get('cache_hit'):
            data = result.get('data') or {}
            emit('cache_response', cache_key=key, mode=request.mode,
                 requested_url=request.url if request.mode == 'url' else None,
                 query=request.query if request.mode == 'query' else None,
                 success=result.get('success'), title=data.get('title'),
                 content_chars=len(str(data.get('content') or '')) if request.mode == 'url' else None,
                 final_url=data.get('final_url') or data.get('finalUrl'),
                 quality_warnings=data.get('quality_warnings', []))
        result.pop('_cacheable', None)
        result['execution_time'] = time.monotonic() - start
        return SearchResponse(**result)

    async def _execute_uncached(self,request):
        if request.mode != 'url':
            return await self._execute_once(request)
        if blocked_url(request.url, domains=DOMAINS) or is_google_redirect(request.url):
            return await self._execute_once(request)
        async def jina_fetch(url):
            return await self._execute_once(request.model_copy(update={'url': url}))
        return await recover(request.url, jina_fetch)

    async def _execute_once(self,request):
        start=time.monotonic()
        if request.mode=='url':
            if blocked_url(request.url,domains=DOMAINS):return dict(blocked_response(),execution_time=0.0)
            if is_google_redirect(request.url):return dict(success=False,mode='url',data=None,error='Unresolved Google URL; search again for a direct URL.',execution_time=0.0)
        result=(await super().execute(request)).model_dump()
        data = result.get('data')
        if request.mode == 'url':
            content = data.get('content', '') if isinstance(data, dict) else ''
            checked = reject_challenge_page(result, reject_google_redirects=True)
            emit('upstream_page', success=result.get('success'), status=result.get('upstream_status'),
                 content_chars=len(str(content or '')), empty_content=not str(content or '').strip(),
                 quality_rejected=bool(result.get('success') and not checked.get('success')),
                 captcha_marker=bool(__import__('re').search(r'are you a robot|captcha|verifying your request|robot challenge', str(content), __import__('re').I)))
        if request.mode=='query' and result.get('success') and any(redirect_paths(result.get('data'))):
            if not hasattr(self,'resolve_semaphore'):self.resolve_semaphore=asyncio.Semaphore(64)
            endpoint=urlsplit(self.config.search_api_url)._replace(path='/resolve',query='',fragment='').geturl()
            resolver_metrics = Counter()
            async with aiohttp.ClientSession() as session:
                result=await resolve_response(result,session=session,endpoint=endpoint,headers=self._headers_for_request(request),semaphore=self.resolve_semaphore,metrics=resolver_metrics)
            emit('redirect_resolution', **dict(resolver_metrics))
        result=filter_response(result)
        result=reject_challenge_page(result,reject_google_redirects=True)
        # Keep resolver's _cacheable flag until the successful-response cache checks it.
        result['execution_time']=time.monotonic()-start
        return result
if __name__=='__main__':
    config=SearchServerConfig(host='0.0.0.0',port=int(os.environ['SEARCH_PORT']),registry=os.environ['REGISTRY_PATH'],jina_api_key=os.environ['JINA_API_KEY'],upstream_timeout=120,upstream_headers={'X-API-KEY':os.environ['SERPER_API_KEY'],'Content-Type':'application/json'})
    uvicorn.run(EvalSearchServer(config).app,host=config.host,port=config.port)
