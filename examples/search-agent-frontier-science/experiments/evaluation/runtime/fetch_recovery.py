"""Bounded Jina and Serper content recovery."""
import asyncio
import random
import re
from urllib.parse import urlsplit
import aiohttp
from eval_web_policy import DOMAINS, blocked_url, filter_response
from serper_resolve import is_google_redirect
from web_quality import reject_challenge_page
from web_metrics import emit

def failure(message):
    return dict(success=False, mode='url', data=None, error=message, execution_time=0.0)

def checked(result):
    return reject_challenge_page(filter_response(result),reject_google_redirects=True)

async def serper_fetch(url):
    import os
    if blocked_url(url, domains=DOMAINS) or is_google_redirect(url):
        return failure('Blocked or unresolved URL')
    key=os.environ.get('SERPER_API_KEY')
    if not key:return failure('SERPER_API_KEY is not configured')
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as session:
            async with session.post('https://scrape.serper.dev',
                    headers={'X-API-KEY':key,'Content-Type':'application/json'},
                    json={'url':url,'includeMarkdown':True}) as response:
                if response.status!=200:return failure(f'Serper scrape HTTP {response.status}')
                body=await response.json(content_type=None)
                if not isinstance(body,dict):return failure('Invalid Serper scrape response')
                content=body.get('markdown') or body.get('text') or ''
                if not isinstance(content,str):return failure('Invalid Serper scrape content')
                metadata=body.get('metadata') or {}
                if not isinstance(metadata,dict):metadata={}
                result=checked(dict(success=True,mode='url',execution_time=0.0,upstream_status=200,
                    data=dict(url=url,final_url=body.get('url') or url,title=metadata.get('title') or body.get('title') or '',
                    content=content,fetch_provider='serper',original_content_characters=len(content),content_truncated=len(content)>64000)))
                if result['success']:result['data']['content']=content[:64000]
                return result
    except (TimeoutError,aiohttp.ClientError,ValueError) as exc:
        return failure(f'Serper scrape failed: {type(exc).__name__}')

def arxiv_pdf(url):
    p=urlsplit(url)
    if p.hostname in {'arxiv.org','www.arxiv.org'} and p.path.startswith('/html/'):
        return 'https://arxiv.org/pdf/'+p.path.split('/html/',1)[1].split('/')[0]
    return None

async def recover(url, jina_fetch, *, scrape=serper_fetch, sleep=asyncio.sleep):
    last=failure('No readable source content')
    target=url
    try:
        async with asyncio.timeout(750):
            for round_index in range(3):
                for provider, fetch in [('jina', jina_fetch), ('serper', scrape)]:
                    attempt=round_index*2 + (provider=='serper')
                    if attempt:
                        await sleep(min(8, 2 ** (attempt-1))+random.uniform(0,.5))
                    try:
                        async with asyncio.timeout(120):
                            last=checked(await fetch(target))
                    except (TimeoutError,aiohttp.ClientError):
                        last=failure(f'{provider} attempt timed out or had a transport error')
                    emit('fetch_attempt',provider=provider,attempt=round_index+1,url=target,
                         success=last['success'],error=last.get('error'))
                    if last['success']:
                        if target!=url:
                            last['data']['requested_url']=url
                            last['data']['fallback_url']=target
                        return last
                # Preserve strict provider alternation; try the corresponding PDF
                # in subsequent rounds if both providers failed on arXiv HTML.
                target=arxiv_pdf(url) or url
    except TimeoutError:
        last=failure('Content recovery exceeded 750-second budget')
    return last
