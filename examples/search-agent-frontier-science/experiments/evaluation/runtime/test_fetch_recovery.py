import asyncio
from fetch_recovery import recover,failure,checked
async def test():
    calls=[];delays=[]
    good=dict(success=True,mode='url',data={'url':'https://example.org','content':'Scientific paper text'},execution_time=0)
    async def jina(url):
        calls.append(url)
        return good if len(calls)==3 else failure('empty')
    async def no_direct(url):return failure('blocked')
    async def sleep(delay):delays.append(delay)
    assert (await recover('https://example.org',jina,scrape=no_direct,sleep=sleep))['success']
    assert len(calls)==3 and len(delays)==4
    calls.clear()
    async def yes_direct(url):return good
    assert (await recover('https://example.org',jina,scrape=yes_direct,sleep=sleep))['success']
    assert len(calls)==1
    calls.clear()
    async def pdf_jina(url):
        calls.append(url)
        return good if '/pdf/' in url else failure('image asset')
    assert (await recover('https://arxiv.org/html/2411.14614',pdf_jina,scrape=no_direct,sleep=sleep))['data']['fallback_url'].endswith('/pdf/2411.14614')
    assert not checked(dict(success=True,mode='url',data={'url':'https://arxiv.org/html/1','final_url':'https://arxiv.org/html/1/fig.svg','content':'caption'}))['success']
    print('PASS: bounded Jina retries/backoff, Serper recovery, arXiv PDF fallback, image rejection')
asyncio.run(test())
