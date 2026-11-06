"""Check fixed model readiness and actual Serper/Jina/terminal requests."""
import asyncio,json,sys,time,os
from pathlib import Path
import httpx
from literegistry_tool_client import SearchClient, FetchClient, TerminalExecutionClient

def check_services():
    for pid in os.getenv("EVAL_SERVICE_PIDS", "").split():
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            raise RuntimeError(f"Evaluation service PID {pid} exited; inspect this attempt’s service logs")

async def main():
    model_url = 'http://127.0.0.1:' + os.getenv('MODEL_PORT','28743')
    gateway = 'http://127.0.0.1:' + os.getenv('GATEWAY_PORT','28742')
    search_url = 'http://127.0.0.1:' + os.getenv('SEARCH_PORT','28741')
    async with httpx.AsyncClient(timeout=10) as client:
        for endpoint in [model_url+'/health', gateway+'/health', search_url+'/health']:
            deadline = time.monotonic() + 1800
            while True:
                check_services()
                try:
                    response = await client.get(endpoint)
                    response.raise_for_status()
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'Service did not become ready: {endpoint}')
                    await asyncio.sleep(5)
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(gateway+'/search', json=dict(mode='url',url='https://huggingface.co/datasets',model_path=os.environ.get('EVAL_SEARCH_MODEL_PATH', 'search:frontier-web-v2')))
        response.raise_for_status()
        blocked = response.json()
        assert blocked.get('success') is False and blocked.get('data') is None, blocked
    search = SearchClient(gateway+'/search', model_path=os.environ.get('EVAL_SEARCH_MODEL_PATH', 'search:frontier-web-v2'), timeout=450,max_retries=2)
    result = await search.execute(query='weak value amplification Aharonov Albert Vaidman',num_results=3)
    assert result.get('success'), result
    hits = result['data'].get('organic') or result['data'].get('results')
    assert hits, result
    fetch = FetchClient(local_search_server_url=gateway+'/search',local_search_model_path=os.environ.get('EVAL_SEARCH_MODEL_PATH', 'search:frontier-web-v2'),timeout=450,max_retries=2)
    page = await fetch.execute(url='https://en.wikipedia.org/wiki/Weak_value')
    assert page.get('success') and page.get('data',{}).get('content'), page
    terminal = TerminalExecutionClient(gateway+'/terminal', timeout=450,max_retries=2)
    result = await terminal.execute(command='head -n 1',contents='frontier-live-search-smoke\n')
    assert 'frontier-live-search-smoke' in json.dumps(result), result
    Path(sys.argv[1], 'preflight.json').write_text(json.dumps(dict(model=True,serper=True,jina=True,terminal=True)))
    print('Model, Serper, Jina and terminal preflight passed', flush=True)
asyncio.run(main())
