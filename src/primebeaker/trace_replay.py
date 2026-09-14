"""Replay recorded search-agent turns against a local vLLM server."""
import concurrent.futures
import json
import random
import threading
import time
import urllib.request
import urllib.error
from pathlib import Path


def prepare(traces, tools, output, conversations=32, turns=4, seed=42, runtime_traces=None):
    """Create a reproducible subset manifest without copying conversation text."""
    import hashlib
    if int(conversations) < 1 or int(turns) < 1:
        raise ValueError('conversations and turns must be positive')
    target=Path(output)
    if target.exists():
        raise FileExistsError(target)
    schema=json.loads(Path(tools).read_text())
    if not isinstance(schema, list) or any(t.get('type') != 'function' or 'function' not in t for t in schema):
        raise ValueError('tools must be a list of OpenAI function-tool schemas')
    source_path=Path(traces).resolve(); offsets=[]; checksum=hashlib.sha256()
    with source_path.open('rb') as source:
        while True:
            offset=source.tell(); line=source.readline()
            if not line: break
            checksum.update(line)
            if line.strip(): offsets.append(offset)
    rng=random.Random(int(seed)); rng.shuffle(offsets); entries=[]; call_count=0
    with source_path.open('rb') as source:
        for offset in offsets:
            source.seek(offset); trace=json.loads(source.readline())
            trajectory=trace.get('info', {}).get('search_agent', {}).get('trajectory', [])
            if not trajectory: continue
            start=rng.randrange(max(1, len(trajectory)-int(turns)+1))
            count=min(int(turns), len(trajectory)-start)
            entries.append({'offset':offset, 'trace_id':trace['id'], 'start_turn':start, 'turn_count':count})
            call_count+=count
            if len(entries)==int(conversations): break
    if not entries: raise ValueError('No search-agent trajectories in traces')
    manifest={'traces':runtime_traces or str(source_path), 'sha256':checksum.hexdigest(),
              'tools':schema, 'entries':entries, 'turns':call_count, 'subset_seed':int(seed)}
    with target.open('x') as stream: stream.write(json.dumps(manifest,indent=2)+'\n')
    return {'manifest':str(target), 'conversations':len(entries), 'calls':call_count}


def request_body(turn, tools):
    messages = []
    for original in turn['prompt']:
        message = {k:v for k,v in original.items() if v is not None and k != 'thinking_blocks'}
        if message.get('tool_calls'):
            calls = []
            for call in message['tool_calls']:
                if 'function' in call:
                    calls.append(call)
                else:
                    arguments = call['arguments']
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    calls.append({'id':call['id'], 'type':'function',
                                  'function':{'name':call['name'], 'arguments':arguments}})
            message['tool_calls'] = calls
        messages.append(message)
    return {'model':'sweep-model', 'messages':messages, 'tools':tools,
            'max_tokens':int(turn['response']['usage']['completion_tokens']),
            'temperature':1.0, 'ignore_eos':True, 'stream':True,
            'stream_options':{'include_usage':True}}


def run(manifest, port, output, concurrency=64, seed=42, warmup=False):
    data=json.loads(Path(manifest).read_text())
    entries=list(data['entries']);random.Random(int(seed)).shuffle(entries)
    if warmup:entries=entries[:1]
    lock=threading.Lock()
    root=Path(output);root.mkdir(exist_ok=True)
    log=root/('replay-warmup.jsonl' if warmup else 'replay-requests.jsonl')
    started=time.monotonic(); completed=0;failed=0;output_tokens=0
    def replay(entry):
        nonlocal completed,failed,output_tokens
        with open(data['traces'],'rb') as source:
            source.seek(entry['offset']);trace=json.loads(source.readline())
        trajectory = trace['info']['search_agent']['trajectory']
        start_turn = entry.get('start_turn', 0)
        selected = trajectory[start_turn:start_turn + entry.get('turn_count', len(trajectory))]
        for index,turn in enumerate(selected, start=start_turn):
            usage=turn.get('response',{}).get('usage',{})
            length=usage.get('completion_tokens',0)
            if not length:continue
            body=request_body(turn, data['tools'])
            before=time.monotonic();row={'trace_id':trace['id'],'turn':index,
                'recorded_prompt_tokens':usage.get('prompt_tokens'), 'target_output_tokens':length,
                'started_at_unix':time.time(),'status':'started'}
            def save(value):
                with lock:
                    with log.open('a') as out:out.write(json.dumps(value)+'\n');out.flush()
            save(row)
            try:
                request=urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                    data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                first=None;chunks=0;actual={}
                with urllib.request.urlopen(request,timeout=180) as response:
                    for line in response:
                        if not line.startswith(b'data: '):continue
                        content=line[6:].strip()
                        if content==b'[DONE]':break
                        event=json.loads(content)
                        if event.get('error'):
                            raise RuntimeError('Streaming API error: '+str(event['error']))
                        if event.get('usage'):actual=event['usage']
                        if event.get('choices'):
                            chunks+=1
                            if first is None:first=time.monotonic()-before
                row.update(status='completed',duration=time.monotonic()-before,ttft_seconds=first,
                           chunks=chunks,usage=actual)
                with lock:completed+=1;output_tokens+=actual.get('completion_tokens',0)
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError):
                    row['http_status'] = error.code
                    row['response_body'] = error.read().decode(errors='replace')[:8000]
                row.update(status='failed',error=str(error),duration=time.monotonic()-before)
                with lock:failed+=1
            save(row)
            if row['status']=='failed' or warmup:break
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(concurrency)) as pool:
        list(pool.map(replay,entries))
    duration=time.monotonic()-started
    result={'completed':completed,'failed':failed,'duration':duration,
            'output_throughput':output_tokens/duration,'total_output_tokens':output_tokens,
            'dataset':'recorded-search-agent-turns','conversations':len(entries)}
    (root/('warmup.json' if warmup else 'benchmark.json')).write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    import fire
    fire.Fire(run)
