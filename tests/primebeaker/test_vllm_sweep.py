import json
from pathlib import Path
import sys

import pytest

from primebeaker.vllm_sweep import SweepCLI, commands, eligible, flags, load, trials


def config(tmp_path, **updates):
    c = {'model': '/model', 'server': {'max_model_len': 128, 'max_num_seqs': 4},
         'grid': {}, 'concurrency': [2], 'workloads': [{'name': 'short', 'input_tokens': 32, 'output_tokens': 8}],
         'requests': 4, 'warmup_requests': 2, 'repeats': 1, 'startup_timeout': 10, 'benchmark_timeout': 10, **updates}
    path = tmp_path/'config.json'
    path.write_text(json.dumps(c))
    return path


def test_deterministic_grid_and_independent_concurrency(tmp_path):
    c = load(config(tmp_path, grid={'max_num_batched_tokens': [16, 64], 'kv_offloading_size': [None, 32]}, repeats=2))
    ts = trials(c)
    assert ts == trials(c) and len(ts) == 8
    assert len({t['id'] for t in ts}) == 8
    server, bench = commands(c, ts[0], 8888, tmp_path)
    assert server[server.index('--max-num-seqs')+1] == '4'
    assert bench[bench.index('--max-concurrency')+1] == '2'
    assert '--ignore-eos' in bench and '--save-result' in bench and '--save-detailed' in bench
    assert flags({'kv_offloading_size': None, 'enable_chunked_prefill': False}) == ['--no-enable-chunked-prefill']


def test_length_includes_shared_prefix_and_output(tmp_path):
    c = load(config(tmp_path, workloads=[{'name':'too-long','input_tokens':64,'prefix_tokens':64,'output_tokens':8}]))
    with pytest.raises(ValueError, match='exceeds max_model_len'):
        trials(c)


def test_partial_success_and_latency_violations_cannot_win(tmp_path):
    c = load(config(tmp_path, latency_limits_ms={'p99_ttft_ms': 500}))
    assert not eligible({'completed':3,'output_throughput':9999,'p99_ttft_ms':1}, c)
    assert not eligible({'completed':4,'output_throughput':9999,'p99_ttft_ms':501}, c)
    assert not eligible({'completed':4,'output_throughput':9999}, c)
    assert eligible({'completed':4,'output_throughput':10,'p99_ttft_ms':10}, c)


def test_report_requires_all_repetitions_and_separates_workloads(tmp_path):
    c = load(config(tmp_path, repeats=2))
    output = tmp_path/'results'; output.mkdir()
    (output/'config.json').write_text(json.dumps(c))
    for i, t in enumerate(trials(c)):
        folder = output/t['id']; folder.mkdir()
        (folder/'result.json').write_text(json.dumps({**t,'status':'completed','eligible':True,'metrics':{'output_throughput':10+2*i}}))
        SweepCLI().report(output)
        ranking = json.loads((output/'ranking.json').read_text())
        assert ranking[0]['eligible'] == (i == 1)
    assert ranking[0]['median_output_tokens_per_second'] == 11


def test_real_subprocess_lifecycle_continues_after_engine_failure(tmp_path, monkeypatch):
    package = tmp_path/'fake'/'vllm'/'entrypoints'/'cli'
    package.mkdir(parents=True)
    for p in (package, package.parent, package.parent.parent):
        (p/'__init__.py').write_text('')
    (package/'main.py').write_text('''
import sys, json, os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
args=sys.argv[1:]
def arg(key): return args[args.index(key)+1]
if args[0]=='serve':
    if arg('--max-num-seqs')=='1': sys.exit(2)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            data=json.dumps({'data':[{'id':'sweep-model'}]}).encode() if self.path=='/v1/models' else b'metrics 1'
            self.send_response(200); self.end_headers(); self.wfile.write(data)
        def do_POST(self):
            self.send_response(200 if os.environ.get('VLLM_SERVER_DEV_MODE') == '1' else 404)
            self.end_headers()
    HTTPServer(('127.0.0.1',int(arg('--port'))),Handler).serve_forever()
else:
    Path(arg('--result-dir'),arg('--result-filename')).write_text(json.dumps({'completed':int(arg('--num-prompts')),'output_throughput':100,'p99_ttft_ms':5}))
''')
    monkeypatch.setenv('PYTHONPATH', str(tmp_path/'fake'))
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'version', lambda _: 'test')
    c = config(tmp_path, grid={'max_num_seqs':[1,4]})
    root = tmp_path/'run'
    first = SweepCLI().run(c, root, max_trials=1)
    assert first['finished_trials'] == 1
    result = SweepCLI().run(c, root, resume=True)
    assert result['finished_trials'] == 2
    rows = [json.loads(p.read_text()) for p in root.glob('*/result.json')]
    assert sorted(r['status'] for r in rows) == ['completed','failed']
    good = next(r for r in rows if r['eligible'])
    assert (root/good['id']/'metrics-after.txt').exists()
    assert json.loads((root/good['id']/'warmup.json').read_text())['completed'] == 2
    with pytest.raises(FileExistsError): SweepCLI().run(c, root)


def test_fire_entrypoint_plans_without_loading_vllm(tmp_path):
    from primebeaker.cli import run
    result = run(['vllm-sweep', 'plan', '--config='+str(config(tmp_path))])
    assert result['trials'] == 1


def test_rexs_rejects_gpu_underallocation(tmp_path):
    path = config(tmp_path, grid={'tensor_parallel_size':[2]})
    with pytest.raises(ValueError, match='requires 2 GPUs'):
        SweepCLI().rexs(path, 'unused', tmp_path/'bundle', 'model-image', gpus=1)


def test_sampling_retains_all_workloads_and_repeats(tmp_path):
    c = load(config(tmp_path, grid={'max_num_batched_tokens': list(range(16, 32))}, sample_settings=3, repeats=2))
    ts = trials(c)
    assert ts == trials(c) and len(ts) == 6
    assert len({json.dumps(t['engine'], sort_keys=True) for t in ts}) == 3
    with pytest.raises(ValueError, match='sample_settings'):
        load(config(tmp_path, sample_settings=0))


def test_fit_holds_out_settings_and_suggests_only_unmeasured(tmp_path):
    pytest.importorskip('numpy')
    c = load(config(tmp_path, grid={'max_num_batched_tokens': list(range(16, 32))}, sample_settings=10, repeats=2))
    root = tmp_path/'fit-results'; root.mkdir()
    (root/'config.json').write_text(json.dumps(c))
    measured = set()
    for t in trials(c):
        n = t['engine']['max_num_batched_tokens']; measured.add(n)
        folder = root/t['id']; folder.mkdir()
        (folder/'result.json').write_text(json.dumps({**t, 'status':'completed', 'eligible':True,
            'metrics':{'output_throughput': 100 + 2*n}}))
    report = SweepCLI().fit(root, source='completed')['workloads'][0]
    assert report['status'] == 'fitted' and report['successful_settings'] == 10
    assert report['cv_log_rmse'] < 0.1
    assert len(report['suggestions']) == 6
    assert all(r['engine']['max_num_batched_tokens'] not in measured for r in report['suggestions'])


def test_interactive_limit_includes_pending_and_other_partitions(monkeypatch):
    from primebeaker.vllm_sweep import check_interactive_slots
    import subprocess
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '11|gpuH200x8-interactive\n12|gpuA40x4-interactive\n13|gpuH200x8\n')
    with pytest.raises(RuntimeError, match='limit 2 reached'):
        check_interactive_slots()
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: '11|gpuH200x8-interactive\n13|gpuH200x8\n')
    assert check_interactive_slots() == ['11']
    monkeypatch.setattr(subprocess, 'check_output', lambda *a, **kw: 'bad output')
    with pytest.raises(RuntimeError, match='Cannot parse'):
        check_interactive_slots()


def test_resume_rejects_changed_config(tmp_path):
    path = config(tmp_path)
    root = tmp_path/'results'; root.mkdir()
    (root/'config.json').write_text(path.read_text())
    path = config(tmp_path, repeats=3)
    with pytest.raises(ValueError, match='same configuration'):
        SweepCLI().run(path, root, resume=True)


def test_analysis_preserves_repeats_failures_and_raw_metrics(tmp_path):
    import csv
    c = load(config(tmp_path, repeats=2))
    root = tmp_path/'data'; root.mkdir()
    (root/'config.json').write_text(json.dumps(c))
    for i, t in enumerate(trials(c)):
        folder = root/t['id']; folder.mkdir()
        (folder/'result.json').write_text(json.dumps({**t, 'status':'failed' if i else 'completed',
            'eligible':not i, 'metrics':{'output_throughput':0 if i else 50, 'ttfts':[0.1,0.2], 'p50_ttft_ms':100}}))
    result = SweepCLI().report(root)
    data = list(csv.DictReader(Path(result['analysis']).open()))
    assert len(data) == 2 and {r['repeat'] for r in data} == {'0','1'}
    assert {r['status'] for r in data} == {'completed','failed'}
    assert all(r['engine.max_num_seqs'] == '4' for r in data)
    raw = json.loads(Path(result['observations']).read_text())
    assert raw['observations'][0]['metrics']['ttfts'] == [0.1,0.2]
    assert all('metrics.p50_ttft_ms' in r for r in data)


def test_compute_counts_inflight_work_and_detects_counter_resets():
    from primebeaker.vllm_sweep import token_counters, summarize_compute
    counters = token_counters(b'vllm:generation_tokens_total{model_name="some model"} 120\nvllm:prompt_tokens_total 800\n')
    result = summarize_compute([
        {'monotonic_seconds': 10, 'counters': {k:0 for k in counters}},
        {'monotonic_seconds': 12, 'counters': counters}])
    assert result['output_tokens_per_second'] == 60
    assert result['input_tokens_per_second'] == 400
    reset = summarize_compute([
        {'monotonic_seconds': 10, 'counters': counters},
        {'monotonic_seconds': 12, 'counters': {k:0 for k in counters}}])
    assert reset['counter_reset'] and 'output_tokens_per_second' not in reset


def test_interrupted_compute_is_ranked_without_completed_requests(tmp_path):
    from primebeaker.vllm_sweep import dump
    c = load(config(tmp_path, repeats=2))
    root=tmp_path/'partial'; root.mkdir(); dump(root/'config.json',c)
    t=trials(c)[0]; folder=root/t['id']; folder.mkdir(); dump(folder/'trial.json',t)
    dump(folder/'compute-samples.json',[
        {'monotonic_seconds':1, 'counters':{'vllm:generation_tokens_total':0}},
        {'monotonic_seconds':11, 'counters':{'vllm:generation_tokens_total':500}}])
    SweepCLI().report(root)
    ranked=json.loads((root/'compute-ranking.json').read_text())
    assert ranked[0]['median_output_tokens_per_second']==50
    assert ranked[0]['measured_repeats']==1 and ranked[0]['planned_repeats']==2
    assert ranked[0]['statuses']==['interrupted']


def test_bounded_measurement_saves_tokens_after_timeout(tmp_path, monkeypatch):
    import subprocess
    import primebeaker.vllm_sweep as sweep
    values=iter([b'vllm:generation_tokens_total 0\n', b'vllm:generation_tokens_total 120\n'])
    monkeypatch.setattr(sweep,'http',lambda *a,**kw: next(values))
    def timeout(*a, **kw): raise subprocess.TimeoutExpired('bench',120)
    monkeypatch.setattr(sweep,'execute',timeout)
    result=sweep.measure_compute(1234,tmp_path,['bench'],120)
    assert result['output_tokens']==120
    assert (tmp_path/'compute-samples.json').exists()
    assert (tmp_path/'compute.json').exists()


def test_trace_replay_preserves_actual_prompts_tools_and_turn_order(tmp_path, monkeypatch):
    from primebeaker.trace_replay import run
    import io
    import urllib.request
    turns=[{'prompt':[{'role':'user','content':'actual question'}], 'response':{'usage':{'completion_tokens':3,'prompt_tokens':20}}},
           {'prompt':[{'role':'user','content':'actual question'},{'role':'assistant','content':'search'},{'role':'tool','content':'actual retrieved evidence'}],
            'response':{'usage':{'completion_tokens':5,'prompt_tokens':40}}}]
    traces=tmp_path/'traces.jsonl';traces.write_text(json.dumps({'id':'real','info':{'search_agent':{'trajectory':turns}}})+'\n')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'traces':str(traces),'entries':[{'offset':0}], 'tools':[{'type':'function','function':{'name':'search'}}]}))
    sent=[]
    def respond(request,**kwargs):
        body=json.loads(request.data);sent.append(body)
        return io.BytesIO(('data: '+json.dumps({'choices':[{'delta':{'content':'x'}}]})+'\n\ndata: '+json.dumps({'usage':{'completion_tokens':body['max_tokens']}})+'\n\ndata: [DONE]\n\n').encode())
    monkeypatch.setattr(urllib.request,'urlopen',respond)
    result=run(manifest,1234,tmp_path/'out',concurrency=1)
    assert [r['messages'] for r in sent]==[t['prompt'] for t in turns]
    assert [r['max_tokens'] for r in sent]==[3,5]
    assert all(r['tools'][0]['function']['name']=='search' for r in sent)
    assert result['total_output_tokens']==8 and result['completed']==2
    data=json.loads(manifest.read_text());data['entries'][0].update(start_turn=1,turn_count=1)
    manifest.write_text(json.dumps(data));sent.clear()
    result=run(manifest,1234,tmp_path/'subset',concurrency=1)
    assert len(sent)==1 and sent[0]['messages']==turns[1]['prompt']
    assert result['total_output_tokens']==5


def test_trace_workload_uses_replay_command(tmp_path):
    c=load(config(tmp_path,workloads=[{'name':'real','replay_manifest':'/weka/replay.json'}]))
    server,bench=commands(c,trials(c)[0],1234,tmp_path)
    assert '--manifest=/weka/replay.json' in bench
    assert '--dataset-name' not in bench


def test_recorded_tool_calls_are_converted_to_openai_schema():
    from primebeaker.trace_replay import request_body
    turn={'prompt':[{'role':'assistant','content':None,'thinking_blocks':None,
           'reasoning_content':'reason','tool_calls':[{'id':'a','name':'search','arguments':'{"query":"q"}'}]},
           {'role':'tool','tool_call_id':'a','content':'actual evidence'}],
          'response':{'usage':{'completion_tokens':10}}}
    body=request_body(turn,[])
    call=body['messages'][0]['tool_calls'][0]
    assert call=={'id':'a','type':'function','function':{'name':'search','arguments':'{"query":"q"}'}}
    assert body['messages'][0]['reasoning_content']=='reason'
    assert body['messages'][1]==turn['prompt'][1]
    assert 'function' not in turn['prompt'][0]['tool_calls'][0]


def test_prepare_manifest_is_reproducible_and_keeps_full_history(tmp_path):
    from primebeaker.trace_replay import prepare
    traces=tmp_path/'traces.jsonl'
    traces.write_text(''.join(json.dumps({'id':str(i),'info':{'search_agent':{'trajectory':[{'prompt':[{'role':'user','content':'q'}]}]*6}}})+'\n' for i in range(10)))
    tools=tmp_path/'tools.json';tools.write_text('[]')
    first=tmp_path/'one.json';second=tmp_path/'two.json'
    result=prepare(traces,tools,first,conversations=3,turns=2,runtime_traces='/data/traces.jsonl')
    prepare(traces,tools,second,conversations=3,turns=2,runtime_traces='/data/traces.jsonl')
    assert first.read_text()==second.read_text()
    assert result['conversations']==3 and result['calls']==6
    data=json.loads(first.read_text());assert data['traces']=='/data/traces.jsonl'
    with traces.open('rb') as f:
        for entry in data['entries']:
            f.seek(entry['offset']);assert json.loads(f.readline())['id']==entry['trace_id']
    with pytest.raises(FileExistsError):prepare(traces,tools,first)
