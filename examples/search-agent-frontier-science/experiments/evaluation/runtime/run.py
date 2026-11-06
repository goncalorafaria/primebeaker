"""JTC search rollouts with the training prompt and canonical live-web clients.

Requires a fixed exported policy served at --inference-url, and a gateway with
Serper/Jina search + terminal workers. Does not use stored candidate answers as
references. Rubric-ready JRecords are written separately for grouped scoring.
"""
import os
import argparse
import asyncio
import json
import sys
from collections import Counter
from web_metrics import trace_metrics, save_summary
sys.path.append('/weka/gfaria/primebeaker/.venv/lib/python3.12/site-packages')
from pathlib import Path

ROOT = Path(__file__).resolve().parent

async def main(args):
    from jtc.common.workflows import search_agent_rollout_workflow as w
    config = json.loads(Path(args.settings).read_text())
    template = Path(config['template_path'])
    if not template.is_absolute():
        template = ROOT / template
        if not template.is_file() and config['template_path'] == 'assets/prompt.json':
            # Direct use from a checkout: resolve the canonical versioned assets.
            template = next((parent / 'src/primebeaker/environments/frontier_science/resources/science/prompt.json'
                             for parent in ROOT.parents
                             if (parent / 'src/primebeaker/environments/frontier_science/resources/science/prompt.json').is_file()), template)
    for name in ('prompt.json', 'search-description.txt', 'webterminal-description.txt', 'final-warning.txt'):
        if not template.with_name(name).is_file():
            raise FileNotFoundError(f'Missing search-agent prompt asset: {template.with_name(name)}')
    config['template_path'] = str(template)
    config['search_model_path'] = os.environ.get('EVAL_SEARCH_MODEL_PATH', config['search_model_path'])
    rows = [json.loads(line) for line in (ROOT / 'questions.jsonl').open()]
    if args.max_rows:
        rows = rows[:args.max_rows]
    tasks = [dict(question=row['input'], answer=None,
                  prompt=w._template_prompt(config['template_path'], row['input']),
                  jtask=dict(kind='JTask', **row)) for row in rows]
    api = w._environment_api()
    parent = api['SearchAgentEnv']
    class TrainingSearchEnv(parent):
        async def search(self, query: str) -> str:
            return await super().search(query=query, num_results=10)

        async def webterminal(self, command: str, asset_store) -> str:
            return await super().webterminal(command=command, asset_store=asset_store)

    assets = Path(config['template_path']).parent
    TrainingSearchEnv.search.__doc__ = (assets / 'search-description.txt').read_text()
    TrainingSearchEnv.webterminal.__doc__ = (assets / 'webterminal-description.txt').read_text()
    api['SearchAgentEnv'] = TrainingSearchEnv
    env, _ = w.build_env(
        tasks=tasks, gateway_url=args.gateway_url,
        local_search_model_path=config['search_model_path'], harness='webterminal',
        max_turns=config['max_turns'], max_tool_calls=config['max_tool_calls'],
        timeout=config['tool_timeout'], max_retries=config['tool_attempts'], terminal_truncation=config['terminal_truncation'],
        visit_max_content_length=config['visit_max_content_length'])
    # Match the actual native training tool schemas, not the legacy callable converter.
    # Snapshot: resume420 training trace, verifiers 0.3.2.dev136.
    from verifiers.types import Tool
    training_tools = json.loads((ROOT / 'training-tools.json').read_text())
    env.tool_defs = [Tool(**{k: v for k, v in tool.items() if k != 'type'})
                     for tool in training_tools]
    from gateway_fetch_recovery import install_gateway_recovery
    install_gateway_recovery(env.webterminal_client.fetch_client)
    # Match the plain-answer training harness, including its final tool warning.
    env.final_tool_call_warning = Path(config['template_path']).with_name('final-warning.txt').read_text().strip()
    env.timeout_seconds = config['rollout_timeout']
    env.set_max_total_completion_tokens(config['max_total_completion_tokens'])
    if args.check_only:
        print('Environment constructed; canonical clients and training prompt loaded', flush=True)
        print(env.tools, flush=True)
        return
    client, model = w.build_rollout_client(
        model=args.model, inference_url=args.inference_url,
        openai_api_key=None, openai_base_url=None,
        timeout=config['rollout_timeout'], max_retries=3)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'traces.jsonl').exists():
        raise FileExistsError('Use a new output directory to preserve prior traces')
    (output / 'manifest.json').write_text(json.dumps(dict(config=config, model=args.model,
        inference_url=args.inference_url, gateway_url=args.gateway_url, questions=len(rows), seed=args.seed, tool_definitions=training_tools,
        tool_definition_source='native-training-resume420'), indent=2))
    sampling = dict(temperature=config['temperature'], max_tokens=config['max_tokens'], tool_choice='auto')
    if args.seed is not None:
        sampling['seed'] = args.seed
    semaphore = asyncio.Semaphore(config['batch_size'])
    web_counts = Counter()
    with (output / 'traces.jsonl').open('w') as traces, (output / 'answers.jrecords.jsonl').open('w') as answers:
        async def one(index):
            async with semaphore:
                trace = await w.rollout_one(env, client, model, tasks[index], index,
                    sampling)
            # Exact match against another model's answer is not a research score.
            trace['rewards'] = {}
            trace.get('info', {}).get('search_agent', {}).pop('rewards', None)
            trace.setdefault('info', {})['evaluation'] = 'rubrics_pending'
            audit = trace_metrics(trace)
            trace.setdefault('info', {})['web_metrics'] = dict(audit)
            web_counts.update(audit)
            web_counts['completed_rollouts'] += 1
            save_summary(output / 'web-metrics-summary.json', web_counts)
            traces.write(json.dumps(trace) + '\n'); traces.flush()
            messages = [node.get('message', {}) for node in trace.get('nodes', [])]
            answer = w.final_assistant_text(messages)
            if answer and not trace.get('errors'):
                record = dict(rows[index], output=answer, reference=None, __jtc_kind='JRecord')
                answers.write(json.dumps(record) + '\n'); answers.flush()
            print(f'Finished {index + 1}/{len(rows)}; errors={trace.get("errors")}', flush=True)
        await asyncio.gather(*(one(i) for i in range(len(rows))))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('model', 'inference-url', 'gateway-url', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--settings', default=str(ROOT / 'settings.json'))
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-rows', type=int)
    parser.add_argument('--check-only', action='store_true')
    asyncio.run(main(parser.parse_args()))
