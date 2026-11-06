#!/usr/bin/env python3
"""Render explicit Beaker specs; submission is a separate, deliberate command."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tomllib
import uuid

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PHASES = ('base-0-400', 'base-400-800', 'instruct-0-800')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def image(uri):
    if uri.startswith('beaker://'):
        return {'beaker': uri.removeprefix('beaker://')}
    if uri.startswith('docker://'):
        return {'docker': uri.removeprefix('docker://')}
    raise ValueError('Images require beaker:// or docker:// prefixes')


def prepare(args):
    target = Path(args.directory).expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError('Use an empty launch directory; existing launches are never overwritten')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.run_name):
        raise ValueError('run-name must be a filename-safe slug')
    if args.phase == 'base-400-800' and not args.wandb_run_id:
        raise ValueError('Resume requires the original --wandb-run-id; keep --run-name and --output-dir unchanged')
    runtime = json.loads(Path(args.runtime).read_text())
    model = args.judge_model or {
        'moe': 'Qwen/Qwen3.5-35B-A3B',
        'qwen9b': 'Qwen/Qwen3.5-9B',
        'pengin9b': '/weka/gfaria/launches/pengin-9b-rl-benchmarks-100step-20261025/9b/step_400/hf-checkpoint',
    }[args.judge]
    if args.judge == 'pengin9b' and not Path(model).is_dir():
        raise ValueError('Pengin requires an existing --judge-model checkpoint directory')
    target.mkdir(parents=True, exist_ok=True)
    (target / 'profiles').mkdir()
    for path in (HERE / 'scripts').iterdir():
        shutil.copy2(path, target / path.name)
    shutil.copy2(HERE / 'policy-system-prompt.txt', target / 'policy-system-prompt.txt')
    run_id = args.wandb_run_id or uuid.uuid4().hex
    service = 'judge:medical-science-' + args.judge + '-' + args.run_name
    substitutions = {
        '__OUTPUT_DIR__': str(Path(args.output_dir).expanduser().resolve()),
        '__RUN_NAME__': args.run_name,
        '__TRAIN_DATASET__': args.train_data,
        '__EVAL_DATASET__': args.eval_data,
        '__POLICY_SYSTEM_PROMPT__': str(target / 'policy-system-prompt.txt'),
        '__JUDGE_MODEL__': model, '__JUDGE_SERVICE__': service,
        '__WANDB_ENTITY__': args.wandb_entity,
    }
    text = (HERE / 'configs' / (args.phase + '.toml')).read_text()
    for key, value in substitutions.items():
        text = text.replace(key, json.dumps(value)[1:-1])
    text = text.replace('"moe35b-judge"', json.dumps(args.judge + '-judge'))
    if args.judge == 'pengin9b':
        text = text.replace('"judge-no-thinking"', '"judge-thinking"')
    if args.judge != 'moe':
        text = text.replace('\"judge-output64k\"', '\"judge-output16k\"')
    config = tomllib.loads(text)
    (target / 'train.toml').write_text(text)
    profile = json.loads((HERE / 'profiles' / (args.judge + '.json')).read_text())
    profile['model'] = model
    if args.judge == 'pengin9b':
        profile['local_search_model_path'] = args.local_search_model
    template = Path(profile['prompt_template_path']).name
    shutil.copy2(HERE / 'profiles' / template, target / 'profiles' / template)
    profile['prompt_template_path'] = str(target / 'profiles' / template)
    write(target / 'profiles' / (args.judge + '.json'), profile)
    settings = dict(judge=args.judge, judge_model=model, judge_context=args.judge_context,
                    judge_service=service, run_name=args.run_name, wandb_run_id=run_id,
                    output_dir=substitutions['__OUTPUT_DIR__'], phase=args.phase,
                    local_search_model=args.local_search_model,
                    runtime=runtime, source_checkout=str(REPO),
                    pengin_prompt_sha256=(hashlib.sha256((target / 'profiles' / template).read_bytes()).hexdigest()
                                           if args.judge == 'pengin9b' else None))
    write(target / 'launch.json', settings)

    def task(role):
        spec = runtime['trainer' if role == 'trainer' else 'judge']
        python = spec['python'] if role != 'model' else 'python3'
        paths = [str(REPO / 'src'), runtime['literegistry_runtime'], *spec['pythonpath']]
        environment = {
            'WORKER_ROLE': role, 'WORKER_PYTHON': python, 'LAUNCH_DIR': str(target),
            'PYTHONPATH': ':'.join(paths), 'HEAD_REGISTRY': args.head_registry,
            'TOOL_HEAD_REGISTRY': args.head_registry, 'EXPERIMENT_NAME': args.run_name,
            'OUTPUT_DIR': settings['output_dir'], 'CONFIG_PATH': str(target / 'train.toml'),
            'VLLM_API_KEY': 'prime-rl', 'WANDB_RUN_ID': run_id,
            'WANDB_SHARED_RUN_ID': run_id, 'PRL_RUN_ID': run_id,
            'WANDB_ENTITY': args.wandb_entity,
            'WANDB_RESUME': 'allow' if args.phase == 'base-400-800' else 'never',
        }
        env = [{'name': k, 'value': v} for k, v in environment.items()]
        if role == 'trainer':
            env += [{'name': 'WANDB_API_KEY', 'secret': args.wandb_secret},
                    {'name': 'HF_TOKEN', 'secret': args.hf_secret}]
        cluster = 'ai2/holmes' if role == 'trainer' else ('ai2/jupiter' if role == 'api' else args.judge_cluster)
        name = ('trainer-' + str(config['deployment']['num_train_gpus']) + 'train'
                + str(config['deployment']['num_infer_gpus']) + 'infer') if role == 'trainer' else 'judge-' + role
        result = {
            'name': name, 'image': image(spec['image']),
            'command': ['bash', str(target / 'bootstrap.sh')], 'envVars': env,
            'datasets': [{'mountPath': '/weka', 'source': {'weka': args.weka_dataset}}],
            'result': {'path': '/tmp/beaker-result'}, 'hostNetworking': True,
            'context': {'priority': 'high', 'minRuntime': '4h', 'autoResume': role == 'model'},
            'constraints': {'cluster': [cluster]},
            'propagateFailure': False, 'propagatePreemption': False,
        }
        if role != 'api':
            result['resources'] = {'gpuCount': 8 if role == 'trainer' else 4}
        return result

    for role in ('model', 'api', 'trainer'):
        write(target / (role + '.beaker.json'), {
            'version': 'v2', 'description': 'Medical/science rubric RL: ' + args.judge + ' / ' + args.phase,
            'tasks': [task(role)]})
    write(target / 'workspaces.json', {
        'model': 'ai2/oe-agents-holmes' if args.judge_cluster == 'ai2/holmes' else 'ai2/oe-agents',
        'api': 'ai2/oe-agents', 'trainer': 'ai2/oe-agents-holmes'})
    print(json.dumps({'directory': str(target), 'run_directory': str(Path(settings['output_dir']) / args.run_name),
                      'wandb_run_id': run_id, 'submitted': False}, indent=2))


def submit(args):
    target = Path(args.directory).resolve()
    workspaces = json.loads((target / 'workspaces.json').read_text())
    roles = ('model', 'api', 'trainer') if args.part == 'all' else (args.part,)
    for role in roles:
        receipt = target / ('submitted-' + role + '.json')
        intent = target / ('intent-' + role + '.json')
        if receipt.exists() or intent.exists():
            raise ValueError(f'{role} already submitted or submission uncertain; inspect its receipt/intent')
        # Persist intent before the network call. Never duplicate an uncertain create.
        with intent.open('x') as stream:
            json.dump({'role': role, 'workspace': workspaces[role]}, stream)
        result = subprocess.run(['beaker', 'experiment', 'create', str(target / (role + '.beaker.json')),
                                 '--workspace', workspaces[role], '--format', 'json'],
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        value = json.loads(result.stdout)
        write(receipt, value)
        print(role, json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--directory', required=True)
    p.add_argument('--run-name', required=True)
    p.add_argument('--judge', choices=('moe', 'qwen9b', 'pengin9b'), required=True)
    p.add_argument('--phase', choices=PHASES, default='base-0-400')
    p.add_argument('--runtime', default=str(HERE / 'runtime.json'))
    p.add_argument('--output-dir', default='/weka/gfaria/prime_sft_beaker_scratch/ugr-runs/medical-science-examples')
    p.add_argument('--train-data', default='/weka/gfaria/prime_sft_beaker_scratch/data/rubrichub-signed-penalties-20260917')
    p.add_argument('--eval-data', default='/weka/gfaria/prime_sft_beaker_scratch/data/rubrichub-signed-penalties-20260917-eval-256/validation.jsonl')
    p.add_argument('--head-registry', default='sqlite:///weka/gfaria/search-agent-repr/registry/control.sqlite3')
    p.add_argument('--judge-model')
    p.add_argument('--judge-context', type=int, default=131072)
    p.add_argument('--judge-cluster', choices=('ai2/jupiter', 'ai2/holmes'), default='ai2/jupiter')
    p.add_argument('--local-search-model', default='localsearch:bc-rl-v1')
    p.add_argument('--weka-dataset', default='oe-adapt-default')
    p.add_argument('--wandb-entity', default='graf')
    p.add_argument('--wandb-run-id')
    p.add_argument('--wandb-secret', default='gfaria_WANDB_API_KEY')
    p.add_argument('--hf-secret', default='HF_TOKEN')
    p.set_defaults(action=prepare)
    p = commands.add_parser('submit')
    p.add_argument('--directory', required=True)
    p.add_argument('--part', choices=('all', 'model', 'api', 'trainer'), default='all')
    p.set_defaults(action=submit)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
