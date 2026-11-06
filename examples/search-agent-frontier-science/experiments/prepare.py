#!/usr/bin/env python3
"""Render an isolated experiment directory. No submission or running-job changes."""
import argparse, ast, json, re, shutil, tomllib, uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD_T='/weka/gfaria/search-agent-repr/policy-resume450-to600-bf16-cp4-12infer-20261105'
OLD_E='/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105'
OLD_RT='/weka/gfaria/search-agent-repr/evals/frontier-structured350-420-training-tools-seed1-20261104/runtime'
def dump(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def envset(task,**values):
 env={x['name']:x for x in task.get('envVars',[])}
 for k,v in values.items():env[k]={'name':k,'value':str(v)}
 task['envVars']=list(env.values())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 cfg=tomllib.loads(a.config.read_text());c=cfg['experiment'];out=a.out.resolve()
 if out.exists() and any(out.iterdir()):raise SystemExit('Output directory must be new or empty; refusing to alter an existing run')
 kind=c['kind'];assert kind in ['training','evaluation'];assert re.fullmatch(r'[A-Za-z0-9_-]+',c['name'])
 out.mkdir(parents=True,exist_ok=True)
 shutil.copytree(ROOT/kind,out,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
 shutil.copytree(ROOT/'judge',out/'judge',dirs_exist_ok=True)
 replacements={OLD_T:str(out),OLD_E:str(out),OLD_RT:str(out/'runtime')}
 for p in out.rglob('*'):
  if p.suffix in ['.sh','.py','.json','.toml'] and p.is_file():
   s=p.read_text()
   for old,new in replacements.items():s=s.replace(old,new)
   p.write_text(s)
 launch=uuid.uuid4().hex;specname='policy' if kind=='training' else 'eval'
 spec=json.loads((out/f'{specname}.beaker.json').read_text())
 for task in spec['tasks']:
  task['constraints']['cluster']=[c['cluster']];task['context'].update(priority=c['priority'],minRuntime=c['min_runtime'])
 if kind=='training':
  runtime_source=Path(OLD_T)/'runtime'
  # Exact source snapshot, validated before copying, excludes bytecode.
  import hashlib
  for record in json.loads((ROOT/'training/runtime-manifest.json').read_text()):
   rel=record['file'].removeprefix('training/runtime/')
   if hashlib.sha256((runtime_source/rel).read_bytes()).hexdigest()!=record['sha256']:
    raise SystemExit('Runtime fingerprint changed: '+rel)
  shutil.copytree(runtime_source,out/'runtime',dirs_exist_ok=True,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
  shutil.copytree(out/'runtime-overrides',out/'runtime',dirs_exist_ok=True)
  run=c['name'];output=out/'outputs';s=(out/'train.toml').read_text()
  previous=tomllib.loads(s)
  s=s.replace(previous['output_dir'],str(output)).replace(previous['run']['name'],run)
  s=re.sub(r'^max_steps = .*',f'max_steps = {c["max_steps"]}',s,flags=re.M)
  s=re.sub(r'^lr = .*',f'lr = {c["learning_rate"]}',s,flags=re.M)
  s=s.replace(previous['resume']['dir'],c['checkpoint'])
  (out/'train.toml').write_text(s)
  assert Path(c['checkpoint']).is_dir(),c['checkpoint']
  for t in spec['tasks']:
   envset(t,OUTPUT_DIR=output,CONFIG_PATH=out/'train.toml',LAUNCH_ID=launch,RENDEZVOUS_DIR=out/'rendezvous'/launch,WANDB_RUN_ID=launch,WANDB_SHARED_RUN_ID=launch)
  p=out/'start-policy.sh';p.write_text(p.read_text().replace('search-agent-mc32-resume450-to600-bf16-20261105',run))
  p=out/'runtime/prime_rl/trainer/rl/train.py';p.write_text(p.read_text().replace('        assert config.optim.lr == 1e-6\n',''))
  judge=json.loads((out/'judges-jupiter.json').read_text());assert cfg['judge']['engines']==4,'This recipe is validated for four TP4 engines'
  for i,t in enumerate(judge['tasks']):
   t['command']=['bash',str(out/'judge/start-jupiter-tp4.sh')];t['context']['priority']=cfg['judge']['priority'];t['constraints']['cluster']=[cfg['judge']['cluster']]
  dump(out/'judges-jupiter.json',judge)
  p=out/'watch-judges.py';s=p.read_text();s=re.sub(r"CLI=\[.*\]", "CLI=__import__('shlex').split(os.environ.get('BEAKER_COMMAND','beaker'))",s,count=1);s=s.replace("'ai2/oe-agents'",repr(cfg['judge']['workspace']));p.write_text(s)
 else:
  import hashlib
  for record in json.loads((ROOT/'data-manifest.json').read_text()):
   source=Path(record['source'])
   if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:
    raise SystemExit('Evaluation dataset changed: '+str(source))
   shutil.copy2(source,out/record['file'].removeprefix('evaluation/'))
  step=int(c['step']);seeds=c['seeds'];assert seeds and len(set(seeds))==len(seeds) and all(type(x)==int and x>=0 for x in seeds)
  assert c['jina_format'] in ['default','markdown'];assert Path(c['model']).is_dir(),c['model']
  assets=ROOT.parents[2]/'src/primebeaker/environments/frontier_science/resources/science'
  shutil.copytree(assets,out/'runtime/assets',dirs_exist_ok=True)
  settings=json.loads((out/'runtime/settings.json').read_text());settings.update(cfg['sampling'])
  settings['template_path']=str(out/'runtime/assets/prompt.json')
  dump(out/'runtime/settings.json',settings)
  for t in spec['tasks']:envset(t,EVAL_OUTPUT_DIR=out/f'step{step}',MODEL_PATH=c['model'],CHECKPOINT_PATH=c['checkpoint'],EVAL_SEED=seeds[0],JINA_FORMAT=c['jina_format'])
  p=out/'start.sh';s=p.read_text();s=s.replace('for seed in 1 2 3 4 5; do','for seed in '+' '.join(map(str,seeds))+'; do');s=s.replace('search:frontier430-jupiter-five-',f'search:{c["name"]}-{launch[:8]}-');p.write_text(s)
  # Dependencies are deliberately external; do not silently copy mutable site-packages.
  (out/'runtime/deps').symlink_to('/weka/gfaria/primebeaker/examples/frontier-science-live-search/runtime-web-v2/deps',target_is_directory=True)
  p=out/'grade.py';s=p.read_text();a0=s.index('DIRS=');a1=s.index('\nMETA=',a0);s=s[:a0]+'DIRS=[Path(p) for p in '+repr([str(out/f'step{step}-seed{x}/full') for x in seeds])+']'+s[a1:];s=s.replace("ROOT=Path('/weka/gfaria/primebeaker/examples/frontier-science-live-search')",f'ROOT=Path({str(out)!r})');p.write_text(s)
  p=out/'grade.sh';p.write_text(p.read_text().replace('ROOT=/weka/gfaria/primebeaker/examples/frontier-science-live-search','ROOT='+str(out)))
  grade=json.loads((out/'grade.beaker.json').read_text());dump(out/'grade.beaker.json',grade)
  dump(out/'manifest.json',[dict(step=step,seed=x,label=f'{step}-seed{x}',output=str(out/f'step{step}-seed{x}'),model=c['model']) for x in seeds])
  dump(out/'jobs.json',[dict(receipt=str(out/'eval.submitted.json'),aliases={})])
  p=out/'watch.py';s=p.read_text().replace("ROOT/'step430'",f"ROOT/'step{step}'").replace("ROOT/f'step430-seed{seed}'",f"ROOT/f'step{step}-seed{{seed}}'").replace('range(1,6)',repr(seeds));p.write_text(s)
 # Point registered vLLM wrapper to the packaged judge code.
 p=out/'judge/start-jupiter-tp4.sh';s=p.read_text().replace('"$ROOT/judge/serve-registered-vllm.py"','"'+str(out/'judge/serve-registered-vllm.py')+'"');p.write_text(s)
 dump(out/f'{specname}.beaker.json',spec);dump(out/'experiment.json',cfg)
 for p in out.rglob('*.py'):ast.parse(p.read_text(),filename=str(p))
 print(out);print('Prepared only. Submit with submit.py --run-dir',out)
if __name__=='__main__':main()
