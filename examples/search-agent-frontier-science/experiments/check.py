#!/usr/bin/env python3
import ast,json,subprocess,sys,tempfile,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='search-agent-example-') as temp:
 for mode in ['training','evaluation']:
  out=Path(temp)/mode
  command=[sys.executable,str(ROOT/'prepare.py'),'--config',str(ROOT/f'{mode}.toml'),'--out',str(out)]
  subprocess.run(command,check=True)
  assert subprocess.run(command,capture_output=True).returncode!=0,'Must refuse nonempty run dir'
  for p in out.rglob('*.py'):ast.parse(p.read_text())
  for p in out.rglob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True)
  if mode=='training':
   c=tomllib.loads((out/'train.toml').read_text());assert c['max_steps']==600 and c['resume']['dir'].endswith('step_450')
   assert c['trainer']['model']['optimization_dtype']==c['trainer']['model']['reduce_dtype']=='bfloat16'
   assert c['trainer']['model']['cp']==4 and c['trainer']['model']['dp_replicate']==1
   assert c['inference']['vllm']['quantization']=='fp8_per_block' and c['inference']['vllm']['kv_cache_dtype']=='fp8'
   assert c['inference']['vllm']['data_parallel_size']==3 and c['trainer']['optim']['lr']==1e-6
   spec=json.loads((out/'policy.beaker.json').read_text());assert sum(t['resources']['gpuCount'] for t in spec['tasks'])==16
   assert all(t['context']['priority']=='high' for t in spec['tasks'])
   assert str(out/'runtime') in (out/'start-policy.sh').read_text()
   judges=json.loads((out/'judges-jupiter.json').read_text());assert len(judges['tasks'])==4 and all(t['resources']['gpuCount']==4 and t['context']['priority']=='urgent' for t in judges['tasks'])
  else:
   spec=json.loads((out/'eval.beaker.json').read_text());assert spec['tasks'][0]['resources']['gpuCount']==4
   assert len(json.loads((out/'manifest.json').read_text()))==5
   assert str(out/'runtime') in (out/'start.sh').read_text()
   assert str(out) in (out/'grade.py').read_text()
   assert 'resources' not in json.loads((out/'grade.beaker.json').read_text())['tasks'][0]
   # Test the actual default-format override with a fake parent, no API requests.
   source=ast.parse((out/'runtime/eval_search_server.py').read_text());cls=next(n for n in source.body if isinstance(n,ast.ClassDef));method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_headers_for_request')
   module=ast.Module(body=[ast.ClassDef(name='Child',bases=[ast.Name(id='Parent',ctx=ast.Load())],keywords=[],body=[method],decorator_list=[])],type_ignores=[]);ast.fix_missing_locations(module)
   from types import SimpleNamespace
   class Parent:
    def _headers_for_request(self,request):return {'Accept':'application/json','X-Return-Format':'markdown'}
   ns={'Parent':Parent,'os':SimpleNamespace(environ={'JINA_FORMAT':'default'})};exec(compile(module,'headers','exec'),ns)
   assert ns['Child']()._headers_for_request(SimpleNamespace(mode='url'))=={'Accept':'application/json'}
   assert ns['Child']()._headers_for_request(SimpleNamespace(mode='query'))['X-Return-Format']=='markdown'
  print('PASS',mode)
# Load the packaged profile with the merged repository's actual profile loader.
import importlib.util, types, hashlib
profile_module=ROOT.parents[2]/'src/primebeaker/judge/model_profiles.py'
spec=importlib.util.spec_from_file_location('example_merged_profiles',profile_module)
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
profiles=ROOT/'judge/profiles';raw=json.loads((profiles/'pengin9b_rl400.json').read_text())
profile=module.load_model_profile(raw['model'],types.SimpleNamespace(model_profiles_dir=str(profiles), **raw))
assert profile.rubric_mode=='pengin_submit' and profile.enable_thinking
assert profile.max_failed_rubric_fraction==0.2 and profile.terminal_truncation==2000
assert Path(profile.prompt_template_path).is_file()
assert hashlib.sha256(Path(profile.prompt_template_path).read_bytes()).hexdigest()=='c4c138d6bd91b2800a53eeaaee7f454078f410fa3156733133aeedae891dcf30'
print('PASS merged Pengin profile contract')
print('No jobs submitted.')
