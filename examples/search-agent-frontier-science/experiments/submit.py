#!/usr/bin/env python3
"""Explicit submission; reruns never silently resubmit an existing receipt."""
import argparse,json,os,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();r=a.run_dir.resolve()
c=json.loads((r/'experiment.json').read_text())['experiment'];cli=shlex.split(os.environ.get('BEAKER_COMMAND','beaker'))
def submit(kind,workspace):
 receipt=r/f'{kind}.submitted.json'
 if receipt.exists():return json.loads(receipt.read_text())
 spec=r/('policy.beaker.json' if kind=='policy' else f'{kind}.beaker.json')
 name=c['name']+'-'+kind
 intent=r/f'{kind}.submit-intent'
 if intent.exists():
  raise SystemExit(f'{intent} exists without receipt; recover by Beaker experiment name {name} before retrying')
 intent.write_text(name)
 data=json.loads(subprocess.run(cli+['experiment','create',str(spec),'--workspace',workspace,'--name',name,'--format','json'],check=True,capture_output=True,text=True).stdout)
 receipt.write_text(json.dumps(data,indent=2));print('https://beaker.org/ex/'+data[0]['id']);return data
if c['kind']=='training':
 data=submit('policy',c['workspace']);(r/'submitted.json').write_text(json.dumps(data,indent=2))
 # Trigger itself takes a lock. A second invocation cannot start a second active watcher.
 with (r/'judge-trigger.log').open('a') as log:
  subprocess.Popen([os.sys.executable,'-u',str(r/'watch-judges.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 print('Judge trigger launched; inspect judge-trigger.json for readiness.')
else:
 submit('eval',c['workspace']);submit('grade','ai2/oe-agents')
