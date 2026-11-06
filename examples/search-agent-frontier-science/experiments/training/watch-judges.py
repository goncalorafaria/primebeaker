import json,subprocess,time,fcntl,datetime,os
from pathlib import Path
R=Path(__file__).resolve().parent
TRAIN=json.loads((R/'submitted.json').read_text())[0]['id']
CLI=['/weka/gfaria/primebeaker/.venv/bin/python','/weka/gfaria/search-agent-repr/policy-16train-16infer-cp4-mc32-resume130/beaker-auth.py']
def call(*args):
 p=subprocess.run(CLI+list(args)+['--format','json'],capture_output=True,text=True,check=True,timeout=45)
 return json.loads(p.stdout)
def save(status,**kw):
 value=dict(status=status,time=datetime.datetime.now(datetime.timezone.utc).isoformat(),training=TRAIN,**kw)
 p=R/'judge-trigger.tmp';p.write_text(json.dumps(value,indent=2));p.replace(R/'judge-trigger.json');print(value,flush=True)
lock=(R/'judge-trigger.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(R/'judge-trigger.pid').write_text(str(os.getpid()))
receipt=R/'judges-jupiter.submitted.json'
name='pengin-resume450-jupiter-'+TRAIN.lower()[-8:]
while True:
 try:
  jobs=call('experiment','get',TRAIN)[0].get('jobs',[])
  latest={}
  for j in sorted(jobs,key=lambda j:j['status'].get('created','')):latest[j['name']]=j
  stopped=[j for j in latest.values() if any(k in j['status'] for k in ['exited','canceled','finalized'])]
  if stopped:
   if receipt.exists():
    jid=json.loads(receipt.read_text())[0]['id']
    subprocess.run(CLI+['experiment','stop',jid],capture_output=True,text=True,check=True,timeout=45)
    jj=call('experiment','get',jid)[0].get('jobs',[])
    if any(not any(k in j['status'] for k in ['exited','canceled','finalized']) for j in jj):
     save('waiting_for_judge_shutdown',judge=jid);time.sleep(30);continue
   save('training_stopped_judges_stopped');break
  trainer=latest.get('trainer')
  if not trainer or not trainer['status'].get('started'):
   save('armed_waiting_for_trainer');time.sleep(30);continue
  if not receipt.exists():
   intent=R/'judges-jupiter.intent'
   if intent.exists():data=call('experiment','get','goncalof/'+name)
   else:
    intent.write_text(name)
    data=call('experiment','create',str(R/'judges-jupiter.json'),'--workspace','ai2/oe-agents','--name',name)
   receipt.write_text(json.dumps(data,indent=2))
  save('judges_submitted_monitoring_training',judge=json.loads(receipt.read_text())[0]['id'],gpus=16)
 except Exception as exc:save('retrying',error=str(exc))
 time.sleep(30)
