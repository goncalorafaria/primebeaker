#!/usr/bin/env python3
"""Incremental eval progress; Beaker state refreshed separately from trace counters."""
import argparse,json,time,subprocess,sys,threading
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
class Reader:
 def __init__(self):self.offset=0;self.inode=None;self.rows={}
 def update(self,p,kind):
  if not p.exists():return self.rows
  st=p.stat()
  if self.inode!=st.st_ino or st.st_size<self.offset:self.offset=0;self.rows={};self.inode=st.st_ino
  with p.open('rb') as f:
   f.seek(self.offset)
   while True:
    start=f.tell();line=f.readline()
    if not line or not line.endswith(b'\n'):break
    try:r=json.loads(line)
    except json.JSONDecodeError:raise ValueError(f'Malformed completed JSON line at {p}:{start}')
    if kind=='traces':
     key=r.get('info',{}).get('jtask',{}).get('id',r.get('id',str(start)));v={'error':bool(r.get('errors')),'tools':sum(len(n.get('message',{}).get('tool_calls') or []) for n in r.get('nodes',[]))}
    else:
     key=r['record_id'];j=r.get('judge_result') or {};v={'error':bool(r.get('error')) or bool(j.get('parse_error')) or j.get('points') is None,'points':j.get('points')}
    self.rows[key]=v;self.offset=f.tell()
  return self.rows
readers={};job_cache={};job_error='';last_poll=-1e9

job_errors={}
def jobs(config):
 receipt=Path(config['receipt'])
 try:
  exp=json.loads(receipt.read_text())[0]['id']
  cmd=['/weka/gfaria/primebeaker/.venv/bin/python','/weka/gfaria/search-agent-repr/policy-16train-16infer-cp4-mc32-resume130/beaker-auth.py','experiment','get',exp,'--format','json']
  result=subprocess.run(cmd,capture_output=True,text=True,timeout=20,check=True);data=json.loads(result.stdout)[0];cache={}
  for j in data.get('jobs',[]):
   s=j['status'];state='QUEUED'
   if s.get('started'):state='RUNNING'
   if s.get('exited'):state='EXITED '+str(s.get('exitCode',''))
   if s.get('canceled'):state='CANCELED '+s.get('canceledFor','')[:65]
   if s.get('finalized') and not s.get('canceled'):state='FINISHED' if s.get('exitCode',0)==0 else 'FAILED'
   env=j.get('execution',{}).get('spec',{}).get('envVars',[])
   out=next((x.get('value') for x in env if x['name']=='EVAL_OUTPUT_DIR'),None)
   if out:cache[config.get('aliases',{}).get(out,out)]=state
  
  shared=cache.get(str(ROOT/'step430'))
  if shared:
   cache.update({str(ROOT/f'step430-seed{seed}'):shared for seed in range(1,6)})
  job_cache.update(cache);job_errors.pop(str(receipt),None)
 except Exception as e:job_errors[str(receipt)]=f'{receipt.parent.name}: Beaker state unavailable/stale ({type(e).__name__})'

def run_rows():
 m=ROOT/'manifest.json'
 if m.exists():
  x=json.loads(m.read_text())
  if isinstance(x,list):return [(r.get("label", f'step{r["step"]}'),Path(r['output'])) for r in x]
 return [(s,ROOT/s) for s in ['original','reformatted']]

latest={}; refreshing={}; workers_started=False

def collect(name,folder):
 p=folder/'full'
 def read(file,kind):
  path=p/file;return readers.setdefault(str(path),Reader()).update(path,kind)
 tr=read('traces.jsonl','traces');gr=read('gpt5-research-judgments.jsonl','grades');good=[x for x in gr.values() if not x['error']];valid=[x for x in tr.values() if not x['error']];zero=sum(x['tools']==0 for x in valid)
 no=f'{zero}/{len(valid)} {100*zero/len(valid):.1f}%' if valid else '—';mean=f'{sum(x["points"] for x in good)/len(good):.2f}' if good else '—';rate=f'{100*sum(x["points"]>=7 for x in good)/len(good):.1f}%' if good else '—'
 ages=[time.time()-f.stat().st_mtime for f in [p/'traces.jsonl',p/'gpt5-research-judgments.jsonl'] if f.exists()];age=f'{min(ages)/60:.1f}m' if ages else '—'
 state=job_cache.get(str(folder),job_cache.get(str(ROOT/'services'),'UNKNOWN'))
 progress='DONE' if len(good)==59 and (p/'INFERENCE_COMPLETE').exists() else 'grading' if (p/'INFERENCE_COMPLETE').exists() else 'awaiting first response' if not tr else 'inference incomplete'
 return f'{name:12} {str(len(tr))+"/59":9} {str(len(good))+"/59":13} {no:17} {mean:9} {rate:8} {str(len(tr)-len(valid))+"/"+str(len(gr)-len(good)):9} {age:7} {state} / {progress}'

def worker(name,folder):
 while True:
  refreshing[name]=time.monotonic()
  try:latest[name]=(collect(name,folder),time.monotonic())
  except Exception as exc:latest[name]=(f'{name:12} ERROR: {type(exc).__name__}: {exc}',time.monotonic())
  refreshing.pop(name,None)
  time.sleep(5)

def job_worker(config):
 while True:
  jobs(config)
  time.sleep(60)

def start_workers():
 global workers_started
 if workers_started:return
 workers_started=True
 for config in json.loads((ROOT/'jobs.json').read_text()):threading.Thread(target=job_worker,args=(config,),daemon=True).start()
 for name,folder in run_rows():threading.Thread(target=worker,args=(name,folder),daemon=True).start()

def show():
 print(datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),ROOT.name)
 print(f'{"EVAL":12} {"ROLLOUTS":9} {"VALID GRADES":13} {"NO TOOLS*":17} {"MEAN/10*":9} {"PASS%*":8} {"ERR T/G":9} {"AGE":7} JOB / PROGRESS')
 for name,folder in run_rows():
  result=latest.get(name)
  if result:
   line,stamp=result;age=time.monotonic()-stamp
   print(line+(f' [snapshot {age:.0f}s old; read pending]' if age>15 else ''))
  else:
   state=job_cache.get(str(folder),job_cache.get(str(ROOT/'services'),'checking job'))
   wait=time.monotonic()-refreshing.get(name,time.monotonic())
   print(f'{name:12} {state} / reading output ({wait:.0f}s); other checkpoints refresh independently')
 print('*Partial valid-response metrics; no-tools percentage is biased toward early completions. AGE = latest file write. ERR T/G = trace/grading errors.')
 for error in list(job_errors.values()):print(error)
 print('A pending Weka read does not imply stalled inference. Ctrl+C exits.',flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--interval',type=float,default=10);parser.add_argument('--once',action='store_true');a=parser.parse_args()
 start_workers()
 if a.once:
  deadline=time.monotonic()+5
  while len(latest)<len(run_rows()) and time.monotonic()<deadline:time.sleep(.1)
 try:
  while True:
   if not a.once and sys.stdout.isatty():print('\033[2J\033[H',end='')
   show()
   if a.once:break
   time.sleep(max(1,a.interval))
 except KeyboardInterrupt:pass
