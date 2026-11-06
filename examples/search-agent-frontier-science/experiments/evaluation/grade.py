"""Regrade saved responses with Steven's original FrontierScience Research rubric."""
import asyncio, dataclasses, fcntl, json, time
from pathlib import Path
from openai import AsyncOpenAI
from research_grader_reference import RESEARCH_JUDGE_PROMPT, parse_research_judge_output
ROOT=Path('/weka/gfaria/primebeaker/examples/frontier-science-live-search')
DATA={str(r['id']):r for r in map(json.loads,(ROOT/'research-original.jsonl').open())}
BASE=Path('/weka/gfaria/search-agent-repr/evals')
DIRS=[Path(p) for p in ['/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/step430-seed1/full', '/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/step430-seed2/full', '/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/step430-seed3/full', '/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/step430-seed4/full', '/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/step430-seed5/full']]
META=dict(model='gpt-5',reasoning_effort='high',max_output_tokens=40000,method='original research rubric, partial credit, Steven prompt and parser',pass_threshold_points=7,concurrency=512)
def read_rows(path):
    if not path.exists(): return []
    text=path.read_text()
    lines=text.splitlines(keepends=True)
    return [json.loads(line) for line in lines if line.endswith('\n') and line.strip()]

async def main():
 for root in DIRS: root.mkdir(parents=True, exist_ok=True)
 lock=(Path('/weka/gfaria/search-agent-repr/evals/frontier-structured430-five-seeds-jupiter-20261105/grade.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 client=AsyncOpenAI(max_retries=3,timeout=1800,default_headers={'Accept-Encoding':'identity'})
 sem=asyncio.Semaphore(512); done={}; pending={}; tasks=set(); start=time.monotonic()
 for root in DIRS:
  p=root/'gpt5-research-judgments.jsonl';done[root]={r['record_id']:r for r in map(json.loads,p.open())} if p.exists() else {};pending[root]=set()
 async def grade(root,r):
  q=DATA[r['id']];prompt=RESEARCH_JUDGE_PROMPT.format(problem=q['question'],rubric=q['answer'],answer=r['output'])
  result=dict(record_id=r['id'],judge_metadata=META,judge_prompt=prompt,error=None)
  try:
   async with sem:
    response=await client.responses.create(model='gpt-5',input=prompt,reasoning={'effort':'high','summary':'detailed'},max_output_tokens=40000)
   result.update(judge_response=response.output_text,judge_result=dataclasses.asdict(parse_research_judge_output(response.output_text)),response_id=response.id,status=response.status,usage=response.usage.model_dump() if response.usage else None)
  except Exception as exc:
   result.update(error=str(exc),judge_response='',judge_result=dataclasses.asdict(parse_research_judge_output('')))
  with (root/'gpt5-research-judgments.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
  done[root][r['id']]=result;pending[root].discard(r['id']);print(root.parent.name,r['id'],result['judge_result'],flush=True)
 while True:
  complete=True
  for root in DIRS:
   answers=read_rows(root/'answers.jrecords.jsonl')
   for r in answers:
    if r['id'] not in done[root] and r['id'] not in pending[root]:
     pending[root].add(r['id']);task=asyncio.create_task(grade(root,r));tasks.add(task);task.add_done_callback(tasks.discard)
   traces=sum(1 for _ in (root/'traces.jsonl').open()) if (root/'traces.jsonl').exists() else 0;finished=(root/'INFERENCE_COMPLETE').exists() and traces>=59 and not pending[root]
   complete &= finished
   values=list(done[root].values());correct=sum(v['judge_result']['correct'] for v in values);points=sum(v['judge_result']['points'] or 0 for v in values)
   summary=dict(META,expected_questions=59,completed_rollouts=traces,generated_answers=len(answers),graded_answers=len(values),pending=len(pending[root]),correct=correct,accuracy_percent=100*correct/59,mean_points=points/59,parse_errors=sum(v['judge_result']['parse_error'] for v in values),request_errors=sum(bool(v.get('error')) for v in values),missing_answers=max(0,59-len(answers)),complete=finished)
   path=root/'summary-gpt5-research.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(summary,indent=2));tmp.replace(path)
  if complete:break
  if time.monotonic()-start>604800:raise TimeoutError('Incomplete after seven days')
  await asyncio.sleep(5)
 await client.close()
if __name__=='__main__':asyncio.run(main())
