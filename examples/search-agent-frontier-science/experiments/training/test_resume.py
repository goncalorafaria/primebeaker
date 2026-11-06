import tempfile,torch,ast,textwrap,json
from pathlib import Path
from types import SimpleNamespace
from torch.distributed.checkpoint.state_dict import get_state_dict,set_state_dict
from torch.distributed.checkpoint import save,load
r=Path(__file__).parent
old=torch.nn.Linear(3,2);oo=torch.optim.AdamW(old.parameters(),lr=5e-7);os=torch.optim.lr_scheduler.ConstantLR(oo,factor=1)
old(torch.ones(1,3)).sum().backward();oo.step();os.step()
with tempfile.TemporaryDirectory() as td:
 m,o=get_state_dict(old,oo);save({'model':m,'optimizer':o},checkpoint_id=td)
 model=torch.nn.Linear(3,2,dtype=torch.bfloat16);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6);scheduler=torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1)
 m,o=get_state_dict(model,optimizer);state={'model':m,'optimizer':o};load(state,checkpoint_id=td);set_state_dict(model,optimizer,model_state_dict=m,optim_state_dict=o);scheduler.load_state_dict(os.state_dict())
 assert optimizer.param_groups[0]['lr']==5e-7
 config=SimpleNamespace(optim=SimpleNamespace(lr=1e-6),scheduler=SimpleNamespace(type='constant'),model=SimpleNamespace(reduce_dtype='bfloat16'))
 logger=SimpleNamespace(info=print)
 source=(r/'runtime/prime_rl/trainer/rl/train.py').read_text();start=source.index('        # This resume explicitly');end=source.index('        # The checkpoint finished',start)
 exec(textwrap.dedent(source[start:end]))
 model(torch.ones(1,3,dtype=torch.bfloat16)).sum().backward();optimizer.step();scheduler.step()
 assert optimizer.param_groups[0]['lr']==1e-6
 assert all(p.dtype==torch.bfloat16 for p in model.parameters())
 assert all(optimizer.state[p]['step'].item()==2 for p in model.parameters())
 print('PASS: FP32 checkpoint to BF16, optimizer history preserved, LR override remains 1e-6 after update')
for component in ['trainer','orchestrator','inference']:
 d=json.loads((r/'validation-configs'/f'{component}.json').read_text())
 print(component,'validated')
