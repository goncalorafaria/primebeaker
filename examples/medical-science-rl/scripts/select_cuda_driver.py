"""Select a usable CUDA driver library before importing vLLM or PrimeRL."""
import os,sys,subprocess
from pathlib import Path
original=[p for p in os.environ.get('LD_LIBRARY_PATH','').split(':') if p and '/compat' not in p and '/stubs' not in p]
host=':'.join(dict.fromkeys(['/usr/local/nvidia/lib64','/usr/local/nvidia/lib',*original,'/usr/local/cuda/lib64']))
compat='/usr/local/cuda-13.0/compat:'+host
probe='''import torch,ctypes
from pathlib import Path
print('TORCH',torch.__version__,'CUDA',torch.version.cuda,flush=True)
p=Path('/proc/driver/nvidia/version');print('KERNEL_DRIVER',p.read_text() if p.exists() else 'unavailable',flush=True)
lib=ctypes.CDLL('libcuda.so.1');print('LOADED_LIBCUDA',[x.strip() for x in open('/proc/self/maps') if 'libcuda.so' in x],flush=True)
x=torch.ones(1,device='cuda');torch.cuda.synchronize();print('CUDA_ALLOCATION_OK',torch.cuda.get_device_name(0),x.item(),flush=True)
'''
for label,path in [('host-driver',host),('cuda13-compat',compat)]:
 env=os.environ.copy();env['LD_LIBRARY_PATH']=path;env['VLLM_ENABLE_CUDA_COMPATIBILITY']='0'
 print('CUDA_DRIVER_PROBE',label,path,file=sys.stderr,flush=True)
 try:p=subprocess.run([sys.executable,'-c',probe],env=env,text=True,capture_output=True,timeout=120)
 except subprocess.TimeoutExpired:
  print('CUDA_DRIVER_PROBE_TIMEOUT',label,file=sys.stderr,flush=True);continue
 print(p.stdout,file=sys.stderr,flush=True);print(p.stderr,file=sys.stderr,flush=True)
 if p.returncode==0:
  print('CUDA_DRIVER_SELECTED',label,file=sys.stderr,flush=True);print(path,flush=True);break
else:raise SystemExit('Neither mounted host driver nor compatibility driver can allocate a CUDA tensor; refusing startup')
