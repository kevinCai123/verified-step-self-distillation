import importlib.metadata as m, json, platform, subprocess, sys
from pathlib import Path
report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{d.metadata['Name']:d.version for d in m.distributions()}}
try:
    import torch
    report['torch_cuda']=torch.version.cuda
    report['cuda_available']=torch.cuda.is_available()
    if report['cuda_available']:
        x=torch.ones(8,device='cuda'); report['cuda_tensor_sum']=x.sum().item(); report['gpu']=torch.cuda.get_device_name()
except Exception as e: report['cuda_error']=repr(e)
report['nvidia_smi']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
Path(sys.argv[1]).write_text(json.dumps(report,indent=2)); print(json.dumps({k:v for k,v in report.items() if k!='packages'},indent=2))
