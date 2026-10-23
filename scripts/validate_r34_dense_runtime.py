"""Run in the materialized directory with the published compatible dependencies."""
import argparse,hashlib,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=a.directory.resolve();os.chdir(root);sys.path.insert(0,str(root))
os.environ['CPSF_PATCH_REPAIR_ROOT']=str(root);os.environ['CPSF_AP_ROOT']=str(root)
import numpy as np
import torch
from tools.cpsf_r34_patch_repair.model import Model
model=Model('h01_global')
folder=root/'results/r34_h01_global'
checkpoint=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
model.load_state_dict(checkpoint['model'],strict=True)
metric=json.loads((folder/'metrics.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(folder/'final.pt')==metric['final_sha256']
assert metric['identity']['protocol_sha256']==sha(root/'protocol.json')
assert metric['identity']['version']=='CPSF_R34_AP10_SUPPORT_PRESERVED_20261023_A1'
with np.load(folder/'val_predictions.npz',allow_pickle=False) as z:
 y,z=z['target'],z['logits'];e=z-y
 m=dict(mae=float(abs(e).mean()),rmse=float(np.sqrt((e**2).mean())),r2=float(1-(e**2).sum()/((y-y.mean())**2).sum()))
for k,v in m.items():assert abs(v-metric['val'][k])<1e-12
sources={}
for name,module in list(sys.modules.items()):
 if not name.startswith(('tools.','utonia.')):continue
 f=getattr(module,'__file__',None)
 if not f:continue
 f=Path(f).resolve();assert f.is_relative_to(root),(name,str(f))
 sources[name]=dict(path=str(f.relative_to(root)),sha256=sha(f))
result=dict(passed=True,strict_checkpoint_load=True,prediction_metric_recomputed=True,
 protocol_sha256=sha(root/'protocol.json'),checkpoint_sha256=sha(folder/'final.pt'),
 metrics=m,parameters=sum(p.numel() for p in model.parameters()),loaded_source_count=len(sources),
 loaded_sources=sources,torch_version=torch.__version__,new_inference_performed=False,
 original_data_or_parent_code_imported=False)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='loaded_sources'},indent=2))
