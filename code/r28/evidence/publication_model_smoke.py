from pathlib import Path,PurePosixPath
import tarfile,sys,os,json,tempfile,hashlib
root=Path(tempfile.mkdtemp(prefix='cpsf_r28_publication_'))
with tarfile.open('/tmp/cpsf_r28_publication_source_overlay_20261020.tar.gz') as t:
 for m in t.getmembers():
  q=PurePosixPath(m.name);assert m.isfile() and not q.is_absolute() and '..' not in q.parts
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(m).read())
sys.path.insert(0,str(root));sys.path.append('/root/megrez-tmp/CPSF_OFFICIAL_CORE_20261016_R6_A1/runtime_deps')
import torch
torch.set_num_threads(1)
from tools.cpsf_raw_point_attention.model import Model as H01
from tools.cpsf_r28_expansion.model import Model as Ext
out=dict(publication_root=str(root),source_overlay_sha256=hashlib.sha256(Path('/tmp/cpsf_r28_publication_source_overlay_20261020.tar.gz').read_bytes()).hexdigest(),tasks={},no_training=True,no_gpu_allocation=True)
for task in ['h01_local','h01_global','flywire_coarse','ted','tlp']:
 m=H01(task,'r9_lpa_attention') if task.startswith('h01') else Ext(task)
 item=dict(parameters=sum(v.numel() for v in m.parameters()),trainable_parameters=sum(v.numel() for v in m.parameters() if v.requires_grad),module_source=str(sys.modules[type(m).__module__].__file__),official_lpa_source=str(root/'vendor/SPoTr/openpoints/models/backbone/spotr.py'),official_utonia_source=str(root/'vendor/Utonia/utonia/model.py'))
 if task.startswith('h01'):weight=Path('/root/megrez-tmp/CPSF_RAW_POINT_ATTENTION_20261019_R28_A1/results')/('r9_lpa_attention_'+task)/'final.pt'
 elif task=='tlp':weight=Path('/root/megrez-tmp/CPSF_R28_TASK_EXPANSION_20261020_A2/results/r28_lpa_attention_tlp/final.pt')
 else:weight=None
 if weight:
  c=torch.load(weight,map_location='cpu',weights_only=False);r=m.load_state_dict(c['model'],strict=True);assert not r.missing_keys and not r.unexpected_keys;item.update(strict_load=True,weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest())
 else:item['strict_load']=None
 out['tasks'][task]=item;del m
print(json.dumps(out))
