"""Configuration-only experiment; immutable per-task parent implementations."""
import json,hashlib,math
from pathlib import Path
import torch
from tools.cpsf_fixed_nredd.common import load,move,digest,modal
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
ROOT=Path(__file__).resolve().parents[2]
TASKS=('ted','tlp')
ARMS=('full','mean_only','no_fusion','skeleton','point','no_scale','no_position')
CONFIGS=('lr_half','slow_decay')
def job_key(task,arm,config):return '_'.join((task,config,arm))
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    assert sha(ROOT/'PROTOCOL.md')==p['document_sha256']
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,('source drift',n)
    for path,h in p['references'].items():assert sha(path)==h,('parent drift',path)
    return p

def rows(task,arm):
    p=json.loads((ROOT/'protocol.json').read_text());path=p['manifests'][task][modal(arm)]
    d=json.loads(Path(path).read_text());assert d['complete']
    rr=d['records'];assert all(r['split'] in ('train','val') for r in rr)
    return rr,sha(path)

def Model(task,arm):
    if task=='ted':
        from tools.cpsf_fixed_nredd.model import Model as M
        return M(task,arm)
    from tools.cpsf_deterministic.model import Model as M
    return M(task,arm,'voxel')

def optimizer(model,task,config,cfg):
    assert config in ('baseline',)+CONFIGS
    lr=cfg['lr']*(.5 if config=='lr_half' else 1.)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=cfg['weight_decay'])
    if task=='ted':
        divisor=3. if config=='slow_decay' else 1.
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:1/(1+epoch/divisor))
    else:sched=torch.optim.lr_scheduler.StepLR(opt,40 if config=='slow_decay' else 20,.1)
    return opt,sched
