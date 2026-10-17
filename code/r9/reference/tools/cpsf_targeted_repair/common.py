import os,json,hashlib
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
from tools.cpsf_fixed_nredd.common import load,move
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/root/megrez-tmp')
TED=BASE/'CPSF_NREDD_CONFIG_ONLY_20261009'
TLP=BASE/'CPSF_TLP_DIRECTIONAL_STAGE6_20261009'
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    for path,h in p['sources'].items():assert sha(ROOT/path)==h,('source drift',path)
    for path,h in p['references'].items():assert sha(path)==h,('reference drift',path)
    assert sha(ROOT/'development_split.json')==p['development_split_sha256']
    return p
def records(task):return json.loads((ROOT/'manifests'/f'{task}.json').read_text())['records']
def sidepath(task,key):return ROOT/'sidecars'/task/(key.replace(':','_')+'.npz')
def item_load(r,task,arm):
    item=load(r['path'])
    if arm=='repair':
        with np.load(sidepath(task,r['key'])) as z:
            field='pf_directional' if task=='ted' else 'pair_geometry'
            item=dict(item,g=dict(item['g'],**{field:torch.from_numpy(z[field].copy())}))
    return item
class RepairCache(Cache):
    def __init__(self,rr,task,arm):
        super().__init__(budget=512*2**20);self.lookup={r['path']:r for r in rr};self.task=task;self.arm=arm
    def get(self,k,make=None):return super().get(k,lambda:item_load(self.lookup[k],self.task,self.arm))
