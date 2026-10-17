"""One fixed final evaluation after the full registered matrix is frozen."""
import json
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
from .data import ROOT,load
from .model import Model
from .train import dataset,evaluate,add_fine

def run(task,arm):
    setup();lock=json.loads((ROOT/'ALL_TRAINED.json').read_text());key=task+'_'+arm
    out=ROOT/'results'/key
    assert sha(out/'final.pt')==lock['checkpoints'][key]
    dest=out/'test';assert not dest.exists();dest.mkdir()
    rows,labels,ready=dataset(arm);test=[r for r in rows if r['split']=='test'];gallery=[r for r in rows if r['split']=='train']
    assert len(test)==1573
    for r in test:assert sha(r['path'])==r['sha256']
    model=Model(len(labels[task]),arm).cuda();ck=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
    assert ck['protocol_sha256']==sha(ROOT/'protocol.json');model.load_state_dict(ck['model'])
    cache=Cache(budget=512*2**20);metrics,x,z=evaluate(model,test,cache,task,len(labels[task]),dest,'test')
    if task=='fine':
        with np.load(out/'train_features.npz') as f:
            assert f['root_id'].tolist()==[r['root_id'] for r in gallery];gx=f['x']
        add_fine(metrics,x,z,test,gallery,gx,labels,rows,dest,'test')
    save(dest/'metrics.json',dict(task=task,arm=arm,seed=42,final_sha256=lock['checkpoints'][key],test=metrics))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--arm',required=True);a=p.parse_args();run(a.task,a.arm)
