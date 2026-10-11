import json,time,argparse,shutil
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import SOURCE,load,save,sha
from .data import whole_geometry,local_geometry,regions
ROOT=Path(__file__).resolve().parents[2]
ORIGINAL=Path('/root/megrez-tmp/CPSF_V21_SEGMENT_UNITS_20260930')

def store(path,g):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(g,tmp);tmp.replace(path)

def main(smoke=False):
    torch.set_num_threads(2)
    split=json.loads((ORIGINAL/'split_manifest.json').read_text())
    rows=split['rows']['outer_train']+split['rows']['outer_val']
    globals=json.loads((SOURCE/'manifest.json').read_text())['cells']
    if smoke:
        rows=[next(r for r in split['rows']['inner_train'] if r['target']==c) for c in (0,1,2)]
        globals=[max((r for r in globals if r['split']=='train'),key=lambda r:r['skeleton_nodes'])]
    cache=ROOT/('smoke_cache' if smoke else 'cache');records=[];start=time.time()
    for task,items in [('local',rows),('global',globals)]:
        for row in items:
            cell=row['cell'];key=row['sample'] if task=='local' else cell
            if task=='local':
                assert sha(row['path'])==row['sha256'];old=load(row['path']);query=old['geometry']['s'][old['query']].numpy()
                target=row['target'];source_hash=row['sha256'];part='train' if cell in split['inner_train_cells']+split['inner_val_cells'] else 'val'
            else:
                path=SOURCE/'inputs'/(cell+'.npz');source_hash=sha(path)
                with np.load(SOURCE/'targets'/(cell+'.npz')) as y:target=float(y['excitatory_input_fraction'])
                part=row['split'];query=None
            for modal in ('joint','point','skeleton'):
                dest=cache/task/modal/(key+'.pt')
                if not dest.exists():
                    assert shutil.disk_usage(ROOT).free>20*2**30
                    if task=='local':g=local_geometry(old['geometry'],modal)
                    else:
                        with np.load(path) as x:g=whole_geometry(x,modal)
                    data=regions(g,modal,query);store(dest,data);del g,data
                records.append(dict(task=task,modal=modal,key=key,cell=cell,split=part,target=target,path=str(dest),sha256=sha(dest),source_sha256=source_hash))
            save(cache/'status.json',dict(state='building',task=task,cell=cell,records=len(records),seconds=time.time()-start))
            print(task,key,len(records),flush=True)
    save(cache/'manifest.json',dict(records=records,seed=42,smoke=smoke,source_split_sha256=sha(ORIGINAL/'split_manifest.json')))
    save(cache/'READY.json',dict(ready=True,smoke=smoke,records=len(records),manifest_sha256=sha(cache/'manifest.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:main(a.smoke)
    except BaseException as e:save(ROOT/'prepare_failure.json',dict(error=repr(e),smoke=a.smoke));raise
