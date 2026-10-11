import argparse,json,shutil,time
import torch
from .common import *

def write(task,records):
    path=ROOT/'data'/task/'manifest.json';p=verify()
    ids={}
    for mode in {r['modal'] for r in records}:
        rr=[r for r in records if r['modal']==mode]
        assert len(rr)==p['tasks'][task]['train']+p['tasks'][task]['val']
        keys=[r['key'] for r in rr];assert len(keys)==len(set(keys))
        a={r['cell'] for r in rr if r['split']=='train'};b={r['cell'] for r in rr if r['split']=='val'};assert not a&b
        ids[mode]=sorted((r['key'],r['split'],r['target']) for r in rr)
    assert all(v==ids['joint'] for v in ids.values())
    save(path,dict(records=records,protocol_sha256=sha(ROOT/'protocol.json'),split_sha256=digest(ids['joint'])))
    save(path.with_name('READY.json'),dict(ready=True,manifest_sha256=sha(path),test_used=False,records=len(records)))

def h01():
    base=json.loads((H01/'cache/manifest.json').read_text())['records']
    res=json.loads((SCALE/'cache/manifest.json').read_text())['records']
    for task in ('local','global'):
        rows=[dict(r) for r in base if r['task']==task]
        if task=='global':rows += [dict(r) for r in res if r['task']==task and r['modal'] in ('thumbnail','uniform_budget')]
        assert all(r['split'] in ('train','val') for r in rows)
        for r in rows:assert sha(r['path'])==r['sha256']
        write('h01_'+task,rows)

def flywire():
    from tools.cpsf_flywire_v23 import data as adapter
    torch.set_num_threads(1);start=time.time()
    joint=json.loads((FW/'cache/manifest.json').read_text())['records']
    joint=[r for r in joint if r['split'] in ('train','val')]
    old=json.loads((OLD/'cache/manifest.json').read_text())['records'];lookup={(r['root_id'],r['variant']):r for r in old}
    candidates={r['root_id']:r for r in json.loads((FW/'candidates.json').read_text())}
    records=[]
    for i,row in enumerate(joint):
        rid=row['root_id']
        for mode in ('joint','point','skeleton','thumbnail','uniform_budget'):
            if mode=='joint':r=dict(row)
            elif row['legacy']:r=dict(lookup[(rid,mode)])
            else:
                dest=ROOT/'cache/flywire'/mode/(rid+'.pt.gz');meta=dest.with_suffix('.json')
                if dest.exists():
                    r=json.loads(meta.read_text());assert sha(dest)==r['sha256']
                else:
                    assert shutil.disk_usage(ROOT).free>35*2**30,'Preserve disk reserve'
                    adapter.DATA=FW/'geometry';q=json.loads((adapter.DATA/'qa'/(rid+'.json')).read_text());assert q['status']=='ok'
                    source=dict(candidates[rid],sample_sha256=q['sample_sha256']);g=adapter.build(source,mode)
                    assert not any(k in g for k in ('target','coarse_target','fine_target','cell_type','root_id','label'))
                    if mode=='point':assert not g['has_g'].any() and g['raw_nodes']==0
                    if mode=='skeleton':assert not g['has_p'].any() and g['patches']==0
                    adapter.store(dest,g);r=adapter.record(source,mode,dest,g);save(meta,r)
            records.append(dict(r,modal=mode,task='flywire_coarse',key=rid,cell=rid,target=row['coarse_target'],legacy=row['legacy']))
        if i%10==0:
            save(ROOT/'prepare_status.json',dict(state='building_flywire',done=i+1,total=len(joint),seconds=time.time()-start));print(i+1,len(joint),flush=True)
    write('flywire_coarse',records);save(ROOT/'prepare_status.json',dict(state='complete',seconds=time.time()-start,test_used=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=['h01','flywire']);a=parser.parse_args();verify()
    h01() if a.dataset=='h01' else flywire()
