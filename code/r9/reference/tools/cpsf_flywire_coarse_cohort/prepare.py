import argparse,json,shutil,time
from collections import Counter,defaultdict
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import save,sha
from tools.cpsf_flywire_v23 import data as adapter
from .plan import ROOT,OLD,DATA


def build(row,folder):
    dest=ROOT/folder/(row['root_id']+'.pt.gz');meta=dest.with_suffix('.json')
    if dest.exists():
        r=json.loads(meta.read_text());assert sha(dest)==r['sha256'];return r
    assert shutil.disk_usage(ROOT).free>45*2**30
    q=json.loads((ROOT/'geometry/qa'/(row['root_id']+'.json')).read_text());assert q['status']=='ok'
    row=dict(row,sample_sha256=q['sample_sha256'])
    adapter.DATA=ROOT/'geometry';g=adapter.build(row,'joint')
    assert not any(k in g for k in ('coarse_target','fine_target','cell_type','root_id','label'))
    assert np.isclose(float(g['surface_mass'].sum()),q['total_area_um2'],rtol=2e-5)
    adapter.store(dest,g);r=adapter.record(row,'joint',dest,g)
    r.update(legacy=False,fine_available=False,relation_pairs=len(g['pr']))
    save(meta,r);return r


def main(pilot):
    torch.set_num_threads(1);plan=json.loads((ROOT/'plan.json').read_text());candidates=json.loads((ROOT/'candidates.json').read_text())
    assert sha(ROOT/'candidates.json')==plan['candidate_sha256']
    if pilot:
        rows=[r for r in candidates if r['root_id'] in plan['pilot_new_training_roots']]
        records=[build(r,'pilot_cache') for r in rows]
        save(ROOT/'pilot_geometry.json',dict(passed=True,records=records,test_used=False));return
    assert json.loads((ROOT/'acquire.json').read_text())['complete']
    qa={};seen={};exclusions=[];new=[]
    for r in candidates:
        if r['legacy']:
            q=json.loads((DATA/'qa'/(r['root_id']+'.json')).read_text());seen[q['absolute_mesh_vertices_sha256']]=r
        else:
            q=json.loads((ROOT/'geometry/qa'/(r['root_id']+'.json')).read_text());qa[r['root_id']]=q
            if q['status']!='ok':exclusions.append(dict(root_id=r['root_id'],reason=q['status']));continue
            new.append(r)
    groups=defaultdict(list)
    for r in new:groups[qa[r['root_id']]['absolute_mesh_vertices_sha256']].append(r)
    retained=[]
    for fingerprint,rs in groups.items():
        rs=sorted(rs,key=lambda r:r['root_id'])
        if fingerprint in seen:
            exclusions.extend(dict(root_id=r['root_id'],reason='duplicate_of_legacy',canonical_root=seen[fingerprint]['root_id']) for r in rs);continue
        if len({r['coarse_target'] for r in rs})>1:
            exclusions.extend(dict(root_id=r['root_id'],reason='duplicate_conflicting_coarse_labels') for r in rs);continue
        retained.append(rs[0]);exclusions.extend(dict(root_id=r['root_id'],reason='duplicate_new',canonical_root=rs[0]['root_id']) for r in rs[1:])
    save(ROOT/'exclusions.json',exclusions)
    for c in range(6):
        total=sum(not r['legacy'] and r['coarse_target']==c for r in candidates)
        good=sum(r['coarse_target']==c for r in retained)
        assert good>=.95*total,('Class-specific acquisition/QA exclusion exceeds 5%',c,good,total)
    oldm=json.loads((OLD/'cache/manifest.json').read_text());records=[dict(r,legacy=True,fine_available=True) for r in oldm['records'] if r['variant']=='joint']
    assert len(records)==8121
    for i,r in enumerate(sorted(retained,key=lambda r:r['root_id'])):
        rec=build(r,'cache/joint');records.append(rec)
        save(ROOT/'prepare_status.json',dict(state='building',new_done=i+1,new_total=len(retained),total_records=len(records)))
        if i%25==0:print('CACHE',i+1,len(retained),flush=True)
    records.sort(key=lambda r:r['root_id']);assert len({r['root_id'] for r in records})==len(records)
    groups={s:{r['root_id'] for r in records if r['split']==s} for s in ('train','val','test')}
    assert not groups['train']&groups['val'] and not groups['train']&groups['test'] and not groups['val']&groups['test']
    m=dict(records=records,labels=oldm['labels'],plan_sha256=sha(ROOT/'plan.json'),source_old_manifest_sha256=sha(OLD/'cache/manifest.json'))
    save(ROOT/'cache/manifest.json',m)
    save(ROOT/'cache/READY.json',dict(ready=True,records=len(records),new_records=len(retained),manifest_sha256=sha(ROOT/'cache/manifest.json'),counts={s:len(v) for s,v in groups.items()},
        per_class={c:dict(Counter(r['split'] for r in records if r['coarse_target']==i)) for i,c in enumerate(plan['coarse_labels'])},old_split_preserved=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
