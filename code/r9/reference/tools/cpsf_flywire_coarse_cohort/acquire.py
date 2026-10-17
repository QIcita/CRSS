"""Use original public acquisition/QA on new roots only, under a new root."""
import argparse,json,time,shutil
from tools.cpsf_flywire_20260930.build import one,save
from .plan import ROOT


def main(pilot):
    plan=json.loads((ROOT/'plan.json').read_text());rows=[r for r in json.loads((ROOT/'candidates.json').read_text()) if not r['legacy']]
    if pilot:rows=[r for r in rows if r['root_id'] in plan['pilot_new_training_roots']]
    geo=ROOT/'geometry'
    for name in ('raw','qa','samples'):(geo/name).mkdir(parents=True,exist_ok=True)
    started=time.time();results=[]
    for i,row in enumerate(rows):
        assert shutil.disk_usage(ROOT).free>50*2**30,'Preserve 50GiB free; no automatic deletion'
        for attempt in range(3):
            try:q=one(row,geo);break
            except Exception as e:
                if isinstance(e,AssertionError):raise
                if attempt<2:time.sleep(2)
                else:
                    q=dict(root_id=row['root_id'],status='failed',error=repr(e));save(geo/'qa'/(row['root_id']+'.json'),q)
        results.append(dict(root_id=row['root_id'],status=q['status']))
        save(ROOT/('pilot_acquire_status.json' if pilot else 'acquire_status.json'),dict(state='acquiring',processed=i+1,total=len(rows),ok=sum(r['status']=='ok' for r in results),seconds=time.time()-started,last_root=row['root_id']))
        print('ACQUIRE',i+1,len(rows),row['root_id'],q['status'],flush=True)
    save(ROOT/('pilot_acquire.json' if pilot else 'acquire.json'),dict(complete=True,results=results,seconds=time.time()-started))
    if pilot:assert all(r['status']=='ok' for r in results),'Pilot QA failed; inspect before full acquisition'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();main(a.pilot)
