"""Verify and pin completed parent manifests. Never writes into data symlinks."""
import argparse,fcntl,time
from .common import ROOT,PARENT,FINE,TASKS,PARENT_SHA,FINE_SHA,read,save,sha,manifest,verify

def main(task):
    lockpath=ROOT/'input_audits'/(task+'.lock');lockpath.parent.mkdir(exist_ok=True)
    lock=open(lockpath,'a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=verify();rr=read(manifest(task))['records']
    source=(read(FINE/'cohort.json')['records'] if task=='flywire_fine' else
            read(PARENT.parent/'CPSF_R28_TASK_EXPANSION_20261020_A2/data'/task/'manifest.json')['records'])
    assert [r['key'] for r in rr]==[r['key'] for r in source],task
    cfg=p['tasks'][task];tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
    assert len(tr)==cfg['train'] and len(va)==cfg['val'] and len(tr)+len(va)==len(rr)
    assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va),'cell split overlap'
    started=time.time();checked=0
    for i,(r,s) in enumerate(zip(rr,source)):
        for k in ('key','cell','split','target'):assert r[k]==s.get(k),(task,k,r['key'])
        assert r['protocol_sha256']==(FINE_SHA if task=='flywire_fine' else PARENT_SHA)
        a=r['observation_audit'];assert a['no_legacy_sample_used'] and not a['legacy_budget_used'] and a['tiles']==0 and a['views']==1
        for f,h in [('path','sha256'),('feature_path','feature_sha256')]:assert sha(r[f])==r[h];checked+=1
        if task=='flywire_fine':
            assert r['fine_target']==r['target']==s['fine_target'] and r['coarse_target']==s['coarse_target']
            original=read(r['parent_record_path']);assert sha(r['parent_record_path'])==r['parent_record_sha256']
            assert r['path']==original['path'] and r['sha256']==original['sha256']
        if i%250==0:save(ROOT/'input_audits'/(task+'_progress.json'),dict(processed=i+1,total=len(rr),state='verifying'))
    save(ROOT/'input_audits'/(task+'.json'),dict(passed=True,task=task,records=len(rr),train=len(tr),development=len(va),
        manifest_path=str(manifest(task)),manifest_sha256=sha(manifest(task)),protocol_sha256=sha(ROOT/'protocol.json'),
        checked_cache_and_sidecar_files=checked,source_lineage='unchanged parent records; exact cache hashes reverified; raw source files not reread',
        split_and_labels_unchanged=True,test_used=False,geometry_copied=False,seconds=time.time()-started))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--task',required=True,choices=TASKS);a=a.parse_args();main(a.task)
