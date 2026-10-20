import numpy as np
from .common import *

def check_one(arm,d):
    f=ROOT/'results'/job_key(arm,d);m=json.loads((f/'metrics.json').read_text());h=json.loads((f/'history.json').read_text())
    i=m['identity'];assert i['protocol_sha256']==sha(ROOT/'protocol.json') and i['descriptor']==d
    assert i['seed']==42 and not m['test_used'] and not m['smoke']
    assert m['epochs']==80 and m['optimizer_steps']==12480 and len(h)==80 and sha(f/'final.pt')==m['final_sha256']
    parent=parent_result(arm,d);pi=json.loads((parent/'identity.json').read_text());ph=json.loads((parent/'history.json').read_text())
    assert i['initial_state_sha256']==pi['initial_state_sha256']
    assert [x['sample_order_sha256'] for x in h]==[x['sample_order_sha256'] for x in ph]
    audit=json.loads((f/'independent_missing_edge_audit.json').read_text())
    assert audit['passed'] and audit['blocks']==62 and audit['macro_ap']==m['val']['missing_edge_macro_ap']
    rr,_=rows('tlp',arm);assert [b['key'] for b in m['val']['blocks']]==[r['key'] for r in rr if r['split']=='val']
    return f,m,h

def replay_gate():
    verify();checks={}
    for arm in ('full','skeleton'):
        d='voxel'
        f,m,h=check_one(arm,d);parent=parent_result(arm,d)
        pm=json.loads((parent/'metrics.json').read_text());ph=json.loads((parent/'history.json').read_text())
        new=torch.load(f/'final.pt',map_location='cpu',weights_only=False)['model']
        old=torch.load(parent/'final.pt',map_location='cpu',weights_only=False)['model']
        assert new.keys()==old.keys() and all(torch.equal(v,old[k]) for k,v in new.items()),('endpoint weight replay failed',d)
        assert [x['loss'] for x in h]==[x['loss'] for x in ph],('endpoint loss replay failed',d)
        assert m['val']==pm['val'],('endpoint metric replay failed',d)
        files=sorted(f.glob('prediction_*.npz'));assert len(files)==62
        for path in files:
            with np.load(path) as a,np.load(parent/path.name) as b:
                assert a.files==b.files and all(np.array_equal(a[k],b[k]) for k in a.files)
        checks[arm]=dict(weights_equal=True,loss_history_equal=True,predictions_equal=True,metrics_equal=True,macro_ap=m['val']['missing_edge_macro_ap'])
    result=dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),epochs=80,seed=42,checks=checks,
        macro_ap={d:v['macro_ap'] for d,v in checks.items()},comparison='stage6 voxel full and skeleton replay vs frozen stage4',test_used=False)
    save(ROOT/'runtime/replay_gate.json',result);return result

def main():
    p=verify();gate=replay_gate();results={};scores={};provenance={}
    for key,ref in p['references'].items():
        m=json.loads((ANCHOR/'results'/key/'metrics.json').read_text())
        scores[key]=m['val']['missing_edge_macro_ap'];provenance[key]=dict(origin='stage4_frozen_reference',experiment=ANCHOR.name,metrics_sha256=ref['files']['metrics.json'])
    for j in p['jobs']:
        a,d=j['arm'],j['descriptor'];f,m,_=check_one(a,d);key=job_key(a,d)
        results[key]=m;scores[key]=m['val']['missing_edge_macro_ap'];provenance[key]=dict(origin='stage6_trained',experiment=ROOT.name,metrics_sha256=sha(f/'metrics.json'))
    differences={d:{a:scores[job_key(a,d)]-scores[job_key(a,'voxel')] for a in ARMS if a!='skeleton'} for d in ('directional_xyz','directional')}
    criteria={d:dict(full_over_voxel=scores[job_key('full',d)]>scores[job_key('full','voxel')],full_over_skeleton=scores[job_key('full',d)]>scores[job_key('skeleton','voxel')],full_over_variants={a:scores[job_key('full',d)]>scores[job_key(a,d)] for a in ('mean_only','no_fusion','no_scale','point')}) for d in ('directional_xyz','directional')}
    save(ROOT/'analysis.json',dict(primary='Missing-edge macro AP',scores=scores,provenance=provenance,new_minus_voxel=differences,criteria=criteria,replay_gate=gate,test_used=False))
    save(ROOT/'summary.json',dict(results=results,stage6_trained_jobs=len(results),references=p['references'],replay_gate=gate,matched_order=True,test_used=False,analysis_sha256=sha(ROOT/'analysis.json')))
    print(json.dumps(dict(scores=scores,criteria=criteria)),flush=True)
if __name__=='__main__':main()
