import numpy as np
from .common import *

def check_one(arm,d):
    f=ROOT/'results'/job_key(arm,d);m=json.loads((f/'metrics.json').read_text());h=json.loads((f/'history.json').read_text())
    i=m['identity'];assert i['protocol_sha256']==sha(ROOT/'protocol.json') and i['descriptor']==d
    assert i['seed']==42 and not m['test_used'] and not m['smoke']
    assert m['epochs']==80 and m['optimizer_steps']==12480 and len(h)==80
    assert sha(f/'final.pt')==m['final_sha256']
    parent=PARENT/'results'/('tlp_'+parent_arm(arm))
    parent_i=json.loads((parent/'identity.json').read_text());parent_h=json.loads((parent/'history.json').read_text())
    assert i['initial_state_sha256']==parent_i['initial_state_sha256']
    assert [x['sample_order_sha256'] for x in h]==[x['sample_order_sha256'] for x in parent_h]
    a=json.loads((f/'independent_missing_edge_audit.json').read_text())
    assert a['passed'] and a['blocks']==62 and a['macro_ap']==m['val']['missing_edge_macro_ap']
    rr,_=rows('tlp',arm);assert [b['key'] for b in m['val']['blocks']]==[r['key'] for r in rr if r['split']=='val']
    return f,m,h

def replay_gate():
    verify();a,ma,ha=check_one('skeleton','voxel');b,mb,hb=check_one('skeleton','neighborhood')
    wa=torch.load(a/'final.pt',map_location='cpu',weights_only=False)['model']
    wb=torch.load(b/'final.pt',map_location='cpu',weights_only=False)['model']
    assert wa.keys()==wb.keys() and all(torch.equal(v,wb[k]) for k,v in wa.items()),'Repaired skeleton weight replay failed'
    assert [r['loss'] for r in ha]==[r['loss'] for r in hb],'Repaired skeleton loss replay failed'
    assert ma['val']==mb['val'],'Repaired skeleton metrics replay failed'
    paths=sorted(a.glob('prediction_*.npz'));assert len(paths)==62
    for path in paths:
        with np.load(path) as x,np.load(b/path.name) as y:
            assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
    result=dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),epochs=80,seed=42,
                weights_equal=True,loss_history_equal=True,predictions_equal=True,metrics_equal=True,
                macro_ap=ma['val']['missing_edge_macro_ap'],comparison='same repaired kernel; descriptor switch on skeleton',test_used=False)
    save(ROOT/'runtime/replay_gate.json',result);return result

def main():
    p=verify();gate=replay_gate();results={};scores={}
    for job in p['jobs']:
        arm,d=job['arm'],job['descriptor'];_,m,_=check_one(arm,d);k=job_key(arm,d)
        results[k]=m;scores[k]=m['val']['missing_edge_macro_ap']
    differences={arm:scores[job_key(arm,'neighborhood')]-scores[job_key(arm,'voxel')]
                 for arm in ARMS if arm!='shuffle_descriptor'}
    criteria={d:{arm:scores[job_key('full',d)]>scores[job_key(arm,d)]
                for arm in ARMS if arm!='full' and job_key(arm,d) in scores} for d in DESCRIPTORS}
    save(ROOT/'analysis.json',dict(primary='Missing-edge macro AP',scores=scores,
        neighborhood_minus_voxel=differences,full_over_variants=criteria,replay_gate=gate,
        historical_stage2_stage3_not_pooled=True,test_used=False))
    save(ROOT/'summary.json',dict(results=results,matched_order=True,replay_gate=gate,test_used=False,analysis_sha256=sha(ROOT/'analysis.json')))
    print(json.dumps(dict(scores=scores,neighborhood_minus_voxel=differences)),flush=True)

if __name__=='__main__':main()
