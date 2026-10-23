"""Matched final-development decision, without combining different task units."""
import numpy as np
from sklearn.metrics import average_precision_score
from .common import ROOT,TASKS,parent_for,read,save,sha,manifest

PRIMARY=dict(h01_local='macro_f1',h01_global='mae',flywire_coarse='macro_f1',flywire_fine='macro_f1',ted='f1',tlp='full_adjacency_macro_ap')

def audit(root,name,task):
    folder=root/'results'/(name+'_'+task);m=read(folder/'metrics.json');ident=m['identity']
    assert ident['version']==root.name and ident['model']==name and ident['task']==task
    assert ident['protocol_sha256']==sha(root/'protocol.json') and ident['seed']==42
    assert not ident['pretrained'] and not m['test_used'] and ident['selection']=='fixed final epoch'
    assert ident['manifest_sha256']==sha(manifest(task))
    cfg=read(root/'protocol.json')['tasks'][task]
    assert m['epochs']==cfg['epochs'] and m['steps']==cfg['updates']
    assert sha(folder/'final.pt')==m['final_sha256']
    pred=read(folder/'prediction_audit.json');assert pred['passed'] and not pred['test_used']
    for f,h in pred['files'].items():assert sha(f)==h
    rr=[r for r in read(manifest(task))['records'] if r['split']=='val']
    if task=='tlp':
        vals=[]
        for r in rr:
            with np.load(folder/'val'/(r['key']+'.npz')) as a:
                off=~np.eye(len(a['target']),dtype=bool)
                assert np.allclose(a['score'],a['score'].T,atol=2e-5)
                y=a['target'][off];s=a['score'][off]
                vals.append(float(average_precision_score(y,s)) if y.any() else 0.)
        value=float(np.mean(vals))
    else:
        with np.load(folder/'val_predictions.npz') as a:
            assert a['keys'].tolist()==[r['key'] for r in rr]
            assert np.array_equal(a['target'],np.array([r['target'] for r in rr]))
            y=a['target'];z=a['logits'];assert np.isfinite(z).all()
            if task=='h01_global':value=float(np.abs(z-y).mean())
            else:
                n=148 if task=='flywire_fine' else 6 if task=='flywire_coarse' else 2 if task=='ted' else 3
                assert z.shape==(len(y),n)
                cm=np.zeros((n,n),np.int64);np.add.at(cm,(y.astype(int),z.argmax(1)),1)
                den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(n),where=den!=0)
                value=float(f1[1] if task=='ted' else f1.mean())
    assert abs(value-m['val'][PRIMARY[task]])<1e-12,(name,task,value,m['val'])
    return m

def main():
    rows=[];waiting=[]
    for task in TASKS:
        paths=[parent_for(task)/'results'/('r28_'+task)/'metrics.json',ROOT/'results'/('r34_'+task)/'metrics.json']
        if not all(p.exists() for p in paths):
            waiting.append(task);continue
        old,new=[read(p) for p in paths];metric=PRIMARY[task]
        a,b=old['val'][metric],new['val'][metric];gain=(a-b) if task=='h01_global' else b-a
        rows.append(dict(task=task,primary=metric,r28=a,r34=b,signed_gain=gain,
            r34_no_worse=gain>=-1e-12,secondary_tlp=None if task!='tlp' else dict(
                r28_missing_edge_macro_ap=old['val']['missing_edge_macro_ap'],r34_missing_edge_macro_ap=new['val']['missing_edge_macro_ap'])))
    if waiting:
        result=dict(state='waiting_matched_results',reference='R28',candidate='R34',waiting=waiting,comparisons=rows,audited=False)
    else:
        for task in TASKS:
            parent=parent_for(task);old=audit(parent,'r28',task);new=audit(ROOT,'r34',task)
            assert read(parent/'protocol.json')['tasks'][task]==read(ROOT/'protocol.json')['tasks'][task]
            assert sha(parent/'orders'/(task+'.json'))==sha(ROOT/'orders'/(task+'.json'))
            ho=read(parent/'results'/('r28_'+task)/'history.json');hn=read(ROOT/'results'/('r34_'+task)/'history.json')
            assert len(ho)==len(hn)
            for a,b in zip(ho,hn):
                for k in ('epoch','steps','lr','sample_order_sha256'):assert a[k]==b[k],(task,k)
            assert old['identity']['initial_state_sha256']==read(ROOT/'contracts'/(task+'.json'))['r28_initial_state_sha256']
        promoted=all(r['r34_no_worse'] for r in rows) and any(r['signed_gain']>1e-12 for r in rows)
        result=dict(state='promoted' if promoted else 'retain_R28_review_tradeoffs',reference='R34' if promoted else 'R28',
            candidate='R34',comparisons=rows,audited=True,seed=42,test_used=False,
            basis='All six paired fixed-final development primary metrics; no mixed-input or test-set selection')
    save(ROOT/'reference_decision.json',result)
    return result

if __name__=='__main__':main()
