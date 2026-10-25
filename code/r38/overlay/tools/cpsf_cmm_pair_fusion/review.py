"""Independently recompute final development metrics and paired identities."""
import numpy as np
from .common import ROOT,PARENT,GEOMETRY,TASKS,MODEL,read,save,sha,verify


def metric(task,y,z):
    if task=='h01_global':
        err=z-y
        return dict(mae=float(np.abs(err).mean()),rmse=float(np.sqrt((err**2).mean())),
            r2=float(1-(err**2).sum()/((y-y.mean())**2).sum()))
    pred=z.argmax(1);cm=np.zeros((3,3),dtype=np.int64)
    np.add.at(cm,(y.astype(int),pred),1)
    den=cm.sum(0)+cm.sum(1);f1=np.divide(2*np.diag(cm),den,out=np.zeros(3),where=den!=0)
    return dict(macro_f1=float(f1.mean()),accuracy=float(np.trace(cm)/cm.sum()),confusion_matrix=cm.tolist())


def audit(root,model,task):
    folder=root/'results'/(model+'_'+task)
    if not (folder/'metrics.json').exists():return None
    r=read(folder/'metrics.json');ident=r['identity'];protocol=read(root/'protocol.json');cfg=protocol['tasks'][task]
    assert ident['version']==root.name and ident['model']==model and ident['task']==task
    assert ident['seed']==42 and not ident['pretrained'] and not ident['test_used']
    assert ident['protocol_sha256']==sha(root/'protocol.json')
    assert r['epochs']==cfg['epochs'] and r['steps']==cfg['updates']
    assert r['final_sha256']==sha(folder/'final.pt')
    manifest=GEOMETRY/'data'/task/'manifest.json';rr=read(manifest)['records'];va=[r for r in rr if r['split']=='val']
    assert ident['manifest_sha256']==sha(manifest)
    hist=read(folder/'history.json');orders=read(ROOT/'orders'/(task+'.json'))
    assert len(hist)==cfg['epochs']
    assert [h['sample_order_sha256'] for h in hist]==[o['sha256'] for o in orders]
    with np.load(folder/'val_predictions.npz') as a:
        assert a['keys'].tolist()==[r['key'] for r in va]
        np.testing.assert_array_equal(a['target'],np.asarray([r['target'] for r in va]))
        val=metric(task,a['target'],a['logits'])
    for k,v in val.items():
        if isinstance(v,list):assert v==r['val'][k]
        else:assert abs(v-r['val'][k])<1e-12,(task,k,v,r['val'][k])
    return dict(passed=True,metrics=val,epochs=r['epochs'],steps=r['steps'],
        final_sha256=r['final_sha256'],predictions_sha256=sha(folder/'val_predictions.npz'),protocol_sha256=ident['protocol_sha256'])


def main():
    verify();results={}
    for task in TASKS:
        results[task]={m:audit(root,m,task) for root,m in ((ROOT,MODEL),(PARENT,'r34'),(GEOMETRY,'r28'))}
    complete=all(results[t][MODEL] for t in TASKS);paired=all(all(v.values()) for v in results.values())
    differences={}
    if complete:
        for t in TASKS:
            k='mae' if t=='h01_global' else 'macro_f1';own=results[t][MODEL]['metrics'][k]
            differences[t]={m:own-results[t][m]['metrics'][k] for m in ('r34','r28') if results[t][m]}
    state='awaiting_candidate_results' if not complete else 'awaiting_matched_reference' if not paired else 'ready_for_comparison'
    report=dict(state=state,results=results,primary_deltas=differences,selection='fixed final epoch development',test_used=False,
        reference='R28 remains reference; this two-task experiment does not promote a six-task model',next_stage_started=False)
    save(ROOT/'independent_review.json',report);return report

if __name__=='__main__':main()
