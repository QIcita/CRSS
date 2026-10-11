"""Same-case effect decomposition; no test access or new checkpoint selection."""
import numpy as np
from .common import *

def get_predictions(path,task):
    path=Path(path);f=path/'val_predictions.npz'
    if f.exists():
        d=np.load(f);return d['key'],d['target'],d['prediction_or_logits']
    d=np.load(path/'val_features.npz');return d['root_id'],d['target'],d['logits']

def classification(y,z,k):
    if not len(y):return dict(n=0)
    pred=z.argmax(1);cm=np.bincount(y.astype(int)*k+pred,minlength=k*k).reshape(k,k)
    support=cm.sum(1);present=support>0;f1=2*cm.diagonal()/np.maximum(cm.sum(0)+support,1)
    return dict(n=len(y),accuracy=float((pred==y).mean()),support=support.tolist(),
        present_class_macro_f1=float(f1[present].mean()),per_class_f1=[float(f1[i]) if present[i] else None for i in range(k)],
        note='Descriptive subsets only; absent classes omitted, primary metric unchanged')

def main():
    p=verify();all_results={}
    for ref in p['references'].values():
        for name,h in ref['hashes'].items():assert sha(Path(ref['path'])/name)==h
    for task in TASKS:
        rr,_=rows(task,'full');train=[r for r in rr if r['split']=='train'];val=[r for r in rr if r['split']=='val']
        ids,y,base=get_predictions(ROOT/'references'/task,task);assert ids.tolist()==[r['key'] for r in val]
        data={}
        for j in p['jobs']:
            if j['task']!=task:continue
            path=ROOT/'results'/(task+'_'+j['arm']);metrics=path/'metrics.json'
            assert metrics.exists();m=json.loads(metrics.read_text());assert m['final_sha256']==sha(path/'final.pt')
            keys,yy,z=get_predictions(path,task);assert np.array_equal(keys,ids) and np.array_equal(yy,y)
            data[j['arm']]=(z,m['val'])
        for arm in ('mean_only','no_fusion','point','skeleton'):
            ref=p['references'][task+'/'+arm];keys,yy,z=get_predictions(ref['path'],task)
            assert np.array_equal(keys,ids) and np.array_equal(yy,y);data[arm]=(z,ref['val'])
        # Train-only size thresholds; no validation-dependent threshold fitting.
        if task=='flywire_coarse':
            profile=CE/'diagnostics/geometry_profiles.jsonl'
            profiles={d['root_id']:d for d in (json.loads(l) for l in profile.read_text().splitlines())}
            size=lambda r:profiles[r['key']]['skeleton_bbox_diagonal_um']
        else:
            sizes={}
            for r in rr:
                g=load_graph(r,task);xyz=g['s'];sizes[r['key']]=float((xyz.max(0).values-xyz.min(0).values).norm())
            size=lambda r:sizes[r['key']]
        bounds=np.quantile([size(r) for r in train],[.25,.5,.75]);q=np.searchsorted(bounds,[size(r) for r in val],side='right')
        masks={'all':np.ones(len(val),bool),**{'size_q'+str(i+1):q==i for i in range(4)}}
        if task=='flywire_coarse':
            masks.update(legacy=np.array([r['legacy'] for r in val]),added=np.array([not r['legacy'] for r in val]))
            from collections import Counter
            counts=Counter(r['cell_type'] for r in train);n=np.array([counts[r['cell_type']] for r in val])
            masks.update(type_unseen=n==0,type_1_4=(n>=1)&(n<=4),type_5_19=(n>=5)&(n<=19),type_20plus=n>=20)
        result={}
        for arm,(z,metric) in data.items():
            sub={}
            for group,mask in masks.items():
                if not mask.any():continue
                if task=='h01_global':
                    a=abs(base-y)[mask];b=abs(z-y)[mask]
                    sub[group]=dict(n=len(a),full_mae=float(a.mean()),variant_mae=float(b.mean()),full_lower_error=int((a<b).sum()),variant_lower_error=int((b<a).sum()))
                else:
                    pa=base.argmax(1);pb=z.argmax(1);k=6 if task=='flywire_coarse' else 3
                    sub[group]=dict(full=classification(y[mask],base[mask],k),variant=classification(y[mask],z[mask],k),
                        full_only_correct=int(((pa==y)&(pb!=y)&mask).sum()),variant_only_correct=int(((pa!=y)&(pb==y)&mask).sum()))
            result[arm]=dict(val=metric,strata=sub)
        # 2x2 effects use a common higher-is-better score; no assumed additivity.
        primary='mae' if task=='h01_global' else 'macro_f1';sign=-1 if primary=='mae' else 1
        values={'full':sign*p['references'][task+'/full']['val'][primary]}
        for arm in ('no_position','no_scale','no_position_scale'):values[arm]=sign*data[arm][1][primary]
        interaction=values['full']-values['no_position']-values['no_scale']+values['no_position_scale']
        all_results[task]=dict(size_thresholds_train_only=bounds.tolist(),comparisons=result,
            position_scale_interaction=dict(higher_better_scores=values,difference_of_differences=interaction,
                note='Nonadditivity description, not a significance test or causal decomposition across all paths'))
    save(ROOT/'analysis.json',dict(results=all_results,test_used=False,version='fixed-residual family only'))
if __name__=='__main__':main()
