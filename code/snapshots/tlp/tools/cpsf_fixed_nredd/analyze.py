import numpy as np
from .common import *
def main():
    p=verify();results={}
    for task in TASKS:
        key='f1' if task=='ted' else 'missing_edge_macro_ap';metrics={}
        for job in p['jobs']:
            if job['task']!=task:continue
            arm=job['arm'];d=ROOT/'results'/(task+'_'+arm);m=json.loads((d/'metrics.json').read_text());metrics[arm]=m['val'][key]
        results[task]=dict(primary=key,values=metrics,full_minus_variant={a:metrics['full']-v for a,v in metrics.items()},
            position_scale_interaction=metrics['full']-metrics['no_position']-metrics['no_scale']+metrics['no_position_scale'])
        if task=='ted':
            with np.load(ROOT/'results'/(task+'_full')/'val_predictions.npz') as z:
                y=z['target'];ids=z['keys'];base=torch.tensor(z['logits']).sigmoid().round().argmax(1).numpy()
            paired={}
            for a in metrics:
                with np.load(ROOT/'results'/(task+'_'+a)/'val_predictions.npz') as z:
                    assert np.array_equal(ids,z['keys']) and np.array_equal(y,z['target']);pred=torch.tensor(z['logits']).sigmoid().round().argmax(1).numpy()
                    paired[a]=dict(full_only_correct=int(((base==y)&(pred!=y)).sum()),variant_only_correct=int(((base!=y)&(pred==y)).sum()))
            results[task]['paired']=paired
        else:
            bas=json.loads((ROOT/'results'/(task+'_full')/'metrics.json').read_text())['val']['blocks']
            paired={}
            for a in metrics:
                var=json.loads((ROOT/'results'/(task+'_'+a)/'metrics.json').read_text())['val']['blocks'];assert [r['key'] for r in bas]==[r['key'] for r in var]
                diff=np.array([x['missing_edge_ap']-y['missing_edge_ap'] for x,y in zip(bas,var)])
                paired[a]=dict(full_better_blocks=int((diff>1e-12).sum()),variant_better_blocks=int((diff< -1e-12).sum()),tied_blocks=int((abs(diff)<=1e-12).sum()))
            results[task]['paired']=paired
    save(ROOT/'analysis.json',dict(results=results,test_used=False,description='fixed seed42 descriptive paired effects; no selected epoch'))
if __name__=='__main__':main()
