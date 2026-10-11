"""Report all configurations, shared-config comparisons, and equal-search envelopes."""
import json
from .common import *

def run():
    p=verify();table=[]
    for task,arms in p['baseline_dirs'].items():
        metric='f1' if task=='ted' else 'missing_edge_macro_ap'
        for arm,base in arms.items():
            for config in ('baseline',)+CONFIGS:
                directory=Path(base) if config=='baseline' else ROOT/'results'/job_key(task,arm,config)
                m=json.loads((directory/'metrics.json').read_text());h=json.loads((directory/'history.json').read_text())
                assert not m['test_used'] and not m['smoke'] and m['epochs']==p['tasks'][task]['epochs']
                if config!='baseline':
                    assert m['identity']['protocol_sha256']==sha(ROOT/'protocol.json')
                    bh=json.loads((Path(base)/'history.json').read_text())
                    assert [r['sample_order_sha256'] for r in h]==[r['sample_order_sha256'] for r in bh]
                    assert m['optimizer_steps']==json.loads((Path(base)/'metrics.json').read_text())['optimizer_steps']
                table.append(dict(task=task,arm=arm,config=config,metric=metric,score=m['val'][metric],metrics_path=str(directory/'metrics.json'),metrics_sha256=sha(directory/'metrics.json')))
    shared=[];envelopes=[]
    for task,arms in p['baseline_dirs'].items():
        rows=[r for r in table if r['task']==task]
        for config in ('baseline',)+CONFIGS:
            group=[r for r in rows if r['config']==config];full=next(r['score'] for r in group if r['arm']=='full')
            shared.append(dict(task=task,config=config,full_minus_controls={r['arm']:full-r['score'] for r in group if r['arm']!='full'}))
        for arm in arms:
            best=max((r for r in rows if r['arm']==arm),key=lambda r:r['score'])
            envelopes.append(best)
    save(ROOT/'analysis.json',dict(complete=True,records=table,shared_config_deltas=shared,equal_search_validation_best=envelopes,test_used=False,
        interpretation='All candidates reported; no config chosen to maximize full-minus-ablation gap. Validation tuning, not an independent test.'))
if __name__=='__main__':run()
