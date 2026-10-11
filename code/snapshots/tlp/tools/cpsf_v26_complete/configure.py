import json
from .common import *

def main():
    assert not (ROOT/'protocol.json').exists()
    parents={}
    for root in (CE,H01):
        p=json.loads((root/'protocol.json').read_text())
        assert json.loads((root/'pipeline_status.json').read_text())['state']=='complete'
        for name,h in p['source_sha256'].items():assert sha(root/name)==h and sha(ROOT/name)==h,name
        parents[str(root/'protocol.json')]=sha(root/'protocol.json')
    for root in (FW,OLD,H01,SCALE):
        path=root/'cache/manifest.json';parents[str(path)]=sha(path)
    parents[str(FW/'candidates.json')]=sha(FW/'candidates.json')
    references={}
    for task in TASKS:
        if task=='flywire_coarse':
            refs={a:CE/'results'/('coarse_'+('full' if a=='fixed_residual' else a)) for a in ('fixed_residual','mean_only')}
        else:refs={'fixed_residual':H01/'results'/('local_baseline' if task=='h01_local' else 'global_baseline')}
        for arm,path in refs.items():
            m=json.loads((path/'metrics.json').read_text());assert not m['smoke'] and m['seed']==42
            references[task+'/'+arm]=dict(path=str(path),val={k:v for k,v in m['val'].items() if k!='rows'},
                hashes={fn:sha(path/fn) for fn in ('metrics.json','initial.pt','final.pt')})
    jobs=[]
    for task in TASKS:
        for arm in ARMS:
            if task=='h01_local' and arm in ('thumbnail','uniform_budget'):continue
            reused=task=='flywire_coarse' and arm in ('fixed_residual','mean_only')
            jobs.append(dict(task=task,arm=arm,role='reused_reference' if reused else 'new_training'))
    save(ROOT/'protocol.json',dict(version='V26 regional gated residual complete matrix',seed=42,jobs=jobs,
        parent_files=parents,references=references,
        hierarchy={'h01_local':'V24 minimum-four adapter','h01_global':'V24 minimum-four adapter','flywire_coarse':'original V23 cache'},
        tasks=dict(h01_local=dict(epochs=25,batch=8,train=716,val=214,round_cells=716,updates=2250),
                   h01_global=dict(epochs=50,batch=4,train=80,val=24,round_cells=80,updates=1000),
                   flywire_coarse=dict(epochs=50,batch=4,train=6039,val=1922,round_cells=4975,updates=62200)),
        loss='ordinary smoothed CE(.05) for classification; train-std normalized MSE for H01 global',
        optimizer=dict(lr=.0003,weight_decay=.01,lr_min=.00003,clip=5),selection='fixed final epoch',
        gate='base mean + 0.1*tanh(alpha)*sigmoid(Linear([LN(graph-point),mean_distance,mean_concentration]))*joint',
        gate_initialization='alpha=0; region linear weight/bias=0; identical initial function to mean_only',
        test_policy='development validation only; never read test inputs or predictions',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        document_sha256=sha(ROOT/'CPSF_V26_COMPLETE_PROTOCOL_20261003.md')))

if __name__=='__main__':main()
