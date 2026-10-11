"""Independent composition readout using a frozen local-trained encoder."""
from .common import *
from .geometry import prepare_geometry,prepare_surface_geometry
from .offload import encode_offloaded
from .train import SingleTask
from .decoders import CellCompositionDecoder,surface_pool,branch_aware_pool
from torch import nn
import argparse,time

@torch.no_grad()
def extract(a):
    setup(a.seed);assert not a.out.exists();a.out.mkdir(parents=True)
    checkpoint=load(a.checkpoint);variant=checkpoint['protocol']['variant'];assert checkpoint['protocol']['task']=='h01'
    assert checkpoint['protocol']['seed']==a.seed
    stage=checkpoint['protocol']['stage'];method=checkpoint['protocol'].get('method','v16')
    model=SingleTask('h01',variant,stage,method).cuda().eval();model.load_state_dict(checkpoint['model'],strict=True)
    cells=sorted(json.loads((SOURCE/'manifest.json').read_text())['cells'],key=lambda r:r['surface_points'])
    cells=[cells[0],cells[-1]]+cells[1:-1]
    if a.smoke:cells=[min((r for r in cells if r['split']==k),key=lambda r:r['surface_points']) for k in ('train','val')]
    protocol=dict(encoder_checkpoint_sha256=sha(a.checkpoint),seed=a.seed,variant=variant,stage=stage,method=method,pooling=a.pool,expected_cells=len(cells),
        expected_points=sum(r['surface_points'] for r in cells),encoder_frozen=True,single_task_readout=True,
        scope='local-to-cell frozen transfer, not multi-task or end-to-end cell training',smoke=a.smoke)
    save(a.out/'protocol.json',protocol);done=[];started=time.time()
    for r in cells:
        begin=time.time();path=SOURCE/'inputs'/(r['cell']+'.npz')
        with np.load(path,allow_pickle=False) as x:
            raw=[x['surface_xyz_um']] if variant=='point' else [x['skeleton_xyz_um'],x['edge_index'],x['surface_xyz_um'] if variant!='skeleton' else np.empty((0,3))]
        g=prepare_surface_geometry(raw[0]) if variant=='point' else prepare_geometry(*raw)
        z=encode_offloaded(model.encoder,g)
        mean_skel=z.skeleton.mean(0) if len(z.skeleton) else z.surface.new_zeros(model.encoder.config.dim)
        mean_feature=torch.cat((mean_skel,surface_pool(z)))
        if variant!='point': branch_skel,branch_surface=branch_aware_pool(z,raw[1])
        else: branch_skel,branch_surface=mean_skel,surface_pool(z)
        branch_feature=torch.cat((branch_skel,branch_surface))
        feature=mean_feature if a.pool=='mean' else branch_feature
        assert feature.shape==(128,) and torch.isfinite(feature).all()
        with np.load(SOURCE/'targets'/(r['cell']+'.npz')) as y:target=float(y['excitatory_input_fraction'])
        output=a.out/(r['cell']+'.pt');torch.save(dict(cell=r['cell'],split=r['split'],feature=feature,
                        mean_feature=mean_feature,branch_feature=branch_feature,target=target),output)
        done.append(dict(cell=r['cell'],split=r['split'],points=r['surface_points'],encoded_unique_points=g['unique_point_count'],
            sha256=sha(output),source_sha256=sha(path),seconds=time.time()-begin))
        save(a.out/'progress.json',dict(cells=len(done),expected_cells=len(cells),points=sum(x['points'] for x in done),seconds=time.time()-started))
        print('CELL',r['cell'],len(done),round(time.time()-begin,1),flush=True);del g,z,raw
    save(a.out/'manifest.json',dict(records=done,protocol=protocol,all_cells=len(done)==104))
    save(a.out/'metrics.json',dict(passed=True,scope='feature extraction only',seconds=time.time()-started))

def fit(a):
    setup(a.seed);assert not a.out.exists();a.out.mkdir(parents=True);m=json.loads((a.features/'manifest.json').read_text())
    assert m['all_cells'] or a.smoke;sets={k:[] for k in ('train','val')}
    for r in m['records']:
        file=a.features/(r['cell']+'.pt');assert sha(file)==r['sha256'];sets[r['split']].append(load(file))
    if not a.smoke:assert [len(sets[k]) for k in ('train','val')]==[80,24]
    assert {r['cell'] for r in sets['train']}.isdisjoint(r['cell'] for r in sets['val'])
    field=a.feature_field if a.feature_field!='auto' else 'feature'
    x={k:torch.stack([r[field] for r in rows]) for k,rows in sets.items()}
    y={k:torch.tensor([r['target'] for r in rows])[:,None] for k,rows in sets.items()}
    mean=x['train'].mean(0);std=x['train'].std(0,unbiased=False).clamp_min(.01);x={k:(v-mean)/std for k,v in x.items()}
    head=CellCompositionDecoder();nn.init.zeros_(head.head[-2].weight)
    avg=y['train'].mean().clamp(1e-6,1-1e-6);nn.init.constant_(head.head[-2].bias,float(torch.logit(avg)))
    optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01);epochs=2 if a.smoke else 200;history=[]
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True);prediction=head.head(x['train']);loss=nn.functional.mse_loss(prediction,y['train'])
        assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),5,error_if_nonfinite=True);optimizer.step();history.append(float(loss.detach()))
    protocol=dict(task='h01_cell',seed=a.seed,epochs=epochs,train_cells=len(sets['train']),val_cells=len(sets['val']),pooling=m['protocol'].get('pooling','mean'),feature_field=field,
        encoder_frozen=True,independent_test=False,smoke_only=a.smoke,selection='fixed epoch200',loss='equal-cell MSE',
        feature_manifest_sha256=sha(a.features/'manifest.json'),variant=m['protocol']['variant'],stage=m['protocol']['stage'],local_encoder=m['protocol'])
    protocol['cohort_sha256']=__import__('hashlib').sha256(json.dumps({k:sorted((r['cell'],r['target']) for r in seq) for k,seq in sets.items()},sort_keys=True).encode()).hexdigest()
    results={}
    with torch.no_grad():
        for k in sets:
            pred=head.head(x[k]).numpy()[:,0];target=y[k].numpy()[:,0]
            results[k]=dict(**regression(target,pred),rows=[dict(cell=r['cell'],target=float(t),prediction=float(p)) for r,t,p in zip(sets[k],target,pred)])
    results['constant_baseline']=regression(y['val'].numpy()[:,0],np.full(len(y['val']),float(avg)))
    torch.save(dict(head=head.state_dict(),mean=mean,std=std,protocol=protocol),a.out/'final.pt')
    save(a.out/'history.json',history);save(a.out/'metrics.json',dict(protocol=protocol,results=results));save(a.out/'status.json',dict(state='complete'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['extract','fit'],required=True);p.add_argument('--pool',choices=['mean','branch','both'],default='mean');p.add_argument('--feature-field',choices=['auto','feature','mean_feature','branch_feature'],default='auto');p.add_argument('--seed',type=int,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--features',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:(extract if a.stage=='extract' else fit)(a)
    except Exception as exc:save(a.out/'failure.json',dict(error=repr(exc)));raise
