"""Independent single-task training. No summed cross-task loss or joint heads."""
from .common import *
from .data import local_sets,ted_sets,tlp_sets,tlp_geometry
from .geometry import prepare_geometry,adapt_ted_event,to_device
from .encoder import UnifiedDynamicHypergraphEncoder,EncoderConfig
from .decoders import LocalCompartmentDecoder,PointLocalCompartmentDecoder,TEDDecoder,TLPDecoder
from torch import nn
from torch.utils.checkpoint import checkpoint
from dataclasses import asdict
import argparse,time,math


class SingleTask(nn.Module):
    def __init__(self,task,variant,stage='clean_core',method='clean_core'):
        super().__init__();self.task=task;self.variant=variant
        if variant=='point' and task!='h01':raise ValueError('No point-only NREDD task is defined')
        if method != 'clean_core':
            raise ValueError('V18 clean-core accepts only --method clean_core')
        self.method=method
        self.encoder=UnifiedDynamicHypergraphEncoder(EncoderConfig(
            stage='clean_core', association='dynamic', clean_core=True))
        decoder=PointLocalCompartmentDecoder if variant=='point' else {'h01':LocalCompartmentDecoder,'ted':TEDDecoder,'tlp':TLPDecoder}[task]
        self.decoder=decoder()

    def encode(self,gs):
        if self.variant=='skeleton':gs=[skeleton_view(g) for g in gs]
        g,offsets=pack_geometry(gs);return unbind(self.encoder(g),offsets)

    def local(self,gs,queries):
        return torch.cat([self.decoder(z,torch.as_tensor(q,device=z.skeleton.device).reshape(1,3) if self.variant=='point'
            else torch.tensor([q],device=z.skeleton.device)) for z,q in zip(self.encode(gs),queries)])

    def ted(self,gs):
        logits=torch.cat([self.decoder(z,torch.tensor([0],device=z.skeleton.device)) for z in self.encode(gs)])
        # Keep released antisymmetric logit decision/loss semantics; head-only.
        return logits-logits.mean(1,keepdim=True)

    def tlp(self,g):
        z=self.encode([g])[0];n=len(z.skeleton);pairs=torch.triu_indices(n,n,device='cuda').T
        outputs=[]
        for q in pairs.split(128):
            if torch.is_grad_enabled():outputs.append(checkpoint(lambda q:self.decoder(z,q),q,use_reentrant=False))
            else:outputs.append(self.decoder(z,q))
        scores=torch.cat(outputs);matrix=scores.new_zeros((n,n));matrix[pairs[:,0],pairs[:,1]]=scores
        return matrix+matrix.T-torch.diag_embed(matrix.diagonal())

def get_local(row,cache):
    def make():
        g=load(row['path'])['geometry']
        # V14 serialized dataclass identity differs; only canonicalize its type,
        # not any tensor, coordinate, candidate, sampling or normalization.
        from .geometry import GeometryConfig
        g['geometry_config']=GeometryConfig(**asdict(g['geometry_config']))
        return g
    return cache.get(row['sample'],make)
def canonical_geometry_config(g):
    from .geometry import GeometryConfig
    if not isinstance(g.get('geometry_config'), GeometryConfig):
        g['geometry_config']=GeometryConfig(**asdict(g['geometry_config']))
    return g

def get_ted(row,cache):
    return canonical_geometry_config(cache.get(row['name'],lambda:prepare_geometry(*adapt_ted_event(row['event']))))
def get_tlp(row,cache):
    return canonical_geometry_config(cache.get(row['name'],lambda:tlp_geometry(row)))

def microgroups(gs,maximum=70000,max_incidence=350000):
    start=0;points=0;inc=0
    for i,g in enumerate(gs):
        size=len(g['point_xyz']);count=(len(g['cross_anchor'])+len(g['cross_segment'])+
                                       len(g['local_receiver']))
        if i>start and (points+size>maximum or inc+count>max_incidence):
            yield start,i;start=i;points=inc=0
        points+=size;inc+=count
    if len(gs)>start:yield start,len(gs)

@torch.no_grad()
def eval_h01(model,rows,cache):
    model.eval();records=[]
    for r in rows:
        q=load(r['path'])['query'];z=model.local([get_local(r,cache)],[q])[0]
        records.append(dict(sample=r['sample'],cell=r['cell'],target=r['target'],prediction=int(z.argmax()),logits=z.cpu().tolist()))
    return dict(**classification([r['target'] for r in records],[r['prediction'] for r in records]),rows=records)

def counts(y,p):
    y=np.asarray(y);p=np.asarray(p);tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum())
    fn=int(((y==1)&(p==0)).sum());tn=int(((y==0)&(p==0)).sum())
    pe=tp/(tp+fp+1e-5);re=tp/(tp+fn+1e-5)
    return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        f1=2*tp/max(2*tp+fp+fn,1),released_epsilon_metrics=dict(precision=pe,recall=re,f1=2*pe*re/(pe+re+1e-5)))

@torch.no_grad()
def eval_ted(model,blocks,cache,out):
    model.eval();all_y=[];all_p=[];rows=[]
    for bi,block in enumerate(blocks,1):
        if bi==1 or bi%5==0 or bi==len(blocks):
            print(json.dumps({'eval':'ted','block':bi,'blocks':len(blocks),'events':len(all_y)},separators=(',',':')),flush=True)
        outputs=[]
        for start in range(0,len(block['events']),32):
            events=block['events'][start:start+32];gs=[get_ted(r,cache) for r in events]
            for a,b in microgroups(gs):outputs.append(model.ted(gs[a:b]).cpu())
        logits=torch.cat(outputs);target=torch.stack([r['target'] for r in block['events']]).argmax(1).numpy()
        pred=logits.sigmoid().round().argmax(1).numpy();all_y.extend(target);all_p.extend(pred)
        rows.append(dict(block=block['block'],**counts(target,pred)))
        np.savez_compressed(out/f'prediction_{Path(block["block"]).stem}.npz',logits=logits.numpy(),target=target)
    return dict(**counts(all_y,all_p),events=len(all_y),blocks=rows)

def ap(y,s):
    y=np.asarray(y,bool);s=np.asarray(s,float)
    if not y.any():return 0.
    order=np.argsort(-s,kind='stable');ends=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    tp=np.cumsum(y[order])[ends];return float(np.sum(np.diff(np.r_[0.,tp/y.sum()])*tp/(ends+1)))

@torch.no_grad()
def eval_tlp(model,rows,cache,out):
    from scipy.stats import rankdata
    from scipy.sparse.csgraph import connected_components
    from dataset.utils import loadswc,get_edge_for_coonect_test
    from check_utils import Rec_adj
    model.eval();records=[];my=[];ms=[];cy=[];cs=[];tot=np.zeros(3,int);exact=0;component_errors=[]
    for r in rows:
        g=get_tlp(r,cache);score=model.tlp(g).sigmoid().cpu().numpy();target=r['target'].numpy()>.5;obs=r['observed'].numpy()>0
        pairs=g['query_pairs'].numpy();n=len(score);off=~np.eye(n,dtype=bool);upper=np.triu(np.ones((n,n),bool),1)
        universe=upper&~obs;missing=universe&target;y=target[universe];s=score[universe];my.append(y);ms.append(s)
        ctarget=target[pairs[:,0],pairs[:,1]];cscore=score[pairs[:,0],pairs[:,1]];cy.append(ctarget);cs.append(cscore)
        fy=target[off];fs=score[off];pos=int(fy.sum());neg=len(fy)-pos
        auc=float((rankdata(fs)[fy].sum()-pos*(pos+1)/2)/(pos*neg))
        swc=loadswc(r['swc']);raw=torch.from_numpy(get_edge_for_coonect_test(swc)).float()
        repaired=np.asarray(Rec_adj(torch.from_numpy(score.copy()),raw,swc,Max_distance=20,lamd_dist=.5))>0
        added=repaired&~obs&upper;tp=int((added&missing).sum());fp=int((added&~missing).sum());fn=int((missing&~added).sum());tot+=[tp,fp,fn]
        npred,cp=connected_components(repaired|repaired.T,directed=False);ntrue,ct=connected_components(target|target.T,directed=False)
        eq=np.array_equal(cp[:,None]==cp[None],ct[:,None]==ct[None]);exact+=int(eq);component_errors.append(abs(npred-ntrue))
        records.append(dict(block=r['name'],missing_ap=ap(y,s),missing_pairs=len(y),missing_positives=int(y.sum()),
            full_ap=ap(fy,fs),full_auc=auc,candidate_ap=ap(ctarget,cscore) if ctarget.any() else None,
            rule_tp=tp,rule_fp=fp,rule_fn=fn,max_asymmetry=float(abs(score-score.T).max())))
        np.savez_compressed(out/f'prediction_{r["name"]}.npz',score=score,target=target,observed=obs,pairs=pairs)
    tp,fp,fn=map(int,tot)
    return dict(missing_edge_macro_ap=float(np.mean([r['missing_ap'] for r in records])),
        missing_edge_micro_ap=ap(np.concatenate(my),np.concatenate(ms)),
        full_matrix_macro_ap=float(np.mean([r['full_ap'] for r in records])),
        full_matrix_macro_auc=float(np.mean([r['full_auc'] for r in records])),
        candidate_micro_ap=ap(np.concatenate(cy),np.concatenate(cs)),
        official_rule=dict(tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),
                           exact_component_partition=exact/len(rows),component_count_mae=float(np.mean(component_errors))),blocks=records)

def train(a):
    setup(a.seed)
    resumed=bool(a.resume)
    if resumed:
        assert a.out.exists(),f'resume output missing: {a.out}'
        assert (a.out/'latest.pt').is_file(),f'resume checkpoint missing: {a.out}/latest.pt'
    else:
        assert not a.out.exists(),f'output already exists: {a.out}'
        a.out.mkdir(parents=True)
    started=time.time()
    sets=local_sets(a.cache,a.smoke) if a.task=='h01' else ted_sets(a.smoke) if a.task=='ted' else tlp_sets(a.smoke)
    model=SingleTask(a.task,a.variant,a.stage,a.method).cuda()
    # NREDD TED/TLP geometry objects are larger than the original V15
    # 600-MiB cache.  Allow the run directory to raise the bounded cache
    # without changing equations, ordering, or the seed-42 protocol.
    disk_cache=os.getenv('CPSF_GEOM_DISK_CACHE') if a.task=='ted' else None
    cache=LRU(int(os.getenv('CPSF_CACHE_MIB','600'))*2**20,disk=disk_cache)
    epochs=2 if a.smoke else {'h01':35,'ted':30,'tlp':80}[a.task]
    if a.task=='h01':
        opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00003)
    else:
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0005 if a.task=='ted' else .001)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:1/(epoch+1)) if a.task=='ted' else torch.optim.lr_scheduler.StepLR(opt,20,.1)
    protocol=dict(framework='V18 clean-core independent single-task',stage='clean_core',task=a.task,variant=a.variant,method=a.method,seed=a.seed,epochs=epochs,
        encoder=asdict(model.encoder.config),parameters=sum(p.numel() for p in model.parameters()),
        single_task=True,smoke_only=a.smoke,selection='fixed final epoch; no validation/test selection',
        train_count=len(sets['train']),test_count=len(sets['val' if a.task=='h01' else 'test']),
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        geometry_disk_cache=disk_cache,
        architecture='one geometry-conditioned bidirectional correspondence operator; no separate ABF-LC or GCHT heads',
        correspondence='masked candidate score from shared skeleton-surface latent space plus geometry bias and null option; same matrix drives both message directions',
        auxiliary_loss=None,
        v17_baseline='same frozen input adapters, task heads, task loss, optimizer and epochs; clean-core encoder is the only architectural intervention',
        registered_capacity_note='single correspondence block with shared score, geometry bias, null score, and bidirectional residual update',
        deterministic=True,tf32=False,all_retained_unique_points=True,
        skeleton_control='same observed graph/stem/update/decoder, empty surface; same registered not active parameter count')
    if a.variant=='point':protocol['point_control']='H01 only: same frozen point support and labels; point-derived origin, physical um scale1, no skeleton nodes/edges/features. Query XYZ only in decoder. Local FOV sampling remains inherited from the frozen skeleton-guided cohort.'
    if a.task=='h01':protocol.update(train_cells=sorted({r['cell'] for r in sets['train']}),val_cells=sorted({r['cell'] for r in sets['val']}),independent_test=False,cache_manifest_sha256=sha(a.cache/'manifest.json'))
    elif a.task=='ted':protocol.update(train_blocks=len(sets['train']),test_blocks=len(sets['test']),
        train_events=sum(len(b['events']) for b in sets['train']),test_events=sum(len(b['events']) for b in sets['test']),
        loss='alpha5 positive-event BCEWithLogits, antisymmetric logits; summed event gradients',
        batch=64,tail='backward without step at epoch tail, carry gradients to next epoch',cache_report_sha256=sha(TED/'construction_report.json'))
    else:protocol.update(train_blocks=len(sets['train']),test_blocks=len(sets['test']),test_order=[r['swc'] for r in sets['test']],
        train_order=[r['swc'] for r in sets['train']],loss='full-adjacency adaptive weighted BCEWithLogits, stable equivalent to BCE(sigmoid)',
        primary='all unobserved unordered pairs, Missing-edge macro AP; not candidate AP')
    if a.task=='h01':
        protocol['cohort']={k:[dict(sample=r['sample'],cell=r['cell'],target=r['target'],sha256=r['sha256']) for r in seq] for k,seq in sets.items()}
    elif a.task=='ted':
        protocol['cohort']={k:[dict(block=b['block'],source_sha256=b['source_sha256'],events=len(b['events'])) for b in seq] for k,seq in sets.items()}
    else:
        protocol['cohort']={k:[dict(block=r['name'],swc_sha256=sha(r['swc']),edge_sha256=sha(r['edge']),bridge_sha256=sha(r['bridge']),
            observed_target_sha256=__import__('hashlib').sha256(r['observed'].numpy().tobytes()+r['target'].numpy().tobytes()).hexdigest()) for r in seq] for k,seq in sets.items()}
    protocol['cohort_sha256']=__import__('hashlib').sha256(json.dumps(protocol['cohort'],sort_keys=True).encode()).hexdigest()
    if resumed:
        saved=load(a.out/'latest.pt')
        old=saved.get('protocol',{})
        for key in ('task','variant','method','stage','seed'):
            assert old.get(key)==protocol.get(key),f'resume protocol mismatch {key}: {old.get(key)} != {protocol.get(key)}'
        model.load_state_dict(saved['model'])
        opt.load_state_dict(saved['optimizer'])
        sched.load_state_dict(saved['scheduler'])
        history=json.loads((a.out/'history.json').read_text()) if (a.out/'history.json').is_file() else []
        start_epoch=int(saved['epoch'])+1
        assert start_epoch<=epochs,f'checkpoint already complete: epoch {start_epoch-1}/{epochs}'
        save(a.out/'protocol.json',dict(protocol,resumed_from=int(saved['epoch']),resume_seed=a.seed))
        print(json.dumps({'resume':True,'from_epoch':start_epoch-1,'epochs':epochs},separators=(',',':')),flush=True)
    else:
        save(a.out/'protocol.json',protocol);history=[];start_epoch=1
    opt.zero_grad(set_to_none=True)
    for epoch in range(start_epoch,epochs+1):
        model.train();begin=time.time();losses=[];grads=[];core_grads=[]
        def finish():
            if a.task=='h01':torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()),'Nonfinite gradient'
            grads.append(float(sum((p.grad.double().square().sum() for p in model.encoder.surface_stem.parameters() if p.grad is not None),torch.zeros((),device='cuda')).sqrt()))
            keys=('query','key','geometry_bias','null_score','value','gather','scatter',
                  'local_value','local_update','edge_update','vertex_update')
            def module_grad(name):
                module=getattr(model.encoder.block,name,None)
                if module is None:
                    return 0.0
                total=sum((p.grad.double().square().sum() for p in module.parameters()
                           if p.grad is not None),torch.zeros((),device='cuda'))
                return float(total.sqrt())
            core_grads.append({key:module_grad(key) for key in keys})
            opt.step();opt.zero_grad(set_to_none=True)
        if a.task=='h01':
            order=np.random.permutation(len(sets['train']))
            for start in range(0,len(order),8):
                rs=[sets['train'][i] for i in order[start:start+8]];gs=[get_local(r,cache) for r in rs]
                qs=[load(r['path'])['query'] for r in rs]
                for lo,hi in microgroups(gs):
                    z=model.local(gs[lo:hi],qs[lo:hi]);y=torch.tensor([r['target'] for r in rs[lo:hi]],device='cuda')
                    loss=nn.functional.cross_entropy(z,y,reduction='sum')/len(rs)
                    assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach()))
                finish()
        elif a.task=='ted':
            pending=[]
            def ted_step(rs,update):
                gs=[get_ted(r,cache) for r in rs]
                for lo,hi in microgroups(gs):
                    z=model.ted(gs[lo:hi]);y=torch.stack([r['target'] for r in rs[lo:hi]]).cuda()
                    loss=(nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none').mean(1)*torch.where(y[:,1]>0,5.,1.)).sum()
                    assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach())/(hi-lo))
                if update:finish()
            for idx in torch.randperm(len(sets['train'])).tolist():
                pending.extend(sets['train'][idx]['events'])
                while len(pending)>=64:ted_step(pending[:64],True);pending=pending[64:]
            if pending:ted_step(pending,False)
        else:
            for idx in torch.randperm(len(sets['train'])).tolist():
                r=sets['train'][idx];g=get_tlp(r,cache);z=model.tlp(g);y=r['target'].cuda()
                weight=torch.ones_like(y);assert y.sum()>0;weight[y==1]=(y.numel()-y.sum())/y.sum()
                loss=nn.functional.binary_cross_entropy_with_logits(z,y,weight=weight)
                assert torch.isfinite(loss)
                loss.backward();losses.append(float(loss.detach()));finish()
        sched.step();row=dict(epoch=epoch,epochs=epochs,loss=float(np.mean(losses)),seconds=time.time()-begin,
            surface_gradient_l2_mean=float(np.mean(grads)),surface_nonzero_step_fraction=float(np.mean(np.array(grads)>0)),
            core_gradient_l2_mean={key:float(np.mean([r[key] for r in core_grads])) for key in core_grads[0]} if core_grads else {},
            association_last_training_batch=model.encoder.diagnostics)
        history.append(row);save(a.out/'history.json',history);save(a.out/'status.json',dict(state='training',**row));print(json.dumps(row),flush=True)
        temp=a.out/'latest.tmp';torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
            epoch=epoch,protocol=protocol,tail_gradients={n:p.grad for n,p in model.named_parameters() if p.grad is not None}),temp);temp.replace(a.out/'latest.pt')
    print(json.dumps({'checkpoint':'final','task':a.task},separators=(',',':')),flush=True)
    # Move tensors to CPU before serialization.  This avoids a CUDA context
    # synchronization/serialization stall on large NREDD evaluation jobs.
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save(dict(model=state,protocol=protocol),a.out/'final.pt',_use_new_zipfile_serialization=True)
    save(a.out/'status.json',dict(state='evaluating',epoch=epochs));print(json.dumps({'state':'evaluating','task':a.task},separators=(',',':')),flush=True)
    if a.task=='h01':result=dict(validation=eval_h01(model,sets['val'],cache),train=eval_h01(model,sets['train'],cache))
    elif a.task=='ted':result=dict(test=eval_ted(model,sets['test'],cache,a.out))
    else:result=dict(test=eval_tlp(model,sets['test'],cache,a.out))
    save(a.out/'metrics.json',dict(protocol=protocol,**result,elapsed_seconds=time.time()-started))
    save(a.out/'status.json',dict(state='complete',seconds=time.time()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['h01','ted','tlp'],required=True);p.add_argument('--variant',choices=['dynamic','fixed','once','skeleton','point'],required=True)
    p.add_argument('--stage',choices=['clean_core'],default='clean_core')
    p.add_argument('--method',choices=['clean_core'],default='clean_core')
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    try:train(a)
    except Exception as exc:save(a.out/'failure.json',dict(error=repr(exc)));raise
