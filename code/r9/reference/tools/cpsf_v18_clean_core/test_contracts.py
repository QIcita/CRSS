"""Engineering gates for every staged intervention; not accuracy evidence."""
import argparse
from dataclasses import replace
from .common import *
from .geometry import prepare_geometry, prepare_surface_geometry, to_device, GeometryConfig
from .encoder import UnifiedDynamicHypergraphEncoder, EncoderConfig, segment_softmax, add
from .offload import encode_offloaded
from .decoders import LocalCompartmentDecoder, PointLocalCompartmentDecoder, TEDDecoder, TLPDecoder, CellCompositionDecoder


def fixture():
    rng=np.random.default_rng(31)
    s=np.array([[0.,0,0],[1.,.01,0],[2.,.03,0],[0.,.32,.01],[1.,.31,0],[2.,.34,.02]])
    e=np.array([[0,1],[1,2],[3,4],[4,5]])
    return s,e,rng.uniform([0.,-.15,-.15],[2.,.45,.15],(93,3))


def close(a,b):
    for key in ('skeleton','surface'):
        torch.testing.assert_close(getattr(a,key).cpu(),getattr(b,key).cpu(),atol=2e-5,rtol=2e-5)


def run(device):
    setup(42);s,e,p=fixture();raw=prepare_geometry(s,e,p);g=to_device(raw,device)
    cfg=EncoderConfig(dim=32,point_chunk=17);results={}
    # Exact initialization and forward replay against V14, not merely a copy
    # of equations. Original source and cached tensors are kept read-only.
    from tools.cpsf_v14_train_20260922.encoder import UnifiedDynamicHypergraphEncoder as OldEncoder, EncoderConfig as OldConfig
    from tools.cpsf_v14_train_20260922.geometry import GeometryConfig as OldGeometry
    setup(42);old=OldEncoder(OldConfig(dim=32,point_chunk=17)).to(device).eval()
    # Legacy replay uses the V14 geometry proposal; V16's default geometry is
    # deliberately multi-scale and is tested separately below.
    legacy_raw=prepare_geometry(s,e,p,OldGeometry());legacy_g=to_device(legacy_raw,device)
    setup(42);legacy=UnifiedDynamicHypergraphEncoder(replace(cfg,stage='legacy',geometry=OldGeometry())).to(device).eval()
    for k,v in old.state_dict().items():torch.testing.assert_close(v,legacy.state_dict()[k],atol=0,rtol=0)
    with torch.no_grad():
        a,b=old(dict(legacy_g,geometry_config=OldGeometry())),legacy(legacy_g)
        torch.testing.assert_close(a.skeleton,b.skeleton,atol=0,rtol=0)
        torch.testing.assert_close(a.surface,b.surface,atol=0,rtol=0)
    results['legacy_bitwise_v14_replay']=True
    # Scalar and arbitrary-degree receiver normalizations, empty/rejected
    # groups and gradients; zero membership must not yield softmax NaNs.
    z=torch.tensor([-.7,.2,.4,1.],device=device,dtype=torch.float64,requires_grad=True)
    ids=torch.tensor([0,0,2,2],device=device);pr=torch.tensor([.2,.8,.4,.5],device=device,dtype=z.dtype,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,y:segment_softmax(x,ids,4,y),(z,pr),eps=1e-6,atol=1e-5)
    zero=segment_softmax(z,ids,4,torch.zeros_like(pr));assert torch.equal(zero,torch.zeros_like(zero))
    (zero.sum()).backward();assert torch.isfinite(z.grad).all()
    results['masked_segment_softmax_gradcheck_and_all_rejected']=True
    for stage in ('residual','routing','role'):
        modes={}
        for mode in ('dynamic','fixed','once'):
            setup(42);model=UnifiedDynamicHypergraphEncoder(replace(cfg,stage=stage,association=mode)).to(device).eval()
            out,tt=model(g,True)
            heads=[LocalCompartmentDecoder(32),TEDDecoder(32),TLPDecoder(32),CellCompositionDecoder(32)]
            heads=[x.to(device) for x in heads];q=torch.tensor([0,1],device=device);pairs=torch.tensor([[0,1],[1,0]],device=device)
            scores=[heads[0](out,q),heads[1](out,q),heads[2](out,pairs),heads[3](out)]
            torch.testing.assert_close(scores[2][0],scores[2][1],atol=1e-6,rtol=1e-5)
            tt[-1]['hyperedge'].retain_grad()
            sum((x*torch.randn_like(x)).sum() for x in scores).backward()
            assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.parameters())
            required=['skeleton_stem','surface_stem','block.local_value','block.edge_update','block.vertex_update']
            if mode!='fixed':required+=['block.score']
            if stage=='residual' and mode!='fixed':required+=['block.null_score']
            if stage in ('routing','role'):required+=['block.gather_score','block.scatter_score','block.receiver_gate']
            if stage=='role':required+=['block.up_role','block.down_role']
            norms={key:float(sum((v.grad.double().square().sum() for k,v in model.named_parameters()
                                 if k.startswith(key) and v.grad is not None),torch.zeros((),device=device)).sqrt()) for key in required}
            assert min(norms.values())>0,(stage,mode,norms)
            assert tt[-1]['hyperedge'].grad.norm()>0
            key='weights' if stage=='residual' else 'membership'
            if mode=='once':torch.testing.assert_close(tt[0][key],tt[1][key],atol=0,rtol=0)
            if mode=='dynamic':assert not torch.equal(tt[0][key],tt[1][key])
            with torch.no_grad():
                ref=model(g)
                torch.testing.assert_close(model(g).skeleton,ref.skeleton,atol=0,rtol=0)
                # Deliberately split anchors while patches overlap these tiles.
                for chunk in (1,3):
                    off=encode_offloaded(model,raw,point_chunk=13,edge_chunk=chunk,state_chunk=2)
                    close(ref,off)
                for ss,ee,pp in ((s,e[:,::-1],p),(s,e,np.r_[p,p[:7]]),(s*1000+37,e,p*1000+37)):
                    close(model(to_device(prepare_geometry(ss,ee,pp),device)),ref)
                perm=np.array([4,2,0,5,3,1]);inv=np.argsort(perm)
                permuted=model(to_device(prepare_geometry(s[perm],inv[e],p[::-1]),device))
                torch.testing.assert_close(permuted.skeleton[inv],ref.skeleton,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(permuted.surface,ref.surface,atol=2e-5,rtol=2e-5)
                for ss,ee,pp in ((s,e,np.empty((0,3))),(s,np.empty((0,2),int),p),(s[:1],np.empty((0,2),int),np.empty((0,3)))):
                    empty=prepare_geometry(ss,ee,pp);direct=model(to_device(empty,device))
                    assert torch.isfinite(direct.skeleton).all() and torch.isfinite(direct.surface).all()
                    close(direct,encode_offloaded(model,empty,edge_chunk=1,state_chunk=2))
                block=model.block;state=torch.randn(7,32,device=device);msg=torch.randn_like(state)
                no=torch.zeros(7,device=device);small=torch.full((7,),1e-7,device=device)
                if stage=='residual':
                    exact=block.cross_finish(state,torch.zeros_like(msg),no)
                    near=block.cross_finish(state,msg*small[:,None],small)
                    block.null_score[-1].bias.fill_(1e5)
                else:
                    exact=block.receive(state,msg,no);near=block.receive(state,msg,small)
                    block.score[-1].weight.zero_();block.score[-1].bias.fill_(-1000.)
                assert torch.equal(exact,state)
                torch.testing.assert_close(near,state,atol=1e-6,rtol=0)
                if mode!='fixed':
                    _,reject=model(g,True)
                    assert all(torch.equal(t['local'],t['state']) for t in reject)
            # Fresh unmodified model for disjoint batching/gradient comparison.
            setup(42);model=UnifiedDynamicHypergraphEncoder(replace(cfg,stage=stage,association=mode)).to(device)
            other=UnifiedDynamicHypergraphEncoder(replace(cfg,stage=stage,association=mode,point_chunk=10000)).to(device)
            other.load_state_dict(model.state_dict())
            gs=[raw,prepare_geometry(s[:3],e[:2],p[:40]),prepare_geometry(s,e,np.empty((0,3)))]
            packed,offsets=pack_geometry(gs,device);outs=unbind(model(packed),offsets)
            refs=[other(to_device(x,device)) for x in gs]
            for a,b in zip(outs,refs):close(a,b)
            probes=[(torch.randn_like(x.skeleton),torch.randn_like(x.surface)) for x in outs]
            for seq in (outs,refs):sum((z.skeleton*a).sum()+(z.surface*b).sum() for z,(a,b) in zip(seq,probes)).backward()
            for (name,a),(_,b) in zip(model.named_parameters(),other.named_parameters()):
                if a.grad is None or b.grad is None:assert a.grad is None and b.grad is None,name
                else:torch.testing.assert_close(a.grad,b.grad,atol=3e-4,rtol=4e-4,msg=name)
            if stage!='residual':
                _,trace=model(g,True)
                a,surf=g['cross_anchor'],g['cross_surface']
                for t in trace:
                    for w,receiver,count in ((t['alpha'],a,g['n_nodes']),(t['beta'],surf,g['n_patches'])):
                        sums=add(w,receiver,count);present=add(torch.ones_like(w),receiver,count)>0
                        torch.testing.assert_close(sums[present],torch.ones_like(sums[present]),atol=1e-6,rtol=1e-6)
                # Each direction has its own scorer; changing downlink cannot
                # change the first-round membership or uplink attention.
                with torch.no_grad():
                    for param in model.block.scatter_score.parameters():param.add_(torch.randn_like(param)*.2)
                    _,changed=model(g,True)
                    # V16's segment-centered path is intentionally coupled:
                    # scatter changes segment->surface context before the
                    # node-centered uplink is scored.  Thus alpha may change
                    # through its *features* even though gather_score itself
                    # is untouched.  Retain the old strict independence gate
                    # for the no-segment case and check normalization here.
                    if g.get('n_segments',0)==0:
                        torch.testing.assert_close(trace[0]['alpha'],changed[0]['alpha'],atol=0,rtol=0)
                    else:
                        for w,receiver,count in ((changed[0]['alpha'],a,g['n_nodes']),
                                                  (changed[0]['beta'],surf,g['n_patches'])):
                            sums=add(w,receiver,count);present=add(torch.ones_like(w),receiver,count)>0
                            torch.testing.assert_close(sums[present],torch.ones_like(sums[present]),atol=1e-6,rtol=1e-6)
                    assert not torch.equal(trace[0]['beta'],changed[0]['beta'])
            modes[mode]=dict(passed=True,branch_gradient_norms=norms,
                gates=['four_heads_backward','last_hyperedge_consumed','pair_symmetry','zero_tolerance_replay',
                       'receiver_complete_offload','unit_translation','permutation','exact_duplicates',
                       'empty_edgeless_isolated','zero_and_near_zero_identity','batch_chunk_forward_backward',
                       'directional_normalization_and_independence' if stage!='residual' else 'legacy_association'])
        results[stage]=modes
    setup(42);model=UnifiedDynamicHypergraphEncoder(cfg).to(device).eval()
    only=prepare_surface_geometry(p);assert only['n_nodes']==0 and only['n_observed_edges']==0
    assert len(only['cross_anchor'])==0 and len(only['point_xyz'])==len(np.unique(p,axis=0))
    only_g=to_device(only,device);z=model(only_g)
    query=torch.zeros(1,3,device=device);head=PointLocalCompartmentDecoder(32).to(device)
    head(z,query).sum().backward()
    assert model.surface_stem.stem[0].weight.grad.norm()>0
    assert model.skeleton_stem[0].weight.grad is None or model.skeleton_stem[0].weight.grad.count_nonzero()==0
    with torch.no_grad():
        close(z,encode_offloaded(model,only,edge_chunk=1,state_chunk=2))
        close(model(to_device(prepare_surface_geometry(np.r_[p,p[:3]]),device)),z)
        head(z,query+100);close(model(only_g),z)
        a=model(to_device(prepare_geometry(s,e,np.empty((0,3))),device))
        old_skeleton=dict(prepare_geometry(s,e,np.empty((0,3))),geometry_config=OldGeometry())
        # Same seeded shared weights were used above, and cross update is
        # exact identity in every stage without any surface support.
        close(a,old(to_device(old_skeleton,device)))
    results['point_only_no_skeleton_and_query_readout_only']=True
    results['skeleton_control_unchanged']=True
    # Reusing completed V14 skeleton baselines also requires unchanged heads
    # and initialized effective parameters, not merely a similar core diagram.
    from tools.cpsf_v14_train_20260922.train import SingleTask as OldTask
    from .train import SingleTask
    for task in ('h01','ted','tlp'):
        setup(42);old_task=OldTask(task,'skeleton')
        for stage in ('residual','routing','role'):
            setup(42);new_task=SingleTask(task,'skeleton',stage)
            for name,value in old_task.state_dict().items():
                torch.testing.assert_close(value,new_task.state_dict()[name],atol=0,rtol=0)
    results['all_task_skeleton_parameters_and_head_initialization_unchanged']=True
    # Role modulation is applied before reduction and actually depends on the
    # incidence; removing its parameters exactly recovers the routing stage.
    setup(42);role=UnifiedDynamicHypergraphEncoder(cfg).to(device).eval()
    routing=UnifiedDynamicHypergraphEncoder(replace(cfg,stage='routing')).to(device).eval()
    routing.load_state_dict(role.state_dict())
    with torch.no_grad():
        assert not torch.equal(role(g).surface,routing(g).surface)
        for layer in (role.block.up_role,role.block.down_role):
            layer.weight.zero_();layer.bias.zero_()
        torch.testing.assert_close(role(g).surface,routing(g).surface,atol=0,rtol=0)
        torch.testing.assert_close(role(g).skeleton,routing(g).skeleton,atol=0,rtol=0)
    results['zero_role_maps_exactly_recover_routing']=True
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=dict(status='running',device=a.device,performance_claim=False,torch=torch.__version__,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    try:report.update(contracts=run(a.device),status='passed')
    except Exception as exc:report.update(status='failed',error=repr(exc));raise
    finally:save(a.out,report);print(json.dumps(report),flush=True)
