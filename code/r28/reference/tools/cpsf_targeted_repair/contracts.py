import time,gc
from .common import *
from .model import Model,pair_geometry,TEDBase,TLPBase
from .train import state_digest,average_precision
from tools.cpsf_missing_edge.objective import missing_edge_loss
from torch.nn import functional as F
from sklearn.metrics import average_precision_score

def main():
    setup();p=verify();report=dict(passed=False,protocol_sha256=sha(ROOT/'protocol.json'),tasks={},test_used=False);begin=time.time()
    assert json.loads((ROOT/'preparation.json').read_text())['passed']
    split=json.loads((ROOT/'development_split.json').read_text())['tasks']
    for task in ['ted','tlp']:
        rr=records(task);tr=[r for r in rr if r['split']=='train'];dev=[r for r in rr if r['split']=='dev']
        assert all(r['parent_split']=='train' for r in rr);assert {r['group'] for r in tr}.isdisjoint(r['group'] for r in dev)
        # Independently verify spatial boxes, not just group names.
        a=[r for r in split[task]['records'] if r['split']=='train'];b=[r for r in split[task]['records'] if r['split']=='dev']
        for x in a:
            for y in b:
                overlap=np.minimum(np.array(x['origin'])+x['shape_xyz'],np.array(y['origin'])+y['shape_xyz'])-np.maximum(x['origin'],y['origin'])
                assert not (overlap>0).all(),('overlapping development boxes',x['key'],y['key'])
        setup();control=Model(task,'control');initial=state_digest(control)
        setup();repair=Model(task,'repair')
        for key,value in control.state_dict().items():
            other=repair.state_dict()[key]
            if task=='tlp' and key=='head.0.weight':assert torch.equal(value,other[:,:384]) and not other[:,384:].any()
            else:assert torch.equal(value,other),(task,key)
        setup();base=TEDBase('ted','full') if task=='ted' else TLPBase('tlp','full','directional_xyz')
        assert initial==state_digest(base)
        assert control.use_position and repair.use_position and control.use_scale and repair.use_scale
        assert control.processor.use_context and repair.processor.use_context
        ordered=sorted(tr,key=lambda r:(r['regions'] if task=='ted' else r['query_nodes'],r['key']))
        selected=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
        results=[];control=control.cuda();repair=repair.cuda();base=base.cuda()
        for r in selected:
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            old=item_load(r,task,'control');new=item_load(r,task,'repair');go=move(old['g']);gn=move(new['g']);y=old['target'].cuda()
            for key,value in old['g'].items():
                if isinstance(value,torch.Tensor):assert torch.equal(value,new['g'][key])
            control.eval();repair.eval();base.eval()
            with torch.no_grad():
                zo=control(go);zb=base(go);zn=repair(gn)
            assert torch.equal(zo,zb)
            if task=='tlp':
                assert (zo-zn).abs().max()<2e-6
                assert (zn-zn.T).abs().max()<1e-6
                q=go['query_xyz'].cpu().numpy();obs=old['observed'].numpy();perm=np.random.RandomState(42).permutation(len(q))
                geo=gn['pair_geometry'].cpu().numpy();np.testing.assert_allclose(pair_geometry(q[perm],obs[np.ix_(perm,perm)]),geo[perm][:,perm],atol=1e-6)
                np.testing.assert_allclose(pair_geometry(q.astype(np.float64)+[20,30,40],obs),geo,atol=1e-6)
                mask=np.triu(~obs,1);labels=old['target'].numpy()[mask]>0;scores=zn.sigmoid().cpu().numpy()[mask]
                assert abs(average_precision(labels,scores)-average_precision_score(labels,scores))<1e-10
            repair.train();repair.zero_grad(set_to_none=True);z=repair(gn)
            loss=F.binary_cross_entropy_with_logits(z,y)*(5 if y[1]>0 else 1) if task=='ted' else missing_edge_loss(z,y,old['observed'].cuda())
            loss.backward();assert torch.isfinite(loss)
            active=[n for n,v in repair.named_parameters() if v.grad is not None and bool((v.grad!=0).any())]
            assert all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in repair.parameters())
            for prefix in ['position.','scale.','joint.','processor.']:assert any(n.startswith(prefix) for n in active),(task,prefix)
            if task=='tlp':assert (repair.head[0].weight.grad[:,384:]!=0).any()
            torch.optim.Adam(repair.parameters(),lr=p['tasks'][task]['lr'],weight_decay=p['tasks'][task]['weight_decay']).step()
            assert all(torch.isfinite(v).all() for v in repair.parameters())
            results.append(dict(key=r['key'],regions=r['regions'],nodes=r.get('query_nodes',r['nodes']),loss=float(loss),parent_output_exact=True,repair_initial_output_maxdiff=float((zo-zn).abs().max()),peak_gpu_bytes=torch.cuda.max_memory_allocated(),active_gradient_parameters=active))
            # Restore matched initial weights before the next independent acceptance case.
            setup();fresh=Model(task,'repair');repair.load_state_dict(fresh.state_dict());del fresh,go,gn,z,zo,zb,zn,old,new
        report['tasks'][task]=dict(initial_control_sha256=initial,shared_initial_weights_equal=True,position_scale_retained=True,spatial_boxes_disjoint=True,samples=results)
        del control,repair,base;gc.collect();torch.cuda.empty_cache()
    report.update(passed=True,seconds=time.time()-begin);save(ROOT/'contracts.json',report);print(json.dumps({t:max(r['peak_gpu_bytes'] for r in v['samples']) for t,v in report['tasks'].items()}),flush=True)
if __name__=='__main__':main()
