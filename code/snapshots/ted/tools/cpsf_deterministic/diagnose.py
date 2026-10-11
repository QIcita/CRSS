import json,time,types
from pathlib import Path
import torch
from tools.cpsf_point_geometry.model import Model
from tools.cpsf_point_geometry.common import setup,load,move
from tools.cpsf_missing_edge.objective import missing_edge_loss

PARENT=Path('/root/megrez-tmp/CPSF_TLP_POINT_GEOMETRY_STAGE3_20261008')
ROOT=Path(__file__).resolve().parents[2]

class FixedAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,x):
        assert not a.requires_grad
        ctx.save_for_backward(a.transpose(0,1).coalesce())
        return torch.sparse.mm(a,x)
    @staticmethod
    def backward(ctx,grad):
        (at,)=ctx.saved_tensors
        return None,torch.sparse.mm(at,grad)

def pool(self,x,g,mass,active):
    row,col=g['bridge_row'],g['bridge_col'];n=g['pair_count']
    weight=g['bridge_weight']*mass[col]*active[col]
    den=x.new_zeros(n).index_add(0,row,weight)
    weight=weight/den[row].clamp_min(1e-12)
    a=torch.sparse_coo_tensor(torch.stack((row,col)),weight,(n,len(x))).coalesce()
    if self.mode=='adjoint':out=FixedAdjoint.apply(a,x)
    elif self.mode=='index_add':out=x.new_zeros(n,x.shape[1]).index_add(0,row,x[col]*weight[:,None])
    else:out=torch.sparse.mm(a,x)
    return out,den>0

def main():
    setup();results={}
    for arm in ['skeleton','full','point']:
        modal=arm if arm!='full' else 'joint'
        rr=json.loads((PARENT/'data/tlp'/modal/'manifest.json').read_text())['records']
        r=max((r for r in rr if r['split']=='train'),key=lambda r:r['query_nodes'])
        item=load(r['path']);g=move(item['g']);y=item['target'].cuda();obs=item['observed'].cuda()
        setup();m=Model('tlp',arm).cuda();m.train();m.corridor_pool=types.MethodType(pool,m)
        reference_output=None;results[arm]={}
        for mode in ['native','adjoint','index_add']:
            m.mode=mode;ref=None;records=[];begin=time.time();torch.cuda.reset_peak_memory_stats()
            for repeat in range(5):
                m.zero_grad(set_to_none=True);z=m(g);loss=missing_edge_loss(z,y,obs);loss.backward()
                out=z.detach().cpu();grad={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None}
                if ref is None:ref=(out,grad)
                if reference_output is None:reference_output=out
                diff={n:float((v-ref[1][n]).abs().max()) for n,v in grad.items() if not torch.equal(v,ref[1][n])}
                records.append(dict(repeat=repeat,output_equal=torch.equal(out,ref[0]),gradient_equal=not diff,max_gradient_delta=max(diff.values(),default=0),different_tensors=len(diff),output_native_max_delta=float((out-reference_output).abs().max())))
                del z,loss
            results[arm][mode]=dict(records=records,seconds=time.time()-begin,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            print(arm,mode,results[arm][mode],flush=True)
        del m,g;torch.cuda.empty_cache()
    out=dict(unix=time.time(),deterministic_algorithms=True,optimizer_steps=0,results=results,test_used=False)
    (ROOT/'runtime/operator_diagnosis.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
