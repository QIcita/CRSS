"""V21: fixed physical arclength units gather graph and surface into shared regions."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .geometry import pack, point_view, skeleton_view

ARMS = ('full', 'no_fusion', 'no_context', 'no_scale', 'point', 'skeleton', 'no_geometry', 'node_units')
D = 96
CHUNK = 128
CONTEXT = 32

def add(x, i, n):
    return x.new_zeros((n, *x.shape[1:])).index_add(0, i, x)

def mlp(a, b):
    return nn.Sequential(nn.Linear(a,b), nn.GELU(), nn.Linear(b,b))

class Surface(nn.Module):
    def __init__(self, d):
        super().__init__(); self.stem=mlp(4,d); self.output=mlp(2*d,d)

    def forward(self,g):
        x=self.stem(g['pf']); w=g['w'][:,None]
        den=g['mass'][:,None].clamp_min(1e-12)
        mu=add(x*w,g['owner'],g['m'])/den
        second=add(x.square()*w,g['owner'],g['m'])/den
        return self.output(torch.cat((mu,(second-mu.square()).clamp_min(0)),-1))

class Exchange(nn.Module):
    def __init__(self,d,residual_scale=1.):
        super().__init__(); self.residual_scale=float(residual_scale); self.q=nn.Linear(d,d); self.k=nn.Linear(d,d)
        self.v=nn.Linear(d,d); self.bias=nn.Sequential(nn.Linear(12,32),nn.GELU(),nn.Linear(32,1))
        self.geometry=mlp(12,d); self.out=nn.Linear(d,d); self.norm=nn.LayerNorm(d)

    def forward(self,h,p,g):
        r,j=g['receiver'],g['sender']; n=len(h)
        if not len(r):return h
        phi=g['phi']; score=(self.q(h)[r]*self.k(p)[j]).sum(-1)/math.sqrt(h.shape[-1])
        score=score+self.bias(phi).squeeze(-1)
        peak=score.new_full((n,),-torch.inf).scatter_reduce(0,r,score.detach(),'amax')
        ex=(score-peak[r]).exp(); a=ex/add(ex,r,n)[r].clamp_min(1e-12)
        u=add(a[:,None]*(self.v(p)[j]+self.geometry(phi)),r,n)
        has=add(torch.ones_like(a),r,n)>0
        return h+self.residual_scale*self.out(self.norm(u))*has[:,None]


class LocalExchange(nn.Module):
    """One common region update, lifted through normalized hyperedge incidence."""
    def __init__(self,d=D,residual_scale=1.):
        super().__init__();self.residual_scale=float(residual_scale)
        self.hnorm=nn.LayerNorm(d);self.pnorm=nn.LayerNorm(d)
        self.geometry=mlp(5,d);self.joint=mlp(2*d,d)
        self.to_h=nn.Linear(d,d,bias=False);self.to_p=nn.Linear(d,d,bias=False)
        nn.init.zeros_(self.to_h.weight);nn.init.zeros_(self.to_p.weight)

    def forward(self,h,p,g,geometry=True):
        if not len(g['pr']):return h,p
        count=g['u'];sr,sn,sw=g['sr'],g['sn'],g['sw'];pr,pj,pw=g['pr'],g['pj'],g['pw']
        graph=add(self.hnorm(h)[sn]*sw[:,None],sr,count)/add(sw,sr,count).clamp_min(1e-12)[:,None]
        values=self.pnorm(p)[pj]
        if geometry:values=values+self.geometry(g['phi'])
        den=add(pw,pr,count)
        surface=add(values*pw[:,None],pr,count)/den.clamp_min(1e-12)[:,None]
        region=self.joint(torch.cat((graph,surface),-1))*(den>0)[:,None]
        # Graph nodes at junctions share adjacent regions; patches use soft
        # partition-of-unity incidence. Neither direction updates the other first.
        gh=add(region[sr]*sw[:,None],sn,len(h))/add(sw,sn,len(h)).clamp_min(1e-12)[:,None]
        gp=add(region[pr]*pw[:,None],pj,len(p))
        return h+self.residual_scale*self.to_h(gh),p+self.residual_scale*self.to_p(gp)

def rotary(x,xyz):
    # x: B,H,L,24; eight channels per axis, four rotary pairs per axis.
    b,h,l,d=x.shape; assert d%6==0
    v=x.reshape(b,h,l,3,d//6,2)
    freq=10000.**(-torch.arange(d//6,device=x.device,dtype=x.dtype)/(d//6))
    phase=xyz[:,None,:,:,None]*freq
    co,si=phase.cos(),phase.sin();a,c=v[...,0],v[...,1]
    return torch.stack((a*co-c*si,a*si+c*co),-1).reshape(b,h,l,d)

def spatial_chunks(xyz, limit=CHUNK):
    """Geometry-only longest-axis median partition; no labels or features."""
    xyz=np.asarray(xyz)
    pending=[np.arange(len(xyz))];out=[]
    while pending:
        ids=pending.pop()
        if len(ids)<=limit:
            if len(ids):out.append(ids)
            continue
        axis=np.ptp(xyz[ids],axis=0).argmax()
        ids=ids[np.argsort(xyz[ids,axis],kind='stable')];mid=len(ids)//2
        pending.extend((ids[mid:],ids[:mid]))
    return out

class Hierarchy(nn.Module):
    def __init__(self,d=D,residual_scale=1.):
        super().__init__();self.residual_scale=float(residual_scale);self.qkv=nn.Linear(d,3*d);self.out=nn.Linear(d,d)
        self.norm=nn.LayerNorm(d);self.ffnorm=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))

    def forward(self,x,xyz,groups,context=True):
        if not len(x):return x
        summaries=torch.stack([x[g].mean(0) for g in groups])
        centers=torch.stack([xyz[g].mean(0) for g in groups])
        if len(groups)>CONTEXT:
            bins=np.array_split(np.arange(len(groups)),CONTEXT)
            sizes=x.new_tensor([len(g) for g in groups])
            summaries=torch.stack([(summaries[z]*sizes[z,None]).sum(0)/sizes[z].sum() for z in bins])
            centers=torch.stack([(centers[z]*sizes[z,None]).sum(0)/sizes[z].sum() for z in bins])
        if not context:summaries=summaries[:0];centers=centers[:0]
        length=max(len(g) for g in groups);ng=len(groups)
        indices=torch.zeros((ng,length),dtype=torch.long,device=x.device)
        valid=torch.zeros((ng,length),dtype=torch.bool,device=x.device)
        for i,g in enumerate(groups):indices[i,:len(g)]=g;valid[i,:len(g)]=True
        local=x[indices];pos=xyz[indices]
        tokens=torch.cat((local,summaries[None].expand(ng,-1,-1)),1)
        coords=torch.cat((pos,centers[None].expand(ng,-1,-1)),1)
        mask=torch.cat((valid,torch.ones((ng,len(summaries)),dtype=torch.bool,device=x.device)),1)
        q,k,v=self.qkv(self.norm(tokens)).chunk(3,-1)
        def heads(t):return t.reshape(ng,-1,4,D//4).transpose(1,2)
        q=rotary(heads(q),coords)[:,:,:length];k=rotary(heads(k),coords);v=heads(v)
        # Explicit math backend gives deterministic backward and honors padding.
        with sdpa_kernel(SDPBackend.MATH):
            y=F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:],dropout_p=0.)
        y=y.transpose(1,2).reshape(ng,length,D)
        y=local+self.residual_scale*self.out(y);y=y+self.residual_scale*self.ff(self.ffnorm(y))
        flat=indices[valid];assert len(flat)==len(x)
        return x.new_zeros(x.shape).index_copy(0,flat,y[valid])

class Encoder(nn.Module):
    def __init__(self,d=D,residual_scale=1.):
        super().__init__();self.d=d;self.surface=Surface(d);self.graph_stem=mlp(4,d)
        self.message=mlp(2*d+4,d);self.graph_update=mlp(2*d,d);self.graph_norm=nn.LayerNorm(d)
        self.exchange=Exchange(d,residual_scale);self.scale=mlp(5,d);self.position=mlp(3,d)
        self.modality=nn.Parameter(torch.zeros(2,d));self.blocks=nn.ModuleList([Hierarchy(d,residual_scale),Hierarchy(d,residual_scale)])

    def stems(self,gs,arm):
        device=next(self.parameters()).device
        clean=[]
        for g in gs:
            if arm=='point':g=point_view(g)
            elif arm=='skeleton':g=skeleton_view(g)
            clean.append(g)
        packed,offsets=pack(clean,device);p=self.surface(packed);h=self.graph_stem(packed['sf'])
        a,b=packed['e'].unbind(1);r=torch.cat((a,b));j=torch.cat((b,a))
        delta=packed['s'][j]-packed['s'][r]
        for _ in range(2):
            msg=self.message(torch.cat((h[r],h[j]-h[r],delta,delta.norm(dim=-1,keepdim=True)),-1))
            den=add(h.new_ones(len(r)),r,len(h)).clamp_min(1)[:,None]
            h=self.graph_norm(h+self.graph_update(torch.cat((h,add(msg,r,len(h))/den),-1)))
        # Former early exchange remains an inactive parameter block for seed fidelity.
        ns,ms=offsets;out=[]
        for i,g in enumerate(clean):
            sh=h[ns[i]:ns[i+1]];sp=p[ms[i]:ms[i+1]]
            sx=packed['s'][ns[i]:ns[i+1]];px=packed['c'][ms[i]:ms[i+1]]
            coords=torch.cat((sx,px)); origin=(coords.amin(0)+coords.amax(0))/2 if len(coords) else coords.new_zeros(3)
            extent=(coords.amax(0)-coords.amin(0)).clamp_min(1e-4) if len(coords) else coords.new_ones(3)
            scale=coords.new_tensor([g['config']['voxel'],g['config']['radius']])
            code=self.scale(torch.cat((extent.log1p(),scale.log1p()))) if arm!='no_scale' else coords.new_zeros(D)
            sh=sh+self.position((sx-origin)/extent)+code+self.modality[0]
            sp=sp+self.position((px-origin)/extent)+code+self.modality[1]
            unit=g.get('units_node' if arm=='node_units' else 'units_segment') if arm in ('full','no_context','no_scale','no_geometry','node_units') else None
            if arm in ('full','no_context','no_scale','no_geometry','node_units') and unit is None:
                from .segments import build_units
                unit=build_units(g,node_units=arm=='node_units')
            unit={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in unit.items()} if unit is not None else None
            out.append(dict(h=sh,p=sp,s=sx,c=px,mass=packed['mass'][ms[i]:ms[i+1]],radius=g['config']['radius'],units=unit))
        return out

    def propagate(self,bundles,arm,offsets=None):
        # All views of ONE cell share context; different cells never do.
        xs=[];positions=[];groups=[];boundaries=[];base=0
        for i,b in enumerate(bundles):
            xyz=torch.cat((b['s'],b['c']));x=torch.cat((b['h'],b['p']))
            if offsets is not None:xyz=xyz+xyz.new_tensor(offsets[i])
            origin=xyz.mean(0) if len(xyz) else xyz.new_zeros(3)
            # Group assignments do not depend on task labels or learned features.
            local_groups=spatial_chunks(xyz.detach().cpu().numpy())
            groups.extend([torch.as_tensor(g,device=x.device)+base for g in local_groups])
            boundaries.append((base,len(b['h']),len(b['p'])))
            xs.append(x);positions.append(xyz/b['radius']);base+=len(x)
        x=torch.cat(xs);pos=torch.cat(positions)
        if len(pos):pos=pos-pos.mean(0)
        # Both modality streams retain their scale-position hierarchy.
        for modal in (0,1):
            ids=torch.cat([torch.arange(a+(n if modal else 0),a+n+(m if modal else 0),device=x.device) for a,n,m in boundaries])
            if not len(ids):continue
            y=x[ids];xyz=pos[ids]
            gs=[torch.as_tensor(g,device=x.device) for g in spatial_chunks(xyz.detach().cpu().numpy())]
            for block in self.blocks:y=block(y,xyz,gs,arm!='no_context')
            x=x.index_copy(0,ids,y)
        result=[]
        for b,(a,n,m) in zip(bundles,boundaries):
            h,p=x[a:a+n],x[a+n:a+n+m]
            if arm in ('full','no_context','no_scale','no_geometry','node_units'):
                h,p=self.local_exchange(h,p,b['units'],arm!='no_geometry')
            result.append(dict(b,h=h,p=p))
        return result

class Model(nn.Module):
    def __init__(self,task,arm,residual_scale=1.):
        super().__init__();assert arm in ARMS;self.task=task;self.arm=arm;self.encoder=Encoder(residual_scale=residual_scale)
        self.readout=nn.Sequential(nn.Linear(2*D,128),nn.GELU())
        self.head=nn.Sequential(mlp(128 if task=='h01' else 256,96),nn.Linear(96,3 if task=='h01' else 2))
        # Construct last so all shared initial weights exactly match V20.1.
        self.encoder.local_exchange=LocalExchange(D,residual_scale)

    def local_feature(self,b,q):
        zero=next(self.parameters()).new_zeros(D)
        h=b['h'][q] if len(b['h']) else zero
        # Query location is the task's given location, not a skeleton feature.
        xyz=b['query_xyz'];dist=(b['c']-xyz).norm(dim=-1)
        ids=torch.nonzero(dist<=2*b['radius']).flatten()
        if len(ids):
            w=(-dist[ids]/b['radius']).softmax(0);p=(b['p'][ids]*w[:,None]).sum(0)
        else:p=zero
        return self.readout(torch.cat((h,p)))

    def global_feature(self,bs):
        zero=next(self.parameters()).new_zeros(D)
        hv=[];pv=[];weights=[]
        for b in bs:
            hv.append(b['h'].mean(0) if len(b['h']) else zero)
            pv.append((b['p']*b['mass'][:,None]).sum(0)/b['mass'].sum().clamp_min(1e-12) if len(b['p']) else zero)
            weights.append(float(b['mass'].sum()))
        h,p=hv[0],pv[0]
        if len(bs)>1:
            h=(h+torch.stack(hv[1:]).mean(0))/2
            w=h.new_tensor(weights[1:]).clamp_min(1);p=(p+(torch.stack(pv[1:])*w[:,None]).sum(0)/w.sum())/2
        return self.readout(torch.cat((h,p)))

    def forward(self,gs,queries=None):
        qs=queries if queries is not None else [0]*len(gs)
        bundles=self.encoder.stems(gs,self.arm);features=[]
        for g,b,q in zip(gs,bundles,qs):
            b=self.encoder.propagate([b],self.arm)[0]
            b['query_xyz']=g['s'][q].to(next(self.parameters()).device)
            z=self.local_feature(b,q)
            if self.task=='ted':z=torch.cat((z,self.global_feature([b])))
            features.append(z)
        logits=self.head(torch.stack(features))
        return logits if self.task=='h01' else logits-logits.mean(1,keepdim=True)
