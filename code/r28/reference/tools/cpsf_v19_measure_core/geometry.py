"""Fixed-unit, label-free geometry. Empirical mass is not mesh area."""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class Config:
    voxel: float = .5
    radius: float = 2.
    unit: str = 'um'

TED_CONFIG=Config(1.,4.,'native_swc')

def tensor(a,dtype=torch.float32):return torch.as_tensor(np.asarray(a),dtype=dtype)

def branches(e,n):
    adj=[[] for _ in range(n)]
    for i,(a,b) in enumerate(e):adj[a].append((b,i));adj[b].append((a,i))
    seen=set();out=np.full(len(e),-1,int);bi=0
    for root in range(n):
        if len(adj[root])==2:continue
        for nxt,ei in adj[root]:
            if ei in seen:continue
            prev,cur=root,nxt
            while ei not in seen:
                seen.add(ei);out[ei]=bi
                if len(adj[cur])!=2:break
                nxt,ej=next(v for v in adj[cur] if v[1]!=ei)
                prev,cur,ei=cur,nxt,ej
            bi+=1
    for ei in range(len(e)):
        if out[ei]<0:out[ei]=bi;bi+=1
    return out

def relations(s,e,c,cfg,degree=None,wrong=False):
    """All patches in the same physical ball; observed segments are soft priors."""
    n=len(s);m=len(c);degree=np.zeros(n) if degree is None else degree
    receivers=[];senders=[];features=[];donors=[]
    edge_br=branches(e,n) if wrong else None
    patch_branch=np.full(m,-1,int)
    # Exact closest-segment assignment only for the wrong-branch diagnostic.
    if wrong and len(e) and m:
        best=np.full(m,np.inf)
        for ei,(a,b) in enumerate(e):
            d=s[b]-s[a];t=np.clip(((c-s[a])*d).sum(1)/max(d@d,1e-20),0,1)
            dist=np.linalg.norm(c-s[a]-t[:,None]*d,axis=1)
            take=dist<best;best[take]=dist[take];patch_branch[take]=edge_br[ei]
    tree=cKDTree(c) if m else None
    incident=[[] for _ in range(n)]
    for a,b in e:incident[a].append(b);incident[b].append(a)
    for q in range(n):
        ids=np.asarray(sorted(tree.query_ball_point(s[q],2*cfg.radius)),int) if m else np.empty(0,int)
        if not len(ids):continue
        delta=c[ids]-s[q];radial=delta.copy();d=np.linalg.norm(delta,axis=1)
        long=np.zeros(len(ids));length=np.zeros(len(ids));has=np.zeros(len(ids));best=d.copy()
        # Coordinate sorting only resolves ties; features themselves are unoriented.
        for b in sorted(incident[q],key=lambda b:tuple(s[b])):
            v=s[b]-s[q];ll=np.linalg.norm(v)
            if ll<1e-12:continue
            t=np.clip((delta*v).sum(1)/(ll*ll),0,1);r=delta-t[:,None]*v;rd=np.linalg.norm(r,axis=1)
            take=rd<best;radial[take]=r[take];best[take]=rd[take]
            long[take]=abs(2*t[take]-1);length[take]=ll;has[take]=1
        phi=np.column_stack((delta/cfg.radius,radial/cfg.radius,best/cfg.radius,long,
            np.log1p(length),np.full(len(ids),np.log(cfg.radius)),np.full(len(ids),np.log1p(degree[q])),has))
        donor=ids.copy()
        if wrong and len(e):
            # Keep recipient relation/measure, perturb evidence everywhere in the only interaction path.
            all_dist=np.linalg.norm(c-s[q],axis=1)
            for branch in np.unique(patch_branch[ids]):
                rows=np.flatnonzero(patch_branch[ids]==branch)
                ok=np.flatnonzero((patch_branch!=branch)&(patch_branch>=0)&(all_dist<=2*cfg.radius))
                if not len(ok):continue
                ok=ok[np.lexsort((ok,all_dist[ok]))];dd=all_dist[ok]
                at=np.searchsorted(dd,d[rows]);lo=np.maximum(at-1,0);hi=np.minimum(at,len(ok)-1)
                chosen=np.where(abs(dd[lo]-d[rows])<=abs(dd[hi]-d[rows]),lo,hi)
                good=abs(dd[chosen]-d[rows])<=np.maximum(cfg.voxel,.2*d[rows])
                donor[rows[good]]=ok[chosen[good]]
        receivers.extend([q]*len(ids));senders.extend(ids);features.extend(phi);donors.extend(donor)
    return dict(receiver=tensor(receivers,torch.long),sender=tensor(senders,torch.long),
                phi=tensor(np.asarray(features).reshape(-1,12)),donor=tensor(donors,torch.long))

def prepare(s,e,p,cfg=Config(),weights=None,wrong=True):
    s=np.asarray(s,dtype=np.float64).reshape(-1,3);e=np.asarray(e,int).reshape(-1,2);p=np.asarray(p,np.float64).reshape(-1,3)
    assert np.isfinite(s).all() and np.isfinite(p).all()
    assert not len(e) or (e.min()>=0 and e.max()<len(s))
    e=np.unique(np.sort(e,axis=1),axis=0)
    assert not len(e) or np.all(e[:,0]!=e[:,1])
    weights=np.ones(len(p)) if weights is None else np.asarray(weights,np.float64)
    assert weights.shape==(len(p),) and np.isfinite(weights).all() and (weights>0).all()
    # Both modalities use a point-derived grid origin. Degree-2 graph sampling cannot move it.
    origin=(p.min(0)+p.max(0))/2 if len(p) else (s.mean(0) if len(s) else np.zeros(3))
    s=s-origin;p=p-origin
    if len(p):
        p,inv=np.unique(p,axis=0,return_inverse=True);w=np.bincount(inv,weights=weights)
        _,owner=np.unique(np.floor(p/cfg.voxel).astype(np.int64),axis=0,return_inverse=True)
        mass=np.bincount(owner,weights=w);c=np.zeros((len(mass),3));np.add.at(c,owner,p*w[:,None]);c/=mass[:,None]
        rel=(p-c[owner])/cfg.voxel;pf=np.column_stack((rel,np.linalg.norm(rel,axis=1)))
    else:
        w=np.empty(0);owner=np.empty(0,int);mass=np.empty(0);c=np.empty((0,3));pf=np.empty((0,4))
    degree=np.bincount(e.ravel(),minlength=len(s)).astype(float)
    ll=np.linalg.norm(s[e[:,0]]-s[e[:,1]],axis=1);total=np.zeros(len(s))
    for side in (0,1):np.add.at(total,e[:,side],ll)
    sf=np.column_stack((np.log1p(degree),np.log1p(total/np.maximum(degree,1)),degree==1,degree>2))
    out=dict(s=tensor(s),e=tensor(e,torch.long),sf=tensor(sf),p=tensor(p),pf=tensor(pf),
        owner=tensor(owner,torch.long),w=tensor(w),mass=tensor(mass),c=tensor(c),
        origin=origin.tolist(),config=asdict(cfg),n=len(s),m=len(c),unit=cfg.unit,
        source_points=int(len(weights)),fov_extent=np.ptp(p,axis=0).tolist() if len(p) else [0,0,0])
    out.update(relations(s,e,c,cfg,degree,wrong))
    return out

def point_view(g,query_xyz=None):
    out=dict(g);out.update(s=g['s'][:0],e=g['e'][:0],sf=g['sf'][:0],n=0)
    for k in ('receiver','sender','donor','phi'):out[k]=g[k][:0]
    if query_xyz is not None:
        q=np.asarray(query_xyz).reshape(-1,3)
        out['query']=relations(q,np.empty((0,2),int),g['c'].numpy(),Config(**g['config']))
        out['query_n']=len(q)
    return out

def skeleton_view(g):
    out=dict(g);out.update(m=0)
    for k in ('p','pf','owner','w','mass','c','receiver','sender','donor','phi'):out[k]=g[k][:0]
    return out

def pack(gs,device):
    ns=np.r_[0,np.cumsum([g['n'] for g in gs])];ms=np.r_[0,np.cumsum([g['m'] for g in gs])]
    out={k:torch.cat([g[k] for g in gs]).to(device) for k in ('s','sf','p','pf','w','mass','c','phi')}
    for k,offset in [('e',ns),('receiver',ns),('sender',ms),('donor',ms),('owner',ms)]:
        out[k]=torch.cat([g[k]+int(offset[i]) for i,g in enumerate(gs)]).to(device)
    out.update(n=int(ns[-1]),m=int(ms[-1]))
    if any('query' in g for g in gs):
        assert all('query' in g for g in gs)
        qs=np.r_[0,np.cumsum([g['query_n'] for g in gs])]
        out['query']={k:torch.cat([g['query'][k]+(int(qs[i]) if k=='receiver' else int(ms[i]) if k in ('sender','donor') else 0) for i,g in enumerate(gs)]).to(device) for k in ('receiver','sender','donor','phi')}
        out['query_n']=int(qs[-1]);out['query_offsets']=qs
    return out,(ns,ms)
