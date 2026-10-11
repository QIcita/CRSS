"""Preserve original skeleton support; never form a budget-thinned induced graph."""
import numpy as np
from tools.cpsf_v21_segment_units.geometry import prepare, tensor
from tools.cpsf_v21_segment_units.model import spatial_chunks

def geometry(s,e,p,weights=None):
    # V21's legacy node/patch exchange is inactive. Its expensive, unused relation
    # table can be empty; active segment relations are still computed unchanged.
    s=np.asarray(s,float).reshape(-1,3);e=np.asarray(e,int).reshape(-1,2)
    p=np.asarray(p,float).reshape(-1,3)
    g=prepare(np.empty((0,3)),np.empty((0,2),int),p,weights=weights,wrong=False)
    if not len(p):g['origin']=s.mean(0).tolist() if len(s) else [0.,0.,0.]
    s=s-np.asarray(g['origin']);e=np.unique(np.sort(e,axis=1),axis=0)
    assert not len(e) or (e.min()>=0 and e.max()<len(s) and (e[:,0]!=e[:,1]).all())
    degree=np.bincount(e.ravel(),minlength=len(s)).astype(float)
    length=np.linalg.norm(s[e[:,0]]-s[e[:,1]],axis=1);total=np.zeros(len(s))
    for side in (0,1):np.add.at(total,e[:,side],length)
    sf=np.column_stack((np.log1p(degree),np.log1p(total/np.maximum(degree,1)),degree==1,degree>2))
    g.update(s=tensor(s),e=tensor(e,__import__('torch').long),sf=tensor(sf),n=len(s))
    return g

def support(s,e,ids):
    ids=np.unique(ids);lut=np.full(len(s),-1,int);lut[ids]=np.arange(len(ids))
    keep=(lut[e[:,0]]>=0)&(lut[e[:,1]]>=0)
    return s[ids],lut[e[keep]],ids

def geometries(x,arm):
    rng=np.random.default_rng(42);empty=np.empty((0,3));edges=np.empty((0,2),int);out=[]
    if arm=='skeleton':
        s=x['skeleton_xyz_um'];e=x['edge_index'].astype(int)
        out.append(geometry(s,e,empty))
        chunks=spatial_chunks(s,max(1,int(np.ceil(len(s)/min(32,max(1,int(np.ceil(len(s)/4096))))))))
        for ids in chunks:
            # Include observed edges incident to a core node, retaining endpoints.
            take=np.zeros(len(s),bool);take[ids]=True
            boundary=e[take[e[:,0]]|take[e[:,1]]]
            ss,ee,_=support(s,e,np.union1d(ids,boundary.ravel()))
            out.append(geometry(ss,ee,empty))
        return out
    p=x['surface_xyz_um'];s=empty if arm=='point' else x['skeleton_xyz_um'];e=edges if arm=='point' else x['edge_index'].astype(int)
    def view(ids,nodeids,points):
        chosen=np.sort(rng.choice(ids,min(len(ids),points),replace=False))
        ss,ee,_=support(s,e,nodeids)
        return geometry(ss,ee,p[chosen],np.full(len(chosen),len(ids)/max(len(chosen),1)))
    out.append(view(np.arange(len(p)),np.arange(len(s)),4096))
    chunks=[np.arange(len(p))];target=min(32,max(1,int(np.ceil(len(p)/16384))))
    while len(chunks)<target:
        k=max(range(len(chunks)),key=lambda i:len(chunks[i]));ids=chunks.pop(k)
        if len(ids)<2:chunks.insert(k,ids);break
        axis=np.ptp(p[ids],axis=0).argmax();ids=ids[np.argsort(p[ids,axis],kind='stable')];mid=len(ids)//2
        chunks[k:k]=[ids[:mid],ids[mid:]]
    for ids in chunks:
        lo=p[ids].min(0)-4.;hi=p[ids].max(0)+4.
        nodeids=np.flatnonzero(((s>=lo)&(s<=hi)).all(1))
        if len(e):
            cross=((np.maximum(s[e[:,0]],s[e[:,1]])>=lo)&(np.minimum(s[e[:,0]],s[e[:,1]])<=hi)).all(1)
            nodeids=np.union1d(nodeids,e[cross].ravel())
        out.append(view(ids,nodeids,2048))
    return out
