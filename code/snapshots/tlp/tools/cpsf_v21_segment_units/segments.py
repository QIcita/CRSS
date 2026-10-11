"""Observed-graph arclength units and soft surface incidence; no labels/IDs."""
import numpy as np
import torch
from scipy.spatial import cKDTree

def chains(s,e):
    adj=[[] for _ in s]
    for k,(a,b) in enumerate(e):adj[a].append((b,k));adj[b].append((a,k))
    key=lambda a:tuple(s[a]);seen=set();paths=[]
    def walk(start,nxt,edge):
        path=[start];cur=start
        while edge not in seen:
            seen.add(edge);path.append(nxt);cur=nxt
            if len(adj[cur])!=2:break
            options=[(a,k) for a,k in adj[cur] if k not in seen]
            if not options:break
            nxt,edge=min(options,key=lambda v:key(v[0]))
        if key(path[-1])<key(path[0]):path.reverse()
        paths.append(path)
    for a in sorted(range(len(s)),key=key):
        if len(adj[a])==2:continue
        for b,k in sorted(adj[a],key=lambda v:key(v[0])):
            if k not in seen:walk(a,b,k)
    while len(seen)<len(e):
        k=min((k for k in range(len(e)) if k not in seen),key=lambda k:tuple(sorted((key(e[k,0]),key(e[k,1])))))
        a,b=e[k]
        if key(b)<key(a):a,b=b,a
        walk(a,b,k)
    return sorted(paths,key=lambda p:tuple(key(i) for i in p)),adj

def build_units(g,node_units=False):
    s=g['s'].numpy().astype(np.float64);e=g['e'].numpy();c=g['c'].numpy().astype(np.float64)
    radius=float(g['config']['radius']);length=2*radius;units=[];members=[]
    # Each unit holds exact pieces of the observed polyline, not its straight chord.
    if node_units:
        for a in range(len(s)):units.append([(s[a],s[a],0.,0.)]);members.append({a:1.})
    else:
        paths,adj=chains(s,e)
        for path in paths:
            total=sum(np.linalg.norm(s[b]-s[a]) for a,b in zip(path,path[1:]))
            if total<=1e-12:continue
            count=max(1,int(np.ceil(total/length-1e-10)))
            # Equal arclength bins avoid a vanishing last interval.
            step=total/count;parts=[[] for _ in range(count)];weights=[{} for _ in range(count)];cursor=0.
            for a,b in zip(path,path[1:]):
                v=s[b]-s[a];ll=np.linalg.norm(v)
                if ll<=1e-12:continue
                first=max(0,min(count-1,int(np.floor(cursor/step+1e-10))))
                last=max(0,min(count-1,int(np.floor((cursor+ll)/step-1e-10))))
                for k in range(first,last+1):
                    left=max(cursor,k*step);right=min(cursor+ll,(k+1)*step)
                    if right-left<=1e-12:continue
                    t0=(left-cursor)/ll;t1=(right-cursor)/ll;partlen=right-left
                    parts[k].append((s[a]+t0*v,s[a]+t1*v,left-k*step,partlen))
                    weights[k][a]=weights[k].get(a,0.)+partlen*(1-(t0+t1)/2)
                    weights[k][b]=weights[k].get(b,0.)+partlen*(t0+t1)/2
                cursor+=ll
            units.extend(parts);members.extend(weights)
        covered={a for w in members for a,weight in w.items() if weight>0}
        for a in range(len(s)):
            if a not in covered:units.append([(s[a],s[a],0.,0.)]);members.append({a:1.})
    center=[];ulength=[];sr=[];sn=[];sw=[];pr=[];pj=[];raw=[];phi=[]
    tree=cKDTree(c) if len(c) else None
    for u,(pieces,member) in enumerate(zip(units,members)):
        assert pieces
        plen=np.asarray([v[3] for v in pieces]);total=float(plen.sum());ulength.append(total)
        mids=np.stack([(a+b)/2 for a,b,_,_ in pieces])
        center.append((mids*plen[:,None]).sum(0)/total if total>1e-12 else mids[0])
        for node,weight in sorted(member.items()):
            if weight>1e-12:sr.append(u);sn.append(node);sw.append(weight)
        if tree is None:continue
        # Broad-phase bounds contain every patch within 2r of any polyline piece.
        vertices=np.stack([x for a,b,_,_ in pieces for x in (a,b)])
        lo=vertices.min(0);hi=vertices.max(0);mid=(lo+hi)/2
        ids=np.asarray(sorted(tree.query_ball_point(mid,float(np.linalg.norm(hi-lo)/2+2*radius))),dtype=int)
        if not len(ids):continue
        best=np.full(len(ids),np.inf);along=np.zeros(len(ids))
        for a,b,offset,ll in pieces:
            v=b-a;den=float(v@v);t=np.clip(((c[ids]-a)*v).sum(1)/max(den,1e-20),0,1)
            dist=np.linalg.norm(c[ids]-a-t[:,None]*v,axis=1);take=dist<best
            best[take]=dist[take];along[take]=offset+t[take]*ll
        keep=best<=2*radius;ids=ids[keep];best=best[keep];along=along[keep]
        axial=abs(2*along/max(total,1e-12)-1) if total>1e-12 else np.zeros(len(ids))
        distance=best/radius
        feature=np.column_stack((distance,distance**2,axial,axial**2,np.full(len(ids),np.log1p(total/radius))))
        pr.extend([u]*len(ids));pj.extend(ids);raw.extend(np.exp(-.5*distance**2));phi.extend(feature)
    def tensor(x,dtype=torch.float32):return torch.as_tensor(np.asarray(x),dtype=dtype)
    # A patch shares its incidence mass across nearby units; gathering is then
    # normalized over each unit. Empirical patch count, not claimed mesh area.
    raw=np.asarray(raw);pj=np.asarray(pj,dtype=int)
    den=np.bincount(pj,weights=raw,minlength=len(c));pw=raw/np.maximum(den[pj],1e-12)
    return dict(u=len(units),center=tensor(np.asarray(center).reshape(-1,3)),length=tensor(ulength),
        sr=tensor(sr,torch.long),sn=tensor(sn,torch.long),sw=tensor(sw),
        pr=tensor(pr,torch.long),pj=tensor(pj,torch.long),pw=tensor(pw),
        phi=tensor(np.asarray(phi).reshape(-1,5)),target_length=length)
