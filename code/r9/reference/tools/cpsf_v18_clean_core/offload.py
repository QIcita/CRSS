"""Exact equations with CPU state storage and bounded CUDA work tiles.

Inference only, common to any dataset. No support/point omission and no change
to neighborhoods at tile boundaries. Not an end-to-end cell training route.
"""
import torch
from .encoder import JointEmbedding, add, mean, sparsemax


@torch.no_grad()
def encode_offloaded(model,g,point_chunk=4096,edge_chunk=1024,state_chunk=16384):
    assert not model.training and g['point_xyz'].device.type=='cpu'
    assert g['geometry_config']==model.config.geometry
    device=next(model.parameters()).device;d=model.config.dim;n=g['n_nodes'];m=g['n_patches'];q=g.get('n_segments',0)
    move=lambda x:x.to(device)
    stem=model.surface_stem;f=g['point_features'];p=g['point_xyz']
    total=torch.zeros(m,d);maximum=torch.full((m,d),-torch.inf)
    for start in range(0,len(f),point_chunk):
        sl=slice(start,start+point_chunk);idx=g['point_neighbors'][sl]
        center=stem.stem(move(f[sl]));other=stem.stem(move(f[idx]));delta=move(p[idx]-p[sl,None])
        rel=stem.relation(torch.cat((center[:,None].expand_as(other),other-center[:,None],delta,delta.norm(dim=-1,keepdim=True)),-1))
        rel=rel.masked_fill(~move(g['neighbor_valid'][sl])[:,:,None],-torch.inf).amax(1)
        token=stem.norm(center+rel).cpu();owner=g['point_patch'][sl]
        total.index_add_(0,owner,token)
        maximum.scatter_reduce_(0,owner[:,None].expand_as(token),token,'amax')
    state=torch.empty(n+m,d)
    for start in range(0,n,state_chunk):
        sl=slice(start,min(n,start+state_chunk));state[sl]=model.skeleton_stem(move(g['skeleton_features'][sl])).cpu()
    for start in range(0,m,state_chunk):
        sl=slice(start,min(m,start+state_chunk));mass=g['patch_point_count'][sl,None].float()
        state[n+start:n+sl.stop]=stem.output(move(torch.cat((total[sl]/mass,maximum[sl],torch.log1p(mass)),-1))).cpu()
    del total,maximum
    block=model.block;receiver=g['local_receiver'];sender=g['local_sender']
    ptr=torch.cat((torch.zeros(1,dtype=torch.long),torch.bincount(receiver,minlength=n+m).cumsum(0)))
    anchor,surface=g['cross_anchor'],g['cross_surface']
    cross_ptr=torch.cat((torch.zeros(1,dtype=torch.long),torch.bincount(anchor,minlength=n).cumsum(0)))
    # Group downlink by its receiver, not by the uplink anchor tile. This is
    # essential: a patch's candidates can span several anchor tiles.
    surface_order=torch.argsort(surface,stable=True)
    surface_ptr=torch.cat((torch.zeros(1,dtype=torch.long),torch.bincount(surface,minlength=m).cumsum(0)))
    seg_anchor=g.get('cross_segment',torch.empty(0,dtype=torch.long))
    seg_surface=g.get('cross_segment_surface',torch.empty(0,dtype=torch.long))
    seg_cross_ptr=torch.cat((torch.zeros(1,dtype=torch.long),torch.bincount(seg_anchor,minlength=q).cumsum(0))) if q else torch.zeros(1,dtype=torch.long)
    seg_surface_order=torch.argsort(seg_surface,stable=True) if len(seg_surface) else seg_surface
    seg_surface_ptr=torch.cat((torch.zeros(1,dtype=torch.long),torch.bincount(seg_surface,minlength=m).cumsum(0))) if q else torch.zeros(1,dtype=torch.long)
    def tiles(ptr,groups,budget,cap):
        start=0
        while start<groups:
            end=min(groups,start+cap)
            limit=int(ptr[start])+budget
            end=max(start+1,min(end,int(torch.searchsorted(ptr,torch.tensor(limit),right=True))-1))
            yield start,end,slice(int(ptr[start]),int(ptr[end]))
            start=end
    edge=state[:n].clone();saved=[];saved_membership=None
    segment=(model.segment_stem(move(g['segment_features'])).cpu() if q else torch.empty((0,d)))
    if q:
        segment=segment+.5*(state[g['observed_edges'][:,0]]+state[g['observed_edges'][:,1]])
    saved_segment_membership=None
    for step in range(model.config.rounds):
        local=torch.empty_like(state)
        for start,end,sl in tiles(ptr,n+m,32768,state_chunk):
            rr=receiver[sl];ss=sender[sl]
            if not model.config.surface_context:
                keep=rr<n;rr=rr[keep];ss=ss[keep]
            target=move(state[rr]);source=move(state[ss])
            value=block.local_messages(target,source,move(g['positions'][ss]-g['positions'][rr]),move((rr>=n).long()))
            total=add(value,move(rr-start),end-start)
            count=add(value.new_ones(len(rr)),move(rr-start),end-start)
            local[start:end]=block.local_finish(move(state[start:end]),total,count).cpu()
        # No in-place mutation of any state still needed by another tile.
        del state
        if model.config.stage != 'legacy' and q and len(seg_anchor):
            seg_membership=torch.empty(len(seg_anchor));new_segment=torch.empty_like(segment)
            seg_support=torch.empty(q)
            for start,end,sl in tiles(seg_cross_ptr,q,32768,edge_chunk):
                aa=seg_anchor[sl];pp=seg_surface[sl];ids=move(aa-start)
                ss=move(segment[start:end]);sp=move(local[n+pp])
                delta=move(g['patch_xyz'][pp]-g['segment_xyz'][aa])
                geo=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
                if model.config.association=='once' and step>0:
                    member=move(saved_segment_membership[sl])
                else:
                    member=block.member_strength(ss,sp,ss,ids,geo,model.config.association)
                seg_membership[sl]=member.cpu()
                features=block.route_features(ss[ids],sp,ss[ids],geo)
                alpha=block.route(features,member,ids,end-start,'up')
                pool=add(block.upward(sp,geo,features)*alpha[:,None],ids,end-start)
                support=mean(member,ids,end-start)
                new_segment[start:end]=block.group_finish(ss,ss,pool,support).cpu()
                seg_support[start:end]=support.cpu()
            active=(seg_support>0)[:,None]
            segment=torch.where(active,new_segment,segment)
            if model.config.association=='once' and step==0:saved_segment_membership=seg_membership.clone()
            # Segment-to-surface downlink, grouped by surface receiver.
            for start,end,sl in tiles(seg_surface_ptr,m,32768,state_chunk):
                idx=seg_surface_order[sl];aa=seg_anchor[idx];pp=seg_surface[idx];ids=move(pp-start)
                ss=move(segment[aa]);sp=move(local[n+pp])
                delta=move(g['patch_xyz'][pp]-g['segment_xyz'][aa])
                geo=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
                features=block.route_features(ss,sp,ss,geo);member=move(seg_membership[idx])
                beta=block.route(features,member,ids,end-start,'down')
                message=add(block.downward(ss,features)*beta[:,None],ids,end-start)
                support=mean(member,ids,end-start)
                local[n+start:n+end]=block.receive(move(local[n+start:n+end]),message,support).cpu()
            # Segment-to-skeleton endpoint feedback.  Aggregate on CPU, then
            # apply the existing local update in bounded CUDA chunks.
            endpoint_ids=torch.cat((g['observed_edges'][:,0],g['observed_edges'][:,1]))
            endpoint_total=add(torch.cat((segment*seg_support[:,None],segment*seg_support[:,None])),endpoint_ids,n)
            endpoint_count=add(endpoint_total.new_ones(len(endpoint_ids))*torch.cat((seg_support,seg_support)),endpoint_ids,n)
            for start in range(0,n,state_chunk):
                end=min(n,start+state_chunk)
                local[start:end]=block.local_finish(move(local[start:end]),move(endpoint_total[start:end]),move(endpoint_count[start:end])).cpu()
            del seg_membership,new_segment,seg_support
        if block.decoupled:
            membership=torch.empty(len(anchor));new_edge=torch.empty_like(edge)
            group_support=torch.empty(n)
            for start,end,sl in tiles(cross_ptr,n,32768,edge_chunk):
                aa=anchor[sl];pp=surface[sl];ids=move(aa-start)
                sa=move(local[start:end]);sp=move(local[n+pp]);eg=move(edge[start:end])
                delta=move(g['patch_xyz'][pp]-g['skeleton_xyz'][aa])
                geo=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
                if model.config.association=='once' and step>0:
                    member=move(saved_membership[sl])
                else:
                    member=block.member_strength(sa,sp,eg,ids,geo,model.config.association)
                membership[sl]=member.cpu()
                features=block.route_features(sa[ids],sp,eg[ids],geo)
                alpha=block.route(features,member,ids,end-start,'up')
                pool=add(block.upward(sp,geo,features)*alpha[:,None],ids,end-start)
                support=mean(member,ids,end-start)
                new_edge[start:end]=block.group_finish(sa,eg,pool,support).cpu()
                group_support[start:end]=support.cpu()
            edge=new_edge
            if model.config.association=='once' and step==0:saved_membership=membership.clone()
            state=torch.empty_like(local)
            for start in range(0,n,state_chunk):
                sl=slice(start,min(n,start+state_chunk))
                state[sl]=block.receive(move(local[sl]),move(edge[sl]),move(group_support[sl])).cpu()
            for start,end,sl in tiles(surface_ptr,m,32768,state_chunk):
                idx=surface_order[sl];aa=anchor[idx];pp=surface[idx];ids=move(pp-start)
                sa=move(local[aa]);sp=move(local[n+pp]);eg=move(edge[aa])
                delta=move(g['patch_xyz'][pp]-g['skeleton_xyz'][aa])
                geo=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
                features=block.route_features(sa,sp,eg,geo);member=move(membership[idx])
                beta=block.route(features,member,ids,end-start,'down')
                message=add(block.downward(eg,features)*beta[:,None],ids,end-start)
                support=mean(member,ids,end-start)
                state[n+start:n+end]=block.receive(move(local[n+start:n+end]),message,support).cpu()
            del local,membership,group_support
            continue
        sums=torch.zeros(m,d);mass=torch.zeros(m);new_edge=torch.empty_like(edge);skeleton_mass=torch.empty(n)
        for ordinal,(start,end,sl) in enumerate(tiles(cross_ptr,n,32768,edge_chunk)):
            aa=anchor[sl];pp=surface[sl];ids=move(aa-start)
            sa=move(local[start:end]);sp=move(local[n+pp]);eg=move(edge[start:end])
            delta=move(g['patch_xyz'][pp]-g['skeleton_xyz'][aa])
            geo=torch.cat((delta,delta.norm(dim=-1,keepdim=True)),-1)
            if model.config.association=='once' and step>0:
                w,null=(move(x) for x in saved[ordinal])
            else:
                w,null=block.associations(sa,sp,eg,ids,geo,model.config.association)
                if model.config.association=='once':saved.append((w.cpu(),null.cpu()))
            updated=block.group_update(sa,sp,eg,ids,geo,w,null)
            new_edge[start:end]=updated.cpu();skeleton_mass[start:end]=(1-null).cpu()
            sums.index_add_(0,pp,(updated[ids]*w[:,None]).cpu());mass.index_add_(0,pp,w.cpu())
        edge=new_edge
        state=torch.empty_like(local)
        for start in range(0,n,state_chunk):
            sl=slice(start,min(n,start+state_chunk));den=move(skeleton_mass[sl])
            state[sl]=block.cross_finish(move(local[sl]),move(edge[sl])*den[:,None],den).cpu()
        for start in range(0,m,state_chunk):
            end=min(m,start+state_chunk);sl=slice(start,end)
            state[n+start:n+end]=block.cross_finish(move(local[n+start:n+end]),move(sums[sl]),move(mass[sl])).cpu()
        del sums,mass,local
    for start in range(0,len(state),state_chunk):
        sl=slice(start,start+state_chunk);state[sl]=model.output(move(state[sl])).cpu()
    return JointEmbedding(state[:n],state[n:],g['skeleton_xyz'],g['patch_xyz'],g['patch_point_count'])
