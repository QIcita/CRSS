import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .common import setup


def add(x, index, count):
    return x.new_zeros((count, *x.shape[1:])).index_add(0, index, x)


class SurfaceCoupling(nn.Module):
    def __init__(self, relation, position, scale):
        super().__init__()
        self.relation=relation; self.position=position; self.scale=scale
        self.query=nn.Linear(256+384+96,16); self.key=nn.Linear(96,16)
        self.value=nn.Linear(96,96); self.score_geometry=nn.Linear(5,1,bias=False)
        self.output=nn.Linear(96,640,bias=False); nn.init.zeros_(self.output.weight)

    def forward(self, branch, topology, patches, x, conditioned):
        g=x['g']; n=g['nregions']; pr,pj=g['pr'],g['pj']
        xyz=g['pos']-g['pos'].mean(0)
        scale=torch.stack((g['length'].log1p(),torch.full_like(g['length'],g['radius']).log1p(),
                           torch.full_like(g['length'],.5).log1p()),1)
        code=self.position(torch.sign(xyz)*torch.log1p(xyz.abs()/g['radius']))+self.scale(scale)
        region_branch=add(branch[x['jb']]*x['jw'][:,None],x['jr'],n)
        # Normalize only the query view; the 640D direct structure is unchanged.
        query=F.linear(F.layer_norm(region_branch,(256,)),self.query.weight[:,:256],self.query.bias)
        query=query+F.linear(F.layer_norm(topology,(384,)),self.query.weight[:,256:640])
        query=query+F.linear(F.layer_norm(code,(96,)),self.query.weight[:,640:])
        key=self.key(patches); value=self.value(patches)
        prior=g['pw']*g['patch_mass'][pj]/g['surface_mass'][pr]
        # Save only scalar edge scores. No E x 96 message is materialized.
        logits=[]
        for lo in range(0,len(pr),16384):
            rr,pp,phi=pr[lo:lo+16384],pj[lo:lo+16384],g['phi'][lo:lo+16384]
            def score(q,k,geo,rr=rr,pp=pp):
                return (q[rr]*k[pp]).sum(-1)/4+self.score_geometry(geo).squeeze(-1)
            s=checkpoint(score,query,key,phi,use_reentrant=False) if torch.is_grad_enabled() else score(query,key,phi)
            logits.append(s)
        logits=torch.cat(logits)
        if conditioned:
            log_weight=prior.log()+logits
            maximum=log_weight.new_full((n,),-torch.inf)
            maximum.scatter_reduce_(0,pr,log_weight.detach(),reduce='amax',include_self=True)
            w=(log_weight-maximum[pr]).exp()
            probability=w/add(w,pr,n)[pr]
        else:
            # Identical parameter structure; this arm disables score influence.
            probability=prior+logits*0.
        weight=g['surface_mass']/g['surface_mass'].sum()
        pooled=patches.new_zeros(96)
        for lo in range(0,len(pr),16384):
            rr,pp=pr[lo:lo+16384],pj[lo:lo+16384]
            phi=g['phi'][lo:lo+16384]; pi=probability[lo:lo+16384]; a=prior[lo:lo+16384]
            def gather(v,prob,base,geo,rr=rr,pp=pp):
                message=prob[:,None]*v[pp]+base[:,None]*self.relation(geo)
                return (message*weight[rr,None]).sum(0)
            pooled=pooled+(checkpoint(gather,value,pi,a,phi,use_reentrant=False) if torch.is_grad_enabled() else gather(value,pi,a,phi))
        return self.output(pooled)


class Model(nn.Module):
    def __init__(self, arm):
        super().__init__(); self.conditioned=arm=='surface_conditioned'; self.surface_enabled=True
        from tools.cpsf_graphformer_controls.models import OfficialHeadModel
        original=OfficialHeadModel(); self.native=original.native
        # Reuse the exact R38 CMM objects; discard its unused graph/head modules.
        from tools.cpsf_r34_fps_rebench.model import FlyModel
        from tools.cpsf_r38_dense.model import install as dense_install
        from tools.cpsf_r38_joint_context.model import install as joint_install
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42)
            parent=joint_install(dense_install(FlyModel().cuda()))
            self.point=parent.point; self.cmm=parent.cmm
            geo=parent.rcm.geometry
            self.coupling=SurfaceCoupling(geo.relation,geo.position,geo.scale).cuda()
        self.checkpoint_samples=True

    def tree_states(self, batch):
        import dgl
        tree=self.native.tree_encoder; tree.cell.init_state=True
        original=batch.graph
        g=dgl.graph(original.edges(),num_nodes=original.num_nodes(),device=original.device)
        feats=tree.mlp1(batch.feats); n=g.num_nodes()
        g.ndata['iouf']=tree.cell.W1_iouf(feats)
        for k in ('h1','c1','h2','c2'):g.ndata[k]=feats.new_zeros((n,tree.h_size))
        dgl.prop_nodes_topo(g,message_func=tree.cell.message_func,reduce_func=tree.cell.reduce_func,apply_node_func=tree.cell.apply_node_func)
        branch=g.ndata.pop('c2')
        return branch[batch.offset.long()],branch

    def encode(self, x):
        roots,branch=self.tree_states(x['batch'])
        root=(roots*x['mass'][:,None]).sum(0)/x['mass'].sum()
        topology=self.native.encode_image(x['image'][None])[0]
        direct=torch.cat((F.normalize(root,dim=-1),F.normalize(topology,dim=-1)))
        if not self.surface_enabled:return direct
        from tools.cpsf_r38_joint_context.support import SurfaceSupport
        patches=self.cmm(SurfaceSupport(x['g'],self.point(x['g']['pf']))).surface
        return direct+self.coupling(branch,topology,patches,x,self.conditioned)

    def forward_group(self, items):
        features=[checkpoint(self.encode,x,use_reentrant=False) if self.training and self.checkpoint_samples else self.encode(x) for x in items]
        return self.native.classifier(torch.stack(features))

    def forward(self,x):
        return self.forward_group([x])[0]


def make(arm):
    setup()
    return Model(arm).cuda()
