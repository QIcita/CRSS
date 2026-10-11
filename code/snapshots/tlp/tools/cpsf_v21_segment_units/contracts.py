"""Unit geometry, common weights, isolation, and mandatory scale/position retention."""
import sys,ast,copy
from .io import *
from .model import Model,ARMS,LocalExchange
from .geometry import prepare
from .segments import build_units
ROOT=Path(__file__).resolve().parents[2]


def geometry_checks():
    # The same observed polyline with different degree-2 sampling.
    points=np.array([[x,y,0.] for x in np.linspace(0,10,21) for y in (-.4,.4)])
    s=np.array([[0.,0.,0.],[10.,0.,0.]]);e=np.array([[0,1]])
    g=prepare(s,e,points,wrong=False);u=build_units(g)
    s2=np.column_stack((np.linspace(0,10,41),np.zeros((41,2))));e2=np.column_stack((np.arange(40),np.arange(1,41)))
    g2=prepare(s2,e2,points,wrong=False);v=build_units(g2)
    assert u['u']==v['u']==3 and max(u['length'])<=4.00001
    for k in ('center','length','pr','pj','pw','phi'):torch.testing.assert_close(u[k],v[k],atol=2e-5,rtol=2e-5)
    for geom,unit in [(g,u),(g2,v)]:
        den=torch.zeros(unit['u']).index_add(0,unit['sr'],unit['sw'])
        center=torch.zeros(unit['u'],3).index_add(0,unit['sr'],geom['s'][unit['sn']]*unit['sw'][:,None])/den[:,None]
        torch.testing.assert_close(center,unit['center'],atol=2e-5,rtol=2e-5)
        mass=torch.zeros(geom['m']).index_add(0,unit['pj'],unit['pw'])
        torch.testing.assert_close(mass[mass>0],torch.ones_like(mass[mass>0]))
    # Disconnected nearby arms must remain separate units; no topology invention.
    s=np.array([[0,0,0],[2,0,0],[4,0,0],[2,2,0],[0,.2,1],[4,.2,1]],float)
    e=np.array([[0,1],[1,2],[1,3],[4,5]])
    g=prepare(s,e,points,wrong=False);u=build_units(g)
    for i in range(u['u']):
        nodes=set(u['sn'][u['sr']==i].tolist());assert not (nodes&{4,5} and nodes&{0,1,2,3})
        assert not ({0,2}<=nodes) # Junction splits the chain.
    perm=np.array([5,2,0,4,3,1]);inv=np.argsort(perm)
    gp=prepare(s[perm],inv[e[:,::-1]],points,wrong=False);up=build_units(gp)
    for k in ('center','length','pr','pj','pw','phi'):torch.testing.assert_close(u[k],up[k],atol=2e-5,rtol=2e-5)
    # Closed loops and isolated vertices are legal observed graphs.
    cyc=prepare(np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[9,9,9]],float),np.array([[0,1],[1,2],[2,3],[3,0]]),points,wrong=False)
    cu=build_units(cyc);assert set(cu['sn'].tolist())==set(range(5))
    empty=prepare(np.empty((0,3)),np.empty((0,2),int),points,wrong=False)
    assert build_units(empty)['u']==0
    return dict(physical_subdivision_same_incidence=True,linear_graph_field_integral_same=True,partition_of_unity=True,junction_and_component_isolation=True,node_and_edge_permutation=True,cycle_isolate_empty=True)


def check(device,g,q,other,oq):
    sys.path.append('/root/megrez-tmp/CPSF_V20_3_STAGED_EXCHANGE_20260930')
    from tools.cpsf_v20_3_staged_exchange.model import Model as Previous
    checks={}
    for arm in ('no_fusion','point','skeleton'):
        setup();old=Previous('h01',arm,.1).to(device).eval();setup();new=Model('h01',arm,.1).to(device).eval()
        for name,value in old.state_dict().items():
            if name.startswith('encoder.local_exchange.'):continue
            torch.testing.assert_close(value,new.state_dict()[name],atol=0,rtol=0)
        with torch.no_grad():torch.testing.assert_close(old([g],[q]),new([g],[q]),atol=0,rtol=0)
        checks[arm+'_common_weights_and_output_exact']=True
        del old,new
    for arm in ARMS:
        setup();m=Model('h01',arm,.1).to(device).eval();opt=torch.optim.AdamW(m.parameters(),lr=.001)
        for step in range(3):
            opt.zero_grad();y=m([g],[q]);y.square().sum().backward();assert torch.isfinite(y).all()
            assert all(t.grad is None or torch.isfinite(t.grad).all() for t in m.parameters());opt.step()
        def grad(module):return sum(float(t.grad.abs().sum()) for t in module.parameters() if t.grad is not None)
        paths={name:grad(module) for name,module in [('graph',m.encoder.graph_stem),('surface',m.encoder.surface),('region',m.encoder.local_exchange),('scale',m.encoder.scale),('position',m.encoder.position),('hierarchy',m.encoder.blocks)]}
        if arm=='full':
            assert all(v>0 for v in paths.values())
            with torch.no_grad():torch.testing.assert_close(m([g,other],[q,oq])[:1],m([g],[q]),atol=2e-5,rtol=2e-5)
        if arm=='no_fusion':assert paths['region']==0
        if arm=='no_scale':assert paths['scale']==0 and paths['position']>0
        if arm=='no_geometry':assert grad(m.encoder.local_exchange.geometry)==0
        if arm=='point':
            assert paths['graph']==paths['region']==0
            poison=copy.deepcopy(g);poison['s']=torch.randn_like(g['s'])*100;poison['s'][q]=g['s'][q];poison['e']=g['e'][:0];poison['sf']=torch.randn_like(g['sf'])
            with torch.no_grad():torch.testing.assert_close(m([g],[q]),m([poison],[q]),atol=0,rtol=0)
        if arm=='skeleton':
            assert paths['surface']==paths['region']==0
            poison=copy.deepcopy(g)
            for key in ('p','pf','mass','w','c'):poison[key]=torch.randn_like(g[key])*100
            with torch.no_grad():torch.testing.assert_close(m([g],[q]),m([poison],[q]),atol=0,rtol=0)
        checks[arm]=dict(finite=True,gradient_paths=paths);del m,opt,y
    setup();full=Model('h01','full',.1).to(device).eval();setup();base=Model('h01','no_fusion',.1).to(device).eval()
    with torch.no_grad():torch.testing.assert_close(full([g],[q]),base([g],[q]),atol=0,rtol=0)
    checks['zero_output_init_equals_no_fusion']=True
    # Explicitly nonzero projections: absent surface still causes no update.
    ex=LocalExchange().to(device);torch.nn.init.normal_(ex.to_h.weight);torch.nn.init.normal_(ex.to_p.weight)
    g0=prepare(np.array([[0,0,0],[1,0,0]],float),np.array([[0,1]]),np.empty((0,3)),wrong=False);u=build_units(g0)
    u={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in u.items()};h=torch.randn(2,96,device=device);p=torch.empty(0,96,device=device)
    a,b=ex(h,p,u);torch.testing.assert_close(a,h,atol=0,rtol=0);assert not len(b)
    return checks


def retention():
    here=Path(__file__).parent;old=Path('/root/megrez-tmp/CPSF_V20_3_STAGED_EXCHANGE_20260930/tools/cpsf_v20_3_staged_exchange')
    def node(path,name):
        tree=ast.parse(path.read_text());n=next(x for x in tree.body if getattr(x,'name',None)==name)
        return ast.dump(n,include_attributes=False)
    for name in ('rotary','Hierarchy','spatial_chunks'):
        assert node(here/'model.py',name)==node(old/'model.py',name)
    assert node(here/'global_task.py','geometries')==node(old/'global_task.py','geometries')
    model=(here/'model.py').read_text()
    for line in ["code=self.scale(torch.cat((extent.log1p(),scale.log1p()))) if arm!='no_scale' else coords.new_zeros(D)","self.position((sx-origin)/extent)","self.position((px-origin)/extent)","if offsets is not None:xyz=xyz+xyz.new_tensor(offsets[i])","positions.append(xyz/b['radius'])"]:
        assert line in model
    return dict(continuous_xyz_rotary_ast_equal=True,hierarchy_ast_equal=True,global_dynamic_tiles_ast_equal=True,scale_and_local_position_present=True,cell_offsets_present=True)


def main():
    from .train import validated_protocol
    validated_protocol();split=json.loads((ROOT/'split_manifest.json').read_text())['rows'];rows=split['inner_train']
    row=max(rows,key=lambda r:r['points']);x=load(row['path']);other=load(min(rows,key=lambda r:r['points'])['path']);checks={}
    checks['geometry']=geometry_checks();checks['scale_retention']=retention()
    for device in ('cpu','cuda'):checks[device]=check(device,x['geometry'],x['query'],other['geometry'],other['query'])
    examples=[]
    for row in rows[:12]:
        g=load(row['path'])['geometry'];u=build_units(g)
        examples.append(dict(sample=row['sample'],nodes=g['n'],patches=g['m'],segments=u['u'],relations=len(u['pr']),max_length=float(u['length'].max()) if len(u['length']) else 0))
    save(ROOT/'contracts.json',dict(passed=True,checks=checks,examples=examples,scope='implementation and input/scale retention, not efficacy'))

if __name__=='__main__':main()
