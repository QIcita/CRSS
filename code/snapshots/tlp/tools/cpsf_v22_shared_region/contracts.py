import argparse,tempfile,json
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tools.cpsf_h01_topology_repair.adapter import geometry
from tools.cpsf_v21_segment_units.io import setup,save
from .data import regions,whole_geometry,local_geometry
from .model import Model,ARMS
from .train import move

def components(n,e):
    return int(connected_components(coo_matrix((np.ones(len(e)),(e[:,0],e[:,1])),shape=(n,n)),directed=False,return_labels=False))

def main(out):
    setup();rng=np.random.default_rng(42)
    # Bent path, fork, closed loop, and isolated component; no straight-chord replacement.
    s=np.array([[0,0,0],[3,3,0],[6,0,0],[6,3,0],[9,0,0],[20,0,0],[22,2,0],[24,0,0],[22,-2,0],[40,0,0]],float)
    e=np.array([[0,1],[1,2],[2,3],[2,4],[5,6],[6,7],[7,8],[8,5]])
    p=s[rng.integers(len(s),size=300)]+rng.normal(0,.15,(300,3))
    g=geometry(s,e,p);u=regions(g,'joint',s[2])
    assert components(len(s),e)==components(u['nregions'],u['adjacency'].numpy())
    assert abs(float(u['length'].sum())-np.linalg.norm(s[e[:,0]]-s[e[:,1]],axis=1).sum())<1e-4
    assert abs(float(u['surface_mass'].sum())-float(g['mass'].sum()))<1e-3
    # Insert a degree-2 node on an existing straight piece; region lengths and
    # centers stay the same, and graph connectivity is preserved.
    ss=np.vstack((s,(s[0]+s[1])/2));ee=np.vstack(([[0,len(s)],[len(s),1]],e[1:]))
    u2=regions(geometry(ss,ee,p),'joint',s[2])
    torch.testing.assert_close(u['pos'],u2['pos'],rtol=1e-5,atol=1e-5)
    torch.testing.assert_close(u['length'],u2['length'],rtol=1e-5,atol=1e-5)
    # Strict modality adapters need only their own raw arrays.
    po=regions(whole_geometry(dict(surface_xyz_um=p),'point'),'point')
    sk=regions(whole_geometry(dict(skeleton_xyz_um=s,edge_index=e),'skeleton'),'skeleton')
    assert not po['has_g'].any() and not sk['has_p'].any()
    checks={}
    for task in ('local','global'):
        for arm in ARMS:
            gg=regions(local_geometry(g,arm),arm,s[2]-np.array(g['origin'])) if arm in ('point','skeleton') else u
            model=Model(task,arm).cuda();data=move(gg);y=model(data)
            loss=y.square().sum() if task=='local' else (y-.73).square()
            loss.backward();assert torch.isfinite(y).all()
            assert model.scale[0].weight.grad is not None and model.position[0].weight.grad is not None
            if arm=='full':
                for name in ('graph','point','joint'):
                    assert getattr(model,name)[0].weight.grad.abs().sum()>0,name
            else:assert model.joint[0].weight.grad is None
            checks[task+'_'+arm]=dict(output_shape=list(y.shape),finite_backward=True,peak_gpu_bytes=torch.cuda.max_memory_allocated())
            del model,data,y,loss;torch.cuda.empty_cache()
    save(out,dict(passed=True,component_count_preserved=True,polyline_arclength_preserved=True,
                  degree2_region_geometry_invariant=True,surface_mass_conserved=True,pure_modality_adapters=True,
                  explicit_scale_position_gradients=True,checks=checks,accuracy_claim=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
