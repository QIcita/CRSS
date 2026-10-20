import json,argparse
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.geometry import prepare
from tools.cpsf_v21_segment_units.model import Model
from tools.cpsf_v21_segment_units.global_task import geometries as legacy
from tools.cpsf_v21_segment_units.io import setup,save
from .adapter import geometry,geometries

def run(out):
    setup();rng=np.random.default_rng(42)
    s=np.column_stack((np.arange(2600)/20,np.zeros((2600,2))));e=np.column_stack((np.arange(2599),np.arange(1,2600)))
    p=s[rng.integers(len(s),size=8000)]+rng.normal(0,.2,(8000,3));x=dict(skeleton_xyz_um=s,edge_index=e,surface_xyz_um=p)
    old=legacy(x,'point');new=geometries(x,'point')
    for a,b in zip(old,new):
        for k in ('p','pf','w','mass','c','owner'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
        assert a['origin']==b['origin']
    joint=geometries(x,'full');assert joint[0]['n']==2600 and len(joint[0]['e'])==2599
    for a,b in zip(joint,new):
        for k in ('p','pf','w','mass','c','owner'):torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    smalls=s[:20];smalle=e[:19];smallp=p[:60];a=prepare(smalls,smalle,smallp,wrong=False);b=geometry(smalls,smalle,smallp)
    for arm in ('full','no_fusion','skeleton','point'):
        model=Model('h01',arm,residual_scale=.1).cuda().eval()
        with torch.no_grad():torch.testing.assert_close(model([a],[4]),model([b],[4]),rtol=0,atol=0)
    # Curved/cyclic/disconnected raw edges are preserved without invented links.
    c=np.array([[0,0,0],[1,1,0],[2,0,0],[1,-1,0],[9,9,9]],float);ce=np.array([[0,1],[1,2],[2,3],[3,0]])
    v=geometries(dict(skeleton_xyz_um=c,edge_index=ce),'skeleton')[0]
    assert v['n']==5 and len(v['e'])==4
    save(out,dict(passed=True,point_draws_and_weights_bitwise_equal=True,full_thumbnail_edges_preserved=True,
                  inactive_relation_omission_output_bitwise_equal=True,cycle_and_isolated_node_preserved=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.out)
