"""Recover current geometry before historical point budgets or stratified FPS."""
import importlib.util, json, struct, sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from .common import BASE,ROOT,read,sha

NREDD=BASE/'NREDD_proj/NREDD'

@lru_cache(maxsize=2)
def full_h01(cell):
    path=BASE/'H01_HIGH_COVERAGE_FULL104_20260920/inputs'/(cell+'.npz')
    with np.load(path) as a:
        x={k:a[k] for k in ('surface_xyz_um','skeleton_xyz_um','edge_index','point_to_edge')}
    return x,{str(path):sha(path)}

def h01(row):
    x,lineage=full_h01(row['cell']);lineage=dict(lineage)
    if row['task']=='h01_global':
        p=x['surface_xyz_um'];s=x['skeleton_xyz_um'];e=x['edge_index'];q=None
    else:
        path=BASE/'H01_HIGH_COVERAGE_FULL104_20260920/local_refs'/(row['key']+'.npz')
        lineage[str(path)]=sha(path)
        with np.load(path) as a:
            nodes=a['node_ids'];pts=a['point_ids'];edges=a['edge_ids'];query=int(a['query_index'])
        edges=np.union1d(edges,np.unique(x['point_to_edge'][pts]))
        nodes=np.r_[nodes,np.setdiff1d(np.unique(x['edge_index'][edges]),nodes)]
        lookup=np.full(len(x['skeleton_xyz_um']),-1,np.int64);lookup[nodes]=np.arange(len(nodes))
        s=x['skeleton_xyz_um'][nodes];e=lookup[x['edge_index'][edges]]
        p=x['surface_xyz_um'][pts];q=s[query].copy()
    return dict(s=np.asarray(s,float),e=np.asarray(e,int),p=np.asarray(p,float),q=q,
        units='um',mass_kind='observed surface record count',lineage=lineage)

def flywire(row):
    root=BASE/('CPSF_FLYWIRE_TASKS_20260930' if row['legacy'] else 'CPSF_FLYWIRE_COARSE_COHORT_20261002/geometry')
    rid=row['key'];q=read(root/'qa'/(rid+'.json'));sp=root/'raw'/rid/'skeleton.bin';mp=sp.with_name('mesh.npz')
    assert sha(sp)==q['skeleton_sha256'] and sha(mp)==q['mesh_arrays_sha256']
    b=sp.read_bytes();n,ne=struct.unpack('<II',b[:8]);assert len(b)==8+16*n+8*ne
    s=np.frombuffer(b,'<f4',3*n,8).reshape(n,3).astype(float)/1000
    e=np.frombuffer(b,'<u4',2*ne,8+12*n).reshape(ne,2).astype(np.int64)
    with np.load(mp) as a:p=a['vertices_nm'].astype(float)/1000;faces=a['faces']
    tri=p[faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    mass=np.zeros(len(p))
    for side in range(3):np.add.at(mass,faces[:,side],area/3)
    keep=mass>0;p=p[keep];mass=mass[keep]
    assert np.isclose(mass.sum(),q['total_area_um2'],rtol=2e-5)
    return dict(s=s,e=e,p=p,mass=mass,q=None,units='um',
        mass_kind='triangle area lumped to vertices and nearest FPS representatives',
        mesh_area=float(area.sum()),lineage={str(sp):sha(sp),str(mp):sha(mp)})

@lru_cache(maxsize=2)
def module(kind):
    path=NREDD/('tools/build_multishell_surface_cache_v22c.py' if kind=='ted' else
        'artifacts/CPSF_GEOMETRY_V2E_TLP_20260917/data/build_tlp_geometry_v2e_cache.py')
    name='geometry_source_'+kind
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

@lru_cache(maxsize=1)
def ted_block(block):
    manifest=read(BASE/'CPSF_FIXED_CORE_NREDD_20261006/split_manifest.json')
    r=next(r for r in manifest['records']['ted'] if r['key']==block)
    for k in ('swc','image','metadata','cache'):assert sha(r[k])==r[k+'_sha256']
    m=module('ted');prepared=m.v21.prepare_volume(Path(r['image']))
    swc=np.loadtxt(r['swc']);swc[:,1]=0
    with np.load(r['metadata']) as a:ids=a['subgraph_node_indices']
    cached=torch.load(r['cache'],map_location='cpu',weights_only=False)
    return r,prepared,swc,ids,cached

def ted(row):
    block,index=row['key'].rsplit(':',1);i=int(index)
    r,prepared,swc,allids,cached=ted_block(block);ids=allids[i];ids=ids[ids>=0]
    s=swc[ids,2:5];e=cached['events'][i]['edges'].numpy();m=module('ted');v=m.v21
    # Preserve duplicate-node event semantics and the original extraction ROI.
    root=s[0];scale=max(float(np.linalg.norm(s-root,axis=1).max())+v.SOURCE_RADIUS,v.SOURCE_RADIUS,1.)
    low,high=v.event_bounds(root,scale,prepared['raw'].shape)
    fine=v.crop_zyx(prepared['smooth'][0],low,high);grid=v.grid_xyz(fine.shape,low)
    distance=v.nearest_segment_distance(grid,s,e.tolist()).reshape(fine.shape);tube=distance<=v.SOURCE_RADIUS
    _,threshold,background,_,_=v.choose_foreground(fine,tube,tube&(distance>=v.SOURCE_RADIUS*.72),tube&(distance<=v.DISTANCE_STRATA[0]))
    levels=m.shell_levels(fine,tube,threshold,background)
    p,_,_=m.extract_shell_vertices(fine,tube,low,prepared['raw'].shape,s,e.tolist(),levels)
    # The native project mapping is not a verified physical micrometre scale.
    return dict(s=(s-root)/2,e=e,p=(p-root)/2,q=np.zeros((1,3)),units='TED historical SWC/2 units; physical calibration unverified',
        mass_kind='deduplicated current isosurface vertex count',
        lineage={r[k]:r[k+'_sha256'] for k in ('swc','image','metadata','cache')})

def tlp_edge_vertices(prepared,start_raw,end_raw,m):
    # Exact extraction prefix of upstream build_edge_patch, ending BEFORE FPS.
    v=m.node_v1;spacing=m.bridge_common.SPACING_XYZ
    low,high=m.crop_bounds_for_segment(start_raw,end_raw,prepared['raw'].shape)
    fine=v.crop_zyx(prepared['smooth'][0],low,high)
    grid=v.grid_xyz(fine.shape,low)*spacing[None,:];start=start_raw*spacing;end=end_raw*spacing
    distance,fraction=m.segment_distance_and_fraction(grid,start,end)
    cap=m.EDGE_CAP_EXTENT/max(float(np.linalg.norm(end-start)),1e-6)
    valid=((distance<=m.TUBE_RADIUS)&(fraction>=-cap)&(fraction<=1+cap)).reshape(fine.shape)
    if not np.any(valid) or float(np.ptp(fine[valid]))<1e-5:return np.empty((0,3))
    core=valid&(distance.reshape(fine.shape)<=m.CORE_RADIUS)
    levels,*_=m.bridge_v1.adaptive_isosurfaces(fine,valid,core,distance.reshape(fine.shape))
    result=[];shape=np.asarray(prepared['raw'].shape[::-1],np.float32)
    for level in levels:
        local,*_=m.measure.marching_cubes(fine,level=float(level),allow_degenerate=False)
        raw=local[:,::-1]+low[None,:];physical=raw*spacing[None,:]
        d,t=m.segment_distance_and_fraction(physical,start,end)
        keep=((raw>=m.BOUNDARY_MARGIN)&(raw<=shape[None,:]-1-m.BOUNDARY_MARGIN)).all(1)
        keep&=(d<=m.TUBE_RADIUS-m.BOUNDARY_MARGIN)&(t>=-cap)&(t<=1+cap)
        result.append(physical[keep])
    return np.concatenate(result)

def tlp(row):
    r=next(r for r in read(BASE/'CPSF_FIXED_CORE_NREDD_20261006/split_manifest.json')['records']['tlp'] if r['key']==row['key'])
    for k in ('swc','image'):assert sha(r[k])==r[k+'_sha256']
    m=module('tlp');swc=np.loadtxt(r['swc']);swc[:,1]=0;s=swc[:,2:5].astype(np.float32)
    prepared=m.v2d.prepare_surface_volume(Path(r['image']));e=m.canonical_edges(swc)
    pieces=[tlp_edge_vertices(prepared,s[a],s[b],m) for a,b in e]
    graph=m.bridge_common.observed_graph(swc);pairs=m.bridge_common.enumerate_candidate_pairs(swc,graph)
    for pair in pairs:
        g=m.bridge_common.candidate_geometry(swc,graph,pair)
        extracted,_=m.bridge_v1.extract_vertices(prepared,g)
        if not extracted:continue
        p=extracted['vertices_xyz']*m.bridge_common.SPACING_XYZ[None,:]
        d,t=m.segment_distance_and_fraction(p,np.asarray(g['endpoint_a'],np.float32),np.asarray(g['endpoint_b'],np.float32))
        cap=float(m.bridge_common.BRANCH_EXTENT)/max(float(g['length']),1e-6)
        pieces.append(p[(d<=m.TUBE_RADIUS-m.BOUNDARY_MARGIN)&(t>=-cap)&(t<=1+cap)])
    p=np.unique(np.concatenate(pieces),axis=0);s=s*m.bridge_common.SPACING_XYZ[None,:]
    return dict(s=s,e=e,p=p,q=s.copy(),units='TLP historical native spacing [1,1,2]; physical calibration unverified',
        mass_kind='deduplicated current isosurface vertex count',pairs=pairs,
        lineage={r[k]:r[k+'_sha256'] for k in ('swc','image')})

def geometry(row):
    task=row['task']
    x=h01(row) if task.startswith('h01') else flywire(row) if task=='flywire_coarse' else ted(row) if task=='ted' else tlp(row)
    if task in ('ted','tlp'):x['p']=np.unique(x['p'],axis=0)
    assert np.isfinite(x['s']).all() and np.isfinite(x['p']).all() and len(x['p'])>0
    return x
