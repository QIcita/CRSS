"""One support-size budget and one full-source FPS; no old sampler metadata."""
import numpy as np
from scipy.spatial import cKDTree
from tools.cpsf_geometry_fps.fps import sample

RULE=dict(grid_size=0.5,reference_occupied_cells=512,reference_points=8192,
          exponent=0.5,minimum=8192,maximum=65536)

def sample_geometry(points, mass=None):
    p=np.asarray(points,np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all():
        raise ValueError('Finite nonempty current XYZ required')
    u,inv=np.unique(p,axis=0,return_inverse=True)
    cells=np.floor((u-u.min(0))/RULE['grid_size']).astype(np.int64)
    k=len(np.unique(cells,axis=0))
    requested=int(np.ceil(RULE['reference_points']*(k/RULE['reference_occupied_cells'])**RULE['exponent']))
    n=min(len(u),max(RULE['minimum'],min(RULE['maximum'],requested)))
    selected,_,ids,audit=sample(u,n)
    w=np.ones(len(p)) if mass is None else np.asarray(mass,np.float64)
    assert w.shape==(len(p),) and np.isfinite(w).all() and (w>=0).all() and w.sum()>0
    source_mass=np.bincount(inv,weights=w,minlength=len(u))
    d,owner=cKDTree(selected).query(u,workers=1)
    weight=np.bincount(owner,weights=source_mass,minlength=n)
    assert np.isclose(weight.sum(),w.sum(),rtol=1e-10)
    # Nonparticipating mesh vertices must be removed by the source adapter.
    assert (weight>0).all()
    audit.update(rule=RULE,occupied_cells=k,size_proxy=k*RULE['grid_size']**2,
        source_records=len(p),source_unique_points=len(u),source_mass=float(w.sum()),
        requested_points=requested,source_limited=n==len(u),
        coverage_p95=float(np.quantile(d,.95)),coverage_max=float(d.max()),
        mass_kind='nearest representative source measure',
        budget_inputs=['current coordinates','fixed rule'],legacy_budget_used=False)
    return selected,weight,ids,audit
