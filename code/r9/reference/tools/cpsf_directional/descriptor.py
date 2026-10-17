import numpy as np
from scipy.spatial import cKDTree

def describe(points,radius=2.,neighbors=16,duplicate_tolerance=.001,voxel=.5):
    """Directional offsets to a real neighborhood centroid in the original axes."""
    points=np.asarray(points,np.float64).reshape(-1,3);n=len(points)
    assert np.isfinite(points).all() and radius>0 and neighbors>=3
    if n==0:return np.empty((0,4),np.float32),np.empty(0,bool),dict(points=0,distinct_support=0,valid_fraction=0.,zero_fraction=0.)
    # Connected components remove near-coincident observations only in descriptor
    # statistics. The original points, patch ownership and empirical mass stay intact.
    parent=np.arange(n);size=np.ones(n,np.int64)
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    pairs=cKDTree(points).query_pairs(duplicate_tolerance,output_type='ndarray')
    for a,b in pairs:
        a=find(a);b=find(b)
        if a==b:continue
        if size[a]<size[b]:a,b=b,a
        parent[b]=a;size[a]+=size[b]
    roots=np.array([find(i) for i in range(n)]);_,inverse=np.unique(roots,return_inverse=True)
    count=np.bincount(inverse);centers=np.zeros((len(count),3));np.add.at(centers,inverse,points);centers/=count[:,None]
    distance,indices=cKDTree(centers).query(centers,k=min(neighbors,len(centers)),distance_upper_bound=2*radius,workers=1)
    if indices.ndim==1:indices=indices[:,None];distance=distance[:,None]
    mask=np.isfinite(distance);safe=np.minimum(indices,len(centers)-1);xyz=centers[safe]
    # Covariance is centered locally; world translation and orientation are not labels.
    counts=mask.sum(1);mean=(xyz*mask[...,None]).sum(1)/np.maximum(counts,1)[:,None]
    delta=(xyz-mean[:,None])*mask[...,None]
    covariance=np.einsum('nki,nkj->nij',delta,delta)/np.maximum(counts,1)[:,None,None]
    eigen=np.maximum(np.linalg.eigvalsh(covariance)[:,::-1],0);l1,l2,l3=eigen.T
    valid=(counts>=3)&(l1>1e-12);den=np.maximum(l1,1e-12)
    offset=(points-mean[inverse])/voxel
    valid_points=valid[inverse]
    out=np.column_stack((offset,np.linalg.norm(offset,axis=1))).astype(np.float32)
    out[~valid_points]=0
    assert np.isfinite(out).all()
    return out,valid_points,dict(points=n,distinct_support=len(centers),valid_fraction=float(valid_points.mean()),
        zero_fraction=float((out==0).all(1).mean()),neighborhood_count_quantiles=np.quantile(counts,[0,.1,.5,.9,1]).tolist(),
        offset_norm_quantiles=np.quantile(out[:,3],[0,.5,.9,.99,1]).tolist(),
        channels=['neighbor_relative_x','neighbor_relative_y','neighbor_relative_z','neighbor_relative_norm'])
