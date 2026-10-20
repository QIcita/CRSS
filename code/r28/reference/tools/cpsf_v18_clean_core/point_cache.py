"""Same frozen H01 samples/real point IDs, but no skeleton-derived encoding.

Query coordinates enter only the downstream localization readout. The sample
field of view was historically selected using skeleton support; this control
does not claim a skeleton-independent sampling protocol.
"""
import argparse,time
from .common import *
from .data import local_sets
from .geometry import prepare_surface_geometry


def build(a):
    assert not a.out.exists();a.out.mkdir(parents=True);(a.out/'local').mkdir()
    rows=[r for seq in local_sets(a.cache,a.smoke).values() for r in seq]
    done=[];start=time.time()
    for cell in sorted({r['cell'] for r in rows}):
        path=SOURCE/'inputs'/(cell+'.npz')
        with np.load(path,allow_pickle=False) as full:
            surface=full['surface_xyz_um'];xyz=full['skeleton_xyz_um']
        digest=sha(path)
        for r in (r for r in rows if r['cell']==cell):
            assert r['source_sha256']==digest
            reference=SOURCE/'local_refs'/(r['sample']+'.npz')
            with np.load(reference) as ref:
                point_ids=ref['point_ids'];query=xyz[ref['node_ids'][int(ref['query_index'])]]
            assert len(point_ids)==r['points']
            g=prepare_surface_geometry(surface[point_ids])
            q=torch.tensor((query-np.asarray(g['origin']))/g['input_scale'],dtype=torch.float32)
            out=a.out/'local'/(r['sample']+'.pt')
            torch.save(dict(geometry=g,query=q,target=r['target'],sample=r['sample'],cell=cell,split=r['split']),out)
            done.append(dict(sample=r['sample'],cell=cell,split=r['split'],target=r['target'],points=r['points'],
                sha256=sha(out),source_sha256=digest,reference_sha256=sha(reference),
                point_ids_sha256=hashlib.sha256(point_ids.tobytes()).hexdigest(),joint_cache_sha256=r['sha256']))
        print('POINT_CACHE',cell,len(done),flush=True)
    save(a.out/'manifest.json',dict(records=done,source_manifest_sha256=sha(a.cache/'manifest.json'),
        smoke=a.smoke,seconds=time.time()-start,point_only_encoding=True,query_in_decoder_only=True,
        inherited_skeleton_guided_sample_support=True,physical_scale_um=1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');build(p.parse_args())
