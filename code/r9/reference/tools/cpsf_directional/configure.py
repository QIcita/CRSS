from .common import *
def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((PARENT/'protocol.json').read_text())
    jobs=[dict(task='tlp',arm=a,descriptor='voxel') for a in ('full','skeleton')]
    jobs += [dict(task='tlp',arm=a,descriptor=d) for d in ('directional_xyz','directional') for a in ('full','mean_only','no_fusion','no_scale','point')]
    references={}
    for a in ARMS:
        k=job_key(a,'voxel');f=ANCHOR/'results'/k
        references[k]=dict(experiment=ANCHOR.name,files={n:sha(f/n) for n in ['metrics.json','identity.json','history.json','final.pt','independent_missing_edge_audit.json']})
    p=dict(old,version='Stage6 direction-preserving real neighborhood coordinates',parent_root=str(PARENT),
       parent_protocol_sha256=sha(PARENT/'protocol.json'),jobs=jobs,references=references,
       change_scope='local point coordinate reference only; same axes, fixed normalization .5, retained deterministic pooling',
       comparison='old voxel vs neighborhood XYZ with old distance vs consistent neighborhood XYZ and norm',
       negative_control='voxel full and skeleton exact replay against retained stage4',
       descriptor=dict(neighbors=16,radius_factor=2,near_coincident_tolerance=.001,voxel=.5,
           center='mean of distinct real neighborhood supports',axes='original coordinate axes',invalid='fallback to original four channels',
           variants=['voxel','directional_xyz','directional']),
       source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},document_sha256=sha(ROOT/'PROTOCOL.md'))
    save(ROOT/'protocol.json',p);verify();print(ROOT,flush=True)
if __name__=='__main__':main()
