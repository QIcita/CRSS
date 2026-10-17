from .common import *

def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((PARENT/'protocol.json').read_text())
    jobs=[dict(task='tlp',arm='full',descriptor=d) for d in ('voxel','neighborhood')]
    jobs += [dict(task='tlp',arm=a,descriptor=d) for d in ('shape_only','extent_only')
             for a in ('full','mean_only','no_fusion','no_scale','point')]
    references={}
    for d in ('voxel','neighborhood'):
        for a in ARMS:
            k=job_key(a,d);f=PARENT/'results'/k
            references[k]=dict(experiment=PARENT.name,files={n:sha(f/n) for n in
                ['metrics.json','identity.json','history.json','final.pt','independent_missing_edge_audit.json']})
    p=dict(old,version='Stage5 channel-factor decomposition under unchanged deterministic core',
           parent_root=str(PARENT),parent_protocol_sha256=sha(PARENT/'protocol.json'),jobs=jobs,
           change_scope='only select old/new first three channels and old/new fourth channel',
           comparison='2x2 descriptor replacement; fixed four-channel width and initialization',
           negative_control='both endpoint full models replay all80 epochs exactly against stage4',
           historical_scores='stage4 references explicitly labeled and hashed; stage2/stage3 not pooled',
           parent_data_manifests={m:sha(DATA_PARENT/'data/tlp'/m/'manifest.json') for m in ['joint','skeleton','point']},
           references=references,descriptor=dict(voxel=['old_xyz','old_radius'],shape_only=['pca_shape','old_radius'],
              extent_only=['old_xyz','new_extent'],neighborhood=['pca_shape','new_extent']),
           source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},
           document_sha256=sha(ROOT/'PROTOCOL.md'))
    p.pop('parent_checkpoint_sha256',None)
    save(ROOT/'protocol.json',p);verify();print(ROOT,flush=True)

if __name__=='__main__':main()
