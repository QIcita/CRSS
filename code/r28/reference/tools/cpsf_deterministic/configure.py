from .common import *

def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((PARENT/'protocol.json').read_text())
    jobs=[dict(task='tlp',arm='skeleton',descriptor=d) for d in DESCRIPTORS]
    jobs += [dict(task='tlp',arm=a,descriptor=d) for d in DESCRIPTORS for a in ARMS
             if a!='skeleton' and (a!='shuffle_descriptor' or d=='neighborhood')]
    p=dict(old,version='Stage4 deterministic region reduction; matched voxel/neighborhood descriptions',
           parent_root=str(PARENT),parent_protocol_sha256=sha(PARENT/'protocol.json'),jobs=jobs,
           change_scope='sparse region pooling numerical implementation only; same mathematical readout',
           comparison='both descriptors retrained with identical deterministic kernel',
           negative_control='voxel_skeleton vs neighborhood_skeleton exact 80 epoch replay under repaired implementation',
           historical_scores='stage2 and stage3 are separately retained; no requirement to reproduce nondeterministic legacy trajectory',
           parent_data_manifests={m:sha(PARENT/'data/tlp'/m/'manifest.json') for m in ['joint','skeleton','point']},
           parent_checkpoint_sha256={a:sha(PARENT/'results'/('tlp_'+a)/'final.pt') for a in ARMS},
           source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},
           document_sha256=sha(ROOT/'PROTOCOL.md'))
    save(ROOT/'protocol.json',p);verify();print(ROOT,flush=True)

if __name__=='__main__':main()
