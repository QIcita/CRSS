from .common import *
def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((PARENT/'protocol.json').read_text())
    p=dict(old,version='Stage3 local point geometry descriptor; unchanged pair-region readout and scale encoding',parent_root=str(PARENT),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),jobs=[dict(task='tlp',arm=a) for a in ARMS],
        change_scope='four-channel surface content descriptor only; original pf preserved for replay',
        parent_data_manifests={m:parent_rows(m)[1] for m in ['joint','skeleton','point']},
        parent_checkpoint_sha256={a:sha(PARENT/'results'/('tlp_'+a)/'final.pt') for a in ARMS if a!='shuffle_descriptor'},
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},document_sha256=sha(ROOT/'PROTOCOL.md'),
        descriptor=dict(channels=['linearity','planarity','scattering','log1p_rms_extent_over_radius'],neighbors=16,radius_factor=2,
            min_distinct_support=3,near_coincident_tolerance=.001,invalid='zero descriptor; no synthetic neighbors',
            source='point-only cached patch centers plus voxel-relative offsets; no skeleton or labels',new_parameters=0))
    save(ROOT/'protocol.json',p);verify();print(ROOT,flush=True)
if __name__=='__main__':main()
