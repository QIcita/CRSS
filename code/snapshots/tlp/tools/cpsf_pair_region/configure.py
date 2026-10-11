from .common import *
def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((PARENT/'protocol.json').read_text())
    p=dict(old,version='Stage2 pair-region readout; unchanged missing-edge objective and encoder',parent_root=str(PARENT),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),jobs=[dict(task='tlp',arm=a) for a in ARMS],
        change_scope='parameter-free pair-conditioned readout only',
        parent_data_manifests={m:parent_rows(m)[1] for m in ['joint','skeleton','point']},
        parent_checkpoint_sha256={a:sha(PARENT/'results'/('tlp_'+a)/'final.pt') for a in ARMS if a!='shuffle_corridor'},
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},document_sha256=sha(ROOT/'PROTOCOL.md'),
        readout=dict(probes=[.25,.5,.75],neighbors_per_probe=8,radius_factor=2,radial_kernel='exp(-0.5*(distance/radius)^2)',
            context='equal mean of global and pair-region feature; global fallback if no active support',new_parameters=0,
            evaluation_universe='all unobserved unordered node pairs; no candidate filtering'))
    save(ROOT/'protocol.json',p);verify();print(ROOT,flush=True)
if __name__=='__main__':main()
