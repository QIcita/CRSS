"""Freeze a new objective protocol; reference the existing data read-only."""
import shutil
from .common import *

def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((PARENT/'protocol.json').read_text())
    assert sha(ROOT/'split_manifest.json')==old['split_sha256']
    for name,h in old['source_sha256'].items():assert sha(ROOT/name)==sha(PARENT/name)==h
    manifests={};reference=None
    for mode in ('joint','skeleton','point'):
        path=PARENT/'data/tlp'/mode/'manifest.json';manifests[mode]=sha(path)
        rr=json.loads(path.read_text())['records'];keys=[(r['key'],r['split'],r['group']) for r in rr]
        if reference is None:reference=keys
        assert keys==reference
        train=sorted([r for r in rr if r['split']=='train'],key=lambda r:(r['nodes'],r['key']))
        pilot=[train[k] for k in sorted({0,len(train)//2,len(train)-1})]
        save(ROOT/'pilot/tlp'/mode/'manifest.json',dict(records=pilot,selection='train only smallest/median/largest node count; label-free',parent_manifest_sha256=manifests[mode]))
    cfg=dict(old['tasks']['tlp']);cfg['loss']='adaptive positive-weight BCE on upper-triangular unobserved pairs only; no self-loops, no observed edges; no candidate filtering'
    p=dict(version='Stage 1 missing-edge objective; unchanged retained fixed core and endpoint readout',parent_root=str(PARENT),
        parent_protocol_sha256=sha(PARENT/'protocol.json'),seed=42,tasks={'tlp':cfg},jobs=[dict(task='tlp',arm=a) for a in ARMS],
        objective_id='unobserved_unordered_adaptive_bce_v1',change_scope='loss mask and positive weighting universe only',
        selection='final epoch80; no validation selection',test_used=False,primary='all unobserved unordered pairs Missing-edge macro AP',
        split_sha256=sha(ROOT/'split_manifest.json'),data_manifests=manifests,
        parent_checkpoint_sha256={a:sha(PARENT/'results'/('tlp_'+a)/'final.pt') for a in ARMS},
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},document_sha256=sha(ROOT/'PROTOCOL.md'),
        geometry=old['geometry']['tlp'],tlp_decoder=old['tlp_decoder'],point_control=old['point_control'],hierarchy=old['hierarchy'])
    save(ROOT/'protocol.json',p);verify();print(ROOT,flush=True)
if __name__=='__main__':main()
