import json
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save, sha

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/root/megrez-tmp/CPSF_V22_SHARED_REGION_20260930')


def main():
    assert not (ROOT/'protocol.json').exists()
    old = json.loads((BASE/'protocol.json').read_text())
    assert json.loads((BASE/'pipeline_status.json').read_text())['state'] == 'complete'
    for name, digest in old['source_sha256'].items():
        assert sha(BASE/name) == digest
        assert sha(ROOT/name) == digest, (name, 'changed inherited implementation')
    # The optimization/evaluation loop is identical except mean_only uses joint data.
    old_train = (BASE/'tools/cpsf_v22_shared_region/train.py').read_text()
    expected = old_train.replace("a.arm in ('full','no_fusion')", "a.arm in ('full','mean_only','no_fusion')")
    assert (ROOT/'tools/cpsf_v23_residual_region/train.py').read_text() == expected
    ready = json.loads((BASE/'cache/READY.json').read_text())
    assert sha(ROOT/'cache/manifest.json') == ready['manifest_sha256']
    baseline = {}
    for task in ('local', 'global'):
        for arm in ('full', 'no_fusion', 'point', 'skeleton'):
            p = BASE/'results'/(task+'_'+arm)
            m = json.loads((p/'metrics.json').read_text())
            assert m['seed'] == 42 and m['epochs'] == old['epochs'][task]
            baseline[task+'_'+arm] = dict(path=str(p),
                sha256={n:sha(p/n) for n in ('metrics.json','initial.pt','final.pt')},
                val={k:v for k,v in m['val'].items() if k != 'rows'})
    old.update(version='V23 evidence-preserving residual region interaction',
        arms=['full','mean_only'], residual_coefficient=.1,
        inherited_protocol_sha256=sha(BASE/'protocol.json'), baselines=baseline,
        cache_manifest_sha256=ready['manifest_sha256'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        protocol_document_sha256=sha(ROOT/'CPSF_V23_RESIDUAL_REGION_PROTOCOL_20261001.md'))
    save(ROOT/'protocol.json', old)


if __name__ == '__main__':
    main()
