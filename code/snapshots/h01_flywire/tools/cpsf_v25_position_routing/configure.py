import json
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save, sha
from .model import JOBS, ARMS

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/root/megrez-tmp/CPSF_V24_PHYSICAL_CONDITIONING_20261001')
V23 = Path('/root/megrez-tmp/CPSF_V23_RESIDUAL_REGION_20261001')


def main():
    assert not (ROOT / 'protocol.json').exists()
    inherited = json.loads((BASE / 'protocol.json').read_text())
    assert json.loads((BASE / 'pipeline_status.json').read_text())['state'] == 'complete'
    for name, digest in inherited['source_sha256'].items():
        assert sha(BASE / name) == digest and sha(ROOT / name) == digest, name
    baselines = {}
    for task in ('local', 'global'):
        path = BASE / 'results' / (task + '_hierarchy_only')
        m = json.loads((path / 'metrics.json').read_text())
        baselines[task + '_baseline'] = dict(path=str(path),
            sha256={n: sha(path / n) for n in ('metrics.json', 'initial.pt', 'final.pt')},
            val={k: v for k, v in m['val'].items() if k != 'rows'})
    ready = json.loads((ROOT / 'cache/READY.json').read_text())
    assert sha(ROOT / 'cache/manifest.json') == ready['manifest_sha256']
    save(ROOT / 'protocol.json', dict(version='V25 spatial incidence routing factorial', seed=42,
        epochs=inherited['epochs'], jobs=JOBS, arms=ARMS, baselines=baselines,
        cache_manifest_sha256=ready['manifest_sha256'], minimum_coarse_groups=4,
        added_parameters=['routing_weight[4]'], routing_init=0., routing_logit_bound=2.,
        routing_features=['distance/radius', '(distance/radius)^2', 'axial', 'axial^2'],
        routing_normalization='within receiving region; fixed original geometric masses and prior support',
        factors=['shared additive region position', 'scalar spatial incidence routing'],
        unchanged=['dynamic point sampling', 'original skeleton', 'path_position', 'relation(phi)', 'scale', 'Processor including 3D RoPE', 'pool', 'residual fusion'],
        optimizer=dict(name='AdamW', lr=.0003, weight_decay=.01, cosine_eta_min=.00003, clip=5., batch={'local':8, 'global':4}),
        initialization='all inherited weights bitwise equal to V23 seed42; four added weights zero',
        evaluation='final fixed epoch; H01 development split; task-supervised, not frozen transfer',
        inherited_protocol_sha256=sha(BASE / 'protocol.json'),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'tools').rglob('*.py')},
        protocol_document_sha256=sha(ROOT / 'CPSF_V25_POSITION_ROUTING_PROTOCOL_20261001.md')))


if __name__ == '__main__':
    main()
