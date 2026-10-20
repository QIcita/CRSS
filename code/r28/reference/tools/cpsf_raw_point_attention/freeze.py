"""Freeze direct R9 lineage and an unchanged-observation raw evidence interface."""
from .common import *


def main():
    assert not (ROOT / 'protocol.json').exists()
    parent = json.loads((PARENT / 'protocol.json').read_text())
    assert sha(PARENT / 'protocol.json') == '7a4a4283470e3bf689e830268a0d9f6e211f5fe2536c7b3b8b467cff96f7afc6'
    sources = dict(parent['source_sha256'])
    for name, h in sources.items():
        assert sha(ROOT / name) == h, ('parent source drift', name)
    for folder in ('tools/cpsf_raw_point_attention', 'vendor/SPoTr'):
        for file in (ROOT / folder).rglob('*'):
            if file.is_file() and '__pycache__' not in str(file):
                sources[str(file.relative_to(ROOT))] = sha(file)
    sources['launch_after_acceptance.py'] = sha(ROOT / 'launch_after_acceptance.py')
    parents = {str(PARENT / 'protocol.json'): sha(PARENT / 'protocol.json')}
    refs = {}
    for task in parent['tasks']:
        directory = PARENT / 'results' / f'patch_rms_010_{task}'
        refs[task] = dict(root=str(directory), metrics=json.loads((directory / 'metrics.json').read_text())['val'])
        assert sha(ROOT / 'data' / task / 'manifest.json') == sha(PARENT / 'data' / task / 'manifest.json')
        for file in (PARENT / 'data' / task / 'manifest.json', directory / 'metrics.json', directory / 'history.json'):
            parents[str(file)] = sha(file)
        sources['data/' + task + '/manifest.json'] = sha(ROOT / 'data' / task / 'manifest.json')
    upstream = dict(Utonia=parent['upstream']['Utonia'], SPoTr=dict(
        repository='https://github.com/mlvlab/SPoTr',
        commit='69ea023d3781372df80e23326ef56b58a9b60cd6',
        paper='CVPR2023 Self-positioning Point-based Transformer for Point Cloud Understanding',
        module='LPAMLP (entire original constructor/forward), new_attention, get_aggregation_features, original conv/activation factories',
        license='MIT', pretrained=False,
        adaptation='original within-patch bipartite raw grouping; original mass and padding logit prior; no normalization; one post-conv; no global SPoTr path; unused official alpha frozen'))
    variants = {kind: dict(pooling='learned_channel_attention' if kind.endswith('attention') else 'original_mass_uniform',
        features='original 96D point MLP output and original patch-relative XYZ',
        original_memberships_and_points=True, singleton='original R9 only',
        official_forward_unmodified=True, norm=None, post_convs=1, alpha=.1,
        initialization='new output projection zero; original R9 stream exact',
        insertion='parallel raw-point correction after protected patch statistics, original Utonia RMS0.1 untouched',
        control_capacity_note='same registered modules; uniform does not train attention projection; report active capacity separately')
        for kind in ('r9_lpa_attention', 'r9_lpa_uniform')}
    save(ROOT / 'protocol.json', dict(version=ROOT.name, parent_root=str(PARENT), seed=42,
        tasks=parent['tasks'], source_sha256=sources, parent_sha256=parents,
        upstream=upstream, variants=variants, comparison_reference=refs,
        selection='fixed final epoch development; no test feedback',
        promotion=dict(both_not_worse=True, local_min_gain=.001, global_mae_min_reduction=.0001),
        pretrained=False, test_used=False,
        engineering='original R9 kernel policy, precision, seed, optimizer, final epochs and cache budget unchanged; lossless grouping metadata added',
        source_boundary='only new tools/cpsf_raw_point_attention and vendor/SPoTr; parent source unchanged'))
    print('protocol', sha(ROOT / 'protocol.json'), flush=True)


if __name__ == '__main__':
    main()
