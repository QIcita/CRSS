"""Immutable direct R28 descendant; data and dependencies remain read-only."""
from .common import *
from .model import VARIANTS


def main():
    assert not (ROOT/'protocol.json').exists()
    pp = json.loads((PARENT/'protocol.json').read_text())
    assert sha(PARENT/'protocol.json') == '1daddf31b92efd7abb57b3047349bff0f4d71ebe3a908a27951fe553949edde7'
    sources = dict(pp['source_sha256'])
    for k, h in sources.items():
        assert sha(ROOT/k) == h, ('R28 source drift', k)
    for f in (ROOT/'tools/cpsf_path_message').glob('*.py'):
        sources[str(f.relative_to(ROOT))] = sha(f)
    parents = dict(pp['parent_sha256'])
    parents[str(PARENT/'protocol.json')] = sha(PARENT/'protocol.json')
    refs, audit = {}, {}
    for task, cfg in pp['tasks'].items():
        mpath = ROOT/'data'/task/'manifest.json'
        assert sha(mpath) == sha(PARENT/'data'/task/'manifest.json')
        records = json.loads(mpath.read_text())['records']
        tr = [r for r in records if r['split']=='train']
        va = [r for r in records if r['split']=='val']
        assert (len(tr),len(va)) == (cfg['train'],cfg['val'])
        assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
        assert all(r['split'] in ('train','val') for r in records)
        for r in records:
            for pathkey, hashkey in (('path','sha256'),('feature_path','feature_sha256')):
                assert sha(r[pathkey]) == r[hashkey], r['key']
                parents[r[pathkey]] = r[hashkey]
        parents[str(mpath)] = sha(mpath)
        parent_result = PARENT/'results'/f'r9_lpa_attention_{task}'
        refs[task] = dict(root=str(parent_result), metrics=json.loads((parent_result/'metrics.json').read_text())['val'])
        for f in ('metrics.json','history.json'):
            parents[str(parent_result/f)] = sha(parent_result/f)
        audit[task] = dict(records=len(records),train=len(tr),val=len(va),same_manifest=True,cell_disjoint=True,all_input_hashes_valid=True)
    r17 = Path('/root/megrez-tmp/CPSF_CONDITIONAL_REGION_20261016_R17_A1')
    provenance = {}
    for name in ('geo_attention.py','conditional.py','protocol.json'):
        f = r17/name if name=='protocol.json' else r17/'tools/cpsf_utonia_context'/name
        assert f.exists()
        parents[str(f)] = sha(f)
        provenance[name] = dict(path=str(f),sha256=sha(f))
    save(ROOT/'input_audit.json',dict(passed=True,tasks=audit,test_used=False,observation_and_labels_unchanged=True))
    save(ROOT/'protocol.json',dict(version=ROOT.name,parent_root=str(PARENT),parent_variant='r9_lpa_attention',seed=42,
        tasks=pp['tasks'],source_sha256=sources,parent_sha256=parents,upstream=pp['upstream'],r17_control_source=provenance,
        comparison_reference=refs,variants={
            'path_message':dict(change='mass-preserving 16D content/path-conditioned centered patch messages',query='observed skeleton region LayerNorm plus shared learned constant'),
            'constant_message':dict(change='identical gated-message parameters and geometry; no observed graph content in NEW query',query='shared learned constant; graph still used by original R28'),
            'ordinary_attention':dict(change='R17 four-head centered-value attention with geometry logits and mass prior; R28 patch input',source='official GeoTransformer-derived R17 adapter; not equal capacity to gated message')},
        geometry='original five-dimensional phi only; no extra geometry or support',
        amplitude=.1,rank=16,edge_chunk=16384,initialization='new output zero, inherited R28 parameters/RNG unchanged',
        new_surface_addition='after inherited shared physical code, before original region joint',
        objective='inherited H01 losses; same labels, budget, orders, engineering and all-parent gradients',
        promotion=dict(both_not_worse_than_r28=True,local_min_gain=.001,global_mae_min_reduction=.0001),
        pretrained=False,test_used=False,selection='fixed final epoch development only',
        stages_2_and_3_enabled=False,automation_unchanged='PAUSED',
        no_leakage_scope='original train/dev cell-disjoint manifest; only input geometry, no target fields passed; development repeatedly used, not independent holdout'))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)


if __name__=='__main__':
    main()
