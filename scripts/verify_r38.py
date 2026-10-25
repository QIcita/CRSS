"""Verify release hashes, frozen variants, completion boundaries and runtime evidence."""
import ast
import hashlib
import json
from pathlib import Path

repo=Path(__file__).resolve().parents[1];root=repo/'code/r38'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
mapping=read(root/'source_map.json')['files'];export=read(root/'server_export.json')
for name,row in mapping.items():
    p=repo/row['repository_path'];assert sha(p)==row['sha256'],name
    if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8-sig'),filename=name)
for name,row in export['sources'].items():assert mapping[name]['sha256']==row['sha256'],name
for name,row in export['artifacts'].items():assert sha(root/'experiments'/name)==row['sha256'],name
p=read(root/'experiments/drw/protocol.json')
assert p['primary_metric']=='macro_f1' and p['loss']['alpha']==.25
assert p['loss']['train_counts']==[102,962,3950,766,28,231]
assert p['tasks']['flywire_coarse']['updates']==62200 and not p['test_used'] and not p['pretrained']
gate=read(root/'experiments/drw/acceptance/r38_flywire_coarse.json')
assert gate['passed'] and gate['protocol_sha256']==sha(root/'experiments/drw/protocol.json')
assert gate['cpu_contract']['passed'] and gate['compatibility']['passed']
assert not (root/'experiments/drw/results/r38_flywire_coarse/metrics.json').exists()
assert not (root/'experiments/fine/results/r38_flywire_fine/metrics.json').exists()
assert (root/'experiments/fine/PAUSED_BY_USER.json').exists()
assert not list(root.rglob('latest.pt')) and not list(root.rglob('run.lock'))
result=read(root/'RESULTS.json')
assert result['flywire_primary_metric']=='macro_f1' and result['seed']==42
assert len(result['records'])==7
for record in result['records']:
    if record['state']!='complete':
        assert record['metrics'] is None and record['checkpoint'] is None
        continue
    metrics=read(root/record['source'])
    assert sha(root/record['checkpoint'])==record['checkpoint_sha256']
    assert record['epochs']==metrics['epochs']
    if record['variant']=='microns':
        assert metrics['complete'] and metrics['test_evaluated_after_training']
        assert record['updates']==metrics['updates']==4750
        assert record['metrics']=={k:metrics[k] for k in ['validation','test']}
        history=[json.loads(line) for line in (root/'experiments/microns/results/r38_dense/history.jsonl').read_text().splitlines()]
        best=min(history,key=lambda x:x['validation']['cell_macro_poisson_deviance'])
        assert len(history)==50 and best['epoch']==record['selected_epoch']==metrics['selected_epoch']==24
        selection=read(root/'experiments/microns/results/r38_dense/best_selection.json')
        assert best['validation']==selection['validation'] and not selection['test_used']
        # The final selected-weight evaluation is saved separately from the
        # in-loop selection record; compare selection to its own frozen trace.
        identity=read(root/'experiments/microns/results/r38_dense/identity.json')
        assert identity['protocol_sha256']==sha(root/'experiments/microns/protocol.json')
        assert identity['training_protocol_sha256']==sha(root/'experiments/microns/training_protocol.json')
    else:
        assert record['metrics']==metrics['val'] and record['updates']==metrics['steps']
        assert not metrics['test_used']
equivalence=read(root/result['h01_local_dense']['source'])
assert equivalence['passed'] and equivalence['all_development_predictions_exact']
assert equivalence['retraining_skipped'] and not equivalence['new_performance_row']
validation=read(root/'evidence/publication_validation.json')
assert validation['passed'] and len(validation['completed_models'])==5
assert not validation['new_inference_performed'] and not validation['original_data_or_parent_code_imported']
for name,row in validation['loaded_sources'].items():assert mapping[row['path']]['sha256']==row['sha256'],name
for model in validation['completed_models']:
    v=model['variant'];task=model['task']
    checkpoint=('results/r38_dense/best.pt' if v=='microns' else 'results/r38_'+task+'/final.pt')
    assert model['strict_checkpoint_load'] and sha(root/'experiments'/v/checkpoint)==model['checkpoint_sha256']
for name,h in read(root/'release_manifest.json').items():assert sha(repo/name)==h,name
print('Verified R38:',len(mapping),'mapped sources, five completed weights, saved predictions, paused fine classification and running DRW snapshot.')
