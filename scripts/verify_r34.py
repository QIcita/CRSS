"""Verify frozen R34 code, six results, two checkpoints and publication hashes."""
import ast
import hashlib
import json
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
root = repo / 'code/r34'
def read(p):
    return json.loads(p.read_text())
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

p = read(root / 'protocol.json')
assert p['version'] == 'CPSF_PATH_MESSAGE_20261020_R34_A2'
assert p['seed'] == 42 and not p['test_used'] and not p['pretrained']
assert not p['stages_2_and_3_enabled']
source = read(root / 'source_map.json')
assert set(p['source_sha256']) == set(source['files']) | set(source['publication_omissions'])
for name, entry in source['files'].items():
    f = repo / entry['repository_path']
    assert sha(f) == entry['sha256'] == p['source_sha256'][name], name
    if f.suffix == '.py':
        ast.parse(f.read_text(encoding='utf-8-sig'), filename=name)
for name, entry in source['publication_omissions'].items():
    assert entry['sha256'] == p['source_sha256'][name]
for name, entry in read(root / 'server_export.json').items():
    assert sha(root / name) == entry['sha256'], name
gate = read(root / 'contracts.json')
assert gate['passed'] and gate['protocol_sha256'] == sha(root / 'protocol.json')
results = read(root / 'RESULTS.json')
assert len(results['records']) == 6 and results['input_protocol'] == 'original_R28_H01_input_not_geometry_FPS_A3'
for q in results['records']:
    d = root / q['result_directory']
    m = read(d / 'metrics.json')
    i = m['identity']
    cfg = p['tasks'][i['task']]
    assert i['version'] == p['version'] and i['protocol_sha256'] == sha(root / 'protocol.json')
    assert i['seed'] == 42 and not i['pretrained'] and not m['test_used']
    assert m['epochs'] == cfg['epochs'] and m['steps'] == cfg['updates']
    assert m['val'] == q['metrics'] and m['independent_metric_check']
    assert q['final_sha256'] == m['final_sha256']
    if q['checkpoint_published']:
        assert i['kind'] == 'path_message' and sha(d / 'final.pt') == m['final_sha256']
    else:
        assert not (d / 'final.pt').exists()
audit = read(root / 'evidence/publication_validation.json')
assert audit['passed'] and len(audit['jobs']) == 6
assert all(x['strict_checkpoint_load'] and x['prediction_metric_recomputed'] for x in audit['jobs'].values())
for name, digest in read(root / 'release_manifest.json').items():
    assert sha(repo / name) == digest, name
print(f'R34 verified: {len(source["files"])} original sources, 6 final H01 results, 2 retained checkpoints; stage 1, original input only.')
