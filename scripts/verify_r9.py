"""Verify published R9 bytes and version separation without CUDA dependencies."""
import ast
import hashlib
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
r = root / 'code/r9'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
meta = read(r / 'server_export.json')
for name, item in meta['files'].items():
    assert sha(r / name) == item['sha256'], name
omitted = meta['publication_omissions']
reference = read(r / 'reference/protocol.json')
for name, digest in reference['source_sha256'].items():
    target = 'reference/' + name
    if (r / target).exists():
        assert sha(r / target) == digest, target
    else:
        assert target in omitted and omitted[target]['sha256'] == digest, target
extension = read(r / 'extension/protocol.json')
for name, digest in extension['source_sha256'].items():
    path = r / 'extension' / name
    if not path.exists(): path = r / 'reference' / name
    assert path.is_file() and sha(path) == digest, name
for task in ('h01_local', 'h01_global'):
    folder = r / 'reference/results' / ('patch_rms_010_' + task)
    metrics = read(folder / 'metrics.json')
    assert sha(folder / 'final.pt') == metrics['final_sha256']
    assert metrics['identity']['version'] == reference['version']
    assert metrics['identity']['protocol_sha256'] == sha(r / 'reference/protocol.json')
results = read(r / 'RESULTS.json')
assert len(results['records']) == 5 and results['seed'] == 42 and not results['test_used']
for record in results['records']:
    if record['status'] != 'completed': assert record['value'] is None
contracts = read(r / 'extension/contracts.json')
assert contracts['passed'] and contracts['protocol_sha256'] == sha(r / 'extension/protocol.json')
for path in r.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
print(f"R9 verified: {len(meta['files'])} original files, two retained checkpoints, five task identities; running tasks have no final score.")
