"""Assemble byte-identical R34 sources for inspection/model loading, without training."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('destination', type=Path, help='New directory; existing directories are refused')
args = parser.parse_args()
repo = Path(__file__).resolve().parents[1]
release = repo / 'code/r34'
mapping = json.loads((release / 'source_map.json').read_text())
destination = args.destination.expanduser().resolve()
assert not destination.exists(), 'Refusing to overwrite an existing directory'
for name, entry in mapping['files'].items():
    assert hashlib.sha256((repo / entry['repository_path']).read_bytes()).hexdigest() == entry['sha256'], name
destination.mkdir(parents=True)
for name, entry in mapping['files'].items():
    target = destination / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / entry['repository_path'], target)
for name in ['protocol.json', 'contracts.json']:
    shutil.copy2(release / name, destination / name)
shutil.copytree(release / 'results', destination / 'results')
(destination / 'PUBLICATION_SCOPE.json').write_text(json.dumps({
    'purpose': 'Frozen-source model inspection/loading; not a portable end-to-end training package',
    'source_count': len(mapping['files']),
    'omitted_nonruntime_sources': mapping['publication_omissions'],
    'raw_data_geometry_and_parent_run_paths_available': False,
    'frozen_code_modified': False,
}, indent=2) + '\n')
print(f'Assembled {len(mapping["files"])} original source files at {destination}')
print('Original training verification still requires original data/parent artifacts; no gates were bypassed.')
