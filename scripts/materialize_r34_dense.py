"""Assemble immutable R34 dense-neighborhood sources and final H01 weight."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('destination',type=Path);a=p.parse_args()
repo=Path(__file__).resolve().parents[1];release=repo/'code/r34_dense'
mapping=json.loads((release/'source_map.json').read_text());dest=a.destination.expanduser().resolve()
assert not dest.exists(),'Refusing to overwrite an existing directory'
for name,row in mapping['files'].items():
 assert hashlib.sha256((repo/row['repository_path']).read_bytes()).hexdigest()==row['sha256'],name
dest.mkdir(parents=True)
for name,row in mapping['files'].items():
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/row['repository_path'],target)
for name in ('protocol.json','variant.json'):
 shutil.copy2(release/name,dest/name)
for name in ('results','acceptance','orders','parents'):
 shutil.copytree(release/name,dest/name)
(dest/'PUBLICATION_SCOPE.json').write_text(json.dumps(dict(
 purpose='Frozen model inspection and strict final-weight loading; not portable full training',
 source_count=len(mapping['files']),frozen_code_modified=False,raw_data_distributed=False,
 original_parent_paths_required_for_training=True),indent=2)+'\n')
print('Assembled',len(mapping['files']),'sources at',dest)
