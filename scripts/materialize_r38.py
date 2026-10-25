"""Assemble hash-verified R38 sources, five completed weights and experiment records."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--variant',choices=['original','dense','fine','drw','microns'],default='drw')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];release=repo/'code/r38'
    dest=args.destination.expanduser().resolve()
    assert not dest.exists(),'Refusing to overwrite an existing directory'
    mapping=json.loads((release/'source_map.json').read_text())['files']
    for name,row in mapping.items():
        assert hashlib.sha256((repo/row['repository_path']).read_bytes()).hexdigest()==row['sha256'],name
    export=json.loads((release/'server_export.json').read_text())
    for name,row in export['artifacts'].items():
        assert hashlib.sha256((release/'experiments'/name).read_bytes()).hexdigest()==row['sha256'],name
    dest.mkdir(parents=True)
    for name,row in mapping.items():
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(repo/row['repository_path'],target)
    shutil.copytree(release/'experiments',dest/'experiments')
    selected=release/'experiments'/args.variant
    for source in selected.iterdir():
        target=dest/source.name
        if source.is_dir():shutil.copytree(source,target)
        else:shutil.copy2(source,target)
    (dest/'PUBLICATION_SCOPE.json').write_text(json.dumps(dict(
        source_count=len(mapping),selected_variant=args.variant,
        purpose='Frozen source inspection, strict checkpoint loading and saved-prediction recomputation',
        frozen_source_modified=False,raw_data_distributed=False,
        original_data_and_parent_paths_required_for_training=True,
        running_checkpoints_distributed=False),indent=2)+'\n')
    print('Assembled',len(mapping),'sources at',dest)

if __name__=='__main__':main()
