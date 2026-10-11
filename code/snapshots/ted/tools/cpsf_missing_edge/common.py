import json,hashlib
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
from tools.cpsf_fixed_nredd.common import load,move,digest

ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('/root/megrez-tmp/CPSF_FIXED_CORE_NREDD_20261006')
ARMS=('full','mean_only','no_fusion','no_scale','skeleton','point')
TASKS=('tlp',)
def modal(arm):return arm if arm in ('point','skeleton') else 'joint'
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    assert p['parent_root']==str(PARENT)
    assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    old=json.loads((PARENT/'protocol.json').read_text())
    for name,value in old['source_sha256'].items():
        assert sha(PARENT/name)==value,('parent source drift',name)
        assert sha(ROOT/name)==value,('copied parent source drift',name)
    for name,value in p['source_sha256'].items():assert sha(ROOT/name)==value,('new source drift',name)
    assert sha(ROOT/'split_manifest.json')==sha(PARENT/'split_manifest.json')==p['split_sha256']
    assert sha(ROOT/'PROTOCOL.md')==p['document_sha256']
    for arm in ARMS:
        name=PARENT/'data/tlp'/modal(arm)/'manifest.json'
        assert sha(name)==p['data_manifests'][modal(arm)]
        assert sha(PARENT/'results'/('tlp_'+arm)/'final.pt')==p['parent_checkpoint_sha256'][arm]
    return p
def rows(task,arm):
    assert task=='tlp' and arm in ARMS
    path=PARENT/'data/tlp'/modal(arm)/'manifest.json';d=json.loads(path.read_text())
    assert d['complete'] and d['protocol_sha256']==sha(PARENT/'protocol.json')
    assert all(r['split'] in ('train','val') for r in d['records'])
    return d['records'],sha(path)
