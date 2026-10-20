import json,hashlib
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
from tools.cpsf_fixed_nredd.common import load,move,digest
ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('/root/megrez-tmp/CPSF_TLP_PAIR_REGION_STAGE2_20261008')
DATA_PARENT=PARENT
ARMS=('full','mean_only','no_fusion','no_scale','skeleton','point','shuffle_descriptor')
TASKS=('tlp',)
def parent_arm(arm):return 'full' if arm=='shuffle_descriptor' else arm
def modal(arm):return arm if arm in ('point','skeleton') else 'joint'
def parent_rows(mode):
    path=DATA_PARENT/'data/tlp'/mode/'manifest.json';d=json.loads(path.read_text())
    assert d['complete'] and all(r['split'] in ('train','val') for r in d['records'])
    return d['records'],sha(path)
def verify():
    p=json.loads((ROOT/'protocol.json').read_text());assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    old=json.loads((PARENT/'protocol.json').read_text())
    for n,h in old['source_sha256'].items():assert sha(PARENT/n)==sha(ROOT/n)==h
    for n,h in p['source_sha256'].items():assert sha(ROOT/n)==h,('source drift',n)
    assert sha(ROOT/'split_manifest.json')==sha(PARENT/'split_manifest.json')==p['split_sha256']
    assert sha(ROOT/'PROTOCOL.md')==p['document_sha256']
    for mode,h in p['parent_data_manifests'].items():assert sha(DATA_PARENT/'data/tlp'/mode/'manifest.json')==h
    for arm,h in p['parent_checkpoint_sha256'].items():assert sha(PARENT/'results'/('tlp_'+arm)/'final.pt')==h
    return p
def rows(task,arm):
    assert task=='tlp' and arm in ARMS
    path=ROOT/'data/tlp'/modal(arm)/'manifest.json';d=json.loads(path.read_text())
    assert d['complete'] and d['protocol_sha256']==sha(ROOT/'protocol.json')
    return d['records'],sha(path)
