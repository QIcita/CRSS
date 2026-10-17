import json,hashlib
from pathlib import Path
import torch
from tools.cpsf_v21_segment_units.io import setup,save,sha,Cache
from tools.cpsf_fixed_nredd.common import load,move,digest

ROOT=Path(__file__).resolve().parents[2]
PARENT=Path('/root/megrez-tmp/CPSF_TLP_DESCRIPTOR_FACTORS_STAGE5_20261008')
DATA_PARENT=Path('/root/megrez-tmp/CPSF_TLP_POINT_GEOMETRY_STAGE3_20261008')
ANCHOR=Path('/root/megrez-tmp/CPSF_TLP_DETERMINISTIC_STAGE4_20261008')
ARMS=('full','mean_only','no_fusion','no_scale','point','skeleton')
TASKS=('tlp',)
DESCRIPTORS=('voxel','directional_xyz','directional')
def modal(arm):return arm if arm in ('point','skeleton') else 'joint'
def job_key(arm,d):return 'tlp_'+d+'_'+arm
def parent_result(arm,d):return ANCHOR/'results'/job_key(arm,'voxel')
def parent_rows(arm):
    path=DATA_PARENT/'data/tlp'/modal(arm)/'manifest.json';return json.loads(path.read_text())['records']

def rows(task,arm):
    assert task=='tlp' and arm in ARMS
    path=ROOT/'data/tlp'/modal(arm)/'manifest.json';d=json.loads(path.read_text())
    assert d['complete'] and d['protocol_sha256']==sha(ROOT/'protocol.json')
    assert all(r['split'] in ('train','val') for r in d['records'])
    return d['records'],sha(path)
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    assert sha(PARENT/'protocol.json')==p['parent_protocol_sha256']
    old=json.loads((PARENT/'protocol.json').read_text())
    for name,h in old['source_sha256'].items():assert sha(PARENT/name)==sha(ROOT/name)==h
    for name,h in p['source_sha256'].items():assert sha(ROOT/name)==h,('source drift',name)
    assert sha(ROOT/'PROTOCOL.md')==p['document_sha256']
    assert sha(ROOT/'split_manifest.json')==sha(PARENT/'split_manifest.json')==p['split_sha256']
    for mode,h in p['parent_data_manifests'].items():assert sha(DATA_PARENT/'data/tlp'/mode/'manifest.json')==h
    for key,reference in p['references'].items():
        for filename,h in reference['files'].items():assert sha(ANCHOR/'results'/key/filename)==h
    return p
