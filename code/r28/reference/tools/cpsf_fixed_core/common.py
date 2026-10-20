import hashlib,json
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import save,sha
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/root/megrez-tmp');PARENT=BASE/'CPSF_V26_COMPLETE_20261003'
CE=BASE/'CPSF_FLYWIRE_V23_CE_EXPANDED_20261003'
ARMS=('full','no_position','no_scale','no_position_scale','thumbnail','uniform_budget','shuffle_regions','no_context')
TASKS=('h01_local','h01_global','flywire_coarse')
def modal(arm):return arm if arm in ('thumbnail','uniform_budget') else 'joint'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(ROOT/name)==h,('source drift',name)
    for path,h in p['parent_files'].items():assert sha(path)==h,('parent drift',path)
    return p
def rows(task,arm):
    path=PARENT/'data'/task/'manifest.json';ready=json.loads(path.with_name('READY.json').read_text())
    assert ready['ready'] and sha(path)==ready['manifest_sha256']
    return [r for r in json.loads(path.read_text())['records'] if r['modal']==modal(arm)],ready
def load_graph(row,task):
    from tools.cpsf_v26_complete.common import load_graph as inherited
    g=inherited(row,task);g=dict(g)
    ids=torch.where(g['has_g']&g['has_p'])[0]
    seed=int(hashlib.sha256(('within-cell-region42|'+row['key']).encode()).hexdigest()[:8],16)
    rng=np.random.RandomState(seed);order=ids[torch.tensor(rng.permutation(len(ids)),dtype=torch.long)]
    perm=torch.arange(g['nregions'])
    if len(ids)>1:perm[order]=torch.roll(order,1)
    g['surface_content_permutation']=perm
    return g
