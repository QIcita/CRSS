import hashlib,json
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/root/megrez-tmp')
FW=BASE/'CPSF_FLYWIRE_COARSE_COHORT_20261002'
CE=BASE/'CPSF_FLYWIRE_V23_CE_EXPANDED_20261003'
OLD=BASE/'CPSF_FLYWIRE_V23_20261001'
H01=BASE/'CPSF_V25_POSITION_ROUTING_20261001'
SCALE=BASE/'CPSF_V23_SCALE_ABLATION_20261001'
ARMS=('full','fixed_residual','mean_only','no_fusion','point','skeleton','gate_global','no_position','no_scale','no_position_scale','thumbnail','uniform_budget')
TASKS=('h01_local','h01_global','flywire_coarse')
def modal(arm):return arm if arm in ('point','skeleton','thumbnail','uniform_budget') else 'joint'
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def verify():
    p=json.loads((ROOT/'protocol.json').read_text())
    for name,h in p['source_sha256'].items():assert sha(ROOT/name)==h,('source drift',name)
    for path,h in p['parent_files'].items():assert sha(path)==h,('parent drift',path)
    return p
def rows(task,arm):
    path=ROOT/'data'/task/'manifest.json';ready=json.loads(path.with_name('READY.json').read_text())
    assert ready['ready'] and sha(path)==ready['manifest_sha256']
    m=json.loads(path.read_text());return [r for r in m['records'] if r['modal']==modal(arm)],ready
def load_graph(row,task):
    if task=='flywire_coarse':
        from tools.cpsf_flywire_v23.data import load
    else:
        from tools.cpsf_v21_segment_units.io import load
    g=load(row['path'])
    if task!='flywire_coarse':
        from tools.cpsf_v24_physical_conditioning.data import adapt
        g=adapt(g)
    return g
