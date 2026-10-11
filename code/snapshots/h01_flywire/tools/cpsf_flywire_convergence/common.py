import hashlib,json
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha

ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/root/megrez-tmp')
COHORT=BASE/'CPSF_FLYWIRE_COARSE_COHORT_20261002'
OLD=BASE/'CPSF_FLYWIRE_V23_20261001'
ARMS=('full','mean_only')

def digest_object(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def verify_sources(root,protocol):
    bad=[name for name,digest in protocol['source_sha256'].items()
         if not (root/name).is_file() or sha(root/name)!=digest]
    assert not bad,('source mismatch',str(root),bad)

def identity(protocol,arm):
    return dict(schema_version=1,experiment_id=ROOT.name,run_id=ROOT.name+'/coarse_'+arm,
        architecture_id='V23-residual-region-FlyWire',architecture_sha256=protocol['architecture_sha256'],
        hierarchy_id='original-V23-cache-no-V24-runtime-adapter',arm=arm,
        dataset_id='FlyWire-coarse9883-v20261002',cache_manifest_sha256=protocol['cache_manifest_sha256'],
        split_sha256=protocol['split_sha256'],label_sha256=protocol['label_sha256'],
        objective_id='unweighted-smoothed-CE-0.05',seed=42,optimizer_steps=62200,
        evaluation_id='expanded-development-validation1922-plus-legacy1573-added349',
        selection='fixed-final-budget-round50',protocol_sha256=sha(ROOT/'protocol.json'),test_used=False)
