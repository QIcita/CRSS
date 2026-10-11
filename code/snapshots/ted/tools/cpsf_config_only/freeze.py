"""Run once on server after upload; never modifies parent runs."""
import hashlib,json,shutil
from pathlib import Path
BASE=Path('/root/megrez-tmp');ROOT=BASE/'CPSF_NREDD_CONFIG_ONLY_20261009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
assert not (ROOT/'protocol.json').exists(),'Never replace an existing frozen protocol'
ted=BASE/'CPSF_FIXED_CORE_NREDD_20261006';tlp=BASE/'CPSF_TLP_DETERMINISTIC_STAGE4_20261008'
p1=read(ted/'protocol.json');p2=read(tlp/'protocol.json')
for parent,protocol in [(ted,p1),(tlp,p2)]:
    for name,h in protocol['source_sha256'].items():
        source=parent/name;target=ROOT/name;assert sha(source)==h
        if target.exists():assert sha(target)==h,('conflicting inherited source',name)
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    assert sha(parent/'split_manifest.json')==protocol['split_sha256']
assert sha(ted/'split_manifest.json')==sha(tlp/'split_manifest.json')
shutil.copy2(ted/'split_manifest.json',ROOT/'split_manifest.json')
arms=['full','mean_only','no_fusion','skeleton','point','no_scale']
baseline_dirs={t:{a:str(parent/'results'/((t+'_'+a) if t=='ted' else 'tlp_voxel_'+a)) for a in arms+(['no_position'] if t=='ted' else [])} for t,parent in [('ted',ted),('tlp',tlp)]}
refs={};manifests={}
for parent in [ted,tlp]:
    for name in ['protocol.json','split_manifest.json']:refs[str(parent/name)]=sha(parent/name)
for task in ['ted','tlp']:
    manifests[task]={}
    parent=ted if task=='ted' else BASE/'CPSF_TLP_POINT_GEOMETRY_STAGE3_20261008'
    refs[str(parent/'protocol.json')]=sha(parent/'protocol.json')
    for mode in ['joint','skeleton','point']:
        path=parent/'data'/task/mode/'manifest.json';d=read(path)
        assert d['complete'] and d['protocol_sha256']==sha(parent/'protocol.json')
        refs[str(path)]=sha(path);manifests[task][mode]=str(path)
    for arm,directory in baseline_dirs[task].items():
        directory=Path(directory)
        for name in ['identity.json','history.json','metrics.json','final.pt']:
            refs[str(directory/name)]=sha(directory/name)
        for prediction in directory.glob('*prediction*.npz'):refs[str(prediction)]=sha(prediction)
        m=read(directory/'metrics.json');assert not m['test_used'] and not m['smoke']
jobs=[dict(task=t,arm=a,config=c) for t in ['tlp','ted'] for c in ['lr_half','slow_decay'] for a in baseline_dirs[t]]
p=dict(version='Configuration only; retained per-task parents; 2 matched one-factor candidates',seed=42,test_used=False,
    tasks={'ted':p1['tasks']['ted'],'tlp':p2['tasks']['tlp']},jobs=jobs,baseline_dirs=baseline_dirs,manifests=manifests,references=refs,
    source_sha256={str(f.relative_to(ROOT)):sha(f) for f in sorted((ROOT/'tools').rglob('*.py'))},document_sha256=sha(ROOT/'PROTOCOL.md'),
    wait_for=str(BASE/'CPSF_TLP_DIRECTIONAL_STAGE6_20261009'),
    model_families={'ted':'20261006 fixed NREDD TED adapter unchanged','tlp':'stage4 deterministic region reduction; voxel descriptor unchanged'},
    selection='All shared-config deltas and equal-search best-per-arm validation scores; no test use and no maximization of full-ablation gap',
    configs={'lr_half':'initial LR x0.5; baseline schedule','slow_decay':'initial LR unchanged; TED inverse decay divisor3; TLP step40 gamma0.1'})
write(ROOT/'protocol.json',p)
for name in ['runtime','logs','contracts','results']:(ROOT/name).mkdir(exist_ok=True)
write(ROOT/'runtime/scheduler_config.json',dict(max_parallel=6,startup_spacing_seconds=45,gpu_job_allowance_mib=5120,gpu_headroom_mib=2048,ram_job_allowance_mib=4096,ram_headroom_mib=6144))
print(json.dumps(dict(root=str(ROOT),jobs=len(jobs),sources=len(p['source_sha256']),references=len(refs),protocol_sha256=sha(ROOT/'protocol.json'))))
