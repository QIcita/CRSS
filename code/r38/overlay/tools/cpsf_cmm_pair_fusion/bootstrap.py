"""Freeze two H01 jobs and pin original datasets without copying geometry."""
import shutil
from .common import ROOT,PARENT,GEOMETRY,PARENT_SHA,TASKS,MODEL,read,save,sha


def main():
    assert not (ROOT/'protocol.json').exists(),'Cannot overwrite an immutable protocol'
    assert sha(PARENT/'protocol.json')==PARENT_SHA
    parent=read(PARENT/'protocol.json');files=dict(parent['files'])
    files[str(PARENT/'protocol.json')]=PARENT_SHA
    for evidence in ('operator_contract_cpu.json','operator_contract_cuda.json','worker_protocol_audit.json','model_preflight.json'):
        assert read(ROOT/evidence)['passed'],evidence
        files[str(ROOT/evidence)]=sha(ROOT/evidence)
    input_reports={}
    for task in TASKS:
        src=GEOMETRY/'data'/task;manifest=src/'manifest.json';records=read(manifest)['records']
        cfg=parent['tasks'][task];tr=[r for r in records if r['split']=='train'];va=[r for r in records if r['split']=='val']
        assert len(tr)==cfg['train'] and len(va)==cfg['val'] and len(records)==len(tr)+len(va)
        assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
        oldgate=read(PARENT/'input_audits'/(task+'.json'))
        assert oldgate['passed'] and sha(manifest)==oldgate['manifest_sha256']
        for r in records:
            assert sha(r['path'])==r['sha256'] and sha(r['feature_path'])==r['feature_sha256']
        dest=ROOT/'data'/task;dest.parent.mkdir(exist_ok=True);dest.symlink_to(src,target_is_directory=True)
        (ROOT/'orders').mkdir(exist_ok=True);shutil.copyfile(PARENT/'orders'/(task+'.json'),ROOT/'orders'/(task+'.json'))
        files[str(manifest)]=sha(manifest)
        input_reports[task]=dict(passed=True,task=task,manifest_sha256=sha(manifest),records=len(records),
            train=len(tr),development=len(va),cell_disjoint=True,checked_cache_files=2*len(records),
            exact_parent_manifest=True,test_used=False,geometry_copied=False)
    for f in list((ROOT/'tools/cpsf_cmm_pair_fusion').glob('*.py'))+list((ROOT/'orders').glob('*.json'))+list(ROOT.glob('CPSF_CMM_THREE_STREAM_FUSION_DESIGN*.md')):
        files[str(f)]=sha(f)
    for f,h in files.items():assert sha(f)==h,('source mismatch',f)
    save(ROOT/'protocol.json',dict(version=ROOT.name,files=files,seed=42,pretrained=False,test_used=False,
        tasks={t:parent['tasks'][t] for t in TASKS},jobs=[dict(id=MODEL+'_'+t,model=MODEL,task=t) for t in TASKS],
        parent=str(PARENT),parent_protocol_sha256=PARENT_SHA,
        architecture='R34 original CMM main effects plus bounded low-rank H-D, H-C and D-C pair interactions; Utonia/SPoTr internals unchanged',
        trainable_parameters_added=9216,operator_initialization='3 input projections use private seed42; 3 output matrices zero; original initialization and RNG unchanged',fusion=dict(rank=16,alpha=.1,epsilon=1e-6,main_weights=[1,.1,.1],chunk=4096,checkpointed=True),
        unchanged=['unified FPS and mass','original SPoTr all paths','Utonia features max-pooling/grouping/coordinates/RoPE/CPE/MLP/inverse and RMS0.1','RCM','position and scale','readout','training protocol'],
        inherited_worker=str(PARENT/'tools/cpsf_r34_fps_rebench/worker.py'),
        worker_changes=['own backend/root and r38 identity','real operator acceptance replaces parent acceptance','r38 evaluation uses same RNG rule as r34'],
        selection='fixed final epoch development; no test feedback',
        comparison='same candidate versus both matched R34 and R28; no promotion if paired results missing',
        stage='first stage only; two H01 jobs, no automatic architecture search',
        concurrency='at most two workers, account for baseline, R34, adaptive-patch and CB2 memory reservations, preserve existing jobs'))
    for task,report in input_reports.items():
        report['protocol_sha256']=sha(ROOT/'protocol.json');save(ROOT/'input_audits'/(task+'.json'),report)
    print('FROZEN',ROOT.name,sha(ROOT/'protocol.json'),flush=True)

if __name__=='__main__':main()
