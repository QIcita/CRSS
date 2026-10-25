import shutil
from pathlib import Path
from .common import ROOT,R38,R38_SHA,R34,R34_SHA,INPUT,INPUT_SHA,TASKS,read,save,sha

def main():
    assert not (ROOT/'protocol.json').exists()
    assert sha(R38/'protocol.json')==R38_SHA and sha(R34/'protocol.json')==R34_SHA and sha(INPUT/'protocol.json')==INPUT_SHA
    parent=read(R34/'protocol.json');files=dict(read(R38/'protocol.json')['files']);manifests={}
    for task in TASKS:
        manifest=INPUT/'data'/task/'manifest.json';rr=read(manifest)['records'];cfg=parent['tasks'][task]
        tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
        assert (len(tr),len(va))==(cfg['train'],cfg['val']) and len(tr)+len(va)==len(rr)
        assert len({r['key'] for r in rr})==len(rr) and {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
        for r in rr:
            assert r['task']==task and r['protocol_sha256']==INPUT_SHA
            for f in ['path','feature_path','graph_path']:assert Path(r[f]).is_file()
        manifests[task]=sha(manifest);dest=ROOT/'data'/task;dest.parent.mkdir(exist_ok=True)
        dest.symlink_to(INPUT/'data'/task,target_is_directory=True)
        (ROOT/'orders').mkdir(exist_ok=True);shutil.copyfile(R34/'orders'/(task+'.json'),ROOT/'orders'/(task+'.json'))
        save(ROOT/'input_audits'/(task+'.json'),dict(passed=True,manifest_sha256=manifests[task],train=len(tr),val=len(va),cell_disjoint=True,test_used=False,canonical_input_unchanged=True,cache_bytes_hash_verified_on_load=True,full_byte_reaudit_before_training=False))
    for f in list((ROOT/'tools/cpsf_r38_dense').glob('*.py'))+list((ROOT/'orders').glob('*.json')):files[str(f)]=sha(f)
    files[str(R38/'protocol.json')]=R38_SHA;files[str(R34/'protocol.json')]=R34_SHA
    for f,h in files.items():assert sha(f)==h,('source drift',f)
    save(ROOT/'protocol.json',dict(version=ROOT.name,files=files,seed=42,pretrained=False,test_used=False,
        tasks={t:parent['tasks'][t] for t in TASKS},jobs=[dict(id='r38_'+t,model='r38',task=t) for t in TASKS],
        input_manifests=manifests,parent_r38=str(R38),parent_r38_protocol_sha256=R38_SHA,parent_training=str(R34),
        architecture='R38 pair fusion with SPoTr dense-neighborhood increment lifted to original fine support before pair fusion and both RCM consumers',
        neighborhood=dict(minimum_size=.5,multiplier=2,target_mean_points=10,max_levels=25,relative_geometry_unit=.5,original_fine_support=True,original_utonia_input=True),
        fusion=dict(rank=16,alpha=.1,extra_parameters=9216,operator_source=str(R38/'tools/cpsf_cmm_pair_fusion/operator.py')),
        training='Exact inherited R38 loop; original per-task R34 budgets, labels, epoch orders, optimizer and fixed final epoch',
        cache='Original caches read-only; derived dense layout in bounded RAM; no duplicate geometry cache',
        selection='fixed final epoch development; no test feedback',
        acceptance='Exact original R38 two-step H01 control; FlyWire disabled-pair R34 control; dense geometry match; active gradients; single backbone calls; same shared fused patch; extreme-shape backward; A-B-A; six real updates discarded'))
    print('FROZEN',sha(ROOT/'protocol.json'),flush=True)
if __name__=='__main__':main()
