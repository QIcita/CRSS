import json,hashlib,time
from pathlib import Path
BASE=Path('/root/megrez-tmp')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
out=Path('/root/megrez-tmp/CPSF_CURRENT_FIVE_TASK_20261010');out.mkdir(exist_ok=True)
rows=[];audits={}
def sources(root,p):
    for name,h in p['source_sha256'].items():assert sha(root/name)==h
    return len(p['source_sha256'])
root=BASE/'CPSF_FIXED_CORE_ABLATION_20261005';p=read(root/'protocol.json');s=read(root/'summary.json')
assert not s['test_used'];n=sources(root,p)
for name,h in p['parent_files'].items():assert sha(Path(name))==h
for key,ref in p['references'].items():
    f=Path(ref['path'])
    for name,h in ref['hashes'].items():assert sha(f/name)==h
    m=read(f/'metrics.json');assert m['val']==ref['val']
    task,arm=key.split('/');metric='mae' if task=='h01_global' else 'macro_f1'
    rows.append(dict(task=task,arm=arm,metric=metric,value=m['val'][metric],origin='frozen_reference',experiment=f.parent.parent.name,path=str(f),metrics_sha256=sha(f/'metrics.json')))
for key,m in s['results'].items():
    f=root/'results'/key;actual=read(f/'metrics.json');assert all(actual[k]==v for k,v in m.items())
    assert not actual['test_used'] and actual['seed']==42
    assert m['identity']['protocol_sha256']==sha(root/'protocol.json') and sha(f/'final.pt')==m['final_sha256']
    task=m['identity']['task'];metric='mae' if task=='h01_global' else 'macro_f1'
    rows.append(dict(task=task,arm=m['identity']['arm'],metric=metric,value=m['val'][metric],origin='trained',experiment=root.name,path=str(f),metrics_sha256=sha(f/'metrics.json')))
assert sha(root/'analysis.json')==s['analysis_sha256']
audits[root.name]=dict(sources=n,reference_runs=len(p['references']),new_runs=len(s['results']),protocol_sha256=sha(root/'protocol.json'))

root=BASE/'CPSF_NREDD_CONFIG_ONLY_20261009';p=read(root/'protocol.json');n=sources(root,p)
for name,h in p['references'].items():assert sha(Path(name))==h
exploration=[]
for f in sorted((root/'results').glob('*/metrics.json')):
 m=read(f);i=m['identity'];task=i['task'];arm=i['arm'];cfg=i['config'];base=Path(p['baseline_dirs'][task][arm]);bm=read(base/'metrics.json')
 assert i['protocol_sha256']==sha(root/'protocol.json') and i['seed']==42 and not m['test_used'] and not m['smoke']
 assert sha(f.parent/'final.pt')==m['final_sha256'] and m['epochs']==p['tasks'][task]['epochs'] and m['optimizer_steps']==bm['optimizer_steps']
 assert i['initial_state_sha256']==bm['identity']['initial_state_sha256'] and i['data_manifest_sha256']==bm['identity']['data_manifest_sha256']
 assert [r['sample_order_sha256'] for r in read(f.parent/'history.json')]==[r['sample_order_sha256'] for r in read(base/'history.json')]
 metric='f1' if task=='ted' else 'missing_edge_macro_ap'
 r=dict(task=task,arm=arm,metric=metric,value=m['val'][metric],config=cfg,origin='trained',experiment=root.name,path=str(f.parent),metrics_sha256=sha(f))
 exploration.append(r)
 if task=='ted' and cfg=='lr_half':rows.append(r)
for arm,d in p['baseline_dirs']['ted'].items():
 f=Path(d)/'metrics.json';m=read(f);exploration.append(dict(task='ted',arm=arm,metric='f1',value=m['val']['f1'],config='baseline',origin='frozen_reference',experiment=Path(d).parent.parent.name,path=d,metrics_sha256=sha(f)))
audits[root.name]=dict(sources=n,references=len(p['references']),protocol_sha256=sha(root/'protocol.json'),pipeline_status=read(root/'pipeline_status.json'))
root=BASE/'CPSF_TLP_DIRECTIONAL_STAGE6_20261009';p=read(root/'protocol.json');n=sources(root,p);a=read(root/'analysis.json');assert a['replay_gate']['passed']
for arm in ['full','mean_only','no_fusion','no_scale','point','skeleton']:
 key='tlp_directional_xyz_'+arm if arm!='skeleton' else 'tlp_voxel_skeleton';f=root/'results'/key;m=read(f/'metrics.json')
 assert sha(f/'metrics.json')==a['provenance'][key]['metrics_sha256'] and sha(f/'final.pt')==m['final_sha256']
 assert m['identity']['seed']==42 and m['epochs']==80 and m['optimizer_steps']==12480 and not m['test_used'] and not m['smoke']
 assert m['identity']['protocol_sha256']==sha(root/'protocol.json')
 independent=read(f/'independent_missing_edge_audit.json');assert independent['passed'] and independent['macro_ap']==m['val']['missing_edge_macro_ap']
 rows.append(dict(task='tlp',arm=arm,metric='missing_edge_macro_ap',value=m['val']['missing_edge_macro_ap'],origin='directional_xyz' if arm!='skeleton' else 'descriptor_independent_exact_replay',experiment=root.name,path=str(f),metrics_sha256=sha(f/'metrics.json')))
audits[root.name]=dict(sources=n,protocol_sha256=sha(root/'protocol.json'),replay_gate=a['replay_gate'])
assert len(rows)==47 and len({(r['task'],r['arm']) for r in rows})==47
value=dict(unix=time.time(),passed=True,rows=rows,audits=audits,configuration_exploration=exploration,directional_analysis=a,test_used=False,scope='Current per-task defaults; no old-config ablations inserted into new TED/TLP defaults')
(out/'results_audit.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(passed=True,current_records=len(rows),exploration_records=len(exploration),source_files=sum(v['sources'] for v in audits.values()),output=str(out/'results_audit.json'))))
