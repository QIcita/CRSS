"""Verify original R28 source, checkpoints, controls and result identities, no ML deps."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1];r=root/'code/r28'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
meta=read(r/'server_export.json')
for name,info in meta['files'].items():assert sha(r/name)==info['sha256'],name
for prefix in ['reference','extension']:
 p=read(r/prefix/'protocol.json')
 for name,h in p['source_sha256'].items():
  target=prefix+'/'+name
  if target in meta['shared_source_resolution']:
   m=meta['shared_source_resolution'][target];assert m['sha256']==h and sha(r/m['resolved'])==h
  elif (r/target).exists():assert sha(r/target)==h,target
  else:assert meta['publication_omissions'][target]['sha256']==h,target
 contracts=read(r/prefix/'contracts.json');assert contracts['passed'] and contracts['protocol_sha256']==sha(r/prefix/'protocol.json')
results=read(r/'RESULTS.json');assert len(results['records'])==5 and results['seed']==42 and not results['test_used'] and not results['pretrained']
assert {q['task'] for q in results['records']}=={'h01_local','h01_global','flywire_coarse','ted','tlp'}
for q in results['records']:
 assert q['architecture_version']=='CPSF_RAW_POINT_ATTENTION_20261019_R28_A1'
 if q['status']!='completed':assert q['value'] is None and q['metrics_path'] is None;continue
 d=r/q['metrics_path'];m=read(d);i=m['identity'];assert i['version']==q['execution_version'] and i['kind']==q['run_kind'] and i['task']==q['task'] and i['seed']==42
 assert not i['pretrained'] and not m['test_used'] and m['val'][q['primary_metric']]==q['value']
 assert q['final_sha256']==m['final_sha256']==sha(d.parent/'final.pt')
 prefix='reference' if q['task'].startswith('h01') else 'extension'
 assert i['protocol_sha256']==sha(r/prefix/'protocol.json')
 cfg=read(r/prefix/'protocol.json')['tasks'][q['task']];assert m['epochs']==cfg['epochs'] and m['steps']==cfg['updates']
comparison=read(r/'BASELINE_COMPARISON.json');assert len(comparison['baselines'])==27 and comparison['r28']==results['records']
assert all(q['model'] not in ['r9','pointmamba','exphormer'] for q in comparison['baselines'])
publications=read(r/'evidence/baseline_publications.json')
assert len(publications)==9
for q in comparison['baselines']:
 p=publications[q['model']]
 assert all(q[k]==p[k] for k in ['display_name','publication_venue','publication_year','source_urls'])
assert publications['sgformer']['publication_venue']=='NeurIPS'
assert publications['octformer']['publication_venue']=='ACM TOG / SIGGRAPH'
tlp_audit=read(r/'evidence/tlp_dev_audit_20261020.json')
assert tlp_audit['passed'] and not tlp_audit['test_predictions_accessed'] and not tlp_audit['test_metrics_used']
assert not tlp_audit['training_launched'] and tlp_audit['r28_target_observed_parity_graphs']==62
assert len(tlp_audit['official_source_parity'])==6
assert tlp_audit['counts']['pairs']==1146183 and tlp_audit['counts']['positives']==283
for name in ['tlpnet','sgformer','polynormer']:
 q=next(q for q in comparison['baselines'] if q['model']==name and q['task']=='tlp')
 assert q['value']==tlp_audit['methods'][name]['macro_ap']
dual=read(r/'evidence/tlp_dual_metrics_20261020.json')
assert dual['passed'] and not dual['observed_scores_overwritten'] and not dual['test_metrics_used']
for name,rows in [('r28',comparison['r28']),('r9',comparison['historical_r9'])]+[(n,comparison['baselines']) for n in ['sgformer','polynormer','tlpnet']]+[('tlpnet_native_objective',comparison['supplemental_tlp_controls'])]:
 q=next(q for q in rows if q['task']=='tlp' and (name in ['r9','r28'] or q['model']==name))
 for k in ['missing_edge_macro_ap','full_adjacency_macro_ap']:assert q['reported_metrics'][k]==dual['methods'][name][k]
 assert q['final_sha256']==dual['methods'][name]['checkpoint_sha256']
assert all(v['entire_manifest_records_equal_original_R9'] for v in read(r/'live_export.json')['audit']['input_lineage'].values())
audit=read(r/'evidence/baseline_live_audit.json');assert audit['passed'] and len(audit['records'])==27
for p in r.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
pub=read(r/'release_manifest.json')
for name,h in pub.items():assert sha(root/name)==h,name
print(f"R28 verified: {len(meta['files'])} original files, 3 completed checkpoints, 5 task identities, 27 baseline records; source/weights/results/published documents consistent.")
