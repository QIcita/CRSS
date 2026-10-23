"""Verify release byte identity and the saved runtime/metric validation evidence."""
import ast,hashlib,json
from pathlib import Path
repo=Path(__file__).resolve().parents[1];root=repo/'code/r34_dense'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=read(root/'protocol.json');digest=sha(root/'protocol.json')
assert digest=='e5ac48d2060320c307fea451c1e10236bb7394efdfd613349d54ccc13a0e74cd'
assert p['seed']==42 and not p['test_used'] and not p['pretrained']
assert p['repair_variant']=='support_preserved'
mapping=read(root/'source_map.json')['files'];export=read(root/'server_export.json')
for name,row in mapping.items():
 f=repo/row['repository_path'];assert sha(f)==row['sha256'],name
 if f.suffix=='.py':ast.parse(f.read_text(encoding='utf-8-sig'),filename=name)
for name,row in export['sources'].items():
 assert mapping[name]['sha256']==row['sha256']==p['files'][row['source_path']],name
for name,row in export['artifacts'].items():assert sha(root/name)==row['sha256'],name
accept=read(root/'acceptance/r34_h01_global.json')
assert accept['passed'] and accept['protocol_sha256']==digest and accept['fine_support_exact']
folder=root/'results/r34_h01_global';m=read(folder/'metrics.json');i=m['identity'];cfg=p['tasks']['h01_global']
assert i==read(folder/'identity.json') and i['version']==p['version'] and i['protocol_sha256']==digest
assert m['steps']==cfg['updates'] and m['epochs']==cfg['epochs'] and not m['test_used']
assert sha(folder/'final.pt')==m['final_sha256']
history=read(folder/'history.json');orders=read(root/'orders/h01_global.json')
assert len(history)==cfg['epochs'] and [h['sample_order_sha256'] for h in history]==[o['sha256'] for o in orders]
validation=read(root/'evidence/publication_validation.json')
assert validation['passed'] and validation['strict_checkpoint_load'] and validation['prediction_metric_recomputed']
assert validation['protocol_sha256']==digest and validation['checkpoint_sha256']==m['final_sha256']
assert validation['metrics']==m['val'] and not validation['new_inference_performed']
for name,row in validation['loaded_sources'].items():assert mapping[row['path']]['sha256']==row['sha256'],name
for name,digest in read(root/'release_manifest.json').items():assert sha(repo/name)==digest,name
print('Verified R34 dense: 513 sources, one final H01 weight, exact export identity, strict runtime load and saved prediction metrics.')
