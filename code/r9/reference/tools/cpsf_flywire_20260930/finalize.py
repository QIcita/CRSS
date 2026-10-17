"""Freeze only after every candidate was attempted, geometry QA and duplicate audit."""
import argparse,json
from pathlib import Path
from collections import Counter,defaultdict
from .configure import assigned,labels,distribution,save,write,sha

def run(root):
 rows=[json.loads(line) for line in (root/'metadata/manifest_provisional.jsonl').read_text().splitlines()]
 qa={}
 for r in rows:
  p=root/'qa'/(r['root_id']+'.json');assert p.is_file(),f'Unattempted candidate: {r["root_id"]}';qa[r['root_id']]=json.loads(p.read_text())
 kept=[];excluded=[];duplicates=defaultdict(list)
 for r in rows:
  q=qa[r['root_id']]
  if q['status']!='ok':excluded.append(dict(root_id=r['root_id'],reason=q['status'],flags=q.get('flags'),error=q.get('error')));continue
  assert sha(root/'samples'/(r['root_id']+'.npz'))==q['sample_sha256']
  duplicates[q['absolute_mesh_vertices_sha256']].append(r)
 for group in duplicates.values():
  if len({r['cell_type'] for r in group})>1:
   excluded.extend(dict(root_id=r['root_id'],reason='identical_absolute_geometry_conflicting_labels') for r in group);continue
  group=sorted(group,key=lambda r:r['root_id']);kept.append(group[0]);excluded.extend(dict(root_id=r['root_id'],reason='identical_absolute_geometry_duplicate',canonical_root_id=group[0]['root_id']) for r in group[1:])
 count=Counter(r['cell_type'] for r in kept);excluded.extend(dict(root_id=r['root_id'],reason='type_support_below5_after_QA') for r in kept if count[r['cell_type']]<5);kept=[r for r in kept if count[r['cell_type']]>=5];assert kept
 if len(kept)<.95*len(rows):
  save(root/'freeze_review.json',dict(ready=False,retained=len(kept),candidates=len(rows),reason='More than5pct excluded: investigate systematic availability/geometry failures before freezing',excluded=excluded));raise RuntimeError('Exclusions exceed5pct; no automatic formal dataset freeze')
 kept=assigned(kept);mapping=labels(kept)
 for r in kept:r.update(fine_target=mapping['fine'].index(r['cell_type']),coarse_target=mapping['coarse'].index(r['super_class']),sample_sha256=qa[r['root_id']]['sample_sha256'])
 out=root/'dataset';save(out/'labels.json',mapping);write(out/'frozen_manifest.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in kept));save(out/'excluded.json',excluded);save(out/'distribution.json',distribution(kept))
 save(out/'READY.json',dict(ready=True,seed=42,cells=len(kept),fine_classes=len(mapping['fine']),coarse_classes=len(mapping['coarse']),manifest_sha256=sha(out/'frozen_manifest.jsonl'),labels_sha256=sha(out/'labels.json'),all_candidates_attempted=True,duplicate_absolute_geometry_checked=True,known_bilateral_homolog_groups_available=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root)
