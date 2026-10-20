"""Immutable cohort, label and provisional split construction; no geometry fiction."""
import argparse,csv,json,hashlib,math
from pathlib import Path
from collections import Counter,defaultdict

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,text):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.read_text()==text,f'Immutable output differs: {path}'
 else:path.write_text(text)
def save(path,obj):write(path,json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def assigned(rows):
 groups=defaultdict(list)
 for r in rows:groups[r['cell_type']].append(r)
 out=[]
 for typ,rs in sorted(groups.items()):
  rs=sorted(rs,key=lambda r:hashlib.sha256(('42|'+r['root_id']).encode()).hexdigest());n=len(rs);nv=max(1,n//5);nt=max(1,n//5);ntr=n-nv-nt
  assert ntr>=3
  for i,row in enumerate(rs):out.append(dict(row,split='train' if i<ntr else 'val' if i<ntr+nv else 'test'))
 return sorted(out,key=lambda r:r['root_id'])
def labels(rows):
 fine=sorted({r['cell_type'] for r in rows});coarse=sorted({r['super_class'] for r in rows});parents={}
 for f in fine:
  p={r['super_class'] for r in rows if r['cell_type']==f};assert len(p)==1;parents[f]=p.pop()
 return dict(fine=fine,coarse=coarse,fine_to_coarse=[coarse.index(parents[f]) for f in fine],fine_parent_names=parents)
def distribution(rows):
 out={};by=defaultdict(list)
 for r in rows:by[r['cell_type']].append(r)
 for c,rs in by.items():out[c]=dict(total=len(rs),super_class=rs[0]['super_class'],splits=dict(Counter(r['split'] for r in rs)))
 return dict(cells=len(rows),cell_types=len(by),splits=dict(Counter(r['split'] for r in rows)),super_class=dict(Counter(r['super_class'] for r in rows)),per_type=out)
def run(source,annotations,out):
 with source.open() as f:rows=list(csv.DictReader(f))
 with annotations.open() as f:ann={r['root_id']:r for r in csv.DictReader(f,delimiter='\t')}
 assert len(rows)==8122 and len({r['root_id'] for r in rows})==8122
 assert len({r['vfb_id'] for r in rows})==8122
 for r in rows:
  a=ann[r['root_id']];assert not a['status'] and all(r[k]==a[k] for k in ('cell_type','super_class','cell_class'))
  assert r['cell_type'] and r['super_class']
 assert min(Counter(r['cell_type'] for r in rows).values())>=5
 mapping=labels(rows);assert len(mapping['fine'])==148 and len(mapping['coarse'])==6
 rows=assigned(rows)
 for r in rows:r.update(fine_target=mapping['fine'].index(r['cell_type']),coarse_target=mapping['coarse'].index(r['super_class']))
 save(out/'labels_provisional.json',mapping)
 write(out/'manifest_provisional.jsonl',''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows))
 stats=distribution(rows);save(out/'cohort_audit.json',dict(**stats,geometry_ready=False,split_final=False,root_and_vfb_unique=True,label_hierarchy_functional=True,seed=42,source_manifest_sha256=sha(source),annotation_sha256=sha(annotations),annotation_commit='ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0',materialization=783,annotation_release='v2.1.0',group_limit='Root-neuron split; no verified bilateral homolog-pair grouping available. This is within-brain known-type instance generalization.'))
 common=dict(seed=42,dataset='FlyWire FAFB m783',candidate_cells=8122,formal_manifest='dataset/frozen_manifest.jsonl',formal_labels='dataset/labels.json',readiness_gate='dataset/READY.json with verified geometry hashes; provisional metadata alone cannot train',encoder='frozen copy of CPSF V19 Encoder class, independently trained per task',encoder_arms=['measure','count','point','skeleton','geometry'],input=dict(unit='um',voxel=.5,association_radius=2.,query_ball_radius=4.,global_points=4096,global_nodes=2048,tile_points=2048,tile_nodes=512,max_tiles=32,target_tile_surface_area_um2=256.,partition='mesh-only longest-axis area-balanced recursive partition; same supports for all arms',sampling='triangle-area proportional; point weight=view area / point count',metadata_not_in_encoder=['cell_type','super_class','cell_class','side','top_nt','soma location','connectivity','hemilineage','root_id']),training=dict(epochs=50,batch_cells=4,optimizer='AdamW',lr=.0003,weight_decay=.0001,scheduler='cosine',min_lr=.00003,loss='ordinary cross entropy; natural class frequencies, no class-balanced resampling',gradient_clip=5,checkpoint_selection='fixed final epoch50; no test or validation model selection',tile_execution='sequential checkpointed tiles; full registered tile budget, no silent subsampling'),evaluation=dict(classification=['top1_accuracy','macro_f1','balanced_accuracy','per_class_precision_recall_f1_support'],fine_additional=['top5_accuracy','head_mid_tail_metrics','coarse_consistency_diagnostic'],long_tail_bins={'tail':'cohort support 5-19','middle':'20-99','head':'>=100'},scope='closed-set within-brain; no cross-brain generalization claim'))
 for task,key in [('coarse','coarse_target'),('fine','fine_target')]:
  save(out/'configs'/(task+'.json'),dict(common,task='whole_cell_'+task+'_classification',target=key,classes_provisional=6 if task=='coarse' else 148,head='128-dimensional cell representation -> Linear64 GELU -> LinearC; separate task checkpoints, no joint hierarchy loss'))
 save(out/'configs/retrieval.json',dict(seed=42,task='within_FlyWire_same_type_retrieval',encoder_checkpoint='fine classifier fixed final checkpoint; pre-logit 128-dimensional representation',training_loss='none added; evaluation of classification-trained embeddings',gallery_split='train',validation_query_split='val',test_query_split='test',strict_positive='same cell_type, different root_id',self_match='excluded by root identity',gallery_normalization='per-vector L2 only',similarity='cosine',metrics=['query_macro_mAP','cell_type_macro_mAP','Recall@1','Recall@5','Recall@10','MRR'],minimum_gallery_positives=3,zero_positive_policy='fail; never silently exclude',tie_break='gallery root_id ascending',coarse_relevance='secondary hierarchical metric only; not a strict cell_type positive',no_extra_dataset=True))
 save(out/'configs/build.json',dict(seed=42,mesh_url='precomputed://https://storage.googleapis.com/flywire_v141_m783',mesh_lod=0,skeleton_base='https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/flywire_skeletons_783',source_coordinate_unit='nm',stored_coordinate_unit='um',workers=1,disk_min_free_gib=50,disk_budget_gib=60,retries=3,**common['input']))
 print(json.dumps({k:stats[k] for k in ('cells','cell_types','splits','super_class')},ensure_ascii=False,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--annotations',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source,a.annotations,a.out)
