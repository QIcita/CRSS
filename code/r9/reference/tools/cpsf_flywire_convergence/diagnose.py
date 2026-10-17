"""Validation-only coverage and paired-error diagnosis, before the new CE runs."""
import json
from collections import Counter
import numpy as np
import torch
from .common import ROOT,COHORT,save,sha,digest_object
from tools.cpsf_flywire_v23.data import load
from tools.cpsf_flywire_v23.model import Model
from tools.cpsf_v21_segment_units.io import setup
from tools.cpsf_flywire_20260930.evaluation import classify

def subset_metrics(y,p):
    if not len(y):return dict(n=0)
    cm=np.bincount(y*6+p,minlength=36).reshape(6,6);n=cm.sum(1);tp=cm.diagonal()
    f=2*tp/np.maximum(n+cm.sum(0),1);present=n>0
    return dict(n=len(y),accuracy=float((y==p).mean()),support=n.tolist(),
        per_class_f1=[float(f[i]) if present[i] else None for i in range(6)],
        descriptive_macro_f1_present_classes=float(f[present].mean()),
        note='Subset diagnostic; absent classes omitted here only; not the formal six-class primary metric')

def main():
    state=json.loads((COHORT/'pipeline_status.json').read_text());assert state['state']=='complete'
    p=json.loads((ROOT/'protocol.json').read_text())
    assert sha(COHORT/'protocol.json')==p['inherited_protocol_sha256']
    manifest=json.loads((COHORT/'cache/manifest.json').read_text());assert sha(COHORT/'cache/manifest.json')==p['cache_manifest_sha256']
    rows=[r for r in manifest['records'] if r['split'] in ('train','val')]
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    assert len(train)==6039 and len(val)==1922
    out=ROOT/'diagnostics';out.mkdir(exist_ok=True)
    # Recoverable per-cell profiles; existing entries must carry the exact cache hash.
    profiles={};profile_path=out/'geometry_profiles.jsonl'
    if profile_path.exists():
        for line in profile_path.read_text().splitlines():
            d=json.loads(line);profiles[d['root_id']]=d
    torch.set_num_threads(1)
    with profile_path.open('a') as fp:
        for i,r in enumerate(rows):
            if r['root_id'] in profiles:
                assert profiles[r['root_id']]['cache_sha256']==r['sha256'];continue
            assert sha(r['path'])==r['sha256'];g=load(r['path'])
            assert len(g['s'])>0 and torch.isfinite(g['s']).all()
            squares=torch.zeros(g['patches']).index_add_(0,g['pj'],g['pw'].square())
            effective=1/squares.clamp_min(1e-12)
            weights=g['patch_mass'];weights=weights/weights.sum().clamp_min(1e-12)
            d=dict(root_id=r['root_id'],cache_sha256=r['sha256'],split=r['split'],legacy=r['legacy'],
                cell_type=r['cell_type'],target=r['coarse_target'],
                skeleton_bbox_diagonal_um=float((g['s'].max(0).values-g['s'].min(0).values).norm()),
                region_length_sum_um=float(g['length'].sum()),regions=g['nregions'],
                nodes=g['raw_nodes'],points=g['surface_points'],tiles=g['tile_count'],
                effective_regions_per_patch=float((effective*weights).sum()),
                hierarchy_adapter_would_change=bool(g['ncoarse']<min(4,g['nregions'])),
                coarse_groups=g['ncoarse'])
            profiles[r['root_id']]=d;fp.write(json.dumps(d)+'\n')
            if i%50==0:
                fp.flush();save(out/'progress.json',dict(stage='train_val_geometry',done=len(profiles),total=len(rows),test_used=False))
    counts=Counter(r['cell_type'] for r in train);oldcounts=Counter(r['cell_type'] for r in train if r['legacy'])
    quartiles={k:np.quantile([profiles[r['root_id']][k] for r in train],[.25,.5,.75]).tolist()
        for k in ('skeleton_bbox_diagonal_um','effective_regions_per_patch')}
    npzs={};predictions={};metrics={};ids=np.array([r['root_id'] for r in val]);y=np.array([r['coarse_target'] for r in val])
    for arm in ('full','mean_only','no_fusion'):
        run=COHORT/'results'/('coarse_'+arm);m=json.loads((run/'metrics.json').read_text())
        assert m['protocol_sha256']==sha(COHORT/'protocol.json') and m['final_sha256']==sha(run/'final.pt')
        z=np.load(run/'val_features.npz');assert np.array_equal(z['root_id'],ids) and np.array_equal(z['target'],y)
        assert abs(classify(y,z['logits'],6)['macro_f1']-m['val']['macro_f1'])<1e-10
        npzs[arm]=z;predictions[arm]=z['logits'].argmax(1);metrics[arm]=m['val']
    # Disable only the residual in the same frozen full checkpoint, retaining all encoded inputs.
    setup();model=Model(6,'full').cuda();ref=COHORT/'results/coarse_full'
    ckpt=torch.load(ref/'final.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model']);model.eval()
    original=[];disabled=[]
    with torch.no_grad():
        for i,r in enumerate(val):
            g={k:v.cuda() if isinstance(v,torch.Tensor) else v for k,v in load(r['path']).items()}
            model.arm='full';original.append(model(g).cpu().numpy())
            model.arm='mean_only';disabled.append(model(g).cpu().numpy())
            if i%50==0:save(out/'progress.json',dict(stage='frozen_full_residual_intervention',done=i,total=len(val),test_used=False))
    original=np.array(original);disabled=np.array(disabled)
    np.testing.assert_allclose(original,npzs['full']['logits'],rtol=2e-5,atol=2e-5)
    assert np.array_equal(original.argmax(1),predictions['full'])
    predictions['full_checkpoint_residual_off']=disabled.argmax(1)
    np.savez_compressed(out/'residual_off_predictions.npz',root_id=ids,target=y,logits=disabled)
    records=[]
    for i,r in enumerate(val):
        d=dict(profiles[r['root_id']],type_train_count=counts[r['cell_type']],old_type_train_count=oldcounts[r['cell_type']],
            prediction={arm:int(pr[i]) for arm,pr in predictions.items()})
        d['type_support_bin']='unseen' if d['type_train_count']==0 else '1-4' if d['type_train_count']<5 else '5-19' if d['type_train_count']<20 else '20+'
        for k,bounds in quartiles.items():d[k+'_quartile']=int(np.searchsorted(bounds,d[k],side='right'))+1
        records.append(d)
    groups={'all':np.ones(len(y),bool),'legacy':np.array([r['legacy'] for r in val]),'added':np.array([not r['legacy'] for r in val])}
    for field in ('type_support_bin','skeleton_bbox_diagonal_um_quartile','effective_regions_per_patch_quartile','target'):
        for v in sorted({r[field] for r in records},key=str):groups[field+'='+str(v)]=np.array([r[field]==v for r in records])
    # Cross strata expose within-class support shifts without treating different classes as comparable.
    for c in range(6):
        for b in ('unseen','1-4','5-19','20+'):
            groups[f'class={c}/type_support={b}']=np.array([r['target']==c and r['type_support_bin']==b for r in records])
    summary={}
    for key,mask in groups.items():
        summary[key]={a:subset_metrics(y[mask],pr[mask]) for a,pr in predictions.items()}
        summary[key]['paired_full_vs_mean']=dict(full_only_correct=int(((predictions['full']==y)&(predictions['mean_only']!=y)&mask).sum()),
            mean_only_correct=int(((predictions['full']!=y)&(predictions['mean_only']==y)&mask).sum()))
    save(out/'validation_records.json',records)
    save(out/'summary.json',dict(passed=True,test_used=False,source_cohort=str(COHORT),
        protocol_sha256=sha(COHORT/'protocol.json'),cache_manifest_sha256=p['cache_manifest_sha256'],
        profile_sha256=sha(profile_path),split_sha256=p['split_sha256'],quartile_threshold_source='6039 training cells only',
        quartiles=quartiles,primary=metrics,residual_off=classify(y,disabled,6),
        residual_off_role='Same full checkpoint inference intervention, not a trained mean_only control',
        hierarchy_adapter_would_change={s:sum(profiles[r['root_id']]['hierarchy_adapter_would_change'] for r in rows if r['split']==s) for s in ('train','val')},
        strata=summary,read_test_geometry=False,read_test_predictions=False))
    save(out/'progress.json',dict(stage='complete',test_used=False))

if __name__=='__main__':main()
