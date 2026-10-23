"""Freeze independent input/training protocol and original implementation lineage."""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
from .common import ROOT,BASE,R28,SOURCES,TASKS,read,save,sha,source_rows
from .sampling import RULE

def main():
    assert not (ROOT/'protocol.json').exists(),'Never overwrite a frozen campaign'
    files={};sources={}
    for family,parent in SOURCES.items():
        dest=ROOT/'backends'/family;dest.mkdir(parents=True,exist_ok=True)
        if not (dest/'tools').exists():shutil.copytree(parent/'tools',dest/'tools',symlinks=False,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        for name in ('runtime','architecture.json'):
            src=parent/name;dst=dest/name
            if src.is_dir() and not dst.exists():shutil.copytree(src,dst,symlinks=True)
            elif src.is_file() and not dst.exists():shutil.copy2(src,dst)
        if not (dest/'vendor').exists():(dest/'vendor').symlink_to(parent/'vendor',target_is_directory=True)
        shutil.copy2(parent/'protocol.json',dest/'protocol.json')
        p=read(parent/'protocol.json');sources[family]=dict(root=str(parent),protocol_sha256=sha(parent/'protocol.json'),upstream=p.get('upstream'),pretrained=False)
        for f in (dest/'tools').rglob('*.py'):files[str(f)]=sha(f)
        if (dest/'runtime').exists():
            for f in (dest/'runtime').rglob('*.py'):files[str(f)]=sha(f)
        if (dest/'architecture.json').exists():files[str(dest/'architecture.json')]=sha(dest/'architecture.json')
        # Pin native files recorded by original protocols, without copying weights.
        records=p.get('source_sha256',{})
        for name,h in records.items():
            f=parent/name
            if '/vendor/' in str(f) and f.is_file():assert sha(f)==h;files[str(f)]=h
        for path,h in p.get('files',{}).items():
            f=Path(path)
            if f.is_file() and (f.suffix=='.py' or '/vendor/' in str(f)):assert sha(f)==h;files[str(f)]=h
        for path,h in p.get('parent_sha256',{}).items():
            f=Path(path)
            if f.is_file() and f.suffix=='.so':assert sha(f)==h;files[str(f)]=h
    taskcfg={**read(BASE/'CPSF_RAW_POINT_ATTENTION_20261019_R28_A1/protocol.json')['tasks'],**read(R28/'protocol.json')['tasks']}
    for task,cfg in taskcfg.items():cfg.setdefault('round_cells',cfg['train'])
    for task in TASKS:
        rr=source_rows(task);tr=[r for r in rr if r['split']=='train'];rng=np.random.RandomState(42);orders=[]
        assert not {r['cell'] for r in tr}&{r['cell'] for r in rr if r['split']=='val'}
        for epoch in range(1,taskcfg[task]['epochs']+1):
            if task=='ted':
                blocks=sorted({r['key'].rsplit(':',1)[0] for r in tr});by={b:[] for b in blocks}
                for r in tr:by[r['key'].rsplit(':',1)[0]].append(r['key'])
                keys=[k for i in np.random.RandomState(42+epoch).permutation(len(blocks)) for k in by[blocks[i]]]
            else:
                ix=(np.random.RandomState(42+epoch) if task=='tlp' else rng).permutation(len(tr))[:taskcfg[task]['round_cells']]
                keys=[tr[i]['key'] for i in ix]
            orders.append(dict(epoch=epoch,sha256=hashlib.sha256('\n'.join(keys).encode()).hexdigest(),samples=len(keys)))
        path=ROOT/'orders'/f'{task}.json';save(path,orders);files[str(path)]=sha(path)
        if task.startswith('h01'):hist=BASE/'CPSF_RAW_POINT_ATTENTION_20261019_R28_A1/results'/('r9_lpa_attention_'+task)/'history.json'
        elif task in ('ted','tlp'):hist=Path(read(BASE/'CPSF_NREDD_BASELINES_20261014/protocol.json')['reference_history'][task])
        else:hist=Path(read(BASE/'CPSF_EXTERNAL_BASELINES_PHASE2_20261010/protocol.json')['references'][task+'/full']['path'])/'history.json'
        assert [o['sha256'] for o in orders]==[h['sample_order_sha256'] for h in read(hist)]
        files[str(hist)]=sha(hist);files[str(R28/'data'/task/'manifest.json')]=sha(R28/'data'/task/'manifest.json')
    jobs=[]
    for task in TASKS:
        names=['r28','sgformer','polynormer']
        names+=['ptv3','litept','octformer','utonia','graphformer'] if task in TASKS[:3] else ['tednet' if task=='ted' else 'tlpnet']
        for name in names:jobs.append(dict(model=name,task=task,id=name+'_'+task))
    nredd=BASE/'NREDD_proj/NREDD'
    for folder in ('tools/v2_surface_v21','tools/tlp_surface_mc','tools/tlp_bridge_surface',
                   'artifacts/CPSF_GEOMETRY_V2_TLP_20260917/data','artifacts/CPSF_GEOMETRY_V2E_TLP_20260917/data'):
        for f in (nredd/folder).glob('*.py'):files[str(f)]=sha(f)
    f=nredd/'tools/build_multishell_surface_cache_v22c.py';files[str(f)]=sha(f)
    for folder in ('cpsf_geometry_rebench','cpsf_geometry_fps'):
        for f in (ROOT/'tools'/folder).glob('*.py'):files[str(f)]=sha(f)
    for f in (ROOT/'deps').rglob('*'):
        if f.is_file() and f.suffix in ('.py','.so'):files[str(f)]=sha(f)
    save(ROOT/'protocol.json',dict(version=ROOT.name,seed=42,pretrained=False,tasks=taskcfg,jobs=jobs,files=files,implementations=sources,
        sampling=RULE,old_counts_and_old_sampled_points_used=False,geometry_support='current full H01 ROI/whole-cell points; FlyWire mesh vertices; NREDD extracted isosurface candidate vertices before old FPS',
        mass='H01 empirical records; FlyWire lumped triangle area; NREDD unique isosurface vertices; nearest representative conservation',
        storage='one canonical geometry cache; point baselines reconstruct the same coordinates and mass from frozen pf/patch centres, no duplicate point-cloud dataset',
        input_units=dict(h01='um',flywire='um',ted='historical SWC/2 units, not verified um',tlp='historical [1,1,2] native spacing, not verified um'),
        task_changes=dict(tlp='full adjacency weighted upper-triangle BCE; report full and missing AP',ted='all methods use argmax logits; previous native sigmoid-round decision not mixed'),
        normalization='native model normalization; current-input BN for point/graph baseline evaluation and training',
        selection='fixed final epoch; development evaluation only; no score-driven rule adjustment',test_used=False,
        excluded_baselines=['PointMamba','Exphormer'],architecture='unchanged R28 CMM+RCM; official baseline backbones unchanged',
        new_geometry_per_task_must_pass_maximum_shape_acceptance=True,old_results_are_historical=True))
    print(json.dumps(dict(frozen=True,jobs=len(jobs),protocol_sha256=sha(ROOT/'protocol.json')),indent=2))

if __name__=='__main__':main()
