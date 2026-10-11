import argparse,time,json
from pathlib import Path
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import setup,load,save,sha,SOURCE
from tools.cpsf_v21_segment_units.model import Model
from tools.cpsf_v21_segment_units.readout import fit
from .adapter import geometries

@torch.no_grad()
def run(a):
    setup();assert not a.out.exists();a.out.mkdir(parents=True)
    ck=load(a.checkpoint);arm=ck['protocol']['arm']
    assert ck['protocol']['task']=='h01' and ck['protocol']['seed']==42 and ck['epoch']==25
    src=Path(__file__).resolve().parents[1]/'cpsf_v21_segment_units'
    assert all(sha(src/name)==digest for name,digest in ck['protocol']['source_sha256'].items())
    model=Model('h01',arm,**ck['protocol']['model_config']).cuda();model.load_state_dict(ck['model']);model.eval()
    rows=sorted(json.loads((SOURCE/'manifest.json').read_text())['cells'],key=lambda r:r['cell'])
    if a.profile:rows=[max((r for r in rows if r['split']=='train'),key=lambda r:r['skeleton_nodes'])]
    vectors=[];targets=[];records=[];start=time.time()
    for row in rows:
        t=time.time();torch.cuda.reset_peak_memory_stats()
        path=SOURCE/'inputs'/(row['cell']+'.npz')
        with np.load(path) as x:
            gs=geometries(x,arm)
            if arm!='point':
                assert gs[0]['n']==len(x['skeleton_xyz_um'])
                assert len(gs[0]['e'])==len(np.unique(np.sort(x['edge_index'],axis=1),axis=0))
        bs=[model.encoder.stems([g],arm)[0] for g in gs]
        bs=model.encoder.propagate(bs,arm,offsets=[g['origin'] for g in gs])
        v=model.global_feature(bs).cpu().numpy();assert v.shape==(128,) and np.isfinite(v).all()
        with np.load(SOURCE/'targets'/(row['cell']+'.npz')) as y:target=float(y['excitatory_input_fraction'])
        vectors.append(v);targets.append(target)
        rec=dict(cell=row['cell'],split=row['split'],source_sha256=sha(path),tiles=len(gs)-1,
                 surface_points=sum(len(g['p']) for g in gs),nodes=sum(g['n'] for g in gs),
                 thumbnail_nodes=gs[0]['n'],thumbnail_edges=len(gs[0]['e']),
                 seconds=time.time()-t,peak_gpu_bytes=torch.cuda.max_memory_allocated())
        records.append(rec);save(a.out/'status.json',dict(state='extracting',cells=len(records),total=len(rows),seconds=time.time()-start,last=rec))
        print(json.dumps(rec),flush=True);del gs,bs;torch.cuda.empty_cache()
    np.savez_compressed(a.out/'features.npz',x=np.array(vectors),y=np.array(targets),cell=np.array([r['cell'] for r in rows]),split=np.array([r['split'] for r in rows]))
    save(a.out/'manifest.json',dict(arm=arm,records=records,checkpoint_sha256=sha(a.checkpoint),encoder_frozen=True,profile=a.profile,
         scope='V21 frozen transfer with unthinned original skeleton support; unchanged point draws, weights and readout'))
    if not a.profile:fit(a.out)
    else:save(a.out/'status.json',dict(state='profile_complete',record=records[0]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--profile',action='store_true');a=p.parse_args()
    try:run(a)
    except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise
