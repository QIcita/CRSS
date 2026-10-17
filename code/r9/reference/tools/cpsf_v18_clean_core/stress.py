"""Real-data stress gates; full point retention, not accuracy evaluation."""
import argparse,time,resource
from .common import *
from .geometry import prepare_geometry,prepare_surface_geometry,to_device
from .encoder import UnifiedDynamicHypergraphEncoder,EncoderConfig
from .offload import encode_offloaded
from .train import SingleTask,get_local,LRU
from .data import local_sets


def run(a):
    setup(42);start=time.time();rows=[]
    if a.kind=='local':
        samples=[r for seq in local_sets(a.cache).values() for r in seq]
        row=max(samples,key=lambda r:r['points']);g=get_local(row,LRU());q=load(row['path'])['query']
        model=SingleTask('h01','dynamic',a.stage).cuda()
        torch.cuda.reset_peak_memory_stats()
        logits=model.local([g],[q]);loss=torch.nn.functional.cross_entropy(logits,torch.tensor([row['target']],device='cuda'))
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        rows.append(dict(sample=row['sample'],points=g['input_point_count'],loss=float(loss.detach()),backward_verified=True))
    else:
        cell=max(json.loads((SOURCE/'manifest.json').read_text())['cells'],key=lambda r:r['surface_points'])
        path=SOURCE/'inputs'/(cell['cell']+'.npz')
        with np.load(path) as x:
            g=prepare_surface_geometry(x['surface_xyz_um']) if a.kind=='point_cell' else prepare_geometry(x['skeleton_xyz_um'],x['edge_index'],x['surface_xyz_um'])
        assert g['input_point_count']==cell['surface_points']
        model=UnifiedDynamicHypergraphEncoder(EncoderConfig(stage=a.stage)).cuda().eval()
        torch.cuda.reset_peak_memory_stats();z=encode_offloaded(model,g)
        assert torch.isfinite(z.skeleton).all() and torch.isfinite(z.surface).all()
        rows.append(dict(cell=cell['cell'],points=g['input_point_count'],unique=g['unique_point_count'],
            patches=g['n_patches'],source_sha256=sha(path),backward_verified=False))
    return dict(passed=True,no_subsampling=True,kind=a.kind,stage=a.stage,rows=rows,seconds=time.time()-start,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['local','cell','point_cell'],required=True)
    p.add_argument('--stage',choices=['residual','routing','role'],required=True);p.add_argument('--cache',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=dict(passed=False,source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},performance_claim=False)
    try:report.update(run(a))
    except Exception as exc:report['error']=repr(exc);raise
    finally:save(a.out,report);print(json.dumps(report),flush=True)
