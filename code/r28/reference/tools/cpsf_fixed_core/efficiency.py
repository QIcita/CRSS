import time
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import setup
from .common import *
from .model import Model
from .train import move

def main():
    setup();p=verify();results={}
    for task in TASKS:
        rr,_=rows(task,'full');selected=sorted([r['key'] for r in rr if r['split']=='train'],key=lambda k:digest(['latency42',k]))[:8]
        result={}
        arms=['full','no_context']+([] if task=='h01_local' else ['thumbnail','uniform_budget'])
        for arm in arms:
            ref=p['references'][task+'/full'] if arm=='full' else None
            path=Path(ref['path']) if ref else ROOT/'results'/(task+'_'+arm)
            ckpt=torch.load(path/'final.pt',map_location='cpu',weights_only=False);model=Model(task,arm).cuda()
            model.load_state_dict({k:ckpt['model'][k] for k in model.state_dict()});model.eval()
            rr,_=rows(task,arm);lookup={r['key']:r for r in rr};measures=[]
            with torch.no_grad():
                for key in selected:
                    g=move(load_graph(lookup[key],task))
                    for _ in range(3):model(g)
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    for _ in range(10):model(g)
                    torch.cuda.synchronize();elapsed=(time.perf_counter()-start)*1000/10
                    measures.append(dict(key=key,milliseconds=elapsed,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                        points=g['surface_points'],nodes=g['raw_nodes'],regions=g['nregions'],coarse_groups=g['ncoarse'],incidences=len(g['pr'])))
                    del g
            result[arm]=dict(parameters=sum(v.numel() for v in model.parameters()),rows=measures,
                median_ms=float(np.median([x['milliseconds'] for x in measures])))
            del model;torch.cuda.empty_cache()
        results[task]=result
    save(ROOT/'efficiency.json',dict(results=results,device=torch.cuda.get_device_name(),batch=1,
        warmup=3,repeats=10,selected_split='train',samples_per_task=8,
        timing='CUDA synchronized resident-input inference; excludes CPU cache loading and preprocessing',test_used=False))
if __name__=='__main__':main()
