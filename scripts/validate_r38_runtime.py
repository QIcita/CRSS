"""CPU-only validation of the assembled release in the compatible Torch environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.directory.resolve();os.chdir(root);sys.path.insert(0,str(root))
    for name in ['CPSF_R38_DENSE_ROOT','CPSF_CMM_PAIR_ROOT','CPSF_R34_FPS_ROOT','CPSF_REBENCH_ROOT','CPSF_DRW_ROOT','MICRONS_ROOT']:
        os.environ[name]=str(root)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from tools.cpsf_r34_fps_rebench.model import H01Model,FlyModel
    from tools.cpsf_cmm_pair_fusion.model import install as pair_install
    from tools.cpsf_r38_dense.model import install as dense_install
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    reports=[]
    for variant,task in [('original','h01_local'),('original','h01_global'),('dense','h01_global'),('dense','flywire_coarse')]:
        exp=root/'experiments'/variant;folder=exp/'results'/('r38_'+task)
        model=H01Model(task) if task.startswith('h01_') else FlyModel()
        model=(pair_install if variant=='original' else dense_install)(model)
        c=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(c['model'],strict=True)
        m=read(folder/'metrics.json');ident=m['identity'];p=read(exp/'protocol.json')
        assert ident==read(folder/'identity.json')==c['identity']
        assert ident['protocol_sha256']==sha(exp/'protocol.json')
        assert sha(folder/'final.pt')==m['final_sha256']
        assert m['epochs']==p['tasks'][task]['epochs'] and m['steps']==p['tasks'][task]['updates']
        assert c['epoch']==m['epochs'] and not m['test_used']
        history=read(folder/'history.json');orders=read(exp/'orders'/(task+'.json'))
        assert len(history)==m['epochs'] and [h['sample_order_sha256'] for h in history]==[x['sha256'] for x in orders]
        with np.load(folder/'val_predictions.npz',allow_pickle=False) as a:y,z=a['target'],a['logits']
        assert np.isfinite(z).all()
        if task=='h01_global':
            error=z-y;computed=dict(mae=float(abs(error).mean()),rmse=float(np.sqrt((error**2).mean())),r2=float(1-(error**2).sum()/((y-y.mean())**2).sum()))
        else:
            n=3 if task=='h01_local' else 6;cm=np.zeros((n,n),dtype=int);np.add.at(cm,(y,z.argmax(1)),1)
            f1=np.divide(2*cm.diagonal(),cm.sum(0)+cm.sum(1),out=np.zeros(n),where=(cm.sum(0)+cm.sum(1))!=0)
            computed=dict(macro_f1=float(f1.mean()),accuracy=float(np.trace(cm)/cm.sum()))
            assert cm.tolist()==m['val']['confusion_matrix']
        for k,v in computed.items():assert abs(v-m['val'][k])<1e-12,(task,k,v,m['val'][k])
        reports.append(dict(variant=variant,task=task,strict_checkpoint_load=True,
             checkpoint_sha256=sha(folder/'final.pt'),protocol_sha256=sha(exp/'protocol.json'),
             parameters=sum(p.numel() for p in model.parameters()),metrics=computed))
        del model,c
    from tools.cpsf_microns.models import CountModel
    from tools.cpsf_microns.metrics import cell_metrics,aggregate
    exp=root/'experiments/microns';folder=exp/'results/r38_dense'
    m=read(folder/'metrics.json');c=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model=CountModel('r38_dense',1.);model.load_state_dict(c['model'],strict=True)
    assert sha(folder/'best.pt')==m['selected_checkpoint_sha256'] and c['epoch']==m['selected_epoch']
    history=[json.loads(x) for x in (folder/'history.jsonl').read_text().splitlines()]
    assert len(history)==m['epochs']==50 and m['updates']==4750
    computed={}
    for subset,field,count in [('val','validation',79),('test','test',80)]:
        cells=[]
        for f in sorted((folder/'predictions'/subset).glob('*.npz')):
            with np.load(f,allow_pickle=False) as a:cells.append(cell_metrics(a['target'],a['expected'],a['length']))
        assert len(cells)==count
        values=aggregate(cells)
        for k,v in values.items():assert abs(v-m[field][k])<1e-10,(field,k,v,m[field][k])
        computed[field]=values
    reports.append(dict(variant='microns',task='regional_input_synapse_distribution',strict_checkpoint_load=True,
        checkpoint_sha256=sha(folder/'best.pt'),selected_epoch=c['epoch'],parameters=sum(p.numel() for p in model.parameters()),metrics=computed))
    sources={}
    for name,module in list(sys.modules.items()):
        if not name.startswith(('tools.','utonia.','openpoints.')):continue
        path=getattr(module,'__file__',None)
        if not path:continue
        path=Path(path).resolve();assert path.is_relative_to(root),(name,str(path))
        sources[name]=dict(path=str(path.relative_to(root)),sha256=sha(path))
    result=dict(passed=True,strict_checkpoint_load=True,prediction_metric_recomputed=True,
        completed_models=reports,loaded_sources=sources,torch_version=torch.__version__,
        new_inference_performed=False,new_test_evaluation_performed=False,
        original_data_or_parent_code_imported=False,gpu_used=False,
        scope='Saved predictions were recomputed; no model forward or training was run.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='loaded_sources'},indent=2))

if __name__=='__main__':main()
