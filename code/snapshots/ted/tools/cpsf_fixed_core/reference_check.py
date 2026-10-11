import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import setup,Cache
from .common import *
from .model import Model
from .train import evaluate,move

def main():
    setup();p=verify();results={}
    for task in TASKS:
        ref=p['references'][task+'/full'];path=Path(ref['path'])
        for name,h in ref['hashes'].items():assert sha(path/name)==h
        model=Model(task,'full').cuda();ckpt=torch.load(path/'final.pt',map_location='cpu',weights_only=False)
        sd=ckpt['model'];allowed={'gate_alpha','region_gate.weight','region_gate.bias'}
        assert set(sd)-set(model.state_dict())<=allowed
        model.load_state_dict({k:sd[k] for k in model.state_dict()})
        rr,_=rows(task,'full');val=[r for r in rr if r['split']=='val'];out=ROOT/'references'/task;out.mkdir(parents=True,exist_ok=False)
        cache=Cache(budget=512*2**20);m=evaluate(model,val,cache,task,out,'val')
        key='mae' if task=='h01_global' else 'macro_f1';assert abs(m[key]-ref['val'][key])<1e-9
        new=np.load(out/'val_predictions.npz');oldfile=path/('val_features.npz' if task=='flywire_coarse' else 'val_predictions.npz');old=np.load(oldfile)
        assert np.array_equal(new['key'],old['root_id' if task=='flywire_coarse' else 'key'])
        np.testing.assert_allclose(new['prediction_or_logits'],old['logits' if task=='flywire_coarse' else 'prediction_or_logits'],rtol=2e-5,atol=2e-5)
        # Translation checks only; scale/rotation invariance is not claimed.
        r=next(r for r in rr if r['split']=='train');g=move(load_graph(r,task));shift=torch.tensor([1.,-2.,3.],device='cuda');translated=dict(g)
        for k in ('s','pos','query'):
            if k in translated:translated[k]=translated[k]+shift
        with torch.no_grad():torch.testing.assert_close(model(g),model(translated),rtol=2e-5,atol=2e-5)
        results[task]=dict(source=str(path),val=m,translation_check=True,gate_parameters_removed=True,
            parameters=sum(x.numel() for x in model.parameters()))
        del model,cache;torch.cuda.empty_cache()
    save(ROOT/'reference_contracts.json',dict(passed=True,results=results,test_used=False,protocol_sha256=sha(ROOT/'protocol.json')))
if __name__=='__main__':main()
