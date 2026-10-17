"""Validate reuse by re-evaluating frozen CE checkpoints with the new control implementation."""
import json
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import setup
from tools.cpsf_flywire_v23.data import load
from tools.cpsf_flywire_20260930.evaluation import classify
from .common import *
from .model import Model
from .train import move

def main():
    setup();p=verify();m=json.loads((FW/'cache/manifest.json').read_text());val=[r for r in m['records'] if r['split']=='val'];results={}
    for arm in ('fixed_residual','mean_only'):
        ref=p['references']['flywire_coarse/'+arm];path=Path(ref['path'])
        for fn,h in ref['hashes'].items():assert sha(path/fn)==h
        model=Model('flywire_coarse',arm).cuda();checkpoint=torch.load(path/'final.pt',map_location='cpu',weights_only=False)
        state=model.state_dict();assert set(checkpoint['model'])==set(state)-{'gate_alpha','region_gate.weight','region_gate.bias'}
        state.update(checkpoint['model']);model.load_state_dict(state);model.eval();z=[]
        with torch.no_grad():
            for r in val:z.append(model(move(load(r['path']))).cpu().numpy())
        z=np.array(z);old=np.load(path/'val_features.npz');assert np.array_equal(old['root_id'],np.array([r['root_id'] for r in val]))
        np.testing.assert_allclose(z,old['logits'],rtol=2e-5,atol=2e-5);assert np.array_equal(z.argmax(1),old['logits'].argmax(1))
        metric=classify([r['coarse_target'] for r in val],z,6);assert abs(metric['macro_f1']-ref['val']['macro_f1'])<1e-10
        results[arm]=dict(source=str(path),macro_f1=metric['macro_f1'],exact_predictions=True)
        del model;torch.cuda.empty_cache()
    save(ROOT/'reference_contracts.json',dict(passed=True,test_used=False,results=results,protocol_sha256=sha(ROOT/'protocol.json')))

if __name__=='__main__':main()
