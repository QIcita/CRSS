"""CPU-only data-path and matched-budget check before large acquisition."""
import json
import numpy as np
import torch
from tools.cpsf_v21_segment_units.io import save,sha
from tools.cpsf_flywire_v23.data import load
from tools.cpsf_flywire_v23.model import Model
from tools.cpsf_flywire_balanced.loss import balanced_loss,weights_from_counts
from .plan import ROOT,OLD,ARMS
from .budget import select_round,order_hash


def main():
    torch.set_num_threads(1);plan=json.loads((ROOT/'plan.json').read_text());rows=json.loads((ROOT/'candidates.json').read_text());train=[r for r in rows if r['split']=='train']
    orders=[]
    for arm in ARMS:
        rng=np.random.RandomState(42);seen=set();hh=[]
        for _ in range(50):
            ids=select_round(train,rng);assert len(ids)==len(set(ids))==4975
            selected=[train[i] for i in ids];seen.update(r['root_id'] for r in selected);hh.append(order_hash(selected))
        assert len(seen)==len(train);orders.append(hh)
    assert orders[0]==orders[1]==orders[2]
    counts=[sum(r['coarse_target']==i for r in train) for i in range(6)];w=weights_from_counts(counts)
    torch.testing.assert_close(w*torch.tensor(counts),torch.full((6,),len(train)/6),rtol=1e-6,atol=.001)
    geometry=json.loads((ROOT/'pilot_geometry.json').read_text());checks=[]
    # Every coarse category has a new real geometry, including fine-unavailable cells.
    for r in geometry['records']:
        assert r['fine_target']==-1 and not r['fine_available'] and r['split']=='train'
        assert sha(r['path'])==r['sha256'];g=load(r['path'])
        assert g['raw_nodes']==r['nodes'] and not any(k in g for k in ('target','coarse_target','fine_target','cell_type'))
        assert torch.isfinite(g['phi']).all()
        torch.manual_seed(42);model=Model(6,'full')
        ref=torch.load(OLD/'results/coarse_full/initial.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,ref[k]) for k,v in model.state_dict().items())
        model.eval()
        with torch.no_grad():z=model(g)
        assert torch.isfinite(z).all() and torch.isfinite(balanced_loss(z[None],torch.tensor([r['coarse_target']]),w))
        checks.append(dict(root_id=r['root_id'],passed=True));del g,model
    save(ROOT/'preflight.json',dict(passed=True,real_new_coarse_geometry=checks,
        full_pool_covered_by_fixed_budget=True,all_arms_same_order=True,optimizer_steps=50*((4975+3)//4),
        old_initialization_preserved=True,new_fine_labels_not_used=True,test_used=False,accuracy_claim=False))


if __name__=='__main__':main()
