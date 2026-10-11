import numpy as np
from .common import *

def main():
    p=verify();results={};parents={};orders=None;initial=None
    rr,_=rows('tlp','full');expected_val=[r['key'] for r in rr if r['split']=='val']
    for arm in ARMS:
        key='tlp_'+arm;folder=ROOT/'results'/key;m=json.loads((folder/'metrics.json').read_text());ident=m['identity']
        assert ident['protocol_sha256']==sha(ROOT/'protocol.json') and ident['objective_id']==p['objective_id']
        assert not m['smoke'] and not m['test_used'] and m['epochs']==80 and m['optimizer_steps']==156*80
        assert m['final_sha256']==sha(folder/'final.pt')
        h=json.loads((folder/'history.json').read_text());order=[r['sample_order_sha256'] for r in h]
        parent_h=json.loads((PARENT/'results'/('tlp_'+parent_arm(arm))/'history.json').read_text())
        assert order==[r['sample_order_sha256'] for r in parent_h]
        if initial is None:initial=ident['initial_state_sha256'];orders=order
        assert ident['initial_state_sha256']==initial and order==orders
        audit=json.loads((folder/'independent_missing_edge_audit.json').read_text())
        assert audit['passed'] and audit['blocks']==62 and abs(audit['macro_ap']-m['val']['missing_edge_macro_ap'])<1e-12
        assert [b['key'] for b in m['val']['blocks']]==expected_val
        results[arm]=m;parents[arm]=json.loads((PARENT/'results'/('tlp_'+parent_arm(arm))/'metrics.json').read_text())['val']['missing_edge_macro_ap']
    scores={a:m['val']['missing_edge_macro_ap'] for a,m in results.items()}
    assert abs(scores['skeleton']-parents['skeleton'])<1e-12, 'Skeleton negative-control drift'
    new_state=torch.load(ROOT/'results/tlp_skeleton/final.pt',map_location='cpu',weights_only=False)['model']
    old_state=torch.load(PARENT/'results/tlp_skeleton/final.pt',map_location='cpu',weights_only=False)['model']
    assert all(torch.equal(v,old_state[k]) for k,v in new_state.items()), 'Skeleton weights changed despite point-only modification'
    pairs={}
    baseline=np.array([r['missing_edge_ap'] for r in results['full']['val']['blocks']])
    for a,m in results.items():
        delta=baseline-np.array([r['missing_edge_ap'] for r in m['val']['blocks']])
        pairs[a]=dict(full_better=int((delta>1e-12).sum()),variant_better=int((delta< -1e-12).sum()),ties=int((abs(delta)<=1e-12).sum()))
    criteria=dict(full_over_shuffled_descriptor=scores['full']>scores['shuffle_descriptor'],full_over_mean=scores['full']>scores['mean_only'],full_over_no_fusion=scores['full']>scores['no_fusion'],full_over_point=scores['full']>scores['point'],full_over_skeleton=scores['full']>scores['skeleton'],full_over_no_scale=scores['full']>scores['no_scale'])
    save(ROOT/'analysis.json',dict(primary='Missing-edge macro AP',scores=scores,paired=pairs,parent_scores_voxel_descriptor=parents,
        new_minus_parent={a:scores[a]-parents[a] for a in ARMS},matched_initialization=True,matched_order=True,criteria=criteria,
        next_step='Review point-content gains under unchanged readout and scale encoding',test_used=False))
    save(ROOT/'summary.json',dict(results={'tlp_'+a:v for a,v in results.items()},matched_order=True,test_used=False,analysis_sha256=sha(ROOT/'analysis.json')))
    lines=['# Stage 3 final validation','', 'seed42; final epoch80; same pair-region readout; neighborhood descriptor compared with voxel-relative descriptor.','',
        '| arm | missing-edge macro AP (%) | stage2 voxel descriptor (%) |','|---|---:|---:|']
    lines += [f'| {a} | {100*scores[a]:.3f} | {100*parents[a]:.3f} |' for a in ARMS]
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(scores),flush=True)
if __name__=='__main__':main()
