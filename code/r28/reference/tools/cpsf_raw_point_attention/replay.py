"""Two actual optimizer updates, also executable against untouched R9."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F

def main():
    a=argparse.ArgumentParser();a.add_argument('--task',required=True);a.add_argument('--kind',required=True);a.add_argument('--out',required=True);a=a.parse_args()
    if a.kind=='parent':
        from tools.cpsf_utonia_context.common import setup,rows,state_sha
        from tools.cpsf_utonia_context.data import load_item,move
        from tools.cpsf_utonia_context.model import Model
        kind='patch_rms_010'
    else:
        from tools.cpsf_raw_point_attention.common import setup,rows,state_sha
        from tools.cpsf_raw_point_attention.data import load_item,move
        from tools.cpsf_raw_point_attention.model import Model
        kind=a.kind
    import numpy as np
    setup();rr=[r for r in rows(a.task) if r['split']=='train'];m=Model(a.task,kind).cuda().train()
    def core_sha(model):
        import hashlib
        h=hashlib.sha256()
        for n,v in model.state_dict().items():
            if not n.startswith(('raw_refine.',)):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial=core_sha(m);mean=float(np.mean([r['target'] for r in rr]));scale=max(float(np.std([r['target'] for r in rr])),.05)
    if a.task=='h01_global':
        with torch.no_grad():m.head.bias.fill_(float(np.log(mean/(1-mean))))
    order=[rr[i] for i in np.random.permutation(len(rr))];batch=8 if a.task=='h01_local' else 4
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.01);trace=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True);group=order[step*batch:(step+1)*batch];losses=[]
        for r in group:
            z=m(move(load_item(r)));y=torch.tensor(r['target'],device='cuda')
            loss=((z-y.float())/scale).square() if a.task=='h01_global' else F.cross_entropy(z[None],y.long()[None],label_smoothing=.05)
            (loss/len(group)).backward();losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        trace.append(dict(loss=losses,state_sha256=core_sha(m),full_state_sha256=state_sha(m),keys=[r['key'] for r in group]))
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(initial=initial,trace=trace),indent=2)+'\n')
    print(a.kind,a.task,'replay complete',flush=True)

if __name__=='__main__':main()
