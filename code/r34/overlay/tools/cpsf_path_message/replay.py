"""Independent two-update traces using the exact inherited optimization recipe."""
import argparse
import hashlib
from .common import *
from .data import load_item, move
from .model import Model
from torch.nn import functional as F


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',required=True);ap.add_argument('--kind',required=True);ap.add_argument('--out',required=True)
    args=ap.parse_args();setup();verify()
    rr=[r for r in rows(args.task) if r['split']=='train']
    if args.kind=='parent':
        from tools.cpsf_raw_point_attention.model import Model as Parent
        model=Parent(args.task,'r9_lpa_attention').cuda().train()
    else:
        model=Model(args.task,args.kind).cuda().train()
    def core_sha():
        h=hashlib.sha256()
        for n,v in model.state_dict().items():
            if not n.startswith('region_message.'):
                h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    initial=core_sha();mean=float(np.mean([r['target'] for r in rr]));scale=max(float(np.std([r['target'] for r in rr])),.05)
    if args.task=='h01_global':
        with torch.no_grad():model.head.bias.fill_(float(np.log(mean/(1-mean))))
    order=[rr[i] for i in np.random.permutation(len(rr))];batch=8 if args.task=='h01_local' else 4
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);trace=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True);group=order[step*batch:(step+1)*batch];losses=[]
        for row in group:
            z=model(move(load_item(row)));y=torch.tensor(row['target'],device='cuda')
            loss=((z-y.float())/scale).square() if args.task=='h01_global' else F.cross_entropy(z[None],y.long()[None],label_smoothing=.05)
            (loss/len(group)).backward();losses.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);opt.step()
        trace.append(dict(loss=losses,keys=[r['key'] for r in group],state_sha256=core_sha(),full_state_sha256=state_sha(model)))
    save(args.out,dict(initial=initial,trace=trace));print(args.kind,args.task,'two updates verified',flush=True)


if __name__=='__main__':main()
