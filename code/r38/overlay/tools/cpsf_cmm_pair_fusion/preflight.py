"""Small real-model integration smoke check before protocol freeze."""
import gc
import torch
from .common import ROOT,PARENT,GEOMETRY,read,save,PREFIX
from .backend import Backend,loss_for
from .acceptance import compatibility,fusion_interface_contract
from .worker import optimizer


def main():
    reports={}
    for task in ('h01_local','h01_global'):
        records=read(GEOMETRY/'data'/task/'manifest.json')['records']
        tr=[r for r in records if r['split']=='train'];cfg=read(PARENT/'protocol.json')['tasks'][task]
        compatibility(task,tr,cfg)
        b=Backend('r38',task,tr);m=b.model;opt,_=optimizer(b,cfg);row=min(tr,key=lambda r:r['points']);active=set()
        for step in range(6):
            m.train();opt.zero_grad(set_to_none=True);g,y,_=b.item(row)
            loss=loss_for(m(g),y,task,b.scale);loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            active.update(n for n,p in m.named_parameters() if p.grad is not None and (p.grad!=0).any())
            torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
        assert {n for n,_ in m.named_parameters() if PREFIX in n}<=active
        reports[task]=fusion_interface_contract(b,row)
        del b,m,opt,g,y,loss;gc.collect();torch.cuda.empty_cache()
    report=dict(passed=True,tasks=reports,test_used=False,weights_saved_or_reused=False)
    save(ROOT/'model_preflight.json',report);print(report,flush=True)


if __name__=='__main__':main()
