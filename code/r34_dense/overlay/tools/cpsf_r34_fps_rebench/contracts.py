"""Prove the new task adapters preserve R28 and the original R34 H01 design."""
import gc,math
import torch
from .common import ROOT,save,sha
from .backend import Backend,digest,loss_for

def compatibility(task,tr,cfg):
    from .worker import optimizer
    from tools.cpsf_raw_point_attention.common import setup
    row=min(tr,key=lambda x:x['points'])
    ref=Backend('r28',task,tr);new=Backend('r34',task,tr)
    reference_initial=digest(ref.model)
    shared={n:v for n,v in new.model.state_dict().items() if not n.startswith('region_message.')}
    assert shared.keys()==ref.model.state_dict().keys()
    assert all(torch.equal(v,ref.model.state_dict()[n]) for n,v in shared.items())
    assert sum(p.numel() for p in new.model.region_message.parameters())==6608
    g,y,_=new.item(row)
    ref.model.eval();new.model.eval()
    with torch.no_grad():
        a=ref.eval_forward(g);b=new.eval_forward(g)
        assert torch.equal(a,b),('zero-init mismatch',task,float((a-b).abs().max()))
    # Disabled-branch replay must reproduce every R28 parameter after two updates.
    new.model.message_enabled=False
    opt0,_=optimizer(ref,cfg);opt1,_=optimizer(new,cfg);replay=[]
    for step in range(2):
        losses=[]
        for backend,opt in ((ref,opt0),(new,opt1)):
            torch.manual_seed(42+step);backend.model.train();opt.zero_grad(set_to_none=True)
            loss=loss_for(backend.model(g),y,task,backend.scale)
            loss.backward()
            if task not in ('ted','tlp'):torch.nn.utils.clip_grad_norm_(backend.model.parameters(),5,error_if_nonfinite=True)
            opt.step();losses.append(float(loss.detach()))
        assert losses[0]==losses[1],('disabled loss mismatch',task,step)
        assert all(torch.equal(v,new.model.state_dict()[n]) for n,v in ref.model.state_dict().items()),('disabled parameter mismatch',task,step)
        replay.append(dict(step=step+1,loss=losses[0],core_parameters_exact=True))
    del ref,new,opt0,opt1,shared,a,b,loss,g,y;gc.collect();torch.cuda.empty_cache()
    parity=None
    if task.startswith('h01_'):
        new=Backend('r34',task,tr)
        from .frozen_r34 import Model as OriginalR34
        setup();old=OriginalR34(task,'path_message').cuda()
        if task=='h01_global':
            with torch.no_grad():old.head.bias.fill_(math.log(new.mean/(1-new.mean)))
        assert digest(old)==digest(new.model),('old R34 initialization mismatch',task)
        g,y,_=new.item(row)
        # Same optimizer/loss; verify the inherited H01 adaptation for two steps.
        opt0,_=optimizer(new,cfg);opt1=torch.optim.AdamW(old.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        for step in range(2):
            losses=[]
            for m,opt in ((new.model,opt0),(old,opt1)):
                torch.manual_seed(42+step);m.train();opt.zero_grad(set_to_none=True)
                loss=loss_for(m(g),y,task,new.scale);loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True)
                opt.step();losses.append(float(loss.detach()))
            assert losses[0]==losses[1] and digest(old)==digest(new.model),('original R34 replay mismatch',task,step)
        parity=dict(original_H01_R34_two_updates_exact=True,state_sha256=digest(old))
        del old,new,opt0,opt1,g,y,loss;gc.collect();torch.cuda.empty_cache()
    save(ROOT/'contracts'/(task+'.json'),dict(passed=True,task=task,protocol_sha256=sha(ROOT/'protocol.json'),
        sample=row['key'],test_used=False,zero_initialization_equals_R28=True,added_parameters=6608,
        r28_initial_state_sha256=reference_initial,
        disabled_branch_two_updates=replay,original_R34_parity=parity,weights_reused_for_training=False))
