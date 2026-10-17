"""Freeze group holdouts using geometry/counts only, before features or training."""
from collections import defaultdict
from .common import *
def split_task(rows,task):
    groups=defaultdict(list)
    for r in rows:groups[r['spatial_group']].append(r)
    keys=sorted(groups);rng=np.random.RandomState(42);k=int(np.ceil(.25*len(keys)))
    weights=np.array([sum(r['events'] if task=='ted' else 1 for r in groups[g]) for g in keys])
    counts=np.array([len(groups[g]) for g in keys]);bins=[0,128,256,np.inf]
    hist=np.array([np.histogram([r['nodes'] for r in groups[g]],bins)[0] for g in keys]);fullhist=hist.sum(0)/hist.sum()
    candidates=[]
    for _ in range(10000):
        ix=np.sort(rng.choice(len(keys),k,replace=False));fraction=weights[ix].sum()/weights.sum()
        dominance=weights[ix].max()/weights[ix].sum()
        if not .15<=fraction<=.25 or dominance>.5 or not (hist[ix].sum(0)>0).all():continue
        score=abs(fraction-.2)+abs(counts[ix].sum()/counts.sum()-.2)+.25*abs(hist[ix].sum(0)/hist[ix].sum()-fullhist).sum()+.05*dominance
        candidates.append((score,tuple(ix)))
    assert candidates,'No predeclared feasible group holdout; stop rather than inspect labels'
    score,ix=min(candidates);dev={keys[i] for i in ix}
    result=[]
    for r in rows:result.append(dict(key=r['key'],group=r['spatial_group'],split='dev' if r['spatial_group'] in dev else 'train',nodes=r['nodes'],events=r.get('events'),origin=r['origin'],shape_xyz=r['shape_xyz']))
    return dict(records=result,score=score,dev_groups=sorted(dev),train_groups=sorted(set(keys)-dev))
def main():
    assert not (ROOT/'protocol.json').exists()
    src=BASE/'CPSF_FIXED_CORE_NREDD_20261006/split_manifest.json';original=json.loads(src.read_text())
    split={t:split_task([r for r in original['records'][t] if r['split']=='train'],t) for t in ['ted','tlp']}
    save(ROOT/'development_split.json',dict(seed=42,assignment_uses_labels=False,original_train_only=True,tasks=split))
    pted=json.loads((TED/'protocol.json').read_text());refs={str(src):sha(src)}
    for root in [TED,TLP]:
        refs[str(root/'protocol.json')]=sha(root/'protocol.json')
    for t in ['ted','tlp']:
        path=Path(pted['manifests']['ted']['joint']) if t=='ted' else TLP/'data/tlp/joint/manifest.json'
        refs[str(path)]=sha(path);lookup={r['key']:r for r in split[t]['records']}
        rr=[]
        for r in json.loads(path.read_text())['records']:
            if r['split']!='train':continue
            s=lookup[r['block']];assert s['group']==r['group']
            rr.append(dict(r,split=s['split'],parent_split='train'))
        assert {r['block'] for r in rr}==set(lookup)
        save(ROOT/'manifests'/f'{t}.json',dict(records=rr,complete=True))
        refs[str(ROOT/'manifests'/f'{t}.json')]=sha(ROOT/'manifests'/f'{t}.json')
    point=Path(pted['manifests']['ted']['point']);refs[str(point)]=sha(point)
    p=dict(experiment_id=ROOT.name,seed=42,selection='fixed final epoch; internal dev comparisons only; no historic val/test evaluation',
        historical_exposure='All internal dev samples belonged to historical training. New models start from scratch; this is not a pristine final test.',
        split_rule='10,000 seed42 random subsets of ceil(25% groups); 15-25% sample mass; largest dev group <=50%; each node-size bin represented; minimize count/mass/bin mismatch, label-free',
        development_split_sha256=sha(ROOT/'development_split.json'),references=refs,
        tasks={'ted':dict(epochs=30,batch=64,lr=.0005,weight_decay=.0005,loss='alpha5 weighted two-logit BCE with summed gradients; final partial step',scheduler='1/(epoch+1)',primary='positive F1'),
               'tlp':dict(epochs=80,batch=1,lr=.001,weight_decay=.001,loss='adaptive weighted BCE all unordered unobserved pairs',scheduler='StepLR20 gamma0.1',primary='missing-edge Macro-AP')},
        interventions={'ted':'Only replace first 3 point channels by directional XYZ from 16 distinct real neighbors within 2*radius; retain fourth channel and fallback',
            'tlp':'Only append abs(delta_xyz)/20, distance/20, min/max observed outward branch continuation, same observed component to pair head; initialize new columns zero'},
        retained=['fixed residual shared core','repaired observed skeleton','existing dynamic resolution and sampled points','physical position and scale encoding','seed42','old task optimizer and budget'],
        jobs=[dict(task=t,arm=a) for t in ['ted','tlp'] for a in ['control','repair']],
        sources={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')})
    save(ROOT/'protocol.json',p);verify()
    for t in split:
        rr=records(t);print(t,{s:dict(samples=sum(r['split']==s for r in rr),blocks=len({r['block'] for r in rr if r['split']==s}),groups=len({r['group'] for r in rr if r['split']==s})) for s in ['train','dev']},flush=True)
if __name__=='__main__':main()
