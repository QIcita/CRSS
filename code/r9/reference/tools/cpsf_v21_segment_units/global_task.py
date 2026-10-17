"""Whole-cell frozen transfer, same V20 token hierarchy, train-only readout CV."""
import argparse,time
from .io import *
from .geometry import prepare
from .model import Model, spatial_chunks
from .readout import fit

def induced(s,e,ids,limit):
    if len(ids)>limit:ids=ids[np.linspace(0,len(ids)-1,limit,dtype=int)]
    lut=np.full(len(s),-1,int);lut[ids]=np.arange(len(ids))
    keep=(lut[e[:,0]]>=0)&(lut[e[:,1]]>=0)
    return s[ids],lut[e[keep]]

def geometries(x,arm):
    rng=np.random.default_rng(42);empty=np.empty((0,3));edges=np.empty((0,2),int);views=[]
    if arm=='skeleton':
        s=x['skeleton_xyz_um'];e=x['edge_index'].astype(int)
        ss,ee=induced(s,e,np.arange(len(s)),2048);views.append(prepare(ss,ee,empty,wrong=False))
        chunks=spatial_chunks(s,max(1,int(np.ceil(len(s)/min(32,max(1,int(np.ceil(len(s)/4096))))))))
        for ids in chunks:
            ss,ee=induced(s,e,ids,512);views.append(prepare(ss,ee,empty,wrong=False))
        return views
    p=x['surface_xyz_um']
    if arm=='point':s=empty;e=edges
    else:s=x['skeleton_xyz_um'];e=x['edge_index'].astype(int)
    def view(ids,node_ids,nodes,points):
        selected=np.sort(rng.choice(ids,min(len(ids),points),replace=False))
        ss,ee=induced(s,e,node_ids,nodes)
        weights=np.full(len(selected),len(ids)/max(len(selected),1))
        return prepare(ss,ee,p[selected],weights=weights,wrong=False)
    views.append(view(np.arange(len(p)),np.arange(len(s)),2048,4096))
    # Median bisection creates up to 32 geometrically compact surface chunks.
    chunks=[np.arange(len(p))]
    target=min(32,max(1,int(np.ceil(len(p)/16384))))
    while len(chunks)<target:
        k=max(range(len(chunks)),key=lambda i:len(chunks[i]));ids=chunks.pop(k)
        if len(ids)<2:chunks.insert(k,ids);break
        axis=np.ptp(p[ids],axis=0).argmax();ids=ids[np.argsort(p[ids,axis],kind='stable')];mid=len(ids)//2
        chunks[k:k]=[ids[:mid],ids[mid:]]
    for ids in chunks:
        lo=p[ids].min(0)-4.;hi=p[ids].max(0)+4.
        node_ids=np.flatnonzero(((s>=lo)&(s<=hi)).all(1))
        views.append(view(ids,node_ids,512,2048))
    return views

@torch.no_grad()
def extract(checkpoint,out,smoke=False):
    from .train import validated_protocol
    protocol=validated_protocol()
    setup();assert not out.exists();out.mkdir(parents=True)
    ck=load(checkpoint);arm=ck['protocol']['arm'];assert ck['protocol']['task']=='h01'
    assert ck['protocol']['source_sha256']==protocol['source_sha256'] and not ck['protocol']['smoke']
    model=Model('h01',arm,**ck['protocol'].get('model_config',{})).cuda();model.load_state_dict(ck['model']);model.eval()
    rows=sorted(json.loads((SOURCE/'manifest.json').read_text())['cells'],key=lambda r:r['cell'])
    if smoke:rows=[min((r for r in rows if r['split']==k),key=lambda r:r['surface_points']) for k in ('train','val')]
    records=[];features=[];ys=[];start=time.time()
    for row in rows:
        path=SOURCE/'inputs'/(row['cell']+'.npz')
        with np.load(path) as x:gs=geometries(x,arm)
        # Stream raw point stems; all compact tokens then communicate jointly.
        bundles=[model.encoder.stems([g],arm)[0] for g in gs]
        bundles=model.encoder.propagate(bundles,arm,offsets=[g['origin'] for g in gs])
        vector=model.global_feature(bundles).cpu().numpy();assert vector.shape==(128,) and np.isfinite(vector).all()
        with np.load(SOURCE/'targets'/(row['cell']+'.npz')) as y:target=float(y['excitatory_input_fraction'])
        features.append(vector);ys.append(target);records.append(dict(cell=row['cell'],split=row['split'],source_sha256=sha(path),tiles=len(gs)-1,surface_points=sum(len(g['p']) for g in gs),nodes=sum(g['n'] for g in gs)))
        save(out/'status.json',dict(state='extracting',cells=len(records),expected=len(rows),seconds=time.time()-start));print('CELL',arm,len(records),flush=True)
    np.savez_compressed(out/'features.npz',x=np.asarray(features),y=np.asarray(ys),cell=np.asarray([r['cell'] for r in records]),split=np.asarray([r['split'] for r in records]))
    save(out/'manifest.json',dict(arm=arm,checkpoint_sha256=sha(checkpoint),records=records,smoke=smoke,encoder_frozen=True,scope='V21 frozen local-to-global; post-hierarchy tokens only',partition='surface median splits for joint/point; skeleton-only uses skeleton spatial splits',pool='thumbnail/local 1:1; node mean, empirical mass surface; learned 128D shared readout'))
    save(out/'status.json',dict(state='extraction_complete',cells=len(records)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    try:
        extract(a.checkpoint,a.out,a.smoke)
        if not a.smoke:fit(a.out)
    except BaseException as e:save(a.out/'failure.json',dict(error=repr(e)));raise
