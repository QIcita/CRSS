"""Explicit long-tail and retrieval denominators; never drop zero-positive queries."""
import numpy as np

def classify(target,logits,nclasses):
 y=np.asarray(target,int);z=np.asarray(logits);assert z.shape==(len(y),nclasses);p=z.argmax(1)
 matrix=np.bincount(y*nclasses+p,minlength=nclasses*nclasses).reshape(nclasses,nclasses);tp=matrix.diagonal();support=matrix.sum(1)
 assert (support>0).all(),'Every formal test class must have support'
 precision=tp/np.maximum(matrix.sum(0),1);recall=tp/support;f1=2*tp/np.maximum(support+matrix.sum(0),1)
 return dict(accuracy=float((p==y).mean()),macro_f1=float(f1.mean()),balanced_accuracy=float(recall.mean()),top5_accuracy=float((np.argsort(-z,axis=1,kind='stable')[:,:min(5,nclasses)]==y[:,None]).any(1).mean()),per_class=[dict(target=i,support=int(support[i]),precision=float(precision[i]),recall=float(recall[i]),f1=float(f1[i])) for i in range(nclasses)],confusion_matrix=matrix.tolist())

def retrieve(query,gallery,qtypes,gtypes,qids,gids):
 q=np.asarray(query,float);g=np.asarray(gallery,float);assert len(q)==len(qtypes)==len(qids) and len(g)==len(gtypes)==len(gids)
 assert len(set(qids))==len(qids) and len(set(gids))==len(gids)
 assert np.isfinite(q).all() and np.isfinite(g).all() and (np.linalg.norm(q,axis=1)>0).all() and (np.linalg.norm(g,axis=1)>0).all()
 q=q/np.linalg.norm(q,axis=1,keepdims=True);g=g/np.linalg.norm(g,axis=1,keepdims=True)
 order0=np.argsort(np.asarray(gids),kind='stable');g=g[order0];gt=np.asarray(gtypes)[order0];gi=np.asarray(gids)[order0];rows=[]
 for i,x in enumerate(q):
  valid=gi!=qids[i];scores=g[valid]@x;relevant=gt[valid]==qtypes[i];n=int(relevant.sum());assert n>=3,('Insufficient gallery positives',qids[i],n)
  order=np.argsort(-scores,kind='stable');hit=relevant[order];at=np.flatnonzero(hit)+1;ap=float((np.arange(1,n+1)/at).mean())
  # Recall@K here is retrieval recall (fraction of all positives), not hit rate.
  rows.append(dict(root_id=qids[i],cell_type=qtypes[i],positives=n,ap=ap,mrr=1/float(at[0]),recall={str(k):float(hit[:k].sum()/n) for k in (1,5,10)},hit={str(k):int(hit[:k].any()) for k in (1,5,10)}))
 return dict(query_macro_mAP=float(np.mean([r['ap'] for r in rows])),cell_type_macro_mAP=float(np.mean([np.mean([r['ap'] for r in rows if r['cell_type']==t]) for t in sorted(set(qtypes))])),MRR=float(np.mean([r['mrr'] for r in rows])),recall={str(k):float(np.mean([r['recall'][str(k)] for r in rows])) for k in (1,5,10)},hit_rate={str(k):float(np.mean([r['hit'][str(k)] for r in rows])) for k in (1,5,10)},queries=len(rows),gallery=len(g),rows=rows)
