import time,shutil
from concurrent.futures import ProcessPoolExecutor
from .common import *
from .model import pair_geometry
from tools.cpsf_directional.data import from_point_cache

def prepare_one(args):
    task,r,aux=args;torch.set_num_threads(1)
    assert sha(r['path'])==r['sha256'];item=load(r['path']);g=item['g'];path=sidepath(task,r['key']);path.parent.mkdir(parents=True,exist_ok=True)
    if task=='ted':
        assert sha(aux['path'])==aux['sha256'];point=load(aux['path'])
        for field in ['pf','point_owner','point_weight','patch_mass']:assert torch.equal(g[field],point['g'][field]),(r['key'],field)
        feature,valid,stats=from_point_cache(point['g'])
        effective=torch.cat((feature[:,:3],g['pf'][:,3:]),1)
        np.savez_compressed(path,pf_directional=feature.numpy(),valid=valid.numpy())
        stats=dict(old_zero=float((g['pf']==0).all(1).float().mean()),new_zero=float((effective==0).all(1).float().mean()),valid_fraction=float(valid.float().mean()))
    else:
        assert sha(aux['swc'])==aux['swc_sha256'];swc=np.loadtxt(aux['swc']);n=len(swc)
        assert np.array_equal(swc[:,0],np.arange(1,n+1))
        obs=np.eye(n,dtype=bool)
        for i,parent in enumerate(swc[:,-1].astype(int)):
            if parent!=-1:obs[i,parent-1]=obs[parent-1,i]=True
        assert np.array_equal(obs,item['observed'].numpy())
        xyz=swc[:,2:5]*[1,1,2];q=g['query_xyz'].numpy().astype(np.float64)
        np.testing.assert_allclose(xyz-xyz.mean(0),q-q.mean(0),atol=.003,rtol=0)
        feature=pair_geometry(q,obs)
        np.savez_compressed(path,pair_geometry=feature)
        stats=dict(nodes=n,channels=feature.shape[-1],max_asymmetry=float(abs(feature-feature.transpose(1,0,2)).max()))
    return dict(key=r['key'],split=r['split'],path=str(path),sha256=sha(path),bytes=path.stat().st_size,**stats)
def main():
    verify();start=time.time();original=json.loads((BASE/'CPSF_FIXED_CORE_NREDD_20261006/split_manifest.json').read_text())
    pp=json.loads((TED/'protocol.json').read_text())['manifests']['ted']['point']
    points={r['key']:r for r in json.loads(Path(pp).read_text())['records'] if r['split']=='train'}
    summary={}
    for task in ['tlp','ted']:
        rr=records(task);aux=points if task=='ted' else {r['key']:r for r in original['records']['tlp'] if r['split']=='train'}
        args=[(task,r,aux[r['key']]) for r in rr];done=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for row in pool.map(prepare_one,args,chunksize=8):
                done.append(row)
                if len(done)%200==0 or len(done)==len(rr):
                    assert shutil.disk_usage(ROOT).free>5*2**30
                    save(ROOT/'prepare_status.json',dict(task=task,processed=len(done),total=len(rr),seconds=time.time()-start));print(task,len(done),len(rr),flush=True)
        save(ROOT/'manifests'/f'{task}_sidecars.json',dict(records=done,complete=True,protocol_sha256=sha(ROOT/'protocol.json')))
        summary[task]={s:dict(samples=sum(r['split']==s for r in done),**({k:float(np.mean([r[k] for r in done if r['split']==s])) for k in ['old_zero','new_zero','valid_fraction']} if task=='ted' else {})) for s in ['train','dev']}
    save(ROOT/'preparation.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),summary=summary,seconds=time.time()-start,features_use_labels=False))
if __name__=='__main__':main()
