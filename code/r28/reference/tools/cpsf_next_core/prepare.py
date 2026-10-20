import time,shutil
from concurrent.futures import ProcessPoolExecutor
from .common import *
from .data import construct

def write_item(task,row,s,e,p,w,q,target):
 data,stats=construct(s,e,p,w,q);folder=ROOT/'data'/task;folder.mkdir(parents=True,exist_ok=True)
 path=folder/(row['key'].replace(':','_')+'.npz');np.savez_compressed(path,**data)
 assert shutil.disk_usage(ROOT).free>4*2**30,'Disk reserve; preserve old datasets/checkpoints'
 return dict(key=row['key'],cell=row.get('cell',row.get('group',row.get('spatial_group'))),split=row['split'],target=target,path=str(path),sha256=sha(path),source=row,**stats)
def prepare_h01(row):
 torch.set_num_threads(1)
 for k in ['graph','point']:assert sha(row[k+'_path'])==row[k+'_sha256']
 with np.load(row['graph_path']) as g,np.load(row['point_path']) as p:
  return [write_item(row['task'],row,g['coord'],g['edge_index'],p['coord'],p['mass'],g['query'],row['target'])]
def prepare_ted(row):
 torch.set_num_threads(1)
 from tools.cpsf_fixed_nredd.data import ted_events
 out=[]
 for ev in ted_events(row):
  r=dict(key=ev['key'],group=row['spatial_group'],split=row['split'],block=row['key'],parent_cache=row['cache'],parent_cache_sha256=row['cache_sha256'],metadata=row['metadata'],metadata_sha256=row['metadata_sha256'])
  out.append(write_item('ted',r,ev['s'],ev['e'],ev['p'],np.ones(len(ev['p'])),ev['q'],int(ev['target'].argmax())))
 return out

def main():
 verify();m=json.loads((ROOT/'input_manifest.json').read_text());start=time.time()
 for task in ['h01_local','h01_global','ted']:
  records=[];source=m['ted_blocks'] if task=='ted' else m['h01'][task];worker=prepare_ted if task=='ted' else prepare_h01
  with ProcessPoolExecutor(max_workers=6) as pool:
   for i,rr in enumerate(pool.map(worker,source,chunksize=1)):
    records+=rr
    if (i+1)%10==0 or i+1==len(source):
     save(ROOT/'prepare_status.json',dict(task=task,blocks=i+1,total_blocks=len(source),samples=len(records),seconds=time.time()-start));print(task,i+1,len(source),len(records),flush=True)
  train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val'];p=verify();cfg=p['tasks'][task]
  assert len(train)==cfg['train'] and len(val)==cfg['val'];assert {r['cell'] for r in train}.isdisjoint(r['cell'] for r in val)
  save(ROOT/'data'/task/'manifest.json',dict(records=records,complete=True,protocol_sha256=sha(ROOT/'protocol.json'),features_use_labels=False))
 save(ROOT/'preparation.json',dict(passed=True,protocol_sha256=sha(ROOT/'protocol.json'),seconds=time.time()-start,test_used=False))
if __name__=='__main__':main()
