from .common import *
from collections import Counter

def main():
    p=json.loads((ROOT/'protocol.json').read_text());summary={}
    for task,cfg in p['tasks'].items():
        rr=rows(task);assert set(r['split'] for r in rr)=={'train','val'}
        tr=[r for r in rr if r['split']=='train'];va=[r for r in rr if r['split']=='val']
        assert (len(tr),len(va))==(cfg['train'],cfg['val'])
        assert {r['cell'] for r in tr}.isdisjoint(r['cell'] for r in va)
        for r in rr:
            assert sha(r['path'])==r['sha256'] and sha(r['feature_path'])==r['feature_sha256']
        summary[task]=dict(records=len(rr),splits=dict(Counter(r['split'] for r in rr)),train_dev_cells_disjoint=True,all_source_and_coordinate_sidecar_hashes_valid=True)
        print(task,summary[task],flush=True)
    save(ROOT/'input_audit.json',dict(passed=True,tasks=summary,test_used=False))
if __name__=='__main__':main()
