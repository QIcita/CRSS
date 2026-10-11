import json
from pathlib import Path
from .data import ROOT,read_dataset
from .model import ARMS
from tools.cpsf_v21_segment_units.io import save,sha

def main():
    assert not (ROOT/'protocol.json').exists()
    rows,labels,ready=read_dataset()
    parents={}
    for dirname in ('CPSF_V23_RESIDUAL_REGION_20261001','CPSF_V23_SCALE_ABLATION_20261001'):
        p=Path('/root/megrez-tmp')/dirname;old=json.loads((p/'protocol.json').read_text())
        for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,(name,'Inherited source drift')
        parents[dirname]=sha(p/'protocol.json')
    save(ROOT/'protocol.json',dict(version='FlyWire V23 full dataset and matched ablations',seed=42,
        tasks=['coarse','fine'],arms=ARMS,epochs=50,batch=4,lr=.0003,weight_decay=.01,
        label_smoothing=.05,lr_min=.00003,clip=5,selection='fixed final epoch',
        initialization='from scratch; all arms within task identical',
        dataset=ready,counts={s:sum(r['split']==s for r in rows) for s in ('train','val','test')},
        inherited_protocol_sha256=parents,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        document_sha256=sha(ROOT/'CPSF_FLYWIRE_V23_PROTOCOL_20261001.md'),
        test_policy='all 20 final checkpoints frozen and successful before one test evaluation per run; no test selection',
        retrieval='fine pre-head 128D cosine; train gallery, val/test queries; same cell_type positives; self-root excluded'))

if __name__=='__main__':main()
