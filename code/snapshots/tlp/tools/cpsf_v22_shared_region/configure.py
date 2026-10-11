from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha
ROOT=Path(__file__).resolve().parents[2]

def main():
    assert not (ROOT/'protocol.json').exists()
    save(ROOT/'protocol.json',dict(version='V22 shared physical region encoder',seed=42,
        arms=['full','no_fusion','point','skeleton'],tasks=['local','global'],epochs={'local':25,'global':50},
        selection='fixed final epoch; no tuning or checkpoint selection on 24 development cells',
        local_objective='cross entropy, label smoothing .05',global_objective='MSE divided by train target variance; sigmoid output; scale floor .05',
        optimizer=dict(name='AdamW',lr=.0003,weight_decay=.01,clip=5),batch={'local':8,'global':4},
        evaluation='separate task-supervised runs; never replace V21 frozen-transfer scores',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        protocol_document_sha256=sha(ROOT/'CPSF_V22_SHARED_REGION_PROTOCOL_20260930.md')))

if __name__=='__main__':main()
