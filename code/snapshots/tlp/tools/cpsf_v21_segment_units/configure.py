"""Freeze a bounded inner-cell search before any candidate is trained."""
from .io import *
ROOT=Path(__file__).resolve().parents[2]
CACHE=Path('/root/megrez-tmp/CPSF_V19_MEASURE_CORE_20260930/cache_h01')
CANDIDATES=[dict(name=arm,arm=arm,weight_decay=.01,label_smoothing=.05,residual_scale=.1)
            for arm in ('full','no_fusion','node_units')]

def main():
    assert not (ROOT/'search_protocol.json').exists()
    original=json.loads((CACHE/'manifest.json').read_text())['records']
    training=[r for r in original if r['split']=='train'];outer=[r for r in original if r['split']=='val']
    cells=sorted({r['cell'] for r in training},key=lambda c:hashlib.sha256(('42|'+c).encode()).hexdigest())
    assert len(cells)==80;held=set(cells[:16]);fit=set(cells[16:]);outer_cells={r['cell'] for r in outer}
    assert len(outer_cells)==24 and not (set(cells)&outer_cells)
    rows={k:[] for k in ('inner_train','inner_val','outer_train','outer_val')}
    for r in training:
        row=dict(r,path=str(CACHE/'local'/(r['sample']+'.pt')))
        assert sha(row['path'])==r['sha256'];rows['outer_train'].append(row)
        rows['inner_val' if r['cell'] in held else 'inner_train'].append(row)
    for r in outer:
        row=dict(r,path=str(CACHE/'local'/(r['sample']+'.pt')));assert sha(row['path'])==r['sha256'];rows['outer_val'].append(row)
    for key in ('inner_train','inner_val'):assert {r['target'] for r in rows[key]}=={0,1,2}
    for left,right in [('inner_train','inner_val'),('outer_train','outer_val')]:
        for key in ('cell','sample','sha256'):
            assert {r[key] for r in rows[left]}.isdisjoint({r[key] for r in rows[right]}),(left,right,key)
    save(ROOT/'split_manifest.json',dict(rows=rows,inner_train_cells=sorted(fit),inner_val_cells=sorted(held),outer_val_cells=sorted(outer_cells),source_manifest_sha256=sha(CACHE/'manifest.json')))
    protocol=dict(version='V21 physical segment region units',seed=42,candidates=CANDIDATES,
                  checkpoints=[15,25,35],max_epochs=35,scheduler_horizon=35,
                  selection='full only: max inner validation Macro-F1; exact ties prefer earlier epoch',
                  gate='selected full must strictly beat BOTH no_fusion and node_units at the same epoch; otherwise stop before outer refit',
                  outer_evaluation='one refit per arm after selection; fixed selected epoch and same selected hyperparameters for eight local arms and six primary global arms; no outer-driven additional search',
                  split_manifest_sha256=sha(ROOT/'split_manifest.json'),
                  sample_counts={k:len(v) for k,v in rows.items()},
                  source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
                  protocol_sha256=sha(ROOT/'CPSF_V21_SEGMENT_UNITS_PROTOCOL_20260930.md'))
    save(ROOT/'search_protocol.json',protocol);print(json.dumps(protocol,indent=2))

if __name__=='__main__':main()
