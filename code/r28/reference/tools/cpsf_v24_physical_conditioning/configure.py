import json
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha
from .model import JOBS,ARMS
ROOT=Path(__file__).resolve().parents[2]
BASE=Path('/root/megrez-tmp/CPSF_V23_SCALE_ABLATION_20261001')
V23=Path('/root/megrez-tmp/CPSF_V23_RESIDUAL_REGION_20261001')

def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((BASE/'protocol.json').read_text())
    for name,digest in old['source_sha256'].items():assert sha(ROOT/name)==digest,(name,'source drift')
    audit=Path('/root/megrez-tmp/CPSF_V23_GEOMETRY_AUDIT_20261001/audit.json')
    assert json.loads(audit.read_text())['passed']
    baselines={}
    for task in ('local','global'):
        for arm in ('full','no_position','no_scale','no_position_scale'):
            p=(V23 if arm=='full' else BASE)/'results'/(task+'_'+arm)
            m=json.loads((p/'metrics.json').read_text());baselines[task+'_'+arm]=dict(path=str(p),
                sha256={n:sha(p/n) for n in ('metrics.json','initial.pt','final.pt')},
                val={k:v for k,v in m['val'].items() if k!='rows'})
    old.update(version='V24 reference-centered gated physical conditioning',jobs=JOBS,arms=ARMS,
        baselines=baselines,source_audit_sha256=sha(audit),minimum_coarse_groups=4,
        added_parameters=['position_alpha','scale_alpha'],gate_init=0,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        inherited_protocol_sha256=sha(BASE/'protocol.json'),
        protocol_document_sha256=sha(ROOT/'CPSF_V24_PHYSICAL_CONDITIONING_PROTOCOL_20261001.md'))
    save(ROOT/'protocol.json',old)

if __name__=='__main__':main()
