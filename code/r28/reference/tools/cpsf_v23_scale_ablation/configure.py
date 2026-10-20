import json
from pathlib import Path
from tools.cpsf_v21_segment_units.io import save,sha
from .model import JOBS
from .prepare import ROOT,BASE


def main():
    assert not (ROOT/'protocol.json').exists()
    old=json.loads((BASE/'protocol.json').read_text())
    assert json.loads((BASE/'pipeline_status.json').read_text())['state']=='complete'
    for name,digest in old['source_sha256'].items():
        assert sha(BASE/name)==digest and sha(ROOT/name)==digest,(name,'inherited source drift')
    baseline={}
    for task in ('local','global'):
        p=BASE/'results'/(task+'_full');m=json.loads((p/'metrics.json').read_text())
        baseline[task]=dict(path=str(p),sha256={n:sha(p/n) for n in ('metrics.json','initial.pt','final.pt')},
            val={k:v for k,v in m['val'].items() if k!='rows'})
    old['inherited_cache_manifest_sha256']=old.pop('cache_manifest_sha256')
    old.update(version='V23 dynamic resolution and explicit physical encoding ablations',
        arms=['no_position','no_scale','no_position_scale','thumbnail','uniform_budget'],jobs=JOBS,
        baselines=baseline,inherited_protocol_sha256=sha(BASE/'protocol.json'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'tools').rglob('*.py')},
        protocol_document_sha256=sha(ROOT/'CPSF_V23_SCALE_ABLATION_PROTOCOL_20261001.md'))
    save(ROOT/'protocol.json',old)


if __name__=='__main__':main()
