from .common import *
def main():
    assert not (ROOT/'protocol.json').exists()
    p=json.loads((PARENT/'protocol.json').read_text());assert json.loads((PARENT/'pipeline_status.json').read_text())['state']=='complete'
    for name,h in p['source_sha256'].items():assert sha(PARENT/name)==h and sha(ROOT/name)==h
    refs={};files={str(PARENT/'protocol.json'):sha(PARENT/'protocol.json')}
    files[str(CE/'diagnostics/geometry_profiles.jsonl')]=sha(CE/'diagnostics/geometry_profiles.jsonl')
    for task in TASKS:
        path=PARENT/'data'/task/'manifest.json';files[str(path)]=sha(path)
        for arm in ('full','mean_only','no_fusion','point','skeleton'):
            if task=='flywire_coarse' and arm in ('full','mean_only'):
                path=CE/'results'/('coarse_'+arm)
            else:path=PARENT/'results'/(task+'_'+('fixed_residual' if arm=='full' else arm))
            m=json.loads((path/'metrics.json').read_text());assert m['seed']==42 and not m['smoke'] and not m['test_used']
            refs[task+'/'+arm]=dict(path=str(path),val=m['val'],hashes={fn:sha(path/fn) for fn in ('initial.pt','final.pt','metrics.json')})
            prediction=path/('val_features.npz' if task=='flywire_coarse' and arm in ('full','mean_only') else 'val_predictions.npz')
            files[str(prediction)]=sha(prediction)
    jobs=[dict(task=t,arm=a) for t in TASKS for a in ARMS if a!='full' and not(t=='h01_local' and a in ('thumbnail','uniform_budget'))]
    p.update(version='Retained fixed residual: encoding, resolution and mechanism completion; no gate',
        jobs=jobs,references=refs,parent_files=files,
        model_family='V23 fixed residual with dataset-specific frozen hierarchy; gate-free code refactor',
        source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},
        document_sha256=sha(ROOT/'CPSF_FIXED_CORE_PROTOCOL_20261005.md'))
    for k in ('gate','gate_initialization'):p.pop(k,None)
    save(ROOT/'protocol.json',p)
if __name__=='__main__':main()
