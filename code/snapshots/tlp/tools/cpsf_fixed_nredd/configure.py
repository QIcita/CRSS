import json
import numpy as np
from tools.spatial_grouped_split.build_tlp_spatial_grouped_split import Block,parse_origin,read_shape_xyz,count_swc_nodes,build_groups,solve_assignment,positive_volume_overlap,SPLITS
from .common import *

def main():
    assert not (ROOT/'protocol.json').exists()
    report=json.loads((TED/'construction_report.json').read_text())
    old=json.loads((ROOT/'tlp_spatial_manifest_reference.json').read_text())
    assignments={name.removesuffix('.swc'):('val' if split=='validation' else split) for split,d in old['splits'].items() for name in d['filenames']}
    allrows={};audits={}
    for task in TASKS:
        raw=[];blocks=[]
        for source in ('train','test'):
            if task=='ted':
                names=[Path(r['block']).stem for r in report['splits'][source]['rows']]
                root=NREDD/'data/error_detect_data'/source
            else:
                root=NREDD/'data/link_pred_data'/source
                names=[p.stem for p in sorted((root/'raw_data').glob('*.swc'))]
            for name in sorted(names):
                swc=root/('labeled_data' if task=='ted' else 'raw_data')/(name+'.swc')
                image=root/'image'/(name+'.tif');shape=read_shape_xyz(image)
                b=Block(name+'.swc',name,source,parse_origin(name),shape,count_swc_nodes(swc));blocks.append(b)
                r=dict(key=name,official_source_split=source,swc=str(swc),swc_sha256=sha(swc),image=str(image),image_sha256=sha(image),origin=list(b.origin),shape_xyz=list(shape),nodes=b.node_count)
                if task=='ted':
                    cache=TED/source/(name+'.pt');meta=root/'event_cloud_cpsf_geometry_v2d_segment_shell'/(name+'.npz')
                    expected=next(x for x in report['splits'][source]['rows'] if Path(x['block']).stem==name)
                    assert sha(cache)==expected['sha256']
                    r.update(cache=str(cache),cache_sha256=sha(cache),metadata=str(meta),metadata_sha256=sha(meta),events=expected['events'])
                else:
                    label=root/'label'/(name+'.txt');edge=TLP/'edge_all'/(name+'.npz');bridge=TLP/'bridge_all'/(name+'.npz')
                    assert label.is_file()
                    r.update(label=str(label),label_sha256=sha(label),edge=str(edge),edge_sha256=sha(edge),bridge=str(bridge),bridge_sha256=sha(bridge))
                raw.append(r)
        groups,overlaps=build_groups(blocks)
        if task=='ted':
            solution=solve_assignment(groups,[.6,.2,.2]);group_splits=[SPLITS[i].replace('validation','val') for i in solution['assignment']]
            for group,split in zip(groups,group_splits):
                for i in group['members']:raw[i]['split']=split;raw[i]['spatial_group']=group['id']
        else:
            for group in groups:
                selected={assignments[raw[i]['key']] for i in group['members']};assert len(selected)==1
                for i in group['members']:raw[i]['split']=next(iter(selected));raw[i]['spatial_group']=group['id']
        assert len({r['key'] for r in raw})==len(raw)
        assert all(raw[a]['split']==raw[b]['split'] for a,b in overlaps)
        for field in ('swc_sha256','image_sha256'):
            values={}
            for r in raw:values.setdefault(r[field],set()).add(r['split'])
            assert all(len(x)==1 for x in values.values()),('cross split duplicate',task,field)
        allrows[task]=raw
        audits[task]=dict(blocks=len(raw),spatial_groups=len(groups),overlap_pairs=len(overlaps),cross_split_overlap_pairs=0,complete_file_cross_split_duplicates=0,
            splits={s:dict(blocks=sum(r['split']==s for r in raw),samples=sum((r.get('events',1)) for r in raw if r['split']==s)) for s in ('train','val','test')})
    save(ROOT/'split_manifest.json',dict(records=allrows,audit=audits,assignment_uses_labels=False,tlp_reference_sha256=sha(ROOT/'tlp_spatial_manifest_reference.json'),
        note='Spatially disjoint within each independently trained task. TED freshly grouped; TLP frozen 156/62/55 grouping. Heldout sets have historical project exposure, not newly unseen data.'))
    jobs=[dict(task=t,arm=a) for a in ARMS for t in TASKS if not(t=='ted' and a in ('thumbnail','uniform_budget'))]
    save(ROOT/'protocol.json',dict(version='Retained fixed core on spatially separated NREDD; first task adapters',seed=42,jobs=jobs,
        selection='fixed final epoch; no validation checkpoint selection',evaluation='development val only; test reserved and not prepared in this run',
        tasks={'ted':dict(epochs=30,batch=64,lr=.001,weight_decay=.0005,scheduler='1/(epoch+1)',loss='alpha5 weighted two-logit BCE, summed gradients; step on final partial batch'),
               'tlp':dict(epochs=80,batch=1,lr=.001,weight_decay=.001,scheduler='StepLR20 gamma .1',loss='full adjacency adaptive positive-weight BCE')},
        primary={'ted':'positive-event F1, sigmoid-round-argmax','tlp':'all unobserved unordered pairs Missing-edge macro AP'},
        geometry={'ted':'native SWC /2; radius2 and voxel.5 equal native radius4 voxel1; micrometer calibration unverified; original 1024-point local event',
                  'tlp':'observed SWC with V2E spacing[1,1,2], no per-cell normalization; radius2 voxel.5; cached real surface union, not full original mesh'},
        hierarchy='minimum-four coarse groups on both tasks, geometry only',
        tlp_decoder='endpoint XYZ query pooling from region embeddings; symmetric sum/absolute difference plus pooled whole-graph context; all node pairs scored, no target candidates',
        point_control='no skeleton feature/edge/anchor enters encoder; task query XYZ allowed in readout; surface support inherited from observed-skeleton-guided cohort',
        clip=None,tail_change='TED final partial batch stepped consistently for all arms; no claim of exact old optimizer trajectory',
        split_sha256=sha(ROOT/'split_manifest.json'),source_sha256={str(f.relative_to(ROOT)):sha(f) for f in (ROOT/'tools').rglob('*.py')},
        document_sha256=sha(ROOT/'CPSF_FIXED_CORE_NREDD_PROTOCOL_20261006.md')))
    print(json.dumps(audits),flush=True)
if __name__=='__main__':main()
