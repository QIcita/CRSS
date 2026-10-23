"""Dense computational neighborhoods need not replace physical evidence support."""
from collections import OrderedDict
import numpy as np
import torch
from .common import AP,read,sha,variant
from tools.cpsf_r28_expansion.data import load_item as parent_load
from tools.cpsf_utonia_context.data import move

DENSE_FIELDS=('point_owner','patch_mass','patches','raw_patch_ids','raw_indices','raw_valid','raw_counts','raw_layout')

def build(row,force_parent=False):
    # Original inputs and AP derived artifact are read-only, with byte provenance.
    for f,h in [('path','sha256'),('feature_path','feature_sha256'),('graph_path','graph_sha256')]:
        assert sha(row[f])==row[h],row[f]
    original=parent_load(row)['g']
    if force_parent:
        original['patch_size']=.5
        return dict(g=original,audit={'forced_parent':True})
    path=AP/'adaptive_cache'/row['task']/(row['key']+'.pt');meta=read(path.with_suffix('.json'))
    assert meta['cache_sha256']==sha(path)
    assert meta['source_sha256']==row['sha256'] and meta['feature_sha256']==row['feature_sha256']
    assert meta['protocol_sha256']==sha(AP/'protocol.json')
    item=torch.load(path,map_location='cpu',weights_only=False);ap=item['g'];kind=variant()
    for k in ('ucoord','ugrid','uinverse','point_weight','s','sf'):
        assert torch.equal(ap[k],original[k]),k
    h=float(ap['patch_size'])
    if kind=='context_control':
        g=dict(ap)
        # Only restore the original nine-channel Utonia input. Its geometry was
        # already identical; all AP10 grouping, statistics and RCM stay intact.
        g['ufeat']=original['ufeat']
        assert all(torch.equal(g[k],ap[k]) for k in ('pf','pr','pj','pw','phi','pos','surface_mass'))
    elif kind=='support_preserved':
        g=dict(original);g['patch_size']=.5
        if h!=.5:
            # Fine physical tokens remain for statistics and graph routing.
            # Only official SPoTr's within-neighborhood aggregation sees AP10.
            g.update({'dense_'+k:ap[k] for k in DENSE_FIELDS})
            # Preserve the official adapter's physical unit (0.5 um). Dividing
            # by the adaptive window size would erase its actual metric extent.
            g['dense_pf']=ap['pf']*(h/.5)
            g['dense_raw_patch_pos']=ap['raw_patch_pos']*(h/.5)
            g['dense_size']=h
        for k in original:
            if isinstance(original[k],torch.Tensor):assert torch.equal(g[k],original[k]),k
    else:raise ValueError(kind)
    assert not {'target','label','split','observed','fine_target','coarse_target'}.intersection(g)
    return dict(g=g,audit=dict(item['audit'],repair=kind,utonia_input_exact=True,
        fine_support_exact=kind=='support_preserved',neighborhood_size=h,
        support_scale=float(g['patch_size']),geometry_unit=.5))

class Cache:
    def __init__(self,force_parent=False):self.items=OrderedDict();self.size=0;self.force_parent=force_parent
    def get(self,row):
        key=(row['task'],row['key'])
        if key in self.items:
            v,n=self.items.pop(key);self.items[key]=(v,n);return v
        v=build(row,self.force_parent)
        n=sum(t.numel()*t.element_size() for t in v['g'].values() if isinstance(t,torch.Tensor))
        while self.items and self.size+n>768*2**20:
            _,(_,size)=self.items.popitem(last=False);self.size-=size
        if n<768*2**20:self.items[key]=(v,n);self.size+=n
        return v
