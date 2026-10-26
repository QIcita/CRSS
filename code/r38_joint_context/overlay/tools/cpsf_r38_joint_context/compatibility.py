"""Explicit old/new tensor namespace migration and optimizer-order compatibility."""
from collections import OrderedDict


PREFIXES = (
    ('patch', 'cmm.morphology'),
    ('native_point', 'cmm.context.hierarchy'),
    ('raw_refine', 'cmm.context.neighborhood'),
    ('cmm_pair_fusion', 'cmm.context.interaction'),
    ('path_position', 'rcm.geometry.path_position'),
    ('relation', 'rcm.geometry.relation'),
    ('position', 'rcm.geometry.position'),
    ('scale', 'rcm.geometry.scale'),
    ('region_message', 'rcm.message'),
    ('joint', 'rcm.joint'),
)


def rename(name, to_legacy=False):
    for old, new in PREFIXES:
        source, target = (new, old) if to_legacy else (old, new)
        if name == source or name.startswith(source + '.'):
            return target + name[len(source):]
    return name


def convert_state_dict(state, to_legacy=False):
    """Only rename keys; preserve tensors, metadata, and strict-load behavior."""
    converted = OrderedDict()
    for name, value in state.items():
        mapped = rename(name, to_legacy)
        if mapped in converted:
            raise ValueError('Mixed/colliding checkpoint namespaces: ' + mapped)
        converted[mapped] = value
    if hasattr(state, '_metadata'):
        converted._metadata = OrderedDict()
        for name, value in state._metadata.items():
            if to_legacy and name in ('cmm', 'cmm.context', 'rcm', 'rcm.geometry'):
                continue
            mapped = rename(name, to_legacy)
            if mapped in converted._metadata:
                raise ValueError('Mixed checkpoint metadata namespaces: ' + mapped)
            converted._metadata[mapped] = value
    return converted


class CompatibilityMixin:
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        pairs = list(super().named_parameters(prefix, recurse, remove_duplicate))
        if recurse:
            stem = prefix + '.' if prefix else ''
            order = {stem + n: i for i, n in enumerate(self._r38_parameter_order)}
            pairs.sort(key=lambda pair: order.get(pair[0], len(order)))
        yield from pairs

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(convert_state_dict(state_dict), strict=strict, assign=assign)

    def legacy_state_dict(self):
        state = convert_state_dict(self.state_dict(), to_legacy=True)
        # Restore old serialization order as well as names for reproducibility.
        ordered = OrderedDict((name, state[name]) for name in self._r38_state_order if name in state)
        ordered.update((name, value) for name, value in state.items() if name not in ordered)
        if hasattr(state, '_metadata'):
            ordered._metadata = state._metadata
        return ordered
