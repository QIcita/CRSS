"""The frozen dense loader with the parent's fine-to-coarse geometry adapter."""
from tools.cpsf_r38_dense.data import Cache as OriginalCache, DENSE_FIELDS, move
from tools.cpsf_r38_dense.data import parent_load as original_parent_load


def native_row(row):
    if row['task'] == 'flywire_fine':
        return dict(row, task='flywire_coarse', target=row['coarse_target'])
    return row


def parent_load(row):
    return original_parent_load(native_row(row))


class Cache(OriginalCache):
    def get(self, row):
        return super().get(native_row(row))
