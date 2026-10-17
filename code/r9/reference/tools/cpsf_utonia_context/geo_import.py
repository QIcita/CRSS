"""Import untouched official GeoTransformer math without unused registration/C++ exports.

This is import plumbing, NOT an implementation of GeoTransformer. A complete,
commit-pinned upstream checkout must be supplied. No weights are loaded.
"""
from pathlib import Path
import importlib
import importlib.util
import sys
import types


def load_official_geotransformer(upstream):
    upstream = Path(upstream).resolve()
    root = upstream / 'geotransformer'
    if not (root / 'modules/geotransformer/geotransformer.py').is_file():
        raise FileNotFoundError(root)
    if 'geotransformer' in sys.modules:
        old = getattr(sys.modules['geotransformer'], '__path__', [])
        if str(root) not in list(old):
            raise RuntimeError('A different geotransformer package is already loaded')
    def package(name, path):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = [str(path)]
            mod.__package__ = name
            sys.modules[name] = mod
        return sys.modules[name]
    package('geotransformer', root)
    package('geotransformer.modules', root / 'modules')
    # Upstream ops.__init__ eagerly imports an unused compiled grid/radius extension.
    ops = package('geotransformer.modules.ops', root / 'modules/ops')
    ops.pairwise_distance = importlib.import_module(
        'geotransformer.modules.ops.pairwise_distance').pairwise_distance
    # Keep official layers and transformer __init__ and every mathematical class intact.
    importlib.import_module('geotransformer.modules.layers')
    importlib.import_module('geotransformer.modules.transformer')
    # Upstream this __init__ eagerly imports registration/procrustes; not our task.
    package('geotransformer.modules.geotransformer', root / 'modules/geotransformer')
    core = importlib.import_module('geotransformer.modules.geotransformer.geotransformer')
    return core.GeometricTransformer
