"""Verify and assemble the C model sources without data or training workers."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import tokenize


def relocate_paths(source, old, new):
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.STRING and old in token.string:
            value = ast.literal_eval(token.string)
            if isinstance(value, str) and old in value:
                token = token._replace(string=repr(value.replace(old, new)))
        tokens.append(token)
    return tokenize.untokenize(tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?')
    args = parser.parse_args()
    release = Path(__file__).resolve().parent
    repo = release.parents[1]
    manifest = json.loads((release / 'source_map.json').read_text())
    sources = {}
    for relative, row in manifest['files'].items():
        value = (repo / row['repository_path']).read_bytes()
        if hashlib.sha256(value).hexdigest() != row['sha256']:
            raise ValueError('Source hash mismatch: ' + relative)
        if relative.endswith('.py'):
            ast.parse(value, filename=relative)
        sources[relative] = value
    if args.destination is None:
        print('Verified', len(sources), 'frozen source and notice files')
        return
    destination = args.destination.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    assembled = {}
    for relative, value in sources.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith('.py'):
            value = relocate_paths(value.decode(), manifest['source_prefix'], str(destination)).encode()
            ast.parse(value, filename=relative)
        target.write_bytes(value)
        assembled[relative] = hashlib.sha256(value).hexdigest()
    entry = '''"""CUDA C model factory; expects prepared geometry, not raw XYZ."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "CPSF_C_TRANSFER_20261031_A2"))


def build_model(task, initial=None):
    """initial is the training target mean (H01) or rate per um (MICrONS)."""
    from tools.cpsf_c_transfer.model import make as transfer_make
    if task.startswith("h01_"):
        from tools.cpsf_h01_progressive_repair.model import make
        return make("regional_scaled_pi", task, .5 if initial is None else initial)
    return transfer_make(task, 1. if initial is None else initial)
'''
    (destination / 'model.py').write_text(entry)
    (destination / 'ASSEMBLED_SOURCES.json').write_text(json.dumps(dict(
        source_map_sha256=hashlib.sha256((release / 'source_map.json').read_bytes()).hexdigest(),
        transformation='Only absolute source-root strings are relocated; frozen repository bytes are unchanged',
        files=assembled), indent=2) + '\n')
    print('Assembled', len(sources), 'model sources at', destination)


if __name__ == '__main__':
    main()
