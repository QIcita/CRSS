"""Assemble the untrained PI candidate alongside the unchanged C factory."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path, nargs='?')
    args = parser.parse_args()
    release = Path(__file__).resolve().parent
    parent = release.parent / 'c_smm_cmm_rcm'
    manifest = json.loads((release / 'source_map.json').read_text())
    for relative, expected in manifest['parent_files'].items():
        if hashlib.sha256((parent / relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Parent release changed: ' + relative)
    sources = {}
    for relative, expected in manifest['files'].items():
        value = (release / 'sources' / relative).read_bytes()
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError('Candidate source changed: ' + relative)
        ast.parse(value, filename=relative)
        sources[relative] = value
    command = [sys.executable, str(parent / 'materialize.py')]
    if args.destination is not None:
        command.append(str(args.destination.expanduser().resolve()))
    subprocess.run(command, check=True)
    if args.destination is None:
        print('Verified', len(sources), 'candidate sources; default C release unchanged')
        return
    destination = args.destination.expanduser().resolve()
    # Keep the parent model.py and its default build_model API byte-identical.
    for relative, value in sources.items():
        target = destination / 'CPSF_C_TRANSFER_20261031_A2' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
    (destination / 'candidate.py').write_text('''"""Untrained PI-basis-residual candidate; prepared C inputs required."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "CPSF_C_TRANSFER_20261031_A2"))


def build_model(task, initial=None):
    from tools.cpsf_pi_basis_residual.model import make
    return make(task, (.5 if task == "h01_global" else 1.) if initial is None else initial)
''')
    (destination / 'CANDIDATE_SOURCES.json').write_text(json.dumps(dict(
        variant='pi_basis_residual', trained=False, performance_evaluated=False,
        source_map_sha256=hashlib.sha256((release / 'source_map.json').read_bytes()).hexdigest(),
        files=manifest['files'], parent_files=manifest['parent_files']), indent=2) + '\n')
    print('Candidate factory:', destination / 'candidate.py')


if __name__ == '__main__':
    main()
