"""Verify frozen model and repository-export manifests (standard library only)."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for base, manifest in [
    (root / 'code', root / 'code/MANIFEST.sha256.json'),
    (root, root / 'provenance/files.sha256.json'),
]:
    records = json.loads(manifest.read_text())
    for name, expected in records.items():
        path = base / name
        if not path.is_file():
            raise SystemExit(f'Missing: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise SystemExit(f'Hash mismatch: {name}')
    print(f'Verified {len(records)} entries: {manifest.relative_to(root)}')
