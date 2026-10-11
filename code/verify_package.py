"""Verify all release files using only the Python standard library."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'MANIFEST.sha256.json').read_text())
for name, expected in manifest.items():
    path = root / name
    assert path.is_file(), ('Missing', name)
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == expected, ('Changed', name)
print('Verified', len(manifest), 'release files.')
