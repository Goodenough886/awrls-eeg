"""Check the public source inventory; no external dependencies required."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
expected = json.loads((ROOT / 'SHA256SUMS.json').read_text(encoding='utf-8'))
failures = []
for name, sha in expected.items():
    path = ROOT / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        failures.append(name)
if failures:
    raise SystemExit('Missing or changed files: ' + ', '.join(failures))
print(f'Verified {len(expected)} published files.')
