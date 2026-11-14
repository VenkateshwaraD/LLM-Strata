"""Shared sys.path setup for the test suite.

`unittest discover -s tests` puts `tests/` itself on sys.path, so test modules
`import _bootstrap` to expose the repo root (for `core.*`) and the attack stage
directory (for the `agent_security` package).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for path in (REPO_ROOT / "layers" / "attack", REPO_ROOT):
    resolved = str(path)
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
