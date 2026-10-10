"""Make ``imageflow`` and the shared test helpers importable however pytest is launched."""

import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent

for path in (ROOT, TESTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
