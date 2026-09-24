"""Shared pytest configuration for the open_webui backend unit tests.

`WEBUI_SECRET_KEY` is a hard requirement when authentication is enabled (see
`open_webui/env.py`), so tests that import backend modules must set it before
any `open_webui.*` import happens. Setting it here — at conftest import time,
which runs before test modules are imported — keeps the requirement satisfied
without leaking the value into product code.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault('WEBUI_SECRET_KEY', 'test-secret-key-for-unit-tests')

# Ensure the `backend/` directory is importable as the package root when pytest
# is invoked from the repository root.
BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
