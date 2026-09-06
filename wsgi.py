"""WSGI entrypoint for Gunicorn deployments."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent / "fake-review-monitor"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from app import app  # noqa: E402, F401
