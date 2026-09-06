"""Root application entrypoint for Render and local deployments."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add fake-review-monitor to Python path
PACKAGE_DIR = Path(__file__).resolve().parent / "fake-review-monitor"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

# Import the Flask application from the package
from app import app  # noqa: E402, F401

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
