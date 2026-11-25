"""Repository-level paths.

Hydra configs live in ``configs/`` at the repository root (next to ``src/``), so
the package must be used from a source checkout (``pip install -e .`` or
``PYTHONPATH=src``). Set ``LWM_CONFIG_DIR`` to use a different config root.
"""

import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_DIR = pathlib.Path(os.environ.get("LWM_CONFIG_DIR", REPO_ROOT / "configs"))

# Base directory for relative asset paths in task configs (the original code used
# the repository root). Override with PROJECT_DIR.
PROJECT_DIR = os.environ.get("PROJECT_DIR", str(REPO_ROOT))
