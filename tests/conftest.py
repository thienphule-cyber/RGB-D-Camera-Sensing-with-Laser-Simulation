"""Pytest configuration shared by all tests."""

import os
import sys

# Headless Linux (e.g. WSL2) needs EGL for offscreen rendering. This must be set
# before `import mujoco`, so it lives in conftest.py, which pytest loads first.
# macOS and Windows use their default OpenGL backend.
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")