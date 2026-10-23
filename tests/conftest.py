"""Run the standalone package tests without installing the research repository."""

from pathlib import Path
import os
import sys


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('COIL_EVAL_TEST_INSTALLED') != '1':
    sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
