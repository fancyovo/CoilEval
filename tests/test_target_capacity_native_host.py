"""Target geometry arithmetic and missed toroidal folds without a CUDA runtime."""
from pathlib import Path
import os
import shutil
import subprocess

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def test_section_polynomial_equivalence_and_toroidal_capacity_aliasing(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++17 compiler unavailable")
    executable = tmp_path / ("capacity-guard.exe" if os.name == "nt" else "capacity-guard")
    build = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-ffp-contract=off",
         "-I" + str(PACKAGE / "native/include"),
         str(PACKAGE / "tests/capacity_guard_test.cpp"), "-o", str(executable)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(executable)], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "section polynomial equivalence and toroidal capacity aliasing passed" in run.stdout
