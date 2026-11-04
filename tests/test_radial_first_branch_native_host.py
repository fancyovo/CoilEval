"""Compile the shared host/device radial guard against independent root oracles."""
from pathlib import Path
import json
import os
import shutil
import subprocess

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def test_native_first_branch_and_frozen_rays(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++17 compiler unavailable")
    executable = tmp_path / ("radial-root.exe" if os.name == "nt" else "radial-root")
    build = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-ffp-contract=off",
         "-I" + str(PACKAGE / "native/include"),
         str(PACKAGE / "tests/radial_first_branch_test.cpp"), "-o", str(executable)],
        capture_output=True, text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    fixture = json.loads((PACKAGE / "tests/fixtures/radial_first_branch.json").read_text())
    rays = fixture["rays"]
    inputs = [f"{len(rays)} {len(rays[0]['polynomial']) - 1} {fixture['level']}"]
    inputs += [" ".join(map(repr, ray["polynomial"] + [ray["expected_first_root"]]))
               for ray in rays]
    run = subprocess.run([str(executable)], input="\n".join(inputs),
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "first-branch host regression passed: 12 frozen rays" in run.stdout
