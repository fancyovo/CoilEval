"""The public package must remain usable outside the training repository."""

import ast
from pathlib import Path
import subprocess
import sys


SOURCE = Path(__file__).resolve().parents[1] / "src"


def test_package_has_no_training_imports():
    package = SOURCE / "coil_eval"
    files = list(package.rglob("*.py"))
    assert files, "The standalone coil_eval package is missing"
    forbidden = {"torch", "flow_matching", "local_surface_evaluator"}
    violations = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                if name.split(".", 1)[0] in forbidden:
                    violations.append(f"{path.name}:{node.lineno}: {name}")
    assert violations == []


def test_public_import_works_with_only_its_own_source_directory():
    script = "\n".join(
        [
            "import pathlib, sys",
            "sys.path.insert(0, sys.argv[1])",
            "import coil_eval",
            "source = pathlib.Path(sys.argv[1]).resolve()",
            "assert pathlib.Path(coil_eval.__file__).resolve().is_relative_to(source)",
            "assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)",
            "assert not any(n == 'flow_matching' or n.startswith('flow_matching.') for n in sys.modules)",
            "assert not any(n == 'local_surface_evaluator' or n.startswith('local_surface_evaluator.') for n in sys.modules)",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(SOURCE)],
        cwd=SOURCE.parent,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
