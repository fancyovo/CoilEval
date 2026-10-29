"""Check optional entry-point requirements before loading a backend."""
from importlib.util import find_spec
import os
import shlex
import subprocess
import sys


_REQUIREMENTS = {
    "cpu": (("torch", "torch>=2.5"), ("scipy", "scipy>=1.10"),
            ("threadpoolctl", "threadpoolctl>=3")),
    "torch": (("torch", "torch>=2.5"),),
}


def require_dependencies(feature):
    """Report all absent direct dependencies; leave broken installs visible."""
    missing = [(name, requirement) for name, requirement in _REQUIREMENTS[feature]
               if find_spec(name) is None]
    if not missing:
        return
    commands = []
    requirements = [requirement for _, requirement in missing]
    if feature == "cpu" and any(name == "torch" for name, _ in missing):
        commands.append([sys.executable, "-m", "pip", "install", "torch>=2.5",
                         "--index-url", "https://download.pytorch.org/whl/cpu"])
        requirements.remove("torch>=2.5")
    if requirements:
        commands.append([sys.executable, "-m", "pip", "install", *requirements])
    quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
    message = (
        f"CoilEval {feature} support requires: {', '.join(name for name, _ in missing)}. "
        "Install these dependencies in the same Python environment:\n  "
        + "\n  ".join(quote(command) for command in commands)
        + f'\nFor a fresh installation, use pip install "./coil_eval-<version>-<tags>.whl[{feature}]". '
        "Replace the filename with your supplied wheel."
    )
    if feature == "torch":
        message += " For GPU evaluation, select the CUDA 12.8 PyTorch build; see docs/installation.md."
    raise ModuleNotFoundError(message, name=missing[0][0])
