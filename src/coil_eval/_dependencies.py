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

_TORCH_INDEX = "https://download.pytorch.org/whl/"
_TORCH_VALIDATED_VERSION = "2.7.1"


def _torch_install_command(build):
    return [sys.executable, "-m", "pip", "install",
            f"torch=={_TORCH_VALIDATED_VERSION}+{build}",
            "--index-url", _TORCH_INDEX + build]


def _format_command(command):
    quote = subprocess.list2cmdline if os.name == "nt" else shlex.join
    return quote(command)


def require_dependencies(feature):
    """Report all absent direct dependencies; leave broken installs visible."""
    missing = [(name, requirement) for name, requirement in _REQUIREMENTS[feature]
               if find_spec(name) is None]
    if not missing:
        return
    commands = []
    requirements = [requirement for _, requirement in missing]
    if "torch>=2.5" in requirements:
        commands.append(_torch_install_command("cpu"))
        requirements.remove("torch>=2.5")
    if requirements:
        commands.append([sys.executable, "-m", "pip", "install", *requirements])
    message = (
        f"CoilEval {feature} support requires: {', '.join(name for name, _ in missing)}. "
        "Install these dependencies in the same Python environment:\n  "
        + "\n  ".join(_format_command(command) for command in commands)
        + f'\nFor a fresh installation, use pip install "./coil_eval-<version>-<tags>.whl[{feature}]". '
        "Replace the filename with your supplied wheel."
    )
    if feature == "torch":
        message += (" For GPU evaluation, select the CUDA 12.8 PyTorch build instead:\n  "
                    + _format_command(_torch_install_command("cu128")))
    raise ModuleNotFoundError(message, name=missing[0][0])


def require_torch_cuda(torch, device):
    """Diagnose CUDA initialization before the first GPU tensor or native call."""
    build = torch.version.cuda
    version = torch.__version__
    if build is None:
        raise RuntimeError(
            f"CoilEval requested {device}, but installed PyTorch {version} has no CUDA build. "
            "Select device='cpu' or install the validated CUDA 12.8 build in this environment:\n  "
            + _format_command(_torch_install_command("cu128"))
        )
    try:
        torch.cuda.init()
    except RuntimeError as error:
        message = (
            f"CoilEval could not initialize {device} with PyTorch {version} (CUDA {build}). "
            f"PyTorch reported: {error}\n"
            "Check NVIDIA driver compatibility and GPU visibility in this process."
        )
        if build != "12.8":
            message += (" The validated CoilEval CUDA 12.8 build can be selected with:\n  "
                        + _format_command(_torch_install_command("cu128")))
        raise RuntimeError(message) from error
    count = torch.cuda.device_count()
    if device.index is not None and device.index >= count:
        raise ValueError(
            f"CoilEval requested {device}, but only {count} CUDA device(s) are visible. "
            "Select a process-local device index after CUDA_VISIBLE_DEVICES masking."
        )
