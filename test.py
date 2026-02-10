import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

print("Python:", sys.version)
print("Python executable:", sys.executable)
print("Python prefix:", sys.prefix)
print("Current working dir:", os.getcwd())
print("OS:", platform.platform())
print("Machine:", platform.machine())
print("Processor:", platform.processor())
print("Python build:", platform.python_build())
print("Python compiler:", platform.python_compiler())

cuda_env = {
    "CUDA_HOME": os.environ.get("CUDA_HOME"),
    "CUDA_PATH": os.environ.get("CUDA_PATH"),
    "NVIDIA_VISIBLE_DEVICES": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
}
print("CUDA env:", cuda_env)

path_entries = os.environ.get("PATH", "").split(os.pathsep)
print("PATH entries (first 10):", path_entries[:10])

def _run_cmd(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return result.stdout.strip() or result.stderr.strip()
    except Exception as e:
        return f"error: {e}"

python_cmd = shutil.which("python") or sys.executable
pip_cmd = shutil.which("pip")
print("python in PATH:", python_cmd)
print("pip in PATH:", pip_cmd)
if pip_cmd:
    print("pip --version:", _run_cmd([pip_cmd, "--version"]))

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA version (torch):", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
        print("GPU capability:", torch.cuda.get_device_capability(0))
    else:
        print("CUDA version (torch):", torch.version.cuda)
except Exception as e:
    print("PyTorch error:", e)

nvsmicmd = shutil.which("nvidia-smi")
print("nvidia-smi in PATH:", nvsmicmd)
if nvsmicmd:
    print("nvidia-smi:", _run_cmd([nvsmicmd, "-L"]))

try:
    import numpy as np
    print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy error:", e)

try:
    print("Installed packages (top 20):")
    pkgs = sorted([f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()])
    for item in pkgs[:20]:
        print("  ", item)
except Exception as e:
    print("Package list error:", e)
