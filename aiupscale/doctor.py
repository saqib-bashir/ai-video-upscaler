"""Explain why the GPU is (not) used:  aiupscale doctor"""

from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field

FIX_HINT_WIN = 'double-click "fix_gpu.bat" in the aiupscale folder'
FIX_CMD = ("python -m pip uninstall -y torch torchvision && python -m pip install torch "
           "torchvision --index-url https://download.pytorch.org/whl/cu128")


@dataclass
class Diagnosis:
    ok: bool                 # a GPU will be used
    device: str              # cuda / mps / cpu
    summary: str             # one line for the GUI header
    details: list[str] = field(default_factory=list)


def _nvidia_smi() -> tuple[bool, str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, "nvidia-smi not found (no NVIDIA driver installed, or GPU switched off)"
    try:
        r = subprocess.run([exe, "--query-gpu=name,driver_version,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    except Exception as e:  # noqa: BLE001
        return False, f"nvidia-smi failed: {e}"
    out = (r.stdout or r.stderr).strip()
    return r.returncode == 0 and bool(r.stdout.strip()), out or f"nvidia-smi exit {r.returncode}"


def diagnose() -> Diagnosis:
    import torch

    cuda_build = torch.version.cuda  # None => CPU-only wheel
    smi_ok, smi = _nvidia_smi()
    d = [
        f"python      {sys.executable}",
        f"torch       {torch.__version__}  (CUDA build: {cuda_build or 'NO - CPU-only build'})",
        f"nvidia-smi  {smi}",
    ]

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        d.append(f"cuda        OK -> {p.name}, {p.total_memory / 2**30:.1f} GB")
        return Diagnosis(True, "cuda",
                         f"GPU: {p.name} ({p.total_memory / 2**30:.0f} GB) - ready", d)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Diagnosis(True, "mps", "GPU: Apple Silicon (MPS) - ready", d)
    if sys.platform == "darwin":
        return Diagnosis(False, "cpu", "No GPU backend available - running on CPU (slow)", d)

    fix = FIX_HINT_WIN if sys.platform == "win32" else f"run: {FIX_CMD}"
    if cuda_build is None:
        if not smi_ok:
            d.append("cause       CPU-only PyTorch, and no working NVIDIA driver either")
            d.append("fix         1) install/enable the NVIDIA driver  2) " + fix)
        else:
            d.append("cause       PyTorch was installed WITHOUT CUDA support (the plain PyPI "
                     "build is CPU-only on Windows)")
            d.append(f"fix         {fix}")
        return Diagnosis(False, "cpu",
                         "No GPU: PyTorch is the CPU-only build - " + fix, d)
    if not smi_ok:
        d.append("cause       PyTorch has CUDA, but the NVIDIA GPU/driver is not reachable. "
                 "On gaming laptops check that Eco / iGPU-only / 'Integrated' GPU mode is "
                 "OFF (Armoury Crate, Lenovo Vantage, MSI Center, ...), then reboot. "
                 "Otherwise install the latest NVIDIA driver.")
        return Diagnosis(False, "cpu",
                         "No GPU: NVIDIA driver not reachable (GPU off / Eco mode / no driver)", d)
    d.append(f"cause       NVIDIA driver works but PyTorch (CUDA {cuda_build}) can't use it - "
             "usually the driver is too old for this CUDA build. Update the NVIDIA driver, "
             f"or reinstall PyTorch: {fix}")
    return Diagnosis(False, "cpu", "No GPU: NVIDIA driver too old for this PyTorch build", d)


def main() -> int:
    dg = diagnose()
    print("\n".join(dg.details))
    print()
    print(dg.summary)
    try:
        from .video import encoder_works

        hw = [e for e in ("hevc_nvenc", "av1_nvenc") if encoder_works(e)]
        print("nvenc       " + (", ".join(hw) if hw else "not available (CPU encoder will be used)"))
    except Exception as e:  # noqa: BLE001
        print(f"ffmpeg      problem: {e}")
    return 0 if dg.ok else 1
