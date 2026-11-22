"""Model registry, download (with SHA-256 pinning) and loading via spandrel.

spandrel understands nearly every community super-resolution architecture
(ESRGAN, Real-ESRGAN Compact, SwinIR, HAT, DAT, OmniSR, SPAN, ...), so any
.pth / .safetensors file from https://openmodeldb.info also works when passed
as a path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger("aiupscale")

_RELEASES = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    file: str
    url: str
    sha256: str
    scale: int
    description: str
    dni_partner: str | None = None  # weak-denoise twin used for --denoise blending


MODELS: dict[str, ModelSpec] = {
    m.name: m
    for m in [
        ModelSpec(
            "general",
            "realesr-general-x4v3.pth",
            f"{_RELEASES}/v0.2.5.0/realesr-general-x4v3.pth",
            "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
            4,
            "Real-ESRGAN general x4 v3 (compact, fast). Best default for AI video.",
            dni_partner="general-wdn",
        ),
        ModelSpec(
            "general-wdn",
            "realesr-general-wdn-x4v3.pth",
            f"{_RELEASES}/v0.2.5.0/realesr-general-wdn-x4v3.pth",
            "1641f8c4464b9f097c9fdda5589273713f67cf59f3d909e0bd688f0cee269dca",
            4,
            "Weak-denoise twin of 'general' (keeps grain/texture).",
        ),
        ModelSpec(
            "x4plus",
            "RealESRGAN_x4plus.pth",
            f"{_RELEASES}/v0.1.0/RealESRGAN_x4plus.pth",
            "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
            4,
            "Real-ESRGAN x4plus (RRDB, heavy). Strongest restoration, ~10x slower.",
        ),
        ModelSpec(
            "x2plus",
            "RealESRGAN_x2plus.pth",
            f"{_RELEASES}/v0.2.1/RealESRGAN_x2plus.pth",
            "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb",
            2,
            "Real-ESRGAN x2plus. Used as the second pass for 8K.",
        ),
        ModelSpec(
            "anime-video",
            "realesr-animevideov3.pth",
            f"{_RELEASES}/v0.2.5.0/realesr-animevideov3.pth",
            "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
            4,
            "Real-ESRGAN anime video v3 (tiny, very fast). For anime/cartoon styles.",
        ),
        ModelSpec(
            "anime",
            "RealESRGAN_x4plus_anime_6B.pth",
            f"{_RELEASES}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
            "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
            4,
            "Real-ESRGAN x4plus anime 6B. Sharper anime lines, slower than anime-video.",
        ),
    ]
}


def model_dir() -> Path:
    d = os.environ.get("AIUPSCALE_MODELS")
    path = Path(d) if d else Path.home() / ".cache" / "aiupscale" / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(spec: ModelSpec) -> Path:
    """Return the local path of a registry model, downloading + verifying if needed."""
    dest = model_dir() / spec.file
    if dest.exists():
        return dest
    log.info("Downloading %s ...", spec.url)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(spec.url, timeout=60) as r:
            shutil.copyfileobj(r, out, 1 << 20)
        digest = _sha256(Path(tmp))
        if digest != spec.sha256:
            raise RuntimeError(
                f"Checksum mismatch for {spec.file}: expected {spec.sha256}, got {digest}"
            )
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def _load_state_dict(path: Path) -> dict:
    from spandrel import ModelLoader

    return ModelLoader().load_state_dict_from_file(str(path))


def _unwrap(sd: dict) -> dict:
    for key in ("params_ema", "params", "state_dict"):
        if key in sd and isinstance(sd[key], dict):
            return sd[key]
    return sd


def load_model(name_or_path: str, denoise: float | None = None):
    """Load a model and return a spandrel ImageModelDescriptor (on CPU, eval mode).

    ``denoise`` (0..1) applies to the 'general' model only: weights are
    interpolated between the strong-denoise model (1.0) and its weak-denoise
    twin (0.0), exactly like Real-ESRGAN's ``--denoise_strength``.
    """
    from spandrel import ImageModelDescriptor, ModelLoader

    spec = MODELS.get(name_or_path)
    if spec is None:
        path = Path(name_or_path).expanduser()
        if not path.exists():
            raise SystemExit(
                f"Unknown model '{name_or_path}'. Use one of: {', '.join(MODELS)} "
                "or a path to a .pth/.safetensors file."
            )
        desc = ModelLoader().load_from_file(str(path))
    elif spec.dni_partner and denoise is not None and denoise < 0.999:
        strong = _unwrap(_load_state_dict(ensure_model(spec)))
        weak = _unwrap(_load_state_dict(ensure_model(MODELS[spec.dni_partner])))
        d = max(0.0, min(1.0, float(denoise)))
        mixed = {k: d * strong[k] + (1.0 - d) * weak[k] for k in strong}
        desc = ModelLoader().load_from_state_dict(mixed)
    else:
        desc = ModelLoader().load_from_file(str(ensure_model(spec)))

    if not isinstance(desc, ImageModelDescriptor):
        raise SystemExit(f"'{name_or_path}' is not a single-image super-resolution model.")
    if desc.input_channels != 3 or desc.output_channels != 3:
        raise SystemExit(f"'{name_or_path}' must be an RGB->RGB model.")
    desc.model.eval()
    return desc


def model_scale(name_or_path: str) -> int:
    spec = MODELS.get(name_or_path)
    if spec:
        return spec.scale
    return load_model(name_or_path).scale


def download_all(names: list[str] | None = None) -> None:
    for n in names or list(MODELS):
        p = ensure_model(MODELS[n])
        print(f"{n:12s} {p}")
