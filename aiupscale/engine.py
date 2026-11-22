"""GPU/CPU inference with tiling, size-requirement padding and OOM fallback."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

log = logging.getLogger("aiupscale")


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _is_oom(e: BaseException) -> bool:
    if isinstance(e, getattr(torch, "OutOfMemoryError", ())):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


class Upscaler:
    """Wraps a spandrel model. Input/output: float32 tensors [1,3,H,W] in 0..1."""

    def __init__(self, desc, device: torch.device, half: bool = True, tile: int = 0,
                 tile_pad: int = 32, name: str = ""):
        self.name = name
        self.scale = int(desc.scale)
        self.device = device
        req = desc.size_requirements
        self.multiple = max(1, int(req.multiple_of))
        self.minimum = max(1, int(req.minimum))
        half = half and device.type in ("cuda", "mps") and desc.supports_half
        self.dtype = torch.float16 if half else torch.float32
        self.model = desc.model.to(device=device, dtype=self.dtype).eval()
        self.params = sum(p.numel() for p in self.model.parameters())
        self.tile = int(tile)  # 0 = auto (VRAM-aware, then OOM fallback)
        self._auto = self.tile <= 0
        self.tile_pad = int(tile_pad)
        if device.type == "cuda":
            torch.backends.cudnn.benchmark = True  # frame size is constant

    def _auto_tile(self, x: torch.Tensor) -> None:
        """Pick a tile size from free VRAM so the first frame does not OOM.

        Rough budget: heavy RRDB nets (x4plus/x2plus, ~17M params) need about
        1.2 GB per output megapixel in fp16, compact nets about 0.25 GB.
        """
        self._auto = False
        if self.device.type != "cuda":
            return
        free, _ = torch.cuda.mem_get_info(self.device)
        per_mp = 1.2e9 if self.params > 5e6 else 0.25e9
        if self.dtype == torch.float32:
            per_mp *= 2
        budget_mp = max(0.25, (free * 0.7) / per_mp)
        h, w = x.shape[-2:]
        out_mp = h * w * self.scale**2 / 1e6
        if out_mp <= budget_mp:
            return  # whole frame fits
        tile_in = int((budget_mp * 1e6) ** 0.5 / self.scale) // 8 * 8
        self.tile = max(64, tile_in)
        log.info("[%s] auto tile %dpx (%.1f GB free VRAM)", self.name, self.tile, free / 1e9)

    # -- public -------------------------------------------------------------
    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self._auto:
            self._auto_tile(x)
        while True:
            try:
                if self.tile <= 0 or max(x.shape[-2:]) <= self.tile:
                    return self._forward(x)
                return self._tiled(x)
            except Exception as e:  # noqa: BLE001 - re-raised unless OOM
                if not _is_oom(e):
                    raise
                self._empty_cache()
                cur = self.tile if self.tile > 0 else max(x.shape[-2:])
                new = (cur // 2) // 8 * 8
                if new < 64:
                    raise RuntimeError(
                        "Out of memory even with 64px tiles. Try --device cpu or a smaller model."
                    ) from e
                log.warning("[%s] out of memory -> tiling at %dpx", self.name, new)
                self.tile = new

    # -- internals ----------------------------------------------------------
    def _empty_cache(self):
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        ph = max(self.minimum, -(-h // self.multiple) * self.multiple) - h
        pw = max(self.minimum, -(-w // self.multiple) * self.multiple) - w
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        y = self.model(x.to(self.dtype))
        y = y[..., : h * self.scale, : w * self.scale]
        return y.float().clamp_(0, 1)

    def _tiled(self, x: torch.Tensor) -> torch.Tensor:
        """Real-ESRGAN style tiling: run each tile with context padding, keep the centre.

        With enough padding the seams are invisible because every kept output
        pixel saw (practically) the same receptive field as a full-frame run.
        """
        _, c, h, w = x.shape
        s, t, p = self.scale, self.tile, self.tile_pad
        out = torch.empty((1, c, h * s, w * s), device=x.device, dtype=torch.float32)
        for y0 in range(0, h, t):
            for x0 in range(0, w, t):
                y1, x1 = min(y0 + t, h), min(x0 + t, w)
                py0, px0 = max(y0 - p, 0), max(x0 - p, 0)
                py1, px1 = min(y1 + p, h), min(x1 + p, w)
                tile_out = self._forward(x[..., py0:py1, px0:px1])
                oy, ox = (y0 - py0) * s, (x0 - px0) * s
                out[..., y0 * s : y1 * s, x0 * s : x1 * s] = tile_out[
                    ..., oy : oy + (y1 - y0) * s, ox : ox + (x1 - x0) * s
                ]
        return out
