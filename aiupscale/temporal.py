"""Motion-compensated temporal stabilisation (anti-flicker / anti-shimmer).

Single-image GAN upscalers hallucinate texture independently per frame, which
shows up as "boiling" skin, hair, foliage and backgrounds - very visible on AI
generated footage. We fix that with a recursive, motion-compensated blend:

1. Optical flow (OpenCV DIS) is estimated on the cheap *low-res* frames.
2. The previous *upscaled* output is warped onto the current frame on the GPU.
3. A per-pixel confidence is derived from the low-res warping error, so
   occlusions, new content and flow failures fall back to the fresh frame
   (no ghosting).
4. A second high-res check rejects the blend where the warped history and
   the fresh frame disagree by more than shimmer-sized amounts (sub-pixel
   flow error on real edges), which keeps moving detail sharp.
5. out = sr + strength * confidence * (warped_prev_out - sr)

Hard scene cuts reset the history.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class TemporalStabilizer:
    def __init__(self, strength: float, device: torch.device, scene_cut: float = 0.12,
                 tolerance: float = 0.035, hr_tolerance: float = 0.08):
        self.strength = float(max(0.0, min(0.95, strength)))
        self.device = device
        self.scene_cut = scene_cut
        self.tolerance = tolerance
        self.hr_tolerance = hr_tolerance
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._grid_cache: dict[tuple[int, int], torch.Tensor] = {}
        self.reset()

    @property
    def enabled(self) -> bool:
        return self.strength > 0

    def reset(self):
        self.prev_lr: np.ndarray | None = None
        self.prev_gray: np.ndarray | None = None
        self.prev_out: torch.Tensor | None = None

    def _base_grid(self, h: int, w: int) -> torch.Tensor:
        key = (h, w)
        if key not in self._grid_cache:
            ys, xs = torch.meshgrid(
                torch.arange(h, device=self.device, dtype=torch.float32),
                torch.arange(w, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            self._grid_cache = {key: torch.stack((xs, ys), dim=-1)[None]}  # [1,h,w,2]
        return self._grid_cache[key]

    @torch.inference_mode()
    def __call__(self, lr: np.ndarray, sr: torch.Tensor) -> torch.Tensor:
        """lr: HxWx3 uint8 RGB (model input). sr: [1,3,h,w] float32 upscaled frame."""
        if not self.enabled:
            return sr
        gray = cv2.cvtColor(lr, cv2.COLOR_RGB2GRAY)
        prev_out = self.prev_out
        if (
            prev_out is None
            or prev_out.shape != sr.shape
            or self.prev_gray.shape != gray.shape
            or np.mean(cv2.absdiff(gray, self.prev_gray)) / 255.0 > self.scene_cut
        ):
            self.prev_lr, self.prev_gray, self.prev_out = lr, gray, sr
            return sr

        # flow such that cur(x) ~= prev(x + flow(x))
        flow = self.dis.calc(gray, self.prev_gray, None)
        H, W = gray.shape
        gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        warped_lr = cv2.remap(self.prev_lr, gx + flow[..., 0], gy + flow[..., 1],
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        err = cv2.absdiff(lr, warped_lr).astype(np.float32).max(axis=2) / 255.0
        err = cv2.dilate(err, np.ones((3, 3), np.uint8))
        conf = np.exp(-((err / self.tolerance) ** 2))
        conf = cv2.GaussianBlur(conf, (5, 5), 0)

        h, w = sr.shape[-2:]
        sx, sy = w / W, h / H
        flow_t = torch.from_numpy(flow).to(self.device).permute(2, 0, 1)[None]
        flow_t = F.interpolate(flow_t, size=(h, w), mode="bilinear", align_corners=False)
        grid = self._base_grid(h, w).clone()
        grid[..., 0] += flow_t[:, 0] * sx
        grid[..., 1] += flow_t[:, 1] * sy
        grid[..., 0] = grid[..., 0] * (2.0 / max(w - 1, 1)) - 1.0
        grid[..., 1] = grid[..., 1] * (2.0 / max(h - 1, 1)) - 1.0
        warped = F.grid_sample(prev_out, grid, mode="bicubic", padding_mode="border",
                               align_corners=True)
        conf_t = torch.from_numpy(conf).to(self.device)[None, None]
        conf_t = F.interpolate(conf_t, size=(h, w), mode="bilinear", align_corners=False)
        # High-res check: GAN shimmer is low-amplitude, misalignment on real edges is not.
        # Reject the blend where warped history and the new frame disagree strongly.
        hr_err = F.avg_pool2d((warped - sr).abs().amax(dim=1, keepdim=True), 3, 1, 1)
        conf_t = conf_t * torch.exp(-((hr_err / self.hr_tolerance) ** 2))

        out = torch.lerp(sr, warped, conf_t * self.strength)
        self.prev_lr, self.prev_gray, self.prev_out = lr, gray, out
        return out
