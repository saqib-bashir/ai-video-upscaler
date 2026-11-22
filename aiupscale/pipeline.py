"""Planning and running an upscale job."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from . import models as M
from .engine import Upscaler, pick_device
from .temporal import TemporalStabilizer
from .video import FrameReader, FrameWriter, VideoInfo, choose_encoder, mux, probe, rgb_to_yuv420

log = logging.getLogger("aiupscale")

TARGETS = {  # long edge x short edge; orientation follows the input
    "720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440),
    "4k": (3840, 2160), "5k": (5120, 2880), "6k": (6144, 3456), "8k": (7680, 4320),
}

QUALITY = {  # primary model, second-pass model (used only when one pass is not enough)
    "fast": ("general", None),
    "balanced": ("general", "x2plus"),
    "max": ("x4plus", "x2plus"),
}


class Cancelled(Exception):
    """Raised when a job is stopped; finished segments are kept for resume."""


# progress(frames_done, frames_total_estimate, fps)
ProgressFn = Callable[[int, int, float], None]


# --------------------------------------------------------------------------- options
@dataclass
class Options:
    target: str = "4k"
    fit: str = "fit"                 # fit | pad | crop | stretch
    quality: str = "balanced"
    model: str | None = None
    model2: str | None = None
    passes: str = "auto"             # auto | 1 | 2
    denoise: float = 0.5
    deflicker: float = 0.5
    sharpen: float = 0.0
    grain: float = 0.0
    codec: str = "hevc"
    encoder: str | None = None
    crf: int | None = None
    preset: str | None = None
    bit_depth: int = 10
    device: str = "auto"
    fp32: bool = False
    tile: int = 0
    start: float | None = None
    duration: float | None = None
    segment_frames: int = 240
    resume: bool = True
    keep_parts: bool = False
    input_matrix: str = "auto"


@dataclass
class Plan:
    in_w: int
    in_h: int
    out_w: int                       # final canvas
    out_h: int
    content_w: int                   # size the picture is resized to before pad/crop
    content_h: int
    models: list[str] = field(default_factory=list)
    model_scale: int = 1             # product of model scales

    def describe(self) -> str:
        chain = " -> ".join(self.models) if self.models else "(resize only)"
        return (f"{self.in_w}x{self.in_h} --[{chain}, x{self.model_scale}]--> "
                f"{self.in_w * self.model_scale}x{self.in_h * self.model_scale} --resize--> "
                f"{self.content_w}x{self.content_h}"
                + ("" if (self.content_w, self.content_h) == (self.out_w, self.out_h)
                   else f" --{'pad/crop'}--> {self.out_w}x{self.out_h}"))


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)


def target_size(target: str, in_w: int, in_h: int) -> tuple[int, int]:
    t = target.lower()
    if t in TARGETS:
        long_, short = TARGETS[t]
        return (long_, short) if in_w >= in_h else (short, long_)
    if "x" in t:
        w, h = t.split("x")
        return int(w), int(h)
    raise SystemExit(f"Unknown target '{target}'. Use {', '.join(TARGETS)} or WIDTHxHEIGHT.")


def make_plan(opt: Options, in_w: int, in_h: int) -> Plan:
    tw, th = target_size(opt.target, in_w, in_h)
    near_same_aspect = abs((in_w / in_h) / (tw / th) - 1) < 0.01  # e.g. 854x480 vs 16:9
    if opt.fit == "stretch" or near_same_aspect:
        cw, ch, ow, oh = tw, th, tw, th
    else:
        s = (min if opt.fit in ("fit", "pad") else max)(tw / in_w, th / in_h)
        cw, ch = _even(in_w * s), _even(in_h * s)
        if opt.fit == "fit":
            ow, oh = cw, ch
        else:
            ow, oh = _even(tw), _even(th)
    need = max(cw / in_w, ch / in_h)

    primary, secondary = QUALITY[opt.quality]
    primary = opt.model or primary
    secondary = opt.model2 or secondary
    s1 = M.model_scale(primary)
    chain = [primary]
    if opt.passes == "2" or (opt.passes == "auto" and secondary and need > s1 * 1.25):
        chain.append(secondary or primary)
    if opt.passes == "auto" and need <= 1.0:
        chain = []  # already at/above target: plain resize
    scale = 1
    for m in chain:
        scale *= M.model_scale(m)
    return Plan(in_w, in_h, ow, oh, cw, ch, chain, scale)


# --------------------------------------------------------------------------- frame ops
def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(sigma * 3))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k**2) / (2 * sigma**2))
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k, groups=3)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.transpose(-1, -2), groups=3)


class FrameProcessor:
    """Everything that happens to one frame between decode and encode."""

    def __init__(self, opt: Options, plan: Plan, device: torch.device):
        self.opt, self.plan, self.device = opt, plan, device
        half = not opt.fp32
        self.stages: list[Upscaler] = []
        for i, name in enumerate(plan.models):
            desc = M.load_model(name, denoise=opt.denoise if i == 0 else None)
            self.stages.append(Upscaler(desc, device, half=half, tile=opt.tile, name=name))
        self.stabilizer = TemporalStabilizer(opt.deflicker, device)

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(frame).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1)[None].float().div_(255)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == 0:
                x = self.stabilizer(frame, x)
        if not self.stages:
            x = self.stabilizer(frame, x)
        p = self.plan
        if x.shape[-2:] != (p.content_h, p.content_w):
            down = x.shape[-1] > p.content_w
            x = F.interpolate(x, size=(p.content_h, p.content_w), mode="bicubic",
                              align_corners=False, antialias=down)
        if self.opt.sharpen > 0:
            x = x + self.opt.sharpen * (x - _gaussian_blur(x, 1.2))
        if self.opt.grain > 0:
            luma = torch.randn((1, 1, *x.shape[-2:]), device=x.device) * (self.opt.grain * 0.02)
            x = x + luma
        x = x.clamp_(0, 1)
        if (p.content_w, p.content_h) != (p.out_w, p.out_h):
            x = _fit_canvas(x, p.out_w, p.out_h)
        return x


def _fit_canvas(x: torch.Tensor, w: int, h: int) -> torch.Tensor:
    ch, cw = x.shape[-2:]
    if cw > w or ch > h:  # crop centre
        y0, x0 = max(0, (ch - h) // 2), max(0, (cw - w) // 2)
        x = x[..., y0 : y0 + min(h, ch), x0 : x0 + min(w, cw)]
        ch, cw = x.shape[-2:]
    if cw < w or ch < h:  # pad with black
        l, t = (w - cw) // 2, (h - ch) // 2
        x = F.pad(x, (l, w - cw - l, t, h - ch - t), value=0.0)
    return x


# --------------------------------------------------------------------------- job
def _job_key(opt: Options, info: VideoInfo) -> str:
    d = asdict(opt)
    for k in ("resume", "keep_parts", "device", "segment_frames"):
        d.pop(k)
    st = info.path.stat()
    d["_src"] = [str(info.path.resolve()), st.st_size, st.st_mtime_ns]
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def upscale_video(src: Path, dst: Path, opt: Options, dry_run: bool = False,
                  progress: ProgressFn | None = None,
                  stop: threading.Event | None = None) -> Path:
    """Upscale one video. ``progress``/``stop`` let a GUI drive the job."""
    from tqdm import tqdm

    info = probe(src, opt.input_matrix)
    plan = make_plan(opt, info.width, info.height)
    bits = 8 if opt.codec == "h264" else opt.bit_depth
    enc = choose_encoder(opt.codec, plan.out_w, plan.out_h, bits, opt.crf, opt.preset, opt.encoder)
    device = pick_device(opt.device)

    total = info.frames
    if opt.start or opt.duration:
        remain = max(0.0, info.duration - (opt.start or 0))
        total = int(round(min(remain, opt.duration or remain) * float(info.fps)))

    log.info("input   %s  %dx%d @ %.3f fps, ~%d frames, matrix=%s", src.name, info.width,
             info.height, float(info.fps), total, info.color_matrix)
    log.info("plan    %s", plan.describe())
    log.info("encoder %s (%d-bit)  device %s  deflicker %.2f  denoise %.2f", enc.name, bits,
             device, opt.deflicker, opt.denoise)
    if dry_run:
        return dst

    dst.parent.mkdir(parents=True, exist_ok=True)
    parts = dst.parent / (dst.name + ".parts")
    manifest_path = parts / "manifest.json"
    key = _job_key(opt, info)
    manifest = {"key": key, "segments": []}
    if opt.resume and manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("key") == key:
            manifest = old
            # keep only segments whose files survived
            manifest["segments"] = [s for s in manifest["segments"]
                                    if (parts / s["file"]).exists()]
        else:
            log.info("settings changed since last run -> starting fresh")
            shutil.rmtree(parts)
    elif parts.exists():
        shutil.rmtree(parts)
    parts.mkdir(parents=True, exist_ok=True)

    done_frames = sum(s["frames"] for s in manifest["segments"])
    if done_frames:
        log.info("resuming at frame %d (%d segments done)", done_frames, len(manifest["segments"]))

    processor = FrameProcessor(opt, plan, device)
    reader = FrameReader(info, start_frame=done_frames, ss=opt.start, t=opt.duration)
    writer = None
    seg_frames = 0
    seg_file = ""
    t0 = time.time()
    bar = tqdm(total=total or None, initial=done_frames, unit="f", dynamic_ncols=True,
               desc=dst.name[:28], disable=progress is not None)
    done = done_frames

    def finish_segment():
        nonlocal writer, seg_frames
        writer.close()
        manifest["segments"].append({"file": seg_file, "frames": seg_frames})
        manifest_path.write_text(json.dumps(manifest, indent=1))
        writer, seg_frames = None, 0

    try:
        for frame in reader:
            if stop is not None and stop.is_set():
                raise Cancelled(f"stopped at frame {done}; run again to resume")
            out = processor(frame)
            planes = rgb_to_yuv420(out, bits)
            if writer is None:
                seg_file = f"seg_{len(manifest['segments']):05d}.mkv"
                writer = FrameWriter(parts / seg_file, plan.out_w, plan.out_h, info.fps, enc)
            writer.write(planes)
            seg_frames += 1
            done += 1
            bar.update(1)
            if progress is not None:
                progress(done, max(total, done), (done - done_frames) / max(time.time() - t0, 1e-6))
            if seg_frames >= opt.segment_frames:
                finish_segment()
        if writer is not None:
            finish_segment()
    except BaseException:
        if writer is not None:
            writer.abort()
        reader.close()
        bar.close()
        raise
    bar.close()
    reader.close()

    segs = [parts / s["file"] for s in manifest["segments"]]
    if not segs:
        raise SystemExit("No frames decoded.")
    mux(segs, info, dst, enc.codec, opt.start, opt.duration)
    if not opt.keep_parts:
        shutil.rmtree(parts, ignore_errors=True)
    n = sum(s["frames"] for s in manifest["segments"])
    dt = time.time() - t0
    log.info("done    %s  (%d frames, %.1f min, %.2f fps)", dst, n, dt / 60,
             (n - done_frames) / max(dt, 1e-6))
    return dst


# --------------------------------------------------------------------------- compare
def compare(src: Path, dst: Path, opt: Options, at: float = 1.0, crop: int = 768) -> Path:
    """Write a PNG: 1:1 crop of plain bicubic (left) vs. the AI pipeline (right)."""
    import cv2

    info = probe(src, opt.input_matrix)
    if info.duration and at >= info.duration * 0.9:
        at = info.duration / 2  # short clip: use the middle frame
    plan = make_plan(opt, info.width, info.height)
    device = pick_device(opt.device)
    proc = FrameProcessor(replace(opt, deflicker=0.0), plan, device)
    reader = FrameReader(info, ss=at, t=1.0)
    frame = next(iter(reader), None)
    reader.close()
    if frame is None:
        raise SystemExit(f"Could not read a frame at {at}s")
    ai = proc(frame)
    x = torch.from_numpy(frame).permute(2, 0, 1)[None].float().div(255).to(device)
    bic = F.interpolate(x, size=(plan.content_h, plan.content_w), mode="bicubic",
                        align_corners=False).clamp(0, 1)
    if (plan.content_w, plan.content_h) != (plan.out_w, plan.out_h):
        bic = _fit_canvas(bic, plan.out_w, plan.out_h)

    def to_np(t):
        return (t[0].permute(1, 2, 0).cpu().numpy() * 255 + 0.5).astype(np.uint8)

    a, b = to_np(bic), to_np(ai)
    h, w = a.shape[:2]
    c = min(crop, h, w // 2 * 2)
    y0, x0 = (h - c) // 2, (w - c) // 2
    pair = np.concatenate([a[y0 : y0 + c, x0 : x0 + c], np.full((c, 8, 3), 255, np.uint8),
                           b[y0 : y0 + c, x0 : x0 + c]], axis=1)
    cv2.putText(pair, "bicubic", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
    cv2.putText(pair, " + ".join(plan.models) or "resize", (c + 24, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), cv2.cvtColor(pair, cv2.COLOR_RGB2BGR))
    full = dst.with_name(dst.stem + "_full.png")
    cv2.imwrite(str(full), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
    log.info("wrote %s and %s", dst, full)
    return dst
