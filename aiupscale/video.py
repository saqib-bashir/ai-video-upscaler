"""ffmpeg-based probing, decoding, encoding and muxing."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("aiupscale")

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


# --------------------------------------------------------------------------- probe
@dataclass
class VideoInfo:
    path: Path
    width: int          # display width after rotation / SAR correction
    height: int
    fps: Fraction
    frames: int         # best estimate, used for progress only
    duration: float
    has_audio: bool
    color_matrix: str   # ffmpeg scale in_color_matrix value
    sar_fix: bool


_MATRIX = {
    "bt709": "bt709", "smpte170m": "bt601", "bt470bg": "bt601", "bt601": "bt601",
    "bt2020nc": "bt2020", "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc",
}


def probe(path: Path, input_matrix: str = "auto") -> VideoInfo:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format",
           str(path)]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout)
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffprobe failed on {path}: {e.stderr.strip()}") from e
    vs = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise SystemExit(f"No video stream in {path}")
    w, h = int(vs["width"]), int(vs["height"])

    sar = Fraction(1)
    if vs.get("sample_aspect_ratio", "1:1") not in ("0:1", "1:1", "N/A"):
        n, d = vs["sample_aspect_ratio"].split(":")
        sar = Fraction(int(n), int(d))
    sar_fix = sar != 1
    if sar_fix:
        w = int(round(w * sar / 2)) * 2

    rot = 0
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    rot = int(vs.get("tags", {}).get("rotate", rot))
    if abs(rot) % 180 == 90:
        w, h = h, w

    fps = Fraction(vs.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(vs.get("r_frame_rate") or "30/1")
    duration = float(vs.get("duration") or data["format"].get("duration") or 0)
    frames = int(vs.get("nb_frames") or 0) or int(round(duration * float(fps)))

    if input_matrix == "auto":
        # Untagged footage: AI generators (and nearly everything modern) encode BT.709.
        matrix = _MATRIX.get(vs.get("color_space", ""), "bt709")
    else:
        matrix = input_matrix
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    return VideoInfo(Path(path), w, h, fps, frames, duration, has_audio, matrix, sar_fix)


# --------------------------------------------------------------------------- reader
class FrameReader:
    """Decodes frames as HxWx3 uint8 RGB numpy arrays in a background thread."""

    def __init__(self, info: VideoInfo, start_frame: int = 0, ss: float | None = None,
                 t: float | None = None, prefetch: int = 8):
        self.w, self.h = info.width, info.height
        filters = []
        if info.sar_fix:
            filters.append(f"scale={self.w}:{info.height},setsar=1")
        if start_frame > 0:
            filters.append(f"select=gte(n\\,{start_frame})")
        filters.append(
            f"scale=w={self.w}:h={self.h}:in_color_matrix={info.color_matrix}:in_range=auto"
            ":flags=bicubic+accurate_rnd+full_chroma_int,format=rgb24"
        )
        cmd = [FFMPEG, "-v", "error", "-nostdin"]
        if ss:
            cmd += ["-ss", f"{ss:.3f}"]
        cmd += ["-i", str(info.path)]
        if t:
            cmd += ["-t", f"{t:.3f}"]
        cmd += ["-map", "0:v:0", "-vf", ",".join(filters), "-fps_mode", "passthrough",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     bufsize=self.w * self.h * 3 * 2)
        self.q: queue.Queue = queue.Queue(maxsize=prefetch)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        n = self.w * self.h * 3
        try:
            while True:
                buf = bytearray(n)  # writable -> zero-copy torch.from_numpy
                view, got = memoryview(buf), 0
                while got < n:
                    k = self.proc.stdout.readinto(view[got:])
                    if not k:
                        break
                    got += k
                if got < n:
                    break
                self.q.put(np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3))
        finally:
            self.q.put(None)

    def __iter__(self):
        while (frame := self.q.get()) is not None:
            yield frame
        self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace").strip()
        if self.proc.returncode not in (0, None) and err:
            log.warning("decoder: %s", err)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


# --------------------------------------------------------------------------- colour
_KR, _KB = 0.2126, 0.0722  # BT.709


def rgb_to_yuv420(rgb: torch.Tensor, bits: int) -> list[np.ndarray]:
    """[1,3,H,W] float 0..1 RGB -> planar BT.709 limited-range Y, U, V (numpy).

    Done on the GPU so we pipe half the bytes of RGB48 and never depend on
    ffmpeg guessing the matrix.
    """
    r, g, b = rgb[0, 0], rgb[0, 1], rgb[0, 2]
    y = _KR * r + (1 - _KR - _KB) * g + _KB * b
    pb = (b - y) / (2 * (1 - _KB))
    pr = (r - y) / (2 * (1 - _KR))
    pb = F.avg_pool2d(pb[None, None], 2)[0, 0]
    pr = F.avg_pool2d(pr[None, None], 2)[0, 0]
    k = float(1 << (bits - 8))
    maxv = (1 << bits) - 1
    planes = [(16 + 219 * y) * k, (128 + 224 * pb) * k, (128 + 224 * pr) * k]
    if bits == 8:
        # +-0.5 LSB random dither hides banding in smooth AI-generated gradients
        out = [(p + torch.rand_like(p) - 0.5).round_().clamp_(0, maxv).to(torch.uint8)
               for p in planes]
        return [p.cpu().numpy() for p in out]
    out = [p.round_().clamp_(0, maxv).to(torch.int16) for p in planes]
    return [p.cpu().numpy().view(np.uint16) for p in out]


# --------------------------------------------------------------------------- encoders
@lru_cache(maxsize=1)
def available_encoders() -> set[str]:
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True,
                         text=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            names.add(parts[1])
    return names


@lru_cache(maxsize=None)
def encoder_works(name: str) -> bool:
    """Hardware encoders are compiled in everywhere; check one really runs."""
    if name not in available_encoders():
        return False
    cmd = [FFMPEG, "-v", "error", "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
           "-frames:v", "1", "-c:v", name, "-f", "null", "-"]
    return subprocess.run(cmd, capture_output=True).returncode == 0


@dataclass
class EncoderConfig:
    name: str
    args: list[str]
    pix_fmt_in: str      # raw format we pipe
    codec: str           # hevc / av1 / h264 / prores


def choose_encoder(codec: str, width: int, height: int, bits: int, crf: int | None,
                   preset: str | None, encoder: str | None = None) -> EncoderConfig:
    big = max(width, height) > 4096
    candidates = {
        "hevc": ["hevc_nvenc", "hevc_videotoolbox", "libx265"],
        "av1": ["av1_nvenc", "libsvtav1", "libaom-av1"],
        "h264": ["libx264"] if big else ["h264_nvenc", "libx264"],
        "prores": ["prores_ks"],
    }[codec]
    if encoder:
        candidates = [encoder]
    name = next((c for c in candidates if encoder_works(c)), None)
    if name is None:
        raise SystemExit(f"No working encoder for {codec} (tried {', '.join(candidates)}).")

    if codec == "h264":
        bits = 8  # 10-bit H.264 plays almost nowhere
    pipe_fmt = "yuv420p10le" if bits == 10 else "yuv420p"
    a: list[str] = ["-c:v", name]
    if name == "libx265":
        a += ["-preset", preset or "medium", "-crf", str(crf if crf is not None else 16),
              "-pix_fmt", pipe_fmt, "-x265-params", "log-level=error:aq-mode=3"]
    elif name == "libx264":
        a += ["-preset", preset or "slow", "-crf", str(crf if crf is not None else 16),
              "-pix_fmt", "yuv420p"]
    elif name.endswith("_nvenc"):
        a += ["-preset", preset or "p6", "-tune", "hq", "-rc", "vbr",
              "-cq", str(crf if crf is not None else 19), "-b:v", "0",
              "-spatial-aq", "1", "-temporal-aq", "1",
              "-pix_fmt", "p010le" if bits == 10 else "nv12"]
        if codec == "hevc":
            a += ["-profile:v", "main10" if bits == 10 else "main"]
    elif name == "hevc_videotoolbox":
        a += ["-q:v", str(crf if crf is not None else 70), "-pix_fmt",
              "p010le" if bits == 10 else "nv12", "-profile:v", "main10" if bits == 10 else "main"]
    elif name == "libsvtav1":
        a += ["-preset", preset or "6", "-crf", str(crf if crf is not None else 22),
              "-pix_fmt", pipe_fmt, "-svtav1-params", "tune=0:film-grain=0"]
    elif name == "libaom-av1":
        a += ["-cpu-used", preset or "6", "-crf", str(crf if crf is not None else 22), "-b:v", "0",
              "-row-mt", "1", "-pix_fmt", pipe_fmt]
    elif name == "prores_ks":
        a += ["-profile:v", preset or "3", "-pix_fmt", "yuv422p10le"]
        pipe_fmt = "yuv420p10le"
    a += ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
          "-color_range", "tv"]
    return EncoderConfig(name, a, pipe_fmt, codec)


# --------------------------------------------------------------------------- writer
class FrameWriter:
    """Pipes planar YUV frames into an ffmpeg encoder in a background thread."""

    def __init__(self, path: Path, width: int, height: int, fps: Fraction,
                 enc: EncoderConfig):
        cmd = [FFMPEG, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", enc.pix_fmt_in,
               "-s", f"{width}x{height}", "-r", f"{fps.numerator}/{fps.denominator}",
               "-i", "-", *enc.args, "-an", str(path)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.q: queue.Queue = queue.Queue(maxsize=3)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while (planes := self.q.get()) is not None:
                for p in planes:
                    self.proc.stdin.write(memoryview(np.ascontiguousarray(p)).cast("B"))
        except BaseException as e:  # noqa: BLE001 - surfaced in write()/close()
            self.error = e
            while self.q.get() is not None:  # drain so producers never block
                pass

    def write(self, planes: list[np.ndarray]):
        if self.error:
            raise RuntimeError(f"encoder failed: {self._stderr()}") from self.error
        self.q.put(planes)

    def _stderr(self) -> str:
        try:
            return self.proc.stderr.read().decode(errors="replace").strip()
        except Exception:  # noqa: BLE001
            return ""

    def close(self):
        self.q.put(None)
        self.thread.join()
        try:
            self.proc.stdin.close()
        except BrokenPipeError:
            pass
        rc = self.proc.wait()
        if rc != 0 or self.error:
            raise RuntimeError(f"encoder exited with {rc}: {self._stderr()}")

    def abort(self):
        self.proc.kill()
        self.proc.wait()


# --------------------------------------------------------------------------- mux
def mux(segments: list[Path], source: VideoInfo, out: Path, codec: str,
        ss: float | None, t: float | None) -> None:
    """Concatenate encoded segments losslessly and copy audio from the source."""
    lst = out.parent / (out.name + ".concat.txt")
    def quote(p: Path) -> str:  # ffmpeg concat syntax: ' -> '\''
        return "'" + p.resolve().as_posix().replace("'", "'\\''") + "'"

    lst.write_text("".join(f"file {quote(p)}\n" for p in segments), encoding="utf-8")
    base = [FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst)]
    if source.has_audio:
        if ss:
            base += ["-ss", f"{ss:.3f}"]
        if t:
            base += ["-t", f"{t:.3f}"]
        base += ["-i", str(source.path), "-map", "0:v:0", "-map", "1:a?"]  # never -shortest: it can drop video frames
    else:
        base += ["-map", "0:v:0"]
    base += ["-c:v", "copy"]
    ext = out.suffix.lower()
    if ext in (".mp4", ".mov", ".m4v"):
        if codec == "hevc":
            base += ["-tag:v", "hvc1"]  # required for QuickTime / iOS playback
        base += ["-movflags", "+faststart"]
    try:
        attempts = [["-c:a", "copy"], ["-c:a", "aac", "-b:a", "320k"]] if source.has_audio else [[]]
        last = None
        for audio in attempts:
            r = subprocess.run(base + audio + [str(out)], capture_output=True, text=True)
            if r.returncode == 0:
                return
            last = r.stderr.strip()
            log.info("mux retry (audio re-encode): %s", last.splitlines()[-1] if last else "")
        raise RuntimeError(f"mux failed: {last}")
    finally:
        lst.unlink(missing_ok=True)
