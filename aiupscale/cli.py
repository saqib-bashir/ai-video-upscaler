"""Command line interface:  aiupscale {upscale,compare,models} ..."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from . import models as M
from .pipeline import QUALITY, TARGETS, Options, compare, upscale_video

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".ts", ".flv", ".wmv"}


def _add_common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("resolution")
    g.add_argument("-t", "--target", default="4k",
                   help=f"{', '.join(TARGETS)} or WxH (default 4k). Presets follow the input's "
                        "orientation, so vertical 9:16 clips become 2160x3840.")
    g.add_argument("--fit", choices=["fit", "pad", "crop", "stretch"], default="fit",
                   help="fit: keep aspect, no bars (default) | pad: exact size, black bars | "
                        "crop: exact size, fill + centre crop | stretch: distort")

    g = p.add_argument_group("model")
    g.add_argument("-q", "--quality", choices=list(QUALITY), default="balanced",
                   help="fast: compact model only | balanced: compact + x2plus 2nd pass for 8K "
                        "(default) | max: x4plus (+ x2plus for 8K)")
    g.add_argument("-m", "--model", help=f"override primary model: {', '.join(M.MODELS)} or a "
                                         ".pth/.safetensors path (any spandrel architecture)")
    g.add_argument("--model2", help="override second-pass model (used for 8K)")
    g.add_argument("--passes", choices=["auto", "1", "2"], default="auto")
    g.add_argument("--denoise", type=float, default=0.5,
                   help="0..1, 'general' model only. 0 keeps grain/texture, 1 = strongest "
                        "cleanup of AI noise/compression (default 0.5)")

    g = p.add_argument_group("enhancement")
    g.add_argument("--deflicker", type=float, default=0.5,
                   help="0..0.95 motion-compensated temporal stabilisation against texture "
                        "shimmer (default 0.5, 0 = off)")
    g.add_argument("--sharpen", type=float, default=0.0, help="unsharp amount, e.g. 0.3")
    g.add_argument("--grain", type=float, default=0.0,
                   help="add fine luma film grain to hide the waxy AI look, e.g. 0.5")

    g = p.add_argument_group("hardware")
    g.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    g.add_argument("--fp32", action="store_true", help="disable fp16 inference")
    g.add_argument("--tile", type=int, default=0,
                   help="tile size in input pixels (0 = automatic, VRAM-aware)")
    g.add_argument("--input-matrix", default="auto",
                   choices=["auto", "bt709", "bt601", "bt2020"],
                   help="colour matrix of the source if it is untagged (auto assumes BT.709)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="aiupscale",
        description="Upscale (AI-generated) low-res videos to 4K / 8K with Real-ESRGAN-family "
                    "models, temporal anti-flicker and 10-bit HEVC/AV1 output.")
    ap.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upscale", help="upscale one or more videos / folders")
    up.add_argument("inputs", nargs="+", type=Path)
    up.add_argument("-o", "--output", type=Path,
                    help="output file (single input) or folder. Default: next to input, "
                         "<name>_<target>.mp4")
    _add_common(up)
    g = up.add_argument_group("encoding")
    g.add_argument("-c", "--codec", choices=["hevc", "av1", "h264", "prores"], default="hevc")
    g.add_argument("--encoder", help="force an ffmpeg encoder, e.g. libx265, hevc_nvenc")
    g.add_argument("--crf", type=int, help="quality (CRF/CQ). Lower = better. "
                                           "Defaults: x265 16, nvenc 19, av1 22")
    g.add_argument("--preset", help="encoder preset (x265: medium, nvenc: p6, svtav1: 6)")
    g.add_argument("--bit-depth", type=int, choices=[8, 10], default=10)
    g = up.add_argument_group("job")
    g.add_argument("--start", type=float, help="start time in seconds")
    g.add_argument("--duration", type=float, help="only process this many seconds (preview)")
    g.add_argument("--segment-frames", type=int, default=240,
                   help="frames per resumable segment (default 240)")
    g.add_argument("--no-resume", action="store_true", help="ignore previous partial output")
    g.add_argument("--keep-parts", action="store_true")
    g.add_argument("--plan", action="store_true", help="print the plan and exit")
    g.add_argument("--overwrite", action="store_true")

    cp = sub.add_parser("compare", help="side-by-side PNG: bicubic vs AI at 100%% zoom")
    cp.add_argument("input", type=Path)
    cp.add_argument("-o", "--output", type=Path, default=Path("compare.png"))
    cp.add_argument("--at", type=float, default=1.0, help="timestamp in seconds")
    cp.add_argument("--crop", type=int, default=768, help="crop size in output pixels")
    _add_common(cp)

    sub.add_parser("doctor", help="check GPU / PyTorch / NVENC setup and explain problems")

    gp = sub.add_parser("gui", help="open the drag-and-drop window")
    gp.add_argument("inputs", nargs="*", help="videos to pre-load")

    mp = sub.add_parser("models", help="list or pre-download models")
    mp.add_argument("--download", nargs="*", metavar="NAME",
                    help="download models (all if no names given)")
    return ap


def _options(a: argparse.Namespace) -> Options:
    o = Options(target=a.target, fit=a.fit, quality=a.quality, model=a.model, model2=a.model2,
                passes=a.passes, denoise=a.denoise, deflicker=a.deflicker, sharpen=a.sharpen,
                grain=a.grain, device=a.device, fp32=a.fp32, tile=a.tile,
                input_matrix=a.input_matrix)
    if hasattr(a, "codec"):
        o.codec, o.encoder, o.crf, o.preset = a.codec, a.encoder, a.crf, a.preset
        o.bit_depth, o.start, o.duration = a.bit_depth, a.start, a.duration
        o.segment_frames, o.resume, o.keep_parts = a.segment_frames, not a.no_resume, a.keep_parts
    return o


def _expand(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in inputs:
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        elif p.exists():
            files.append(p)
        else:
            raise SystemExit(f"Not found: {p}")
    return files


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(message)s", stream=sys.stderr)

    if a.cmd == "doctor":
        from .doctor import main as doctor_main

        return doctor_main()

    if a.cmd == "gui":
        from .gui import main as gui_main

        return gui_main(a.inputs)

    if a.cmd == "models":
        if a.download is not None:
            M.download_all(a.download or None)
        else:
            for spec in M.MODELS.values():
                have = "x" if (M.model_dir() / spec.file).exists() else " "
                print(f"[{have}] {spec.name:12s} x{spec.scale}  {spec.description}")
            print(f"\nmodel folder: {M.model_dir()}  (override with AIUPSCALE_MODELS)")
        return 0

    opt = _options(a)
    if a.cmd == "compare":
        compare(a.input, a.output, opt, at=a.at, crop=a.crop)
        return 0

    files = _expand(a.inputs)
    if not files:
        raise SystemExit("No input videos found.")
    out_dir_mode = len(files) > 1 or (a.output is not None and a.output.suffix == "")
    failures = 0
    for f in files:
        name = f"{f.stem}_{opt.target.lower()}.mp4"
        if a.output is None:
            dst = f.with_name(name)
        elif out_dir_mode:
            dst = a.output / name
        else:
            dst = a.output
        if dst.resolve() == f.resolve():
            raise SystemExit("Output would overwrite the input.")
        if dst.exists() and not a.overwrite and not a.plan:
            logging.info("skip    %s exists (use --overwrite)", dst)
            continue
        try:
            upscale_video(f, dst, opt, dry_run=a.plan)
        except KeyboardInterrupt:
            logging.info("\ninterrupted - rerun the same command to resume")
            return 130
        except Exception as e:  # noqa: BLE001 - continue the batch
            if len(files) == 1:
                raise
            failures += 1
            logging.error("FAILED  %s: %s", f, e)
    return 1 if failures else 0
