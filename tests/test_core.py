"""Fast CPU tests. Run: pytest -q"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from aiupscale import models as M
from aiupscale.engine import Upscaler
from aiupscale.pipeline import Options, _fit_canvas, make_plan, upscale_video
from aiupscale.temporal import TemporalStabilizer
from aiupscale.video import probe, rgb_to_yuv420


# ----------------------------------------------------------------- planning
@pytest.mark.parametrize(
    "size,target,fit,out,chain",
    [
        ((854, 480), "4k", "fit", (3840, 2160), ["general"]),
        ((854, 480), "8k", "fit", (7680, 4320), ["general", "x2plus"]),
        ((480, 854), "4k", "fit", (2160, 3840), ["general"]),          # vertical 9:16
        ((640, 480), "4k", "fit", (2880, 2160), ["general"]),          # 4:3 keeps aspect
        ((640, 480), "4k", "pad", (3840, 2160), ["general"]),
        ((640, 480), "4k", "crop", (3840, 2160), ["general", "x2plus"]),  # needs 6x
        ((1920, 1080), "1080p", "fit", (1920, 1080), []),              # nothing to do
    ],
)
def test_plan(size, target, fit, out, chain):
    p = make_plan(Options(target=target, fit=fit), *size)
    assert (p.out_w, p.out_h) == out
    assert p.models == chain
    assert p.out_w % 2 == 0 and p.out_h % 2 == 0


def test_plan_fast_quality_single_pass_8k():
    p = make_plan(Options(target="8k", quality="fast"), 854, 480)
    assert p.models == ["general"] and (p.out_w, p.out_h) == (7680, 4320)


def test_fit_canvas_pad_and_crop():
    x = torch.ones(1, 3, 10, 6)
    y = _fit_canvas(x, 10, 10)
    assert y.shape[-2:] == (10, 10) and y[..., :, 0].sum() == 0 and y[0, 0, 5, 5].item() == 1
    z = _fit_canvas(torch.ones(1, 3, 12, 12), 8, 10)
    assert z.shape[-2:] == (10, 8)


# ----------------------------------------------------------------- engine
def test_tiled_matches_full_frame():
    desc = M.load_model("general", denoise=0.5)
    full = Upscaler(desc, torch.device("cpu"), half=False, tile=0)
    tiled = Upscaler(M.load_model("general", denoise=0.5), torch.device("cpu"), half=False,
                     tile=48)
    torch.manual_seed(0)
    x = torch.rand(1, 3, 70, 90)
    a, b = full(x), tiled(x)
    assert a.shape == b.shape == (1, 3, 280, 360)
    d = (a - b).abs()
    assert d.max().item() < 2 / 255 and d.mean().item() < 1e-4  # seams invisible


def test_denoise_interpolation_changes_weights():
    a = M.load_model("general", denoise=1.0).model.state_dict()
    b = M.load_model("general", denoise=0.0).model.state_dict()
    k = next(iter(a))
    assert not torch.allclose(a[k], b[k])


# ----------------------------------------------------------------- colour
def test_yuv_conversion_bt709_reference_values():
    white = torch.ones(1, 3, 4, 4)
    y, u, v = rgb_to_yuv420(white, 10)
    assert y.dtype == np.uint16 and y.shape == (4, 4) and u.shape == (2, 2)
    assert (y == 940).all() and (u == 512).all() and (v == 512).all()
    red = torch.zeros(1, 3, 2, 2)
    red[:, 0] = 1
    y, u, v = rgb_to_yuv420(red, 8)
    assert abs(int(y[0, 0]) - 63) <= 1 and abs(int(v[0, 0]) - 240) <= 1


# ----------------------------------------------------------------- temporal
def test_stabilizer_static_scene_reduces_variance_and_resets_on_cut():
    st = TemporalStabilizer(0.6, torch.device("cpu"))
    rng = np.random.default_rng(0)
    lr = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
    base = torch.rand(1, 3, 128, 128)
    outs = []
    for _ in range(8):
        sr = (base + 0.03 * torch.randn_like(base)).clamp(0, 1)  # shimmering GAN output
        outs.append(st(lr, sr))
    raw_flicker = 0.03 * np.sqrt(2)
    flicker = (outs[-1] - outs[-2]).std().item()
    assert flicker < raw_flicker * 0.8
    cut = np.zeros_like(lr)
    sr = torch.zeros(1, 3, 128, 128)
    assert torch.equal(st(cut, sr), sr)  # scene cut -> untouched


# ----------------------------------------------------------------- end to end
@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("v") / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=96x54:r=12:d=1",
         "-f", "lavfi", "-i", "sine=f=440:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(p)], check=True)
    return p


def test_end_to_end_with_audio_and_resume_manifest(clip, tmp_path):
    out = tmp_path / "out.mp4"
    opt = Options(target="384x216", segment_frames=5, keep_parts=True, codec="hevc",
                  encoder="libx265", preset="ultrafast")
    upscale_video(clip, out, opt)
    info = probe(out)
    assert (info.width, info.height) == (384, 216)
    assert info.has_audio
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames,pix_fmt", "-of", "json",
                        str(out)], capture_output=True, text=True, check=True).stdout
    s = json.loads(n)["streams"][0]
    assert int(s["nb_read_frames"]) == 12 and s["pix_fmt"] == "yuv420p10le"
    manifest = json.loads((tmp_path / "out.mp4.parts" / "manifest.json").read_text())
    assert [x["frames"] for x in manifest["segments"]] == [5, 5, 2]


def test_short_audio_never_truncates_video(tmp_path):
    src = tmp_path / "short_audio.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x36:r=24:d=1",
         "-f", "lavfi", "-i", "sine=f=440:d=0.3", "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], check=True)
    out = tmp_path / "o.mp4"
    upscale_video(src, out, Options(target="128x72", encoder="libx265", preset="ultrafast"))
    r = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                       capture_output=True, text=True, check=True)
    assert int(r.stdout.strip()) == 24


# ----------------------------------------------------------------- gui
def test_gui_output_naming(tmp_path):
    tk = pytest.importorskip("tkinter")  # noqa: F841 - GUI module needs tkinter
    from aiupscale.gui import output_path

    src = tmp_path / "My clip.mp4"
    assert output_path(src, "4k", "") == tmp_path / "My clip_4k.mp4"
    assert output_path(src, "8k", str(tmp_path / "out")) == tmp_path / "out" / "My clip_8k.mp4"


def test_compare_on_clip_shorter_than_timestamp(clip, tmp_path):
    from aiupscale.pipeline import compare

    opt = Options(target="192x108", deflicker=0.7)
    out = compare(clip, tmp_path / "c.png", opt, at=5.0)  # clip is only 1 s long
    assert out.exists() and opt.deflicker == 0.7  # caller's options untouched
