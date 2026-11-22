# aiupscale: 480p AI video → 4K / 8K

A local video upscaler tuned for **AI-generated clips** (Sora, Veo, Kling, Runway, Wan, Hunyuan, Pika, etc.)
that come out at 480p. It is built on the Real-ESRGAN model family and adds a video pipeline designed for the
weaknesses of AI footage:

| Problem with AI 480p video | What aiupscale does |
|---|---|
| Texture "boiling" / shimmer when upscaled frame-by-frame | **Motion-compensated temporal stabiliser** (optical flow + two-level confidence, no ghosting, resets on scene cuts) |
| Compression blocks, generator noise | Real-ESRGAN `general` model with **adjustable denoise** (weight blending of the strong/weak denoise models) |
| 4K/8K is 4.5× / 9× of 480p, models are 4× | **Pass planner**: 4× model (+ 2× model for 8K), then exact high-quality resize to 3840×2160 / 7680×4320 |
| Vertical 9:16 clips, 4:3, odd 854×480 | Orientation-aware targets, `fit` / `pad` / `crop` / `stretch`, snaps near-16:9 to exact 4K/8K |
| Banding in smooth AI gradients | **10-bit** output end-to-end (RGB→BT.709 YUV done on the GPU, no 8-bit bottleneck) |
| 8K encodes are huge and slow | NVENC HEVC/AV1 auto-selected (RTX 40 encodes 8K in hardware), x265/SVT-AV1 fallback |
| Long 8K jobs crash / get interrupted | **Resumable**: work is saved in segments; rerun the same command to continue |
| Command lines are a hassle | **Drag-and-drop window** with before/after preview, progress/ETA, stop & resume |
| Limited VRAM (e.g. 8 GB laptop GPUs) | **VRAM-aware auto tiling** with 32px overlap (seamless), plus OOM fallback |
| "Waxy" over-smoothed look | Optional `--grain` and `--sharpen` |

Audio is copied untouched (re-encoded to AAC only if the container requires it). HEVC output is tagged `hvc1` so it plays
on iPhone/macOS/QuickTime.

## Why not just use an existing repo?

I surveyed the main options (Nov 2026):

* **video2x** (22.8k★): great general framework (C++/Vulkan), but it upscales frame-by-frame with no temporal
  stabilisation, has no 4K/8K exact-size planning, and is hard to extend.
* **Real-ESRGAN** (xinntao): excellent models, but the official video script is a demo. It is 8-bit, has no
  anti-flicker or resume, and uses hard-cut tiles (10px overlap).
* **SeedVR2 / FlashVSR / STAR**: diffusion-based, state-of-the-art restoration, but they need 24–80 GB VRAM for
  high resolutions and are far too slow/heavy to produce 8K on a laptop. (They pair well with this tool; see tips.)

aiupscale keeps the proven Real-ESRGAN weights (loaded through [spandrel](https://github.com/chaiNNer-org/spandrel),
so **any** OpenModelDB model works too) and replaces everything around them.

## Install (Windows + NVIDIA, e.g. RTX 4070 Laptop 8 GB)

1. Install **Python 3.10 – 3.13** from python.org (tick *Add python.exe to PATH*).
2. Double-click **`install.bat`**. It installs ffmpeg (via winget) if needed, creates `.venv`, installs PyTorch with
   CUDA 12.8, and downloads the models (SHA-256 verified). If it installed ffmpeg it will ask you to run it a second
   time.
3. The last line should say `CUDA OK: True NVIDIA GeForce RTX 4070 Laptop GPU`.
4. Double-click **`AI Upscaler.bat`** to open the window.

(The window uses Tkinter, which the python.org installer includes by default.)

Linux / macOS: `./install.sh` (needs `ffmpeg` on PATH). Apple Silicon uses the `mps` backend.

## Use

### The window (easiest)

Double-click **`AI Upscaler.bat`** (or run `aiupscale-gui`). Drag videos or whole folders into the list, pick
**Resolution** (4K / 8K), **Quality**, and **Content** (realistic or anime), and press **Start upscaling**.

* **Preview frame (before / after)** renders one frame at 100% zoom so you can tune *Denoise*, *Anti-flicker* and
  *Film grain* before a long job.
* **Preview only: first N seconds** makes a short test clip (`name_4k_preview.mp4`).
* **Stop** keeps every finished segment. Press Start again with the same settings and it resumes.
* The top line shows which GPU is in use (it should say *RTX 4070 Laptop GPU*). Your settings are remembered.

### Batch files and command line

**Drag & drop onto a batch file:** drop videos (or a folder) onto `upscale_4k.bat` / `upscale_8k.bat`. The output is written next to the input as
`name_4k.mp4`. Drop one onto `compare.bat` to get a 100%-zoom bicubic-vs-AI PNG.

**Command line** (`.venv\Scripts\activate` first):

```bash
aiupscale upscale clip.mp4 --target 4k
```

```bash
aiupscale upscale clip.mp4 --target 8k -o clip_8k.mp4
```

```bash
aiupscale upscale clip.mp4 -t 4k --duration 3 -o preview.mp4
```

```bash
aiupscale upscale "D:\AI clips" -t 4k -o "D:\AI clips\4k"
```

```bash
aiupscale compare clip.mp4 --at 2.5 -t 4k --denoise 0.8 -o cmp.png
```

```bash
aiupscale upscale clip.mp4 -t 8k --plan
```

The fourth command processes a whole folder. The last one prints the processing plan and exits without doing any work. For example:

```text
plan    854x480 --[general -> x2plus, x8]--> 6832x3840 --resize--> 7680x4320
encoder hevc_nvenc (10-bit)  device cuda  deflicker 0.50  denoise 0.50
```

Interrupted? Run **the same command again** and it resumes from the last finished segment.

### Main options

| Option | Default | Meaning |
|---|---|---|
| `-t/--target` | `4k` | `720p 1080p 1440p 4k 5k 6k 8k` or `WxH`. Presets follow input orientation (9:16 → 2160×3840). |
| `--fit` | `fit` | `fit` keep aspect · `pad` exact size with bars · `crop` exact size, fill · `stretch` |
| `-q/--quality` | `balanced` | `fast`: compact model only (8K = 4× + resize) · `balanced`: compact + x2plus for 8K · `max`: x4plus (+x2plus) |
| `-m/--model` | | `general`, `x4plus`, `anime-video`, `anime`, or any `.pth`/`.safetensors` path |
| `--model2` | | second-pass model for 8K (e.g. `general` for a faster 8K) |
| `--denoise` | `0.5` | 0 = keep grain/texture … 1 = strongest cleanup (general model) |
| `--deflicker` | `0.5` | temporal stabilisation 0–0.95 (0 = off). 0.6–0.8 for very shimmery clips |
| `--grain` / `--sharpen` | `0` | e.g. `--grain 0.4` to de-wax faces, `--sharpen 0.3` |
| `-c/--codec` | `hevc` | `hevc`, `av1`, `h264` (8-bit, max compatibility), `prores` (editing) |
| `--crf` | auto | lower = better/larger (nvenc CQ 19, x265 16, AV1 22) |
| `--bit-depth` | `10` | `8` for old players |
| `--tile` | `0` | 0 = auto from free VRAM. Force e.g. `256` if you run other GPU apps |
| `--device` | `auto` | `cuda`, `cuda:1`, `mps`, `cpu` |

## Recommended settings for an RTX 4070 Laptop (8 GB)

* **4K, most AI clips:** defaults. `aiupscale upscale clip.mp4 -t 4k`
* **8K:** defaults (`balanced`). Auto-tiling keeps the x2plus pass inside 8 GB. Try `--model2 general` if it is too slow.
* **Heavily compressed / noisy source:** `--denoise 0.8` or `-q max`
* **Anime / cartoon / stylised:** `-m anime-video` (very fast) or `-m anime`
* **Faces look plastic:** `--denoise 0.3 --grain 0.4`
* Plug the laptop in and use the high-performance power profile; laptop GPUs throttle hard on battery.

**Rough speed expectations** (estimated from model FLOPs; not benchmarked on that exact laptop):
4K `balanced` runs at roughly 10–25 fps, so a 5 s / 24 fps clip takes well under a minute. 8K `balanced` is dominated by
the x2plus pass at roughly 0.3–1 fps, so the same 5 s clip takes about 2–6 minutes. `-q max` is several times slower than
`balanced` at 4K. Run `--duration 2` first to measure your real speed.

## "No GPU found"?

Run **`fix_gpu.bat`** (or `aiupscale doctor` to see the reason). Typical causes:

* **CPU-only PyTorch.** PyTorch from PyPI is CPU-only on Windows. `fix_gpu.bat` reinstalls the CUDA build.
* **GPU switched off.** Gaming laptops have an Eco / iGPU-only / "Integrated" mode (Armoury Crate, Lenovo Vantage,
  MSI Center, ...) that disables the RTX GPU. Switch to Standard/Hybrid or dGPU mode and reboot.
* **Old NVIDIA driver.** Install the latest Game Ready or Studio driver from nvidia.com.

## How it works

```text
ffmpeg decode (correct BT.709/601 matrix, SAR, rotation)
  → [GPU] model pass 1 (4×, fp16, tiled if needed)
  → [GPU] temporal stabiliser  (DIS flow on 480p → warp previous output → confidence-gated blend)
  → [GPU] model pass 2 (2×, only when the target needs > 5×)
  → [GPU] antialiased bicubic to exact size → optional sharpen/grain → fit/pad/crop
  → [GPU] RGB → BT.709 limited-range YUV 4:2:0 10-bit
  → ffmpeg encode (NVENC/x265/SVT-AV1) in resumable segments → lossless concat + audio copy
```

Decode, GPU work and encode run concurrently in separate threads.

## Tips for the best result

* Always try `compare` on a frame with a face or fine texture before a long job and tune `--denoise`.
* If the source has bad morphing or garbled detail, a diffusion restorer gives a bigger jump. On 8 GB, run **SeedVR2**
  (ComfyUI, 3B GGUF + block-swap) to 1080p first, then use aiupscale for the 4K/8K step, which is where SeedVR2 runs out of VRAM.
* Upload targets: YouTube accepts 8K HEVC/AV1 10-bit directly. For editing, use `-c prores`.

## Tests

```bash
pip install -e .[test]
```

```bash
pytest -q
```

## Credits & licences

Models: [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al. (BSD-3-Clause). Loader:
[spandrel](https://github.com/chaiNNer-org/spandrel) (MIT). This tool: BSD-3-Clause.
