"""Simple drag-and-drop desktop window for aiupscale (Tkinter).

Launch with ``aiupscale-gui``, ``aiupscale gui`` or the "AI Upscaler.bat" file.
Real drag-and-drop from Explorer/Finder uses the optional ``tkinterdnd2``
package; without it the "Add files" / "Add folder" buttons still work.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

try:  # optional: native drag & drop
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:  # noqa: BLE001 - any import problem -> buttons only
    TkinterDnD = None
    DND_FILES = None

from . import __version__

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".gif", ".ts", ".flv", ".wmv"}
SETTINGS_FILE = Path.home() / ".aiupscale_gui.json"

RESOLUTIONS = {"4K (3840x2160)": "4k", "8K (7680x4320)": "8k", "1440p": "1440p",
               "1080p": "1080p", "5K": "5k", "6K": "6k"}
QUALITIES = {"Balanced (recommended)": "balanced", "Fast": "fast",
             "Max quality (slow)": "max"}
CONTENT = ["Realistic / cinematic", "Anime / cartoon"]
CODECS = {"HEVC / H.265 (10-bit)": "hevc", "AV1 (10-bit)": "av1",
          "H.264 (max compatibility)": "h264", "ProRes (for editing)": "prores"}
FITS = {"Keep aspect": "fit", "Exact size, black bars": "pad", "Exact size, crop": "crop"}


def open_path(p: Path) -> None:
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except Exception:  # noqa: BLE001
        pass


def output_path(src: Path, target: str, out_dir: str) -> Path:
    name = f"{src.stem}_{target}.mp4"
    return (Path(out_dir) / name) if out_dir else src.with_name(name)


class _QueueLog(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record):
        self.q.put(("log", self.format(record)))


class App:
    def __init__(self, root: tk.Tk, files: list[str]):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.pipeline = None  # imported lazily (torch import takes a few seconds)
        self.last_output: Path | None = None
        self.files: list[Path] = []

        root.title(f"AI Video Upscaler {__version__}")
        root.minsize(760, 640)
        self._build()
        self._load_settings()
        self.add_paths(files)

        handler = _QueueLog(self.q)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger("aiupscale").addHandler(handler)
        logging.getLogger("aiupscale").setLevel(logging.INFO)

        threading.Thread(target=self._init_backend, daemon=True).start()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.after(100, self._poll)

    # ------------------------------------------------------------------ layout
    def _build(self):
        pad = {"padx": 8, "pady": 4}
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="both", expand=True)

        self.device_lbl = ttk.Label(top, text="Loading AI engine ...", foreground="#666")
        self.device_lbl.pack(anchor="w")

        # --- drop zone / queue
        drop = ttk.LabelFrame(top, text="Videos  (drag & drop files or folders here)", padding=6)
        drop.pack(fill="both", expand=True, pady=(6, 4))
        lb_frame = ttk.Frame(drop)
        lb_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(lb_frame, height=7, selectmode="extended", activestyle="none",
                                  font=("Segoe UI", 10) if sys.platform == "win32" else None)
        sb = ttk.Scrollbar(lb_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=sb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.hint = ttk.Label(self.listbox, text="⬇  Drop videos here  ⬇", foreground="#888",
                              font=("", 14))
        btns = ttk.Frame(drop)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Add files...", command=self.pick_files).pack(side="left")
        ttk.Button(btns, text="Add folder...", command=self.pick_folder).pack(side="left", padx=4)
        ttk.Button(btns, text="Remove selected", command=self.remove_selected).pack(side="left")
        ttk.Button(btns, text="Clear", command=self.clear).pack(side="left", padx=4)
        if TkinterDnD is not None:
            for w in (self.root, self.listbox):
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", self._on_drop)

        # --- settings
        st = ttk.LabelFrame(top, text="Settings", padding=6)
        st.pack(fill="x", pady=4)
        st.columnconfigure(1, weight=1)
        st.columnconfigure(3, weight=1)

        self.v_res = tk.StringVar(value="4K (3840x2160)")
        self.v_quality = tk.StringVar(value="Balanced (recommended)")
        self.v_content = tk.StringVar(value=CONTENT[0])
        self.v_codec = tk.StringVar(value="HEVC / H.265 (10-bit)")
        self.v_fit = tk.StringVar(value="Keep aspect")
        self.v_denoise = tk.DoubleVar(value=0.5)
        self.v_deflicker = tk.DoubleVar(value=0.5)
        self.v_grain = tk.DoubleVar(value=0.0)
        self.v_outdir = tk.StringVar(value="")
        self.v_preview = tk.BooleanVar(value=False)
        self.v_preview_s = tk.IntVar(value=3)
        self.v_overwrite = tk.BooleanVar(value=False)

        def combo(row, col, label, var, values):
            ttk.Label(st, text=label).grid(row=row, column=col, sticky="w", **pad)
            cb = ttk.Combobox(st, textvariable=var, values=list(values), state="readonly",
                              width=26)
            cb.grid(row=row, column=col + 1, sticky="ew", **pad)
            return cb

        combo(0, 0, "Resolution", self.v_res, RESOLUTIONS)
        combo(0, 2, "Quality", self.v_quality, QUALITIES)
        combo(1, 0, "Content", self.v_content, CONTENT).bind(
            "<<ComboboxSelected>>", lambda e: self._sync_state())
        combo(1, 2, "Format", self.v_codec, CODECS)
        combo(2, 0, "Aspect", self.v_fit, FITS)

        def slider(row, col, label, var, hi, tip):
            ttk.Label(st, text=label).grid(row=row, column=col, sticky="w", **pad)
            f = ttk.Frame(st)
            f.grid(row=row, column=col + 1, sticky="ew", **pad)
            val = ttk.Label(f, width=4)
            s = ttk.Scale(f, from_=0, to=hi, variable=var,
                          command=lambda _=None: val.configure(text=f"{var.get():.2f}"))
            s.pack(side="left", fill="x", expand=True)
            val.pack(side="left")
            val.configure(text=f"{var.get():.2f}")
            var.trace_add("write", lambda *_: val.configure(text=f"{var.get():.2f}"))
            _Tooltip(s, tip)
            return s

        self.s_denoise = slider(3, 0, "Denoise", self.v_denoise, 1.0,
                                "0 = keep grain/texture, 1 = strongest cleanup of AI noise")
        slider(3, 2, "Anti-flicker", self.v_deflicker, 0.95,
               "Stops textures 'boiling' between frames. 0 = off, 0.5 default, 0.8 strong")
        slider(4, 0, "Film grain", self.v_grain, 1.0,
               "Adds fine grain to hide the waxy/plastic AI look (0.3-0.5 is subtle)")

        ttk.Label(st, text="Save to").grid(row=5, column=0, sticky="w", **pad)
        of = ttk.Frame(st)
        of.grid(row=5, column=1, columnspan=3, sticky="ew", **pad)
        self.out_entry = ttk.Entry(of, textvariable=self.v_outdir)
        self.out_entry.pack(side="left", fill="x", expand=True)
        ttk.Button(of, text="Browse...", command=self.pick_outdir).pack(side="left", padx=4)
        ttk.Button(of, text="Same as input", command=lambda: self.v_outdir.set("")).pack(
            side="left")

        opts = ttk.Frame(st)
        opts.grid(row=6, column=0, columnspan=4, sticky="w", **pad)
        ttk.Checkbutton(opts, text="Preview only: first", variable=self.v_preview).pack(
            side="left")
        ttk.Spinbox(opts, from_=1, to=60, width=4, textvariable=self.v_preview_s).pack(
            side="left", padx=4)
        ttk.Label(opts, text="seconds").pack(side="left")
        ttk.Checkbutton(opts, text="Overwrite existing outputs",
                        variable=self.v_overwrite).pack(side="left", padx=(24, 0))

        # --- actions
        act = ttk.Frame(top)
        act.pack(fill="x", pady=6)
        self.btn_start = ttk.Button(act, text="▶  Start upscaling", command=self.start,
                                    state="disabled")
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(act, text="■  Stop", command=self.request_stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=4)
        self.btn_compare = ttk.Button(act, text="🔍  Preview frame (before / after)",
                                      command=self.compare, state="disabled")
        self.btn_compare.pack(side="left", padx=4)
        ttk.Button(act, text="Open output folder", command=self.open_output).pack(side="right")

        # --- progress
        pr = ttk.Frame(top)
        pr.pack(fill="x")
        self.status = ttk.Label(pr, text="Idle")
        self.status.pack(anchor="w")
        self.pb_file = ttk.Progressbar(pr, mode="determinate")
        self.pb_file.pack(fill="x", pady=2)
        self.pb_all = ttk.Progressbar(pr, mode="determinate")
        self.pb_all.pack(fill="x", pady=2)

        self.log = ScrolledText(top, height=8, state="disabled", font=("Consolas", 9)
                                if sys.platform == "win32" else ("TkFixedFont", 9))
        self.log.pack(fill="both", expand=False, pady=(6, 0))

    # ------------------------------------------------------------------ queue
    def add_paths(self, paths):
        for raw in paths:
            p = Path(raw)
            items = (sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
                     if p.is_dir() else [p] if p.suffix.lower() in VIDEO_EXT else [])
            for f in items:
                if f not in self.files:
                    self.files.append(f)
                    self.listbox.insert("end", f"  {f.name}      —  {f.parent}")
        self._sync_state()

    def _on_drop(self, event):
        self.add_paths(self.root.tk.splitlist(event.data))

    def pick_files(self):
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXT))
        self.add_paths(filedialog.askopenfilenames(
            title="Choose videos", filetypes=[("Videos", exts), ("All files", "*.*")]))

    def pick_folder(self):
        d = filedialog.askdirectory(title="Folder with videos")
        if d:
            self.add_paths([d])

    def pick_outdir(self):
        d = filedialog.askdirectory(title="Save upscaled videos to")
        if d:
            self.v_outdir.set(d)

    def remove_selected(self):
        for i in reversed(self.listbox.curselection()):
            self.listbox.delete(i)
            del self.files[i]
        self._sync_state()

    def clear(self):
        self.listbox.delete(0, "end")
        self.files.clear()
        self._sync_state()

    # ------------------------------------------------------------------ state
    @property
    def busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _sync_state(self):
        if self.files:
            self.hint.place_forget()
        else:
            self.hint.place(relx=0.5, rely=0.5, anchor="center")
        ready = self.pipeline is not None and not self.busy
        has = bool(self.files)
        self.btn_start.configure(state="normal" if ready and has else "disabled")
        self.btn_compare.configure(state="normal" if ready and has else "disabled")
        self.btn_stop.configure(state="normal" if self.busy else "disabled")
        anime = self.v_content.get() == CONTENT[1]
        self.s_denoise.state(["disabled"] if anime else ["!disabled"])

    def _options(self):
        P = self.pipeline
        quality = QUALITIES[self.v_quality.get()]
        model = None
        if self.v_content.get() == CONTENT[1]:
            model = "anime" if quality == "max" else "anime-video"
        return P.Options(
            target=RESOLUTIONS[self.v_res.get()], fit=FITS[self.v_fit.get()],
            quality=quality, model=model, denoise=round(self.v_denoise.get(), 2),
            deflicker=round(self.v_deflicker.get(), 2), grain=round(self.v_grain.get(), 2),
            codec=CODECS[self.v_codec.get()],
            duration=float(self.v_preview_s.get()) if self.v_preview.get() else None)

    # ------------------------------------------------------------------ backend
    def _init_backend(self):
        try:
            from . import pipeline
            from .doctor import diagnose

            dg = diagnose()
            if not dg.ok:
                self.q.put(("log", "GPU check (also available as: aiupscale doctor)"))
                for line in dg.details:
                    self.q.put(("log", "  " + line))
            self.q.put(("ready", pipeline, dg.summary, dg.device))
        except Exception as e:  # noqa: BLE001
            self.q.put(("fatal", f"Could not load the AI engine: {e}"))

    def _run_jobs(self, jobs, opt, overwrite: bool):
        # runs in a worker thread: never touch Tk widgets/variables here, only self.q
        P = self.pipeline
        n = len(jobs)
        ok = failed = skipped = 0
        for i, (src, dst) in enumerate(jobs):
            if self.stop.is_set():
                break
            self.q.put(("file", i, n, src.name))
            if dst.exists() and not overwrite:
                self.q.put(("log", f"skip    {dst.name} already exists"))
                skipped += 1
                continue

            def progress(done, total, fps, i=i):
                self.q.put(("progress", i, n, done, total, fps))
            try:
                P.upscale_video(src, dst, opt, progress=progress, stop=self.stop)
                self.last_output = dst
                ok += 1
            except P.Cancelled as e:
                self.q.put(("log", f"stopped {src.name}: {e}"))
                break
            except BaseException as e:  # noqa: BLE001 - keep the batch going
                failed += 1
                self.q.put(("log", f"FAILED  {src.name}: {e}"))
        self.q.put(("finished", ok, failed, skipped, self.stop.is_set()))

    # ------------------------------------------------------------------ actions
    def start(self):
        if self.busy or not self.files:
            return
        self._save_settings()
        opt = self._options()
        out_dir = self.v_outdir.get().strip()
        jobs = []
        for f in self.files:
            dst = output_path(f, opt.target, out_dir)
            if opt.duration:
                dst = dst.with_name(dst.stem + "_preview.mp4")
            jobs.append((f, dst))
        self.stop.clear()
        self._n_jobs = len(jobs)
        self.pb_all.configure(maximum=len(jobs), value=0)
        self.worker = threading.Thread(target=self._run_jobs, args=(jobs, opt, self.v_overwrite.get()),
                                       daemon=True)
        self.worker.start()
        self._sync_state()

    def request_stop(self):
        self.stop.set()
        self.status.configure(text="Stopping after the current frame ... "
                                   "(finished segments are kept - Start again to resume)")

    def compare(self):
        if self.busy or not self.files:
            return
        sel = self.listbox.curselection()
        src = self.files[sel[0] if sel else 0]
        opt = self._options()
        dst = src.with_name(f"{src.stem}_compare.png")
        out_dir = self.v_outdir.get().strip()
        if out_dir:
            dst = Path(out_dir) / dst.name

        def run():
            try:
                self.pipeline.compare(src, dst, opt, at=1.0)
                self.q.put(("compare", dst))
            except BaseException as e:  # noqa: BLE001
                self.q.put(("error", f"Preview failed: {e}"))
            self.q.put(("idle",))

        self.status.configure(text=f"Rendering preview frame of {src.name} ...")
        self.worker = threading.Thread(target=run, daemon=True)
        self.worker.start()
        self._sync_state()

    def open_output(self):
        if self.last_output is not None:
            open_path(self.last_output.parent)
        elif self.v_outdir.get().strip():
            open_path(Path(self.v_outdir.get().strip()))
        elif self.files:
            open_path(self.files[0].parent)

    def show_image(self, path: Path):
        win = tk.Toplevel(self.root)
        win.title(f"Before (left)  vs  after (right) - {path.name}")
        try:
            img = tk.PhotoImage(file=str(path))
            max_w = int(self.root.winfo_screenwidth() * 0.9)
            factor = 1
            while img.width() / factor > max_w:
                factor += 1
            if factor > 1:
                img = img.subsample(factor)
            lbl = ttk.Label(win, image=img)
            lbl.image = img
            lbl.pack()
            ttk.Label(win, text=f"100% crop. Full frame saved next to it. ({path})").pack(pady=4)
        except tk.TclError:
            win.destroy()
            open_path(path)

    # ------------------------------------------------------------------ events
    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg.rstrip() + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _poll(self):
        try:
            while True:
                self._handle(self.q.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, m):
        kind = m[0]
        if kind == "log":
            self._log(m[1])
        elif kind == "ready":
            _, self.pipeline, text, dev = m
            self.device_lbl.configure(text=text,
                                      foreground="#1a7f37" if dev != "cpu" else "#b35900")
            self._sync_state()
        elif kind == "fatal":
            self.device_lbl.configure(text=m[1], foreground="#c00")
            messagebox.showerror("AI Video Upscaler", m[1])
        elif kind == "file":
            _, i, n, name = m
            self.pb_file.configure(value=0, maximum=1)
            self.status.configure(text=f"[{i + 1}/{n}] {name}: loading models ...")
        elif kind == "progress":
            _, i, n, done, total, fps = m
            self.pb_file.configure(maximum=max(total, 1), value=done)
            self.pb_all.configure(maximum=n, value=i + done / max(total, 1))
            eta = (total - done) / fps if fps > 0 else 0
            self.status.configure(
                text=f"[{i + 1}/{n}] frame {done}/{total}   {fps:.2f} fps   "
                     f"ETA {int(eta // 60):02d}:{int(eta % 60):02d}")
        elif kind == "finished":
            _, ok, failed, skipped, stopped = m
            if not stopped:
                self.pb_all.configure(maximum=self._n_jobs, value=self._n_jobs)
                self.pb_file.configure(maximum=1, value=1)
            parts = [f"{ok} done"] + ([f"{failed} failed"] if failed else []) + (
                [f"{skipped} skipped"] if skipped else [])
            self.status.configure(text=("Stopped. " if stopped else "Finished. ")
                                  + ", ".join(parts))
            self.worker = None
            self._sync_state()
            if ok and not stopped:
                self.root.bell()
        elif kind == "compare":
            self.show_image(m[1])
            self.status.configure(text=f"Preview saved: {m[1]}")
        elif kind == "error":
            self._log(m[1])
            self.status.configure(text=m[1].splitlines()[0][:160])
        elif kind == "idle":
            self.worker = None
            self._sync_state()

    # ------------------------------------------------------------------ settings
    _KEYS = ("v_res", "v_quality", "v_content", "v_codec", "v_fit", "v_denoise",
             "v_deflicker", "v_grain", "v_outdir")

    def _load_settings(self):
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            for k in self._KEYS:
                if k in data:
                    getattr(self, k).set(data[k])
        except Exception:  # noqa: BLE001 - first run / corrupt file
            pass

    def _save_settings(self):
        try:
            SETTINGS_FILE.write_text(json.dumps({k: getattr(self, k).get() for k in self._KEYS}))
        except Exception:  # noqa: BLE001
            pass

    def on_close(self):
        if self.busy:
            if not messagebox.askyesno("AI Video Upscaler",
                                       "A job is running. Stop it and quit?\n"
                                       "(Finished segments are kept so you can resume.)"):
                return
            self.stop.set()
            self.worker.join(timeout=30)
        self._save_settings()
        self.root.destroy()


class _Tooltip:
    def __init__(self, widget, text):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _):
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=self.text, background="#ffffe0", relief="solid",
                 borderwidth=1, padx=6, pady=3).pack()

    def hide(self, _):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def main(argv: list[str] | None = None) -> int:
    if sys.platform == "win32":
        try:  # crisp text on high-DPI laptop screens
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:  # noqa: BLE001
            pass
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    style = ttk.Style(root)
    if sys.platform.startswith("linux") and "clam" in style.theme_names():
        style.theme_use("clam")
    App(root, list(sys.argv[1:] if argv is None else argv))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
