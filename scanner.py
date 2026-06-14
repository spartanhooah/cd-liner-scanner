#!/usr/bin/env python3
"""CD liner booklet scanner — guides the user through scanning, deskews pages, compiles PDF."""

import io
import os
import re
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import pytesseract
from PIL import Image, ImageTk
from deskew import determine_skew
import img2pdf
import numpy as np
from rapidfuzz import fuzz


MUSIC_DIR = Path.home() / "Music"
MATCH_THRESHOLD = 70
PREVIEW_MAX = (480, 360)


# ---------------------------------------------------------------------------
# Scanner detection
# ---------------------------------------------------------------------------

def detect_scanners() -> list[tuple[str, str]]:
    """Return list of (device_name, display_label) pairs from scanimage -L."""
    try:
        result = subprocess.run(
            ["scanimage", "-L"],
            capture_output=True, text=True, timeout=15
        )
        devices = []
        for line in result.stdout.splitlines():
            # Lines look like: device `xerox_mfp:libusb:...' is a Xerox B205 ...
            m = re.match(r"device `([^']+)' is a (.+)", line)
            if m:
                devices.append((m.group(1), m.group(2).strip()))
        return devices
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return []


# ---------------------------------------------------------------------------
# Music directory matching
# ---------------------------------------------------------------------------

def find_best_match(query: str) -> tuple[Path, int] | None:
    """Walk MUSIC_DIR and return (best_path, score) or None."""
    if not MUSIC_DIR.exists():
        return None
    best_path, best_score = None, 0
    for dirpath in MUSIC_DIR.rglob("*"):
        if not dirpath.is_dir():
            continue
        rel_parts = " ".join(dirpath.relative_to(MUSIC_DIR).parts)
        score = fuzz.token_sort_ratio(query.lower(), rel_parts.lower())
        if score > best_score:
            best_score = score
            best_path = dirpath
    if best_score >= MATCH_THRESHOLD:
        return best_path, best_score
    return None


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def deskew_image(img: Image.Image) -> Image.Image:
    """Return a deskewed copy of img, or img unchanged if skew is negligible."""
    grayscale = np.array(img.convert("L"))
    angle = determine_skew(grayscale)
    if angle is None or abs(angle) < 0.5 or abs(angle) > 45:
        return img
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))


_ROTATION_ANGLES = {
    "90° clockwise":         270,  # PIL rotates CCW, so CW = 270
    "90° counter-clockwise": 90,
    "180°":                  180,
}

def auto_orient(
    img: Image.Image, hint: int | None = None, mode: str = "auto"
) -> tuple[Image.Image, int | None]:
    """Rotate portrait images to landscape when the booklet was placed sideways.

    mode is the rotation_var value from the UI. When it is not 'auto' the angle
    is applied directly without OCR. hint carries the OCR result across pages so
    artwork-only pages stay consistent with text-rich ones.
    """
    w, h = img.size

    # Explicit rotation: always apply regardless of aspect ratio.
    # Cover panels are nearly square so h <= w*1.3, but they still need
    # the same rotation as interior spreads when the user picked a direction.
    if mode in _ROTATION_ANGLES:
        angle = _ROTATION_ANGLES[mode]
        return img.rotate(angle, expand=True), angle

    # auto: skip images that are already landscape or nearly square
    if h <= w * 1.3:
        return img, hint

    # auto — try OCR if we have no hint yet
    if hint is not None:
        return img.rotate(hint, expand=True), hint

    best_angle, best_count = 90, -1
    for angle in (90, 270):
        thumb = img.rotate(angle, expand=True).copy()
        thumb.thumbnail((1200, 1200), Image.LANCZOS)
        try:
            data = pytesseract.image_to_data(
                thumb, output_type=pytesseract.Output.DICT, config="--psm 3"
            )
            count = sum(
                1 for conf, text in zip(data["conf"], data["text"])
                if int(conf) > 30 and text.strip()
            )
        except Exception:
            count = 0
        if count > best_count:
            best_count, best_angle = count, angle

    if best_count > 0:
        return img.rotate(best_angle, expand=True), best_angle
    return img, hint


def autocrop(img: Image.Image, margin: int = 30) -> Image.Image:
    """Crop to the bounding box of non-background content, leaving a small margin.

    Samples actual border pixels to set the threshold adaptively, so JPEG
    compression artifacts in the scanner glass don't fool a fixed cutoff.
    """
    arr = np.array(img.convert("L"))
    h, w = arr.shape

    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg_level = float(np.percentile(border, 95))
    threshold = max(bg_level - 20, 0)

    mask = arr < threshold
    if not mask.any():
        return img

    # Require ≥5% of pixels in a row/column to be non-background.
    # Scanner glass columns contain JPEG quantization noise and platen
    # imperfections that can reach ~1-2% density; real booklet content
    # columns are typically 20%+ dark, so 5% cleanly separates them.
    rows = np.where(mask.mean(axis=1) > 0.05)[0]
    cols = np.where(mask.mean(axis=0) > 0.05)[0]
    if not len(rows) or not len(cols):
        return img
    top    = max(0, rows[0]  - margin)
    bottom = min(h, rows[-1] + margin + 1)
    left   = max(0, cols[0]  - margin)
    right  = min(w, cols[-1] + margin + 1)
    return img.crop((left, top, right, bottom))


def make_thumbnail(img: Image.Image) -> ImageTk.PhotoImage:
    thumb = img.copy()
    thumb.thumbnail(PREVIEW_MAX, Image.LANCZOS)
    return ImageTk.PhotoImage(thumb)


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class LinerScannerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CD Liner Scanner")

        self.device: str | None = None
        self.temp_dir: str = tempfile.mkdtemp(prefix="liner_scan_")
        self.scanned_paths: list[str] = []
        self.save_dir: Path | None = None
        self.album_name: str = ""
        self._photo: ImageTk.PhotoImage | None = None  # keep reference

        self._build_ui()
        self.after(100, self._detect_scanner)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        pad = {"padx": 12, "pady": 6}

        # Top status bar
        self.status_var = tk.StringVar(value="Detecting scanner…")
        status_frame = tk.Frame(self, bg="#2b2b2b")
        status_frame.pack(fill=tk.X)
        tk.Label(
            status_frame, textvariable=self.status_var,
            bg="#2b2b2b", fg="#e0e0e0", font=("Sans", 11),
            anchor="w", **pad
        ).pack(fill=tk.X)

        # Phase 1 — album setup
        self.setup_frame = tk.Frame(self)
        self.setup_frame.pack(fill=tk.BOTH, expand=True, **pad)
        self._build_setup_frame()

        # Phase 2 — scanning (hidden initially)
        self.scan_frame = tk.Frame(self)
        self._build_scan_frame()

        # Phase 3 — processing (hidden initially)
        self.proc_frame = tk.Frame(self)
        self._build_proc_frame()

    def _build_setup_frame(self):
        f = self.setup_frame
        tk.Label(f, text="Album / Artist name:", font=("Sans", 11)).grid(
            row=0, column=0, sticky="w", pady=(8, 2))
        self.album_entry = tk.Entry(f, font=("Sans", 11), width=40)
        self.album_entry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.album_entry.bind("<Return>", lambda _: self._resolve_directory())

        tk.Button(
            f, text="Find Album Directory", font=("Sans", 11),
            command=self._resolve_directory
        ).grid(row=2, column=0, sticky="w")

        # Match result area
        self.match_var = tk.StringVar()
        tk.Label(f, textvariable=self.match_var, font=("Sans", 10),
                 wraplength=460, justify="left", fg="#444").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 2))

        self.dir_var = tk.StringVar()
        dir_frame = tk.Frame(f)
        dir_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        tk.Label(dir_frame, text="Save to:", font=("Sans", 10)).pack(side=tk.LEFT)
        tk.Entry(dir_frame, textvariable=self.dir_var, font=("Sans", 10),
                 width=38, state="readonly").pack(side=tk.LEFT, padx=(6, 4))
        tk.Button(dir_frame, text="Browse…", command=self._browse_dir).pack(side=tk.LEFT)

        rot_frame = tk.Frame(f)
        rot_frame.grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 4))
        tk.Label(rot_frame, text="Page rotation:", font=("Sans", 10)).pack(side=tk.LEFT)
        self.rotation_var = tk.StringVar(value="auto")
        rot_menu = ttk.Combobox(
            rot_frame, textvariable=self.rotation_var, state="readonly",
            font=("Sans", 10), width=24,
            values=["auto", "90° clockwise", "90° counter-clockwise", "180°"],
        )
        rot_menu.pack(side=tk.LEFT, padx=(6, 0))

        self.start_btn = tk.Button(
            f, text="Start Scanning", font=("Sans", 11, "bold"),
            state=tk.DISABLED, command=self._start_scanning
        )
        self.start_btn.grid(row=6, column=0, sticky="w", pady=(4, 8))

    def _build_scan_frame(self):
        f = self.scan_frame
        self.instr_var = tk.StringVar()
        tk.Label(f, textvariable=self.instr_var, font=("Sans", 11),
                 wraplength=520, justify="left").pack(anchor="w", padx=12, pady=(10, 4))

        self.preview_canvas = tk.Canvas(
            f, bg="#d0d0d0", width=PREVIEW_MAX[0], height=PREVIEW_MAX[1],
            highlightthickness=0
        )
        self.preview_canvas.pack(padx=12, pady=6)

        self.page_var = tk.StringVar(value="No pages scanned yet")
        tk.Label(f, textvariable=self.page_var, font=("Sans", 10), fg="#555").pack(
            anchor="w", padx=12)

        btn_frame = tk.Frame(f)
        btn_frame.pack(padx=12, pady=10, anchor="w")
        self.scan_btn = tk.Button(
            btn_frame, text="Scan Page", font=("Sans", 12, "bold"),
            width=16, command=self._do_scan
        )
        self.scan_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.done_btn = tk.Button(
            btn_frame, text="Done — Compile PDF", font=("Sans", 12),
            width=20, state=tk.DISABLED, command=self._compile_pdf
        )
        self.done_btn.pack(side=tk.LEFT)

    def _build_proc_frame(self):
        f = self.proc_frame
        self.proc_var = tk.StringVar(value="Processing…")
        tk.Label(f, textvariable=self.proc_var, font=("Sans", 11)).pack(pady=(30, 10))
        self.progress = ttk.Progressbar(f, mode="determinate", length=400)
        self.progress.pack(pady=6)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _show_phase(self, phase: int):
        for frame in (self.setup_frame, self.scan_frame, self.proc_frame):
            frame.pack_forget()
        if phase == 1:
            self.setup_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        elif phase == 2:
            self.scan_frame.pack(fill=tk.BOTH, expand=True)
        elif phase == 3:
            self.proc_frame.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Scanner detection
    # ------------------------------------------------------------------

    def _detect_scanner(self):
        self.status_var.set("Detecting scanner…")
        devices = detect_scanners()
        if not devices:
            self.status_var.set("No scanner found")
            messagebox.showerror(
                "Scanner not found",
                "No SANE-compatible scanner was detected.\n\n"
                "Make sure 'sane' is installed and you are in the 'scanner' group:\n"
                "  sudo pacman -S sane\n"
                "  sudo usermod -aG scanner $USER\n\n"
                "Then log out and back in."
            )
            return
        if len(devices) == 1:
            self.device = devices[0][0]
            self.status_var.set(f"Scanner: {devices[0][1]}")
        else:
            self._pick_scanner(devices)
        self._show_phase(1)

    def _pick_scanner(self, devices: list[tuple[str, str]]):
        win = tk.Toplevel(self)
        win.title("Select Scanner")
        win.grab_set()
        tk.Label(win, text="Multiple scanners found. Choose one:",
                 font=("Sans", 11)).pack(padx=16, pady=(12, 6))
        choice = tk.StringVar(value=devices[0][0])
        for dev, label in devices:
            tk.Radiobutton(win, text=label, variable=choice, value=dev,
                           font=("Sans", 10)).pack(anchor="w", padx=24)
        def confirm():
            self.device = choice.get()
            win.destroy()
        tk.Button(win, text="OK", command=confirm).pack(pady=10)
        self.wait_window(win)

    # ------------------------------------------------------------------
    # Phase 1 — album setup
    # ------------------------------------------------------------------

    def _resolve_directory(self):
        query = self.album_entry.get().strip()
        if not query:
            return
        self.album_name = query
        result = find_best_match(query)
        if result:
            path, score = result
            self.dir_var.set(str(path))
            self.save_dir = path
            self.match_var.set(f"Match found ({score}% confidence): {path}")
            self.start_btn.config(state=tk.NORMAL)
        else:
            self.match_var.set("No match found in ~/Music — please choose a directory.")
            self._browse_dir()

    def _browse_dir(self):
        chosen = filedialog.askdirectory(
            title="Choose save directory",
            initialdir=str(MUSIC_DIR) if MUSIC_DIR.exists() else str(Path.home())
        )
        if chosen:
            self.save_dir = Path(chosen)
            self.dir_var.set(chosen)
            self.start_btn.config(state=tk.NORMAL)

    def _start_scanning(self):
        if not self.save_dir:
            return
        self.scanned_paths.clear()
        self._update_instruction()
        self._show_phase(2)
        self.status_var.set(f"Scanning: {self.album_name}")

    # ------------------------------------------------------------------
    # Phase 2 — scanning
    # ------------------------------------------------------------------

    def _update_instruction(self):
        n = len(self.scanned_paths)
        if n == 0:
            msg = "Place the front cover (single panel) face-down on the scanner, then click Scan Page."
        elif n == 1:
            msg = "Place the first interior spread (double panel) face-down, then click Scan Page."
        else:
            msg = (
                f"Place the next spread face-down, then click Scan Page.  "
                f"When the entire booklet is scanned, click Done."
            )
        self.instr_var.set(msg)

    def _do_scan(self):
        if not self.device:
            messagebox.showerror("Error", "No scanner device selected.")
            return
        self.scan_btn.config(state=tk.DISABLED)
        self.done_btn.config(state=tk.DISABLED)
        self.status_var.set("Scanning… please wait")
        threading.Thread(target=self._scan_worker, daemon=True).start()

    def _scan_worker(self):
        # Talk eSCL directly so we can explicitly request Platen and JPEG
        # (the SANE eSCL backend mis-selects ADF with this scanner).
        if "escl:" not in self.device:
            self.after(0, self._scan_error,
                       f"Device '{self.device}' does not appear to be an eSCL scanner.\n\n"
                       "Only eSCL network scanners are supported by this app.")
            return
        base_url = self.device.split("escl:", 1)[1]  # "https://192.168.2.187:443"
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        idx = len(self.scanned_paths)
        out_path = os.path.join(self.temp_dir, f"page_{idx:03d}.tiff")

        body = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<scan:ScanSettings'
            ' xmlns:pwg="http://www.pwg.org/schemas/2010/12/sm"'
            ' xmlns:scan="http://schemas.hp.com/imaging/escl/2011/05/03">'
            "<pwg:Version>2.0</pwg:Version>"
            "<scan:Intent>Photo</scan:Intent>"
            "<pwg:InputSource>Platen</pwg:InputSource>"
            "<scan:DocumentFormatExt>image/jpeg</scan:DocumentFormatExt>"
            "<scan:ColorMode>RGB24</scan:ColorMode>"
            "<scan:XResolution>300</scan:XResolution>"
            "<scan:YResolution>300</scan:YResolution>"
            "</scan:ScanSettings>"
        ).encode()

        try:
            req = urllib.request.Request(
                f"{base_url}/eSCL/ScanJobs",
                data=body,
                headers={"Content-Type": "text/xml"},
                method="POST",
            )
            with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
                location = resp.headers.get("Location", "")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Scanner busy with a stuck job — check its status
                try:
                    with urllib.request.urlopen(
                        urllib.request.Request(f"{base_url}/eSCL/ScannerStatus"),
                        context=ctx, timeout=10
                    ) as sr:
                        status_xml = sr.read().decode()
                    state = re.search(r"<pwg:State>(\w+)</pwg:State>", status_xml)
                    state = state.group(1) if state else "Unknown"
                except Exception:
                    state = "Unknown"
                self.after(0, self._scan_error,
                           f"Scanner is busy (state: {state}).\n\n"
                           "A previous scan job may be stuck. Wait a moment and try again.")
            else:
                self.after(0, self._scan_error, f"Could not start scan (HTTP {e.code}).")
            return
        except Exception as e:
            self.after(0, self._scan_error, f"Could not connect to scanner:\n{e}")
            return

        if not location:
            self.after(0, self._scan_error, "Scanner did not return a job location.")
            return

        job_url = location if location.startswith("http") else f"{base_url}{location}"
        doc_url = f"{job_url}/NextDocument"

        def _delete_job():
            try:
                urllib.request.urlopen(
                    urllib.request.Request(job_url, method="DELETE"), context=ctx, timeout=5
                )
            except Exception:
                pass

        jpeg_data = None
        deadline = time.time() + 120
        # This scanner returns 404/410 during warm-up instead of 503.
        # Only treat no-document as a real failure after the warm-up window.
        no_doc_since = None
        NO_DOC_TIMEOUT = 30
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(doc_url), context=ctx, timeout=15
                ) as resp:
                    jpeg_data = resp.read()
                break
            except urllib.error.HTTPError as e:
                if e.code == 503:
                    time.sleep(1)
                elif e.code in (404, 410):
                    if no_doc_since is None:
                        no_doc_since = time.time()
                    elif time.time() - no_doc_since > NO_DOC_TIMEOUT:
                        _delete_job()
                        self.after(0, self._scan_error,
                                   "Scanner returned no image.\n\n"
                                   "Make sure the booklet is placed face-down on the flatbed glass.")
                        return
                    time.sleep(1)
                else:
                    _delete_job()
                    self.after(0, self._scan_error, f"Scan error (HTTP {e.code})")
                    return
            except Exception as e:
                _delete_job()
                self.after(0, self._scan_error, f"Network error:\n{e}")
                return

        _delete_job()

        if jpeg_data is None:
            self.after(0, self._scan_error, "Scan timed out waiting for document.")
            return

        try:
            img = Image.open(io.BytesIO(jpeg_data))
            img.save(out_path, format="TIFF", compression="lzw")
        except Exception as e:
            self.after(0, self._scan_error, f"Could not save scan:\n{e}")
            return

        self.after(0, self._scan_done, out_path)

    def _scan_done(self, path: str):
        self.scanned_paths.append(path)
        count = len(self.scanned_paths)
        self.page_var.set(f"{count} page{'s' if count != 1 else ''} scanned")
        try:
            img = Image.open(path)
            self._photo = make_thumbnail(img)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                PREVIEW_MAX[0] // 2, PREVIEW_MAX[1] // 2,
                anchor="center", image=self._photo
            )
        except Exception:
            pass
        self._update_instruction()
        self.status_var.set(f"Scanner ready — {count} page{'s' if count != 1 else ''} scanned")
        self.scan_btn.config(state=tk.NORMAL)
        self.done_btn.config(state=tk.NORMAL)

    def _scan_error(self, msg: str):
        self.status_var.set("Scan failed — see error dialog")
        messagebox.showerror("Scan failed", msg)
        self.scan_btn.config(state=tk.NORMAL)
        if self.scanned_paths:
            self.done_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Phase 3 — deskew + PDF compilation
    # ------------------------------------------------------------------

    def _compile_pdf(self):
        if not self.scanned_paths:
            messagebox.showwarning("Nothing to compile", "No pages have been scanned yet.")
            return
        self._show_phase(3)
        self.progress["maximum"] = len(self.scanned_paths) + 1
        self.progress["value"] = 0
        rotation_mode = self.rotation_var.get()
        threading.Thread(target=self._compile_worker, args=(rotation_mode,), daemon=True).start()

    def _compile_worker(self, rotation_mode: str):
        corrected: list[str] = []
        rotation_hint: int | None = None
        for i, src in enumerate(self.scanned_paths):
            n = len(self.scanned_paths)
            self.after(0, self.proc_var.set, f"Processing page {i + 1} of {n}…")
            img = Image.open(src)

            pre = img.size
            try:
                img = autocrop(img)
            except Exception as e:
                self.after(0, self.proc_var.set, f"Page {i+1}: crop failed — {e}")
            else:
                self.after(0, self.proc_var.set,
                           f"Page {i+1}: crop {pre} → {img.size}")

            try:
                img, rotation_hint = auto_orient(img, rotation_hint, rotation_mode)
            except Exception as e:
                self.after(0, self.proc_var.set, f"Page {i+1}: orientation failed — {e}")

            try:
                img = deskew_image(img)
            except Exception as e:
                self.after(0, self.proc_var.set, f"Page {i+1}: deskew failed — {e}")

            try:
                img = autocrop(img, margin=0)
            except Exception as e:
                self.after(0, self.proc_var.set, f"Page {i+1}: final crop failed — {e}")

            dest = src.replace(".tiff", "_processed.tiff")
            img.save(dest, format="TIFF", compression="lzw")
            corrected.append(dest)
            self.after(0, self._progress_step)

        self.after(0, self.proc_var.set, "Compiling PDF…")
        if not self.save_dir:
            self.after(0, messagebox.showerror, "No save directory",
                       "No save directory is set. Please go back and select a directory.")
            self.after(0, self._show_phase, 2)
            return
        safe_name = re.sub(r'[\\/:*?"<>|]', "_", self.album_name) or "liner_notes"
        pdf_path = self.save_dir / f"{safe_name}.pdf"
        try:
            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(corrected))
        except Exception as e:
            self.after(0, messagebox.showerror, "PDF failed",
                       f"Could not write PDF:\n{e}")
            self.after(0, self._show_phase, 2)
            return

        self.after(0, self._progress_step)
        self.after(0, self._compile_done, str(pdf_path))

    def _progress_step(self):
        self.progress["value"] += 1

    def _compile_done(self, pdf_path: str):
        self.status_var.set("Done")
        answer = messagebox.askquestion(
            "PDF saved",
            f"PDF saved to:\n{pdf_path}\n\nScan another booklet?",
            icon="info"
        )
        self._cleanup_temp()
        self.temp_dir = tempfile.mkdtemp(prefix="liner_scan_")
        self.scanned_paths.clear()
        if answer == "yes":
            self.album_entry.delete(0, tk.END)
            self.match_var.set("")
            self.dir_var.set("")
            self.save_dir = None
            self.start_btn.config(state=tk.DISABLED)
            self.preview_canvas.delete("all")
            self.page_var.set("No pages scanned yet")
            self._show_phase(1)
            self.status_var.set("Ready")
        else:
            self.destroy()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup_temp(self):
        if self.temp_dir and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def destroy(self):
        self._cleanup_temp()
        super().destroy()


if __name__ == "__main__":
    app = LinerScannerApp()
    app.mainloop()
