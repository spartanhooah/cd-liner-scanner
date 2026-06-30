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
    # Clamp to ±15°: genuine flatbed skew is always small. Larger angles indicate
    # a major orientation error that deskew should not try to fix.
    if angle is None or abs(angle) < 0.5 or abs(angle) > 15:
        return img
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))



# Standard CD booklet panel: 120 mm × 120 mm at 300 dpi.
# Interior spreads are two panels side by side (240 mm wide).
_CD_PANEL_PX  = round(120 * 300 / 25.4)   # 1417 px
_CD_SPREAD_PX = round(240 * 300 / 25.4)   # 2835 px

# Fallback crop constants used when auto-detection fails.
# _SCAN_RIGHT_GAP_PX: scanner-frame margin (px) between img.width and the physical
#                     booklet edge after 90° CW rotation.  Adjust if detection
#                     keeps failing and content is consistently cut or padded.
_SCAN_RIGHT_GAP_PX = 50

# Threshold below which a pixel is considered "booklet content" rather than
# blank scanner glass.  240 catches coloured artwork and black text.
_CONTENT_THRESHOLD = 240

# How close (px) to the expected right edge detected content must be to count
# as "reaching the physical booklet edge" rather than an interior text margin.
_EDGE_TOLERANCE_PX = 200


def _detect_right_edge(paths: list[str]) -> int:
    """Return the right_edge_px of the booklet in the 90°-CW-rotated images.

    The booklet is pressed flush to the scanner glass short edge, so its right
    edge in the landscape output is nearly constant across pages.  Detected from
    the cover (paths[0]) or back (paths[-1]) first — those have full-bleed art
    that reaches the physical panel edge.  Interior low-contrast spreads are
    skipped; the rightmost ink pixel sits inside the panel, not at its edge.
    Falls back to img.width - _SCAN_RIGHT_GAP_PX if all detection fails.

    Panel height is NOT detected here.  CD booklet panels are a fixed physical
    size (120 mm square at 300 dpi = _CD_PANEL_PX px).  Detecting height from
    content rows risks including scanner-edge shadows or album-art borders that
    push the boundary beyond the actual panel edge, adding whitespace at the
    bottom.  height = _CD_PANEL_PX always.
    """
    best_var, best_path = -1.0, paths[0]
    img_width = None
    for p in paths:
        img = Image.open(p).rotate(270, expand=True)
        if img_width is None:
            img_width = img.width
            print(f"[crop-debug] rotated size {img.width}×{img.height}")
        thumb = img.copy()
        thumb.thumbnail((400, 300), Image.LANCZOS)
        v = float(np.array(thumb.convert("L")).var())
        if v > best_var:
            best_var, best_path = v, p

    right = None
    for p in dict.fromkeys([paths[0], paths[-1], best_path]):
        img = Image.open(p).rotate(270, expand=True)
        a = np.array(img.convert("L"))
        cols = np.where((a < _CONTENT_THRESHOLD).any(axis=0))[0]
        if len(cols) < 2:
            continue
        candidate = int(cols[-1])
        # Accept only if content reaches within _EDGE_TOLERANCE_PX of img.width.
        # A farther position means we found an interior ink pixel, not the edge.
        if img.width - candidate <= _EDGE_TOLERANCE_PX:
            right = candidate
            print(f"[crop-debug] right edge {right} from {Path(p).name}")
            break

    if right is None:
        right = (img_width or 0) - _SCAN_RIGHT_GAP_PX
        print(f"[crop-debug] right edge fallback: {right}")

    return right


def _find_vertical_top(img: Image.Image) -> int:
    """Slide the _CD_PANEL_PX-tall window to the best vertical position for this page.

    - If content spans most of _CD_PANEL_PX, anchor at the first content row
      (the physical top edge of the panel is visible in the scan).
    - If content is narrower (text with margins, sparse ink), centre the window
      over the detected content band.
    - If nothing is detectable (white paper on white glass), centre in the image.
    """
    arr = np.array(img.convert("L"))
    rows = np.where((arr < _CONTENT_THRESHOLD).any(axis=1))[0]

    if len(rows) == 0:
        top = (img.height - _CD_PANEL_PX) // 2
    else:
        top_row = int(rows[0])
        bot_row = int(rows[-1])
        if bot_row - top_row >= _CD_PANEL_PX * 0.8:
            # Full-bleed page — physical top edge detected
            top = top_row
        else:
            # Margins or sparse content — centre the window on what we found
            mid = (top_row + bot_row) // 2
            top = mid - _CD_PANEL_PX // 2

    return max(0, min(top, img.height - _CD_PANEL_PX))


def crop_to_booklet(
    img: Image.Image, top: int, right: int, is_spread: bool
) -> Image.Image:
    """Crop img to booklet dimensions using per-page top and session-wide right edge."""
    crop_w = _CD_SPREAD_PX if is_spread else _CD_PANEL_PX
    left   = max(0, right - crop_w)
    bottom = min(img.height, top + _CD_PANEL_PX)
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
        threading.Thread(target=self._compile_worker, daemon=True).start()

    def _compile_worker(self):
        corrected: list[str] = []
        n = len(self.scanned_paths)

        # Detect right edge once for the whole session.
        # Panel height is always _CD_PANEL_PX — detecting it from content rows
        # risks picking up scanner-edge shadows that inflate the value and add
        # whitespace at the bottom.  Vertical position is found per-page.
        self.after(0, self.proc_var.set, "Analysing pages for crop bounds…")
        try:
            right = _detect_right_edge(self.scanned_paths)
        except Exception as e:
            print(f"[crop-debug] right-edge detection failed ({e}), using fallback")
            right = None

        for i, src in enumerate(self.scanned_paths):
            self.after(0, self.proc_var.set, f"Processing page {i + 1} of {n}…")
            img = Image.open(src)

            img = img.rotate(270, expand=True)  # 90° clockwise (PIL rotates CCW)

            page_right = right if right is not None else img.width - _SCAN_RIGHT_GAP_PX

            try:
                img = deskew_image(img)
            except Exception as e:
                self.after(0, self.proc_var.set, f"Page {i+1}: deskew failed — {e}")

            # Slide the _CD_PANEL_PX-tall window to this page's content.
            top = _find_vertical_top(img)

            # First and last pages are single panels; interior pages are spreads.
            img = crop_to_booklet(img, top, page_right, is_spread=(0 < i < n - 1))

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
