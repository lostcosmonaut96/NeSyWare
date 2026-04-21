import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Optional

from PIL import Image, ImageTk

import os
os.chdir(Path(__file__).parent)

from inference import NeSyWareInference, PREDICATE_LABELS, CATEGORY_NAMES
from pe_to_image import generate_visualization_pair


CLR_BG          = "#1e1e2e"  
CLR_PANEL       = "#2a2a3e"  
CLR_BORDER      = "#3a3a5c"  
CLR_ACCENT_BLUE = "#7aa2f7" 
CLR_ACCENT_GREEN= "#9ece6a"   
CLR_ACCENT_RED  = "#f7768e"   
CLR_ACCENT_YLW  = "#e0af68"   
CLR_TEXT        = "#cdd6f4"  
CLR_SUBTEXT     = "#7f849c"  
CLR_BTN_LOAD    = "#414868"  
CLR_BTN_ANALYZE = "#7aa2f7"   

FONT_TITLE  = ("Segoe UI", 16, "bold")
FONT_HEADER = ("Segoe UI", 11, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI",  9)
FONT_MONO   = ("Consolas",  9)

WIN_W, WIN_H = 820, 760


def _label(parent, text, font=FONT_BODY, fg=CLR_TEXT, bg=None, **kw):
    return tk.Label(parent, text=text, font=font,
                    fg=fg, bg=bg or CLR_BG, **kw)


def _separator(parent, bg=CLR_BORDER, height=1, pady=4):
    f = tk.Frame(parent, bg=bg, height=height)
    f.pack(fill="x", padx=8, pady=pady)
    return f


def _confidence_bar(parent, label: str, value: float, color: str):

    row = tk.Frame(parent, bg=CLR_PANEL)
    row.pack(fill="x", padx=12, pady=1)

    tk.Label(row, text=label, font=FONT_MONO, fg=CLR_TEXT,
             bg=CLR_PANEL, width=32, anchor="w").pack(side="left")


    bar_w = 180
    bar_h = 12
    cv = tk.Canvas(row, width=bar_w, height=bar_h,
                   bg=CLR_BORDER, highlightthickness=0)
    cv.pack(side="left", padx=(4, 6))
    fill_w = max(2, int(bar_w * value))
    cv.create_rectangle(0, 0, fill_w, bar_h, fill=color, outline="")


    tk.Label(row, text=f"{value:.2f}", font=FONT_MONO,
             fg=CLR_SUBTEXT, bg=CLR_PANEL, width=5).pack(side="left")

    return row



class NeSyWareApp:

    def __init__(self, root: tk.Tk):
        self.root       = root
        self.engine     = NeSyWareInference()
        self.filepath:  Optional[str] = None
        self._img_ref:  Optional[ImageTk.PhotoImage] = None   

        self._build_window()
        self._build_header()
        self._build_toolbar()
        self._build_status()
        self._build_content()

        self._set_status("Initialising models...", CLR_ACCENT_YLW)
        self.btn_analyze.config(state="disabled")
        threading.Thread(target=self._load_models, daemon=True).start()
        self.root.after(200, self._show_disclaimer)

    def _show_disclaimer(self):
        dlg = tk.Toplevel(self.root)
        dlg.title("Disclaimer")
        dlg.configure(bg=CLR_PANEL)
        dlg.resizable(False, False)
        dlg.grab_set()  # modal

        
        self.root.update_idletasks()
        x = self.root.winfo_x() + (WIN_W - 480) // 2
        y = self.root.winfo_y() + (WIN_H - 300) // 2
        dlg.geometry(f"480x300+{x}+{y}")

        tk.Label(dlg, text="⚠  DISCLAIMER",
                font=("Segoe UI", 13, "bold"),
                fg=CLR_ACCENT_YLW, bg=CLR_PANEL).pack(pady=(20, 8))

        msg = (
            "NeSyWare is an AI-based research prototype and may produce\n"
            "incorrect or incomplete classifications.\n\n"
            "Results should always be cross-validated with:\n"
            "  •  VirusTotal or equivalent multi-engine scanners\n"
            "  •  Static analysis tools (e.g. IDA Pro, Ghidra, PE-bear)\n"
            "  •  Dynamic / sandbox analysis (e.g. ANY.RUN, Cuckoo)\n\n"
            "Symbolic predicates reflect visual correlates of behaviour —\n"
            "not verified runtime actions.\n\n"
            "Do not use as a sole basis for security decisions."
        )
        tk.Label(dlg, text=msg,
                font=FONT_SMALL, fg=CLR_TEXT, bg=CLR_PANEL,
                justify="left").pack(padx=24, pady=(0, 16))

        tk.Button(dlg, text="  I Understand  ",
                font=FONT_BODY,
                fg="#1e1e2e", bg=CLR_ACCENT_BLUE,
                relief="flat", cursor="hand2",
                activebackground="#5a82d7",
                command=dlg.destroy,
                padx=10, pady=4).pack(pady=(0, 20))

        dlg.protocol("WM_DELETE_WINDOW", dlg.destroy)
        self.root.wait_window(dlg)

    def _build_window(self):
        self.root.title("NeSyWare v2.0 — Malware Analysis System")
        self.root.configure(bg=CLR_BG)
        self.root.geometry(f"{WIN_W}x{WIN_H}")
        self.root.minsize(WIN_W, WIN_H)
        self.root.resizable(True, True)

    def _build_header(self):
        hdr = tk.Frame(self.root, bg="#13131f", pady=12)
        hdr.pack(fill="x")

        tk.Label(hdr, text="NeSyWare v2.0",
                 font=("Segoe UI", 18, "bold"),
                 fg=CLR_ACCENT_BLUE, bg="#13131f").pack()
        tk.Label(hdr, text="Neuro-Symbolic Malware Analysis System",
                 font=FONT_SMALL, fg=CLR_SUBTEXT, bg="#13131f").pack()

    def _build_toolbar(self):
        bar = tk.Frame(self.root, bg=CLR_PANEL, pady=8)
        bar.pack(fill="x")

        self.btn_load = tk.Button(
            bar, text="  Load Executable  ",
            font=FONT_BODY, fg=CLR_TEXT, bg=CLR_BTN_LOAD,
            relief="flat", cursor="hand2",
            activebackground="#5a5f8a", activeforeground=CLR_TEXT,
            command=self._load_file, padx=8, pady=4,
        )
        self.btn_load.pack(side="left", padx=(12, 8))

        self.lbl_path = tk.Label(
            bar, text="No file selected",
            font=FONT_MONO, fg=CLR_SUBTEXT, bg=CLR_PANEL,
            anchor="w",
        )
        self.lbl_path.pack(side="left", fill="x", expand=True)


        self.btn_analyze = tk.Button(
            bar, text="  ▶  Analyze  ",
            font=("Segoe UI", 10, "bold"),
            fg="#1e1e2e", bg=CLR_BTN_ANALYZE,
            relief="flat", cursor="hand2",
            activebackground="#5a82d7", activeforeground="#1e1e2e",
            command=self._start_analysis, padx=10, pady=4,
        )
        self.btn_analyze.pack(side="right", padx=12)

    def _build_status(self):
        self.status_frame = tk.Frame(self.root, bg="#13131f", pady=4)
        self.status_frame.pack(fill="x")

        self.lbl_status = tk.Label(
            self.status_frame, text="",
            font=FONT_SMALL, fg=CLR_SUBTEXT, bg="#13131f", anchor="w",
        )
        self.lbl_status.pack(side="left", padx=12)

        self.progress = ttk.Progressbar(
            self.status_frame, mode="indeterminate", length=160,
        )
        self.progress.pack(side="right", padx=12)

    def _build_content(self):

        outer = tk.Frame(self.root, bg=CLR_BG)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, bg=CLR_BG, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame = tk.Frame(canvas, bg=CLR_BG)
        self.scroll_window = canvas.create_window(
            (0, 0), window=self.scroll_frame, anchor="nw"
        )

        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(self.scroll_window, width=canvas.winfo_width())

        self.scroll_frame.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_configure)


        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)


        self._build_visualization_section()
        self._build_results_section()

    def _build_visualization_section(self):
        sec = tk.Frame(self.scroll_frame, bg=CLR_PANEL,
                       relief="flat", bd=0)
        sec.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(sec, text="BINARY VISUALISATION",
                 font=("Segoe UI", 9, "bold"),
                 fg=CLR_SUBTEXT, bg=CLR_PANEL).pack(anchor="w", padx=10, pady=(8, 4))
        _separator(sec, pady=0)


        row = tk.Frame(sec, bg=CLR_PANEL)
        row.pack(fill="x", padx=10, pady=(6, 10))


        self.img_canvas = tk.Canvas(
            row, width=200, height=200,
            bg=CLR_BG, highlightthickness=1,
            highlightbackground=CLR_BORDER,
        )
        self.img_canvas.pack(side="left")
        self.img_canvas.create_text(
            100, 100, text="No image",
            fill=CLR_SUBTEXT, font=FONT_SMALL,
        )


        self.file_info_frame = tk.Frame(row, bg=CLR_PANEL)
        self.file_info_frame.pack(side="left", fill="both",
                                  expand=True, padx=(16, 0))
        self.lbl_file_info = tk.Label(
            self.file_info_frame,
            text="Load a PE executable to begin analysis.",
            font=FONT_BODY, fg=CLR_SUBTEXT, bg=CLR_PANEL,
            justify="left", anchor="nw",
        )
        self.lbl_file_info.pack(anchor="nw")

    def _build_results_section(self):
        self.results_outer = tk.Frame(self.scroll_frame, bg=CLR_BG)
        self.results_outer.pack(fill="x", padx=12, pady=(4, 12))


        hdr = tk.Frame(self.results_outer, bg=CLR_PANEL)
        hdr.pack(fill="x")
        tk.Label(hdr, text="ANALYSIS RESULTS",
                 font=("Segoe UI", 9, "bold"),
                 fg=CLR_SUBTEXT, bg=CLR_PANEL).pack(anchor="w", padx=10, pady=(8, 4))
        _separator(hdr, pady=0)


        self.results_frame = tk.Frame(self.results_outer, bg=CLR_PANEL)
        self.results_frame.pack(fill="x")

        tk.Label(self.results_frame,
                 text="Awaiting analysis...",
                 font=FONT_BODY, fg=CLR_SUBTEXT, bg=CLR_PANEL,
                 pady=20).pack()



    def _load_file(self):
        path = filedialog.askopenfilename(
            title="Select executable",
            filetypes=[
                ("PE Executables", "*.exe *.dll *.sys *.scr *.com"),
                ("All files",      "*.*"),
            ],
        )
        if not path:
            return

        self.filepath = path
        short = Path(path).name
        self.lbl_path.config(text=short, fg=CLR_TEXT)

        size_bytes = Path(path).stat().st_size
        size_str   = _format_size(size_bytes)
        self.lbl_file_info.config(
            text=f"File:  {short}\nSize:  {size_str}\nPath:  {path}",
            fg=CLR_TEXT,
        )

        threading.Thread(
            target=self._render_preview, args=(path,), daemon=True,
        ).start()

        self._set_status("File loaded. Press ▶ Analyse to start.", CLR_TEXT)

    def _render_preview(self, path: str):
        try:
            _, _, thumb = generate_visualization_pair(path)
            self.root.after(0, self._update_preview, thumb)
        except Exception as e:
            self.root.after(0, self._set_status,
                            f"Preview error: {e}", CLR_ACCENT_RED)

    def _update_preview(self, img: Image.Image):
        photo = ImageTk.PhotoImage(img)
        self._img_ref = photo
        self.img_canvas.delete("all")
        self.img_canvas.create_image(100, 100, image=photo)

    def _load_models(self):
        try:
            self.root.after(0, self.progress.start, 8)
            self.engine.load(
                progress_callback=lambda m:
                    self.root.after(0, self._set_status, m, CLR_ACCENT_YLW)
            )
            self.root.after(0, self.progress.stop)
            self.root.after(0, self.btn_analyze.config, {"state": "normal"})
            self.root.after(0, self._set_status,
                            "System ready.", CLR_ACCENT_GREEN)
        except Exception as e:
            self.root.after(0, self.progress.stop)
            self.root.after(0, self._set_status,
                            f"Model loading error: {e}", CLR_ACCENT_RED)

    def _start_analysis(self):
        if not self.filepath:
            messagebox.showwarning("Attenzione", "Seleziona prima un file eseguibile.")
            return

        self.btn_analyze.config(state="disabled")
        self.btn_load.config(state="disabled")
        self._set_status("Analysis in progress...", CLR_ACCENT_YLW)
        self.progress.start(8)

        for w in self.results_frame.winfo_children():
            w.destroy()
        tk.Label(self.results_frame, text="Analysis in progress...",
                 font=FONT_BODY, fg=CLR_ACCENT_YLW, bg=CLR_PANEL,
                 pady=20).pack()

        threading.Thread(
            target=self._analysis_thread,
            args=(self.filepath,),
            daemon=True,
        ).start()

    def _analysis_thread(self, path: str):
        result = self.engine.analyze(path)
        self.root.after(0, self._show_results, result)



    def _show_results(self, result: dict):
        self.progress.stop()
        self.btn_analyze.config(state="normal")
        self.btn_load.config(state="normal")


        for w in self.results_frame.winfo_children():
            w.destroy()

        if result.get("error"):
            self._set_status(f"Error: {result['error'][:80]}", CLR_ACCENT_RED)
            tk.Label(self.results_frame,
                     text=f"Analysis error:\n{result['error']}",
                     font=FONT_BODY, fg=CLR_ACCENT_RED, bg=CLR_PANEL,
                     pady=20, justify="left").pack(padx=12)
            return

        is_malware   = result["is_malware"]
        s1_conf      = result["stage1_confidence"] * 100
        s1_level     = result.get("stage1_level", "high")
        is_uncertain = result.get("is_uncertain", False)
        is_suspicious     = result.get("suspicious", False)
        suspicious_reason = result.get("suspicious_reason", "")

        self._section_header("STAGE 1 — BINARY CLASSIFICATION")

        if not is_malware and not is_suspicious:

            if s1_level == 'high':
                vbg, vfg = "#1a2e1a", CLR_ACCENT_GREEN
            elif s1_level == 'medium':
                vbg, vfg = "#2a2a1a", CLR_ACCENT_YLW
            else:
                vbg, vfg = "#2a1e14", "#ff9e64"

            self._set_status("Analysis complete — Benign file", vfg)
            verdict_frame = tk.Frame(self.results_frame, bg=vbg, relief="flat")
            verdict_frame.pack(fill="x", padx=10, pady=(6, 4))
            tk.Label(verdict_frame,
                     text="  ✓  BENIGN",
                     font=("Segoe UI", 14, "bold"),
                     fg=vfg, bg=vbg,
                     pady=10).pack(side="left", padx=8)
            tk.Label(verdict_frame,
                     text=f"Confidence: {s1_conf:.1f}%",
                     font=FONT_BODY, fg=CLR_TEXT, bg=vbg,
                     pady=10).pack(side="right", padx=16)
            return


        if is_suspicious:
            CLR_SUSPICIOUS_BG  = "#2a1a2e"   
            CLR_SUSPICIOUS_FG  = "#bb9af7" 
            self._set_status("Analysis complete — SUSPICIOUS (manual review required)",
                             CLR_SUSPICIOUS_FG)
            verdict_frame = tk.Frame(self.results_frame, bg=CLR_SUSPICIOUS_BG,
                                     relief="flat")
            verdict_frame.pack(fill="x", padx=10, pady=(6, 4))
            tk.Label(verdict_frame,
                     text="  \u26a0  SUSPICIOUS",
                     font=("Segoe UI", 14, "bold"),
                     fg=CLR_SUSPICIOUS_FG, bg=CLR_SUSPICIOUS_BG,
                     pady=10).pack(side="left", padx=8)
            tk.Label(verdict_frame,
                     text=f"Stage 1: {s1_conf:.1f}% benign",
                     font=FONT_BODY, fg=CLR_TEXT, bg=CLR_SUSPICIOUS_BG,
                     pady=10).pack(side="right", padx=16)
            tk.Label(self.results_frame,
                     text=f"\u26a0  {suspicious_reason}",
                     font=FONT_SMALL, fg=CLR_SUSPICIOUS_FG, bg=CLR_PANEL,
                     anchor="w", wraplength=680, justify="left",
                     pady=4).pack(fill="x", padx=18)
            tk.Label(self.results_frame,
                     text="\u2139  Stage 2/3 details shown below — manual review strongly recommended.",
                     font=FONT_SMALL, fg=CLR_SUSPICIOUS_FG, bg=CLR_PANEL,
                     anchor="w", pady=2).pack(fill="x", padx=18, pady=(0, 4))

        if not is_suspicious:
            if is_uncertain:
                vbg, vfg = "#2a1e14", "#ff9e64"
                verdict_text = "  ?  UNCERTAIN — Manual Review Recommended"
                status_msg   = "Analysis complete — Uncertain (low confidence)"
            elif s1_level == 'medium':
                vbg, vfg = "#2e2a1a", CLR_ACCENT_YLW
                verdict_text = "  ✗  MALWARE DETECTED"
                status_msg   = "Analysis complete — MALWARE detected (medium confidence)"
            else:
                vbg, vfg = "#2e1a1a", CLR_ACCENT_RED
                verdict_text = "  ✗  MALWARE DETECTED"
                status_msg   = "Analysis complete — MALWARE detected"

            self._set_status(status_msg, vfg)

            verdict_frame = tk.Frame(self.results_frame, bg=vbg, relief="flat")
            verdict_frame.pack(fill="x", padx=10, pady=(6, 4))
            tk.Label(verdict_frame,
                     text=verdict_text,
                     font=("Segoe UI", 14, "bold"),
                     fg=vfg, bg=vbg,
                     pady=10).pack(side="left", padx=8)
            tk.Label(verdict_frame,
                     text=f"Confidence: {s1_conf:.1f}%",
                     font=FONT_BODY, fg=CLR_TEXT, bg=vbg,
                     pady=10).pack(side="right", padx=16)

            if s1_level != 'high':
                legend = {
                    'medium':   ("⚠  Medium confidence — result may be unreliable", CLR_ACCENT_YLW),
                    'uncertain':("⚠  Low confidence — automated classification unreliable; manual review required", "#ff9e64"),
                }[s1_level]
                tk.Label(self.results_frame,
                         text=legend[0],
                         font=FONT_SMALL, fg=legend[1], bg=CLR_PANEL,
                         anchor="w", pady=4).pack(fill="x", padx=18)

            if is_uncertain:
           
                tk.Label(self.results_frame,
                         text="ℹ  Detailed analysis is shown below for reference only.",
                         font=FONT_SMALL, fg="#ff9e64", bg=CLR_PANEL,
                         anchor="w", pady=2).pack(fill="x", padx=18, pady=(0, 4))

        _separator(self.results_frame)


        self._section_header("STAGE 2 — CATEGORY CLASSIFICATION")

        cat      = result["category"] or "—"
        cat_conf = (result["category_conf"] or 0) * 100

        cat_row = tk.Frame(self.results_frame, bg=CLR_PANEL)
        cat_row.pack(fill="x", padx=10, pady=(4, 8))

        tk.Label(cat_row, text="Category:",
                 font=FONT_BODY, fg=CLR_SUBTEXT,
                 bg=CLR_PANEL).pack(side="left", padx=(8, 4))
        tk.Label(cat_row, text=cat,
                 font=("Segoe UI", 12, "bold"),
                 fg=CLR_ACCENT_BLUE, bg=CLR_PANEL).pack(side="left", padx=4)
        tk.Label(cat_row, text=f"{cat_conf:.1f}%",
                 font=("Segoe UI", 12, "bold"),
                 fg=CLR_ACCENT_YLW, bg=CLR_PANEL).pack(side="right", padx=12)


        top5_cat = result.get("top5_categories", [])
        if top5_cat:
            self._section_subheader("Top-5 candidates  (✓ consistent with predicted family)")
            dist_frame = tk.Frame(self.results_frame, bg=CLR_PANEL)
            dist_frame.pack(fill="x", padx=10, pady=(0, 6))
            for rank, entry in enumerate(top5_cat, 1):
                cname, cprob = entry[0], entry[1]
                tag  = entry[2] if len(entry) > 2 else ""
                clr  = CLR_ACCENT_BLUE if rank == 1 else CLR_BORDER
                row  = tk.Frame(dist_frame, bg=CLR_PANEL)
                row.pack(fill="x", padx=12, pady=1)
                tk.Label(row, text=f"{rank}.", font=FONT_MONO,
                         fg=CLR_SUBTEXT, bg=CLR_PANEL, width=2).pack(side="left")
                tk.Label(row, text=cname, font=FONT_MONO,
                         fg=CLR_TEXT if rank > 1 else CLR_ACCENT_BLUE,
                         bg=CLR_PANEL, width=22, anchor="w").pack(side="left")
                bar_w = 140
                cv = tk.Canvas(row, width=bar_w, height=12,
                               bg=CLR_BORDER, highlightthickness=0)
                cv.pack(side="left", padx=(4, 6))
                cv.create_rectangle(0, 0, max(2, int(bar_w * cprob)),
                                    12, fill=clr, outline="")
                tk.Label(row, text=f"{cprob*100:5.1f}%",
                         font=FONT_MONO, fg=CLR_SUBTEXT,
                         bg=CLR_PANEL, width=6).pack(side="left")
                if tag:
                    tk.Label(row, text=tag, font=FONT_MONO,
                             fg=CLR_ACCENT_GREEN if tag == "✓" else CLR_ACCENT_YLW,
                             bg=CLR_PANEL).pack(side="left", padx=4)

        _separator(self.results_frame)

        self._section_header("STAGE 3 — FAMILY PROBABILITY PROFILE")

        profile       = result.get("family_profile", [])
        profile_label = result.get("family_profile_label", "")
        is_anchor     = result["is_anchor_family"]

        _PROFILE_CFG = {
            "high_confidence": ("  ● High-confidence match",     CLR_ACCENT_GREEN),
            "moderate":        ("  ◑ Moderate confidence",       CLR_ACCENT_YLW),
            "ambiguous":       ("  ○ Ambiguous — inspect manually", "#ff9e64"),
            "inconclusive":    ("  ✕ Inconclusive — family unidentified", CLR_SUBTEXT),
        }
        badge_text, badge_color = _PROFILE_CFG.get(
            profile_label, ("", CLR_SUBTEXT))
        if badge_text:
            tk.Label(self.results_frame,
                     text=badge_text,
                     font=("Segoe UI", 10, "bold"),
                     fg=badge_color, bg=CLR_PANEL, anchor="w",
                     pady=4).pack(fill="x", padx=18)

        if profile_label == "inconclusive":
            tk.Label(self.results_frame,
                     text="  Probability spread too flat for reliable family assignment.\n"
                          "  No single family exceeds the minimum confidence threshold (20%).\n"
                          "  The sample may belong to an unknown/unseen family.",
                     font=FONT_SMALL, fg=CLR_SUBTEXT, bg=CLR_PANEL,
                     anchor="w", justify="left", pady=4).pack(fill="x", padx=18)

        elif is_anchor:
            tk.Label(
                self.results_frame,
                text="  ⚑  Top candidate is an anchor family — high-confidence behavioural profile",
                font=FONT_SMALL, fg=CLR_ACCENT_GREEN, bg=CLR_PANEL, anchor="w",
            ).pack(fill="x", padx=18, pady=(0, 2))

        if profile:
            n_shown = len(profile)
            self._section_subheader(
                f"Candidate families >= 8%  [{n_shown} shown]"
                "  (✓ consistent with predicted category)")
            fam_container = tk.Frame(self.results_frame, bg=CLR_PANEL)
            fam_container.pack(fill="x", padx=10, pady=(0, 6))

            top1_prob = profile[0][1] if profile else 1.0

            for rank, (fname, fconf, tag) in enumerate(profile, 1):
                is_top = (rank == 1)
                clr    = CLR_ACCENT_RED if is_top else CLR_BORDER
                row    = tk.Frame(fam_container, bg=CLR_PANEL)
                row.pack(fill="x", padx=12, pady=1)

                tk.Label(row, text=f"{rank}.", font=FONT_MONO,
                         fg=CLR_SUBTEXT, bg=CLR_PANEL, width=2).pack(side="left")
                tk.Label(row, text=fname, font=FONT_MONO,
                         fg=CLR_TEXT if not is_top else CLR_ACCENT_RED,
                         bg=CLR_PANEL, width=22, anchor="w").pack(side="left")

                bar_w    = 160
                bar_fill = max(2, int(bar_w * (fconf / top1_prob)))
                cv = tk.Canvas(row, width=bar_w, height=12,
                               bg=CLR_BORDER, highlightthickness=0)
                cv.pack(side="left", padx=(4, 6))
                cv.create_rectangle(0, 0, bar_fill, 12, fill=clr, outline="")

                tk.Label(row, text=f"{fconf*100:5.1f}%",
                         font=("Consolas", 9, "bold") if is_top else FONT_MONO,
                         fg=CLR_ACCENT_RED if is_top else CLR_SUBTEXT,
                         bg=CLR_PANEL, width=6).pack(side="left")
                tk.Label(row, text=tag, font=FONT_MONO,
                         fg=CLR_ACCENT_GREEN if tag == "✓" else CLR_ACCENT_YLW,
                         bg=CLR_PANEL).pack(side="left", padx=4)
        elif profile_label != "inconclusive":
            tk.Label(self.results_frame,
                     text="No family exceeds the 8% probability threshold.",
                     font=FONT_SMALL, fg=CLR_SUBTEXT, bg=CLR_PANEL,
                     pady=6).pack(padx=18, anchor="w")

        _separator(self.results_frame)


        active = result.get("active_predicates", [])
        self._section_header(f"SYMBOLIC PREDICATES — ACTIVE  ({len(active)})")

        if not active:
            tk.Label(self.results_frame,
                     text="No predicate exceeds the activation threshold.",
                     font=FONT_SMALL, fg=CLR_SUBTEXT, bg=CLR_PANEL,
                     pady=8).pack(padx=12, anchor="w")
        else:
            pred_container = tk.Frame(self.results_frame, bg=CLR_PANEL)
            pred_container.pack(fill="x", padx=0, pady=(2, 12))
            for pname, pval in active:
                label = PREDICATE_LABELS.get(pname, pname.replace("_", " ").title())
                if pval >= 0.80:
                    bar_color = CLR_ACCENT_RED
                elif pval >= 0.60:
                    bar_color = CLR_ACCENT_YLW
                else:
                    bar_color = CLR_ACCENT_BLUE
                _confidence_bar(pred_container, label, pval, bar_color)

        tk.Frame(self.results_frame, bg=CLR_PANEL, height=12).pack()

    def _section_subheader(self, title: str):
        tk.Label(self.results_frame, text=title,
                 font=FONT_SMALL, fg=CLR_SUBTEXT,
                 bg=CLR_PANEL, anchor="w").pack(fill="x", padx=22, pady=(2, 1))

    def _section_header(self, title: str):
        row = tk.Frame(self.results_frame, bg=CLR_PANEL)
        row.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(row, text=title,
                 font=("Segoe UI", 9, "bold"),
                 fg=CLR_SUBTEXT, bg=CLR_PANEL).pack(anchor="w", padx=2)
        _separator(self.results_frame, bg=CLR_BORDER, height=1, pady=1)



    def _set_status(self, msg: str, color: str = CLR_SUBTEXT):
        self.lbl_status.config(text=f"  {msg}", fg=color)


def _format_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"

def main():
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=CLR_BORDER,
        background=CLR_ACCENT_BLUE,
        thickness=6,
    )
    style.configure(
        "Vertical.TScrollbar",
        background=CLR_PANEL,
        troughcolor=CLR_BG,
        arrowcolor=CLR_SUBTEXT,
    )

    app = NeSyWareApp(root)
    root.mainloop()



if __name__ == "__main__":
    main()
