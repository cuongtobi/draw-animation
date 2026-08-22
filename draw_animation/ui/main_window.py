from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from draw_animation.config import RenderConfig
from draw_animation.services.batch_render_service import BatchRenderService


class MainWindow(tk.Tk):
    POLL_MS = 80

    def __init__(self, batch_service: BatchRenderService | None = None) -> None:
        super().__init__()
        self.title("Draw Animation")
        self.geometry("1020x780")
        self.minsize(900, 680)

        self._batch_service = batch_service or BatchRenderService()
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker: threading.Thread | None = None

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.duration_var = tk.DoubleVar(value=8.0)
        self.fps_var = tk.StringVar(value="30")
        self.max_edge_var = tk.StringVar(value="1080")
        self.path_mode_var = tk.StringVar(value="skeleton")
        self.grid_var = tk.IntVar(value=8)
        self.skeleton_spacing_var = tk.DoubleVar(value=2.5)
        self.skeleton_min_points_var = tk.IntVar(value=8)
        self.ink_radius_var = tk.IntVar(value=4)
        self.show_hand_var = tk.BooleanVar(value=True)
        self.hand_image_var = tk.StringVar()
        self.hand_height_var = tk.IntVar(value=420)
        self.hand_anchor_x_var = tk.DoubleVar(value=0.0)
        self.hand_anchor_y_var = tk.DoubleVar(value=0.0)
        self.match_bg_var = tk.BooleanVar(value=True)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.after(self.POLL_MS, self._drain_events)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=14)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(9, weight=1)

        ttk.Label(root, text="Input image folder").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(root, textvariable=self.input_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Browse…", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(root, text="Output folder").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(root, textvariable=self.output_var).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(root, text="Browse…", command=self._browse_output).grid(row=1, column=2)

        options = ttk.LabelFrame(root, text="Render settings", padding=10)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 8))
        for column in range(8):
            options.columnconfigure(column, weight=1 if column % 2 else 0)

        ttk.Label(options, text="Duration/image (s)").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(options, from_=1.0, to=120.0, increment=0.5, textvariable=self.duration_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 18))
        ttk.Label(options, text="FPS").grid(row=0, column=2, sticky="w")
        ttk.Combobox(options, values=("24", "25", "30", "50", "60"), textvariable=self.fps_var, state="readonly", width=7).grid(row=0, column=3, sticky="w", padx=(6, 18))
        ttk.Label(options, text="Max long edge").grid(row=0, column=4, sticky="w")
        ttk.Combobox(options, values=("480", "720", "1080", "1440", "2160"), textvariable=self.max_edge_var, state="readonly", width=8).grid(row=0, column=5, sticky="w", padx=(6, 18))
        ttk.Label(options, text="Path mode").grid(row=0, column=6, sticky="w")
        ttk.Combobox(options, values=("skeleton", "grid"), textvariable=self.path_mode_var, state="readonly", width=10).grid(row=0, column=7, sticky="w", padx=(6, 0))

        ttk.Label(options, text="Grid px").grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Spinbox(options, from_=4, to=32, textvariable=self.grid_var, width=8).grid(row=1, column=1, sticky="w", padx=(6, 18), pady=(10, 0))
        ttk.Label(options, text="Skeleton spacing").grid(row=1, column=2, sticky="w", pady=(10, 0))
        ttk.Spinbox(options, from_=0.5, to=12.0, increment=0.5, textvariable=self.skeleton_spacing_var, width=8).grid(row=1, column=3, sticky="w", padx=(6, 18), pady=(10, 0))
        ttk.Label(options, text="Skeleton min pts").grid(row=1, column=4, sticky="w", pady=(10, 0))
        ttk.Spinbox(options, from_=2, to=100, textvariable=self.skeleton_min_points_var, width=8).grid(row=1, column=5, sticky="w", padx=(6, 18), pady=(10, 0))
        ttk.Label(options, text="Ink radius").grid(row=1, column=6, sticky="w", pady=(10, 0))
        ttk.Spinbox(options, from_=1, to=32, textvariable=self.ink_radius_var, width=8).grid(row=1, column=7, sticky="w", padx=(6, 0), pady=(10, 0))

        ttk.Checkbutton(options, text="Show hand / pen", variable=self.show_hand_var).grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Checkbutton(options, text="Match image background to paper", variable=self.match_bg_var).grid(row=2, column=4, columnspan=4, sticky="w", pady=(10, 0))

        hand = ttk.LabelFrame(root, text="Real hand PNG (optional)", padding=10)
        hand.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(0, 8))
        hand.columnconfigure(1, weight=1)

        ttk.Label(hand, text="PNG asset").grid(row=0, column=0, sticky="w")
        ttk.Entry(hand, textvariable=self.hand_image_var).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(hand, text="Browse PNG…", command=self._browse_hand).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(hand, text="Clear", command=self._clear_hand).grid(row=0, column=3)

        hand_settings = ttk.Frame(hand)
        hand_settings.grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(hand_settings, text="Height px").pack(side=tk.LEFT)
        ttk.Spinbox(hand_settings, from_=32, to=1200, textvariable=self.hand_height_var, width=8).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(hand_settings, text="Tip anchor X (0..1)").pack(side=tk.LEFT)
        ttk.Spinbox(hand_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.hand_anchor_x_var, width=7).pack(side=tk.LEFT, padx=(6, 18))
        ttk.Label(hand_settings, text="Tip anchor Y (0..1)").pack(side=tk.LEFT)
        ttk.Spinbox(hand_settings, from_=0.0, to=1.0, increment=0.05, textvariable=self.hand_anchor_y_var, width=7).pack(side=tk.LEFT, padx=(6, 0))

        ttk.Label(
            hand,
            text="If no PNG is selected, the renderer uses the built-in procedural hand/pen. Anchor is the pen-tip position inside the cropped PNG.",
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(8, 0))

        action_bar = ttk.Frame(root)
        action_bar.grid(row=4, column=0, columnspan=3, sticky="ew", pady=6)
        self.start_button = ttk.Button(action_bar, text="Start batch", command=self._start)
        self.start_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(action_bar, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=8)

        ttk.Progressbar(root, variable=self.progress_var, maximum=100.0).grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 3))
        ttk.Label(root, textvariable=self.progress_text_var).grid(row=6, column=0, columnspan=3, sticky="w")

        ttk.Label(root, text="Log").grid(row=8, column=0, columnspan=3, sticky="w", pady=(12, 4))
        self.log_widget = ScrolledText(root, height=18, wrap=tk.WORD, state=tk.DISABLED)
        self.log_widget.grid(row=9, column=0, columnspan=3, sticky="nsew")

    def _browse_input(self) -> None:
        selected = filedialog.askdirectory(title="Select input image folder")
        if not selected:
            return
        self.input_var.set(selected)
        if not self.output_var.get().strip():
            self.output_var.set(str(Path(selected) / "output"))

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(title="Select output folder")
        if selected:
            self.output_var.set(selected)

    def _browse_hand(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select transparent hand / pen PNG",
            filetypes=(("PNG image", "*.png"), ("All files", "*.*")),
        )
        if selected:
            self.hand_image_var.set(selected)

    def _clear_hand(self) -> None:
        self.hand_image_var.set("")

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        try:
            hand_path = self.hand_image_var.get().strip() or None
            config = RenderConfig(
                duration_seconds=float(self.duration_var.get()),
                fps=int(self.fps_var.get()),
                max_long_edge=int(self.max_edge_var.get()),
                grid_size=int(self.grid_var.get()),
                ink_path_mode=self.path_mode_var.get(),
                skeleton_min_points=int(self.skeleton_min_points_var.get()),
                skeleton_spacing=float(self.skeleton_spacing_var.get()),
                ink_reveal_radius=int(self.ink_radius_var.get()),
                show_hand=bool(self.show_hand_var.get()),
                hand_image_path=hand_path,
                hand_height=int(self.hand_height_var.get()),
                hand_tip_anchor_x=float(self.hand_anchor_x_var.get()),
                hand_tip_anchor_y=float(self.hand_anchor_y_var.get()),
                match_background=bool(self.match_bg_var.get()),
            )
            config.validate()
            input_folder = Path(self.input_var.get().strip())
            if not self.input_var.get().strip():
                raise ValueError("Select an input folder first")
            output_text = self.output_var.get().strip()
            output_folder = Path(output_text) if output_text else input_folder / "output"
        except Exception as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return

        self._cancel_event.clear()
        self.progress_var.set(0)
        self.progress_text_var.set("Starting…")
        self._set_running(True)
        self._append_log("=" * 72)
        self._append_log(f"Starting batch — path mode: {config.ink_path_mode}")
        if config.show_hand:
            hand_label = config.hand_image_path or "procedural fallback"
            self._append_log(f"Hand overlay: {hand_label}")

        self._worker = threading.Thread(
            target=self._run_batch,
            args=(input_folder, output_folder, config),
            daemon=True,
        )
        self._worker.start()

    def _run_batch(self, input_folder: Path, output_folder: Path, config: RenderConfig) -> None:
        try:
            results = self._batch_service.render_folder(
                input_folder=input_folder,
                output_folder=output_folder,
                config=config,
                progress=lambda value, text: self._events.put(("progress", (value, text))),
                log=lambda text: self._events.put(("log", text)),
                cancel_check=self._cancel_event.is_set,
            )
            self._events.put(("done", results))
        except Exception as exc:
            self._events.put(("error", str(exc)))

    def _cancel(self) -> None:
        self._cancel_event.set()
        self.cancel_button.configure(state=tk.DISABLED)
        self._append_log("Cancellation requested…")

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self._events.get_nowait()
                if kind == "progress":
                    value, text = payload  # type: ignore[misc]
                    self.progress_var.set(float(value) * 100.0)
                    self.progress_text_var.set(f"{float(value) * 100:5.1f}% — {text}")
                elif kind == "log":
                    self._append_log(str(payload))
                elif kind == "done":
                    results = payload  # type: ignore[assignment]
                    self._set_running(False)
                    successes = sum(1 for result in results if result.success)
                    failures = sum(1 for result in results if not result.success)
                    cancelled = self._cancel_event.is_set()
                    status = "Cancelled" if cancelled else "Completed"
                    self.progress_text_var.set(f"{status}: {successes} success, {failures} failed")
                    if not cancelled:
                        self.progress_var.set(100.0 if failures == 0 else self.progress_var.get())
                    messagebox.showinfo("Draw Animation", self.progress_text_var.get())
                elif kind == "error":
                    self._set_running(False)
                    self.progress_text_var.set("Error")
                    self._append_log(f"FATAL: {payload}")
                    messagebox.showerror("Draw Animation", str(payload))
        except queue.Empty:
            pass
        finally:
            self.after(self.POLL_MS, self._drain_events)

    def _append_log(self, text: str) -> None:
        self.log_widget.configure(state=tk.NORMAL)
        self.log_widget.insert(tk.END, text + "\n")
        self.log_widget.see(tk.END)
        self.log_widget.configure(state=tk.DISABLED)

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.cancel_button.configure(state=tk.NORMAL if running else tk.DISABLED)
