# Draw Animation

Python desktop tool that batch-converts still images into whiteboard-style drawing videos.

The rendering model is inspired by the workflow of `geeklee/srt-whiteboard-animation`, but this repository is implemented as a standalone folder-to-video desktop application: no SRT, no manual annotation, and no browser preview are required.

## What it does

- Input: a folder containing `.png`, `.jpg`, `.jpeg`, `.webp`, or `.bmp` images.
- Output: one `.mp4` per image, preserving the image stem (`scene-01.png` -> `scene-01.mp4`).
- Default output: `<input folder>/output`.
- Desktop UI built with the Python standard-library `tkinter`.
- Background worker thread keeps the UI responsive.
- Global progress bar and per-file log.
- Cancel current batch.
- Two drawing-path modes: `skeleton` for line-following motion and `grid` for robust fallback rendering.
- Optional real hand/pen PNG with configurable size and normalized pen-tip anchor.
- Procedural hand/pen remains available when no PNG is selected.
- Configurable duration, FPS, output resolution, path precision, hand overlay, and paper-background matching.

## Drawing mechanism

For each image the renderer:

1. Resizes the source to an even video-friendly resolution.
2. Samples the image corners and optionally maps the original background to warm paper (`#F5EBD7`).
3. Extracts dark line work with adaptive thresholding.
4. Builds a drawing path using the selected mode:
   - `skeleton`: Zhang-Suen thinning -> 1px skeleton -> 8-neighbor stroke tracing -> resampling -> Chaikin smoothing -> nearest-stroke ordering.
   - `grid`: connected grid cells -> continuous DFS paths.
5. Preserves pen lifts between disconnected skeleton strokes so the renderer does not draw fake connector lines.
6. Starts from a blank paper canvas and progressively reveals only line pixels along the active path.
7. Moves either a real PNG hand/pen overlay or the procedural fallback with the active stroke tip.
8. Reveals the original colors with a soft wave/contour-style wipe.
9. Holds the completed image and writes the result to MP4.

If `skeleton` cannot produce a usable stroke plan, the path service automatically falls back to `grid`.

This is image-reveal animation driven by extracted stroke paths; it is not generative video and it does not redraw the source with an AI model.

## Real hand PNG

Use a transparent PNG when possible. RGB PNG files are also accepted; near-white pixels are treated as background.

The UI exposes:

- **PNG asset**: hand/pen image file.
- **Height px**: rendered height of the overlay.
- **Tip anchor X / Y**: normalized position (`0..1`) of the pen tip inside the cropped PNG.

Examples:

- `(0.0, 0.0)`: pen tip is at the top-left of the visible asset.
- `(0.5, 0.7)`: pen tip is near the lower center.

The selected anchor is aligned exactly to the current drawing point. If no PNG is selected, the tool keeps using the built-in procedural hand/pen.

## Architecture

```text
app.py
└── draw_animation/
    ├── config.py
    ├── models.py
    ├── services/
    │   ├── image_discovery_service.py
    │   ├── stroke_path_service.py
    │   ├── hand_overlay_service.py
    │   ├── video_render_service.py
    │   └── batch_render_service.py
    └── ui/
        └── main_window.py
```

The UI only collects settings and publishes UI events. File discovery, stroke planning, hand overlay loading, video rendering, and batch orchestration are isolated services and can be tested without starting Tkinter.

## Install

Python 3.10+ is recommended.

### Windows

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

> Linux distributions may require the OS Tk package, for example `sudo apt install python3-tk`.

## Use

1. Run `python app.py`.
2. Choose the input image folder.
3. Keep the automatically selected `output` folder or choose another output folder.
4. Choose **Path mode**. `skeleton` is recommended for clean line-art; `grid` is more tolerant of noisy/dense images.
5. Optionally choose a real hand/pen PNG and adjust the pen-tip anchor.
6. Set duration/FPS/resolution if needed.
7. Click **Start batch**.
8. Follow rendering status in the progress bar and log panel.

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
pytest
```

Test layout:

- `tests/unit`: config validation, image discovery/output naming, grid/skeleton stroke planning, skeleton fallback, hand PNG loading/alpha stamping.
- `tests/functional`: renders real temporary MP4 files and verifies OpenCV can open/read them; includes a `skeleton + real hand PNG` render and batch image-to-video coverage.

## Notes

- The `mp4v` encoder is used through the official `opencv-python` wheel for broad Windows compatibility.
- `skeleton` mode gives the most convincing result on clean illustrations and line-art because the hand follows the actual line center rather than grid-cell centers.
- Dense photographic textures produce many detected ink pixels and may render more slowly; `grid` mode can be a better fallback for these inputs.
- If two source images have the same stem (`a.jpg` and `a.png`), the batch is rejected to prevent one `a.mp4` from overwriting the other.
