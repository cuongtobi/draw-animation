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
- Configurable duration, FPS, output resolution, grid precision, procedural hand/pen overlay, and paper-background matching.

## Drawing mechanism

For each image the renderer:

1. Resizes the source to an even video-friendly resolution.
2. Samples the image corners and optionally maps the original background to warm paper (`#F5EBD7`).
3. Extracts dark line work with adaptive thresholding.
4. Splits detected ink into a grid and finds connected components.
5. Builds continuous DFS stroke paths through active grid cells.
6. Starts from a blank paper canvas and progressively reveals only line pixels along the stroke path.
7. Moves a procedural pen/hand overlay with the active stroke tip.
8. Reveals the original colors with a soft wave/contour-style wipe.
9. Holds the completed image and writes the result to MP4.

This is image-reveal animation driven by extracted stroke paths; it is not generative video and it does not redraw the source with an AI model.

## Architecture

```text
app.py
└── draw_animation/
    ├── config.py
    ├── models.py
    ├── services/
    │   ├── image_discovery_service.py
    │   ├── stroke_path_service.py
    │   ├── video_render_service.py
    │   └── batch_render_service.py
    └── ui/
        └── main_window.py
```

The UI only collects settings and publishes UI events. File discovery, stroke planning, video rendering, and batch orchestration are isolated services and can be tested without starting Tkinter.

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
4. Set duration/FPS/resolution if needed.
5. Click **Start batch**.
6. Follow rendering status in the progress bar and log panel.

## Tests

Install development dependencies:

```bash
pip install -r requirements-dev.txt
pytest
```

Test layout:

- `tests/unit`: config validation, image discovery/output naming, stroke planning.
- `tests/functional`: renders a real temporary MP4 and verifies OpenCV can open/read it; batch test verifies multiple image-to-video jobs.

## Notes

- The `mp4v` encoder is used through the official `opencv-python` wheel for broad Windows compatibility.
- Clean illustrations and line-art produce the most convincing drawing motion. Dense photographic textures produce many detected ink cells and may render more slowly.
- If two source images have the same stem (`a.jpg` and `a.png`), the batch is rejected to prevent one `a.mp4` from overwriting the other.
