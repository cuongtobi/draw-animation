from pathlib import Path

import cv2
import numpy as np

from draw_animation.config import RenderConfig
from draw_animation.services.batch_render_service import BatchRenderService


def _image(path: Path, offset: int) -> None:
    canvas = np.full((80, 120, 3), 248, dtype=np.uint8)
    cv2.line(canvas, (10 + offset, 10), (100, 65), (10, 10, 10), 3)
    ok, encoded = cv2.imencode(".png", canvas)
    assert ok
    encoded.tofile(str(path))


def test_batch_renders_all_images_with_matching_names(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "videos"
    input_dir.mkdir()
    _image(input_dir / "one.png", 0)
    _image(input_dir / "two.png", 5)
    logs: list[str] = []
    progress_values: list[float] = []

    results = BatchRenderService().render_folder(
        input_folder=input_dir,
        output_folder=output_dir,
        config=RenderConfig(
            duration_seconds=0.8,
            fps=24,
            max_long_edge=240,
            grid_size=8,
            hold_seconds=0.15,
            show_hand=False,
        ),
        progress=lambda value, _: progress_values.append(value),
        log=logs.append,
    )

    assert [result.output_path.name for result in results] == ["one.mp4", "two.mp4"]
    assert all(result.success for result in results)
    assert all(result.output_path.exists() for result in results)
    assert progress_values[-1] == 1.0
    assert any("DONE" in line for line in logs)
