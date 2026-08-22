from pathlib import Path

import cv2
import numpy as np

from draw_animation.config import RenderConfig
from draw_animation.services.video_render_service import VideoRenderService


def _write_test_image(path: Path) -> None:
    image = np.full((96, 160, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (70, 75), (20, 20, 20), 3)
    cv2.circle(image, (115, 48), 26, (30, 30, 30), 3)
    cv2.line(image, (70, 48), (89, 48), (40, 40, 40), 2)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    encoded.tofile(str(path))


def _write_hand(path: Path) -> None:
    rgba = np.zeros((42, 58, 4), dtype=np.uint8)
    cv2.rectangle(rgba, (3, 3), (54, 38), (30, 120, 220, 255), -1)
    cv2.line(rgba, (3, 3), (28, 20), (15, 15, 15, 255), 4)
    ok, encoded = cv2.imencode(".png", rgba)
    assert ok
    encoded.tofile(str(path))


def _assert_playable(output: Path) -> None:
    assert output.exists()
    assert output.stat().st_size > 0
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 20
        ok, frame = capture.read()
        assert ok
        assert frame is not None
    finally:
        capture.release()


def test_renderer_creates_playable_mp4(tmp_path: Path) -> None:
    source = tmp_path / "scene.png"
    output = tmp_path / "scene.mp4"
    _write_test_image(source)

    config = RenderConfig(
        duration_seconds=1.0,
        fps=24,
        max_long_edge=320,
        grid_size=8,
        hold_seconds=0.2,
        show_hand=False,
    )
    VideoRenderService().render(source, output, config)
    _assert_playable(output)


def test_skeleton_renderer_with_real_hand_png_creates_mp4(tmp_path: Path) -> None:
    source = tmp_path / "scene-skeleton.png"
    hand = tmp_path / "hand.png"
    output = tmp_path / "scene-skeleton.mp4"
    _write_test_image(source)
    _write_hand(hand)

    config = RenderConfig(
        duration_seconds=1.0,
        fps=24,
        max_long_edge=320,
        ink_path_mode="skeleton",
        skeleton_min_points=3,
        skeleton_spacing=2.0,
        hold_seconds=0.2,
        show_hand=True,
        hand_image_path=str(hand),
        hand_height=60,
        hand_tip_anchor_x=0.0,
        hand_tip_anchor_y=0.0,
    )
    VideoRenderService().render(source, output, config)
    _assert_playable(output)
