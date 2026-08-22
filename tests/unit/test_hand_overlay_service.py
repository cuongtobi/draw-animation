from pathlib import Path

import cv2
import numpy as np
import pytest

from draw_animation.services.hand_overlay_service import HandOverlayService


def _write_rgba_hand(path: Path) -> None:
    rgba = np.zeros((20, 30, 4), dtype=np.uint8)
    rgba[4:18, 5:28, :3] = (10, 80, 220)
    rgba[4:18, 5:28, 3] = 255
    ok, encoded = cv2.imencode(".png", rgba)
    assert ok
    encoded.tofile(str(path))


def test_loads_rgba_png_and_stamps_at_anchor(tmp_path: Path) -> None:
    hand_path = tmp_path / "hand.png"
    _write_rgba_hand(hand_path)

    overlay = HandOverlayService().load(
        hand_path,
        target_height=40,
        tip_anchor_x=0.0,
        tip_anchor_y=0.0,
    )
    frame = np.full((80, 100, 3), 255, dtype=np.uint8)
    overlay.stamp(frame, (20, 25))

    assert overlay.image.shape[0] == 40
    assert np.any(frame[25:, 20:] != 255)


def test_rejects_non_png_hand_asset(tmp_path: Path) -> None:
    path = tmp_path / "hand.jpg"
    path.write_bytes(b"not-used")
    with pytest.raises(ValueError, match="PNG"):
        HandOverlayService().load(path, 100, 0.0, 0.0)
