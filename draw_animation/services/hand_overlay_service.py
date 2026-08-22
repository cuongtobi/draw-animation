from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class HandOverlay:
    image: np.ndarray
    alpha: np.ndarray
    anchor_x: int
    anchor_y: int

    def stamp(self, frame: np.ndarray, tip: tuple[int, int]) -> None:
        """Alpha-blend the hand image so its configured tip anchor touches tip."""
        x, y = tip
        overlay_h, overlay_w = self.image.shape[:2]
        left = x - self.anchor_x
        top = y - self.anchor_y
        frame_h, frame_w = frame.shape[:2]

        x0 = max(0, left)
        y0 = max(0, top)
        x1 = min(frame_w, left + overlay_w)
        y1 = min(frame_h, top + overlay_h)
        if x1 <= x0 or y1 <= y0:
            return

        sx0, sy0 = x0 - left, y0 - top
        sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
        source = self.image[sy0:sy1, sx0:sx1].astype(np.float32)
        alpha = self.alpha[sy0:sy1, sx0:sx1, None].astype(np.float32)
        target = frame[y0:y1, x0:x1].astype(np.float32)
        frame[y0:y1, x0:x1] = np.clip(target * (1.0 - alpha) + source * alpha, 0, 255).astype(np.uint8)


class HandOverlayService:
    """Load a real PNG hand/pen asset and prepare an anchored overlay."""

    def load(
        self,
        path: Path,
        target_height: int,
        tip_anchor_x: float,
        tip_anchor_y: float,
    ) -> HandOverlay:
        path = Path(path)
        if path.suffix.lower() != ".png":
            raise ValueError("Hand image must be a PNG file")
        if not path.is_file():
            raise ValueError(f"Hand PNG does not exist: {path}")

        raw_bytes = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Could not read hand PNG: {path}")

        if image.ndim == 3 and image.shape[2] == 4:
            bgr = image[:, :, :3]
            mask = image[:, :, 3]
        elif image.ndim == 3 and image.shape[2] == 3:
            bgr = image
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        else:
            raise ValueError("Hand PNG must be RGB or RGBA")

        ys, xs = np.where(mask > 2)
        if xs.size == 0 or ys.size == 0:
            raise ValueError("Hand PNG has no visible pixels")
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        bgr = bgr[y0:y1, x0:x1]
        mask = mask[y0:y1, x0:x1]

        scale = target_height / max(1, bgr.shape[0])
        target_width = max(1, int(round(bgr.shape[1] * scale)))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        bgr = cv2.resize(bgr, (target_width, target_height), interpolation=interpolation)
        mask = cv2.resize(mask, (target_width, target_height), interpolation=interpolation)
        alpha = mask.astype(np.float32) / 255.0

        anchor_x = int(round((target_width - 1) * float(np.clip(tip_anchor_x, 0.0, 1.0))))
        anchor_y = int(round((target_height - 1) * float(np.clip(tip_anchor_y, 0.0, 1.0))))
        return HandOverlay(bgr, alpha, anchor_x, anchor_y)
