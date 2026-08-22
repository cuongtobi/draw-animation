from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from draw_animation.config import RenderConfig
from draw_animation.services.stroke_path_service import GridStrokePathService, StrokePlan

ProgressCallback = Callable[[float, str], None]
CancelCheck = Callable[[], bool]


class RenderCancelled(RuntimeError):
    pass


class VideoRenderService:
    """Render one still image as a progressive whiteboard drawing MP4."""

    def __init__(self, path_service: GridStrokePathService | None = None) -> None:
        self._path_service = path_service or GridStrokePathService()

    def render(
        self,
        image_path: Path,
        output_path: Path,
        config: RenderConfig,
        progress: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> Path:
        config.validate()
        image_path = Path(image_path)
        output_path = Path(output_path)
        self._emit(progress, 0.0, "Loading image")

        source = self._read_image(image_path)
        source = self._resize_even(source, config.max_long_edge)
        paper_bgr = self._hex_to_bgr(config.paper_color)
        source = self._match_background(source, paper_bgr, config) if config.match_background else source
        ink_mask, ink_image = self._extract_ink(source)
        plan = self._path_service.build(ink_mask, config.grid_size)
        self._emit(progress, 0.04, f"Detected {plan.active_cells} ink cells")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = self._open_writer(output_path, config.fps, source.shape[1], source.shape[0])
        try:
            self._render_frames(
                writer=writer,
                source=source,
                ink_mask=ink_mask,
                ink_image=ink_image,
                plan=plan,
                paper_bgr=paper_bgr,
                config=config,
                progress=progress,
                cancel_check=cancel_check,
            )
        except Exception:
            writer.release()
            output_path.unlink(missing_ok=True)
            raise
        else:
            writer.release()

        self._emit(progress, 1.0, "Done")
        return output_path

    def _render_frames(
        self,
        writer: cv2.VideoWriter,
        source: np.ndarray,
        ink_mask: np.ndarray,
        ink_image: np.ndarray,
        plan: StrokePlan,
        paper_bgr: np.ndarray,
        config: RenderConfig,
        progress: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> None:
        h, w = source.shape[:2]
        total_frames = max(3, int(round(config.duration_seconds * config.fps)))
        hold_frames = min(total_frames - 2, max(1, int(round(config.hold_seconds * config.fps))))
        animation_frames = total_frames - hold_frames

        ink_frames = max(1, int(round(animation_frames * config.ink_ratio)))
        color_frames = max(1, int(round(animation_frames * config.color_ratio)))
        if ink_frames + color_frames > animation_frames:
            color_frames = max(1, animation_frames - ink_frames)
        pre_hold_frames = max(0, animation_frames - ink_frames - color_frames)

        canvas = np.empty_like(source, dtype=np.uint8)
        canvas[:] = paper_bgr

        flat_path = plan.flattened
        if not flat_path:
            flat_path = [(w // 2, h // 2)]

        revealed_cells: set[tuple[int, int]] = set()
        last_path_index = -1
        for frame_index in range(ink_frames):
            self._check_cancel(cancel_check)
            target_index = self._mapped_index(frame_index, ink_frames, len(flat_path))
            for path_index in range(last_path_index + 1, target_index + 1):
                self._reveal_ink_cell(
                    canvas,
                    ink_image,
                    ink_mask,
                    flat_path[path_index],
                    config.grid_size,
                    revealed_cells,
                )
            last_path_index = max(last_path_index, target_index)
            tip = flat_path[target_index]
            frame = canvas.copy()
            if config.show_hand:
                self._draw_hand_and_pen(frame, tip)
            writer.write(frame)
            self._emit(progress, 0.04 + 0.61 * ((frame_index + 1) / ink_frames), "Drawing ink")

        for frame_index in range(color_frames):
            self._check_cancel(cancel_check)
            p = (frame_index + 1) / color_frames
            frame = self._contour_wipe(canvas, source, p)
            tip = self._wipe_tip(w, h, p, frame_index)
            if config.show_hand:
                self._draw_hand_and_pen(frame, tip)
            writer.write(frame)
            self._emit(progress, 0.65 + 0.28 * p, "Revealing color")

        final_frame = source.copy()
        for frame_index in range(pre_hold_frames + hold_frames):
            self._check_cancel(cancel_check)
            writer.write(final_frame)
            p = (frame_index + 1) / max(1, pre_hold_frames + hold_frames)
            self._emit(progress, 0.93 + 0.06 * p, "Holding final image")

    @staticmethod
    def _read_image(path: Path) -> np.ndarray:
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            raise ValueError(f"Could not read image: {path}")
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unsupported or corrupted image: {path}")
        return image

    @staticmethod
    def _resize_even(image: np.ndarray, max_long_edge: int) -> np.ndarray:
        h, w = image.shape[:2]
        scale = min(1.0, max_long_edge / max(h, w))
        new_w = max(2, int(round(w * scale)))
        new_h = max(2, int(round(h * scale)))
        new_w -= new_w % 2
        new_h -= new_h % 2
        if (new_w, new_h) == (w, h):
            return image.copy()
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _extract_ink(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        threshold = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,
            9,
        )
        ink_mask = threshold < 128
        ink_image = np.repeat(threshold[:, :, None], 3, axis=2)
        return ink_mask, ink_image

    @staticmethod
    def _match_background(image: np.ndarray, paper_bgr: np.ndarray, config: RenderConfig) -> np.ndarray:
        result = image.copy()
        h, w = result.shape[:2]
        margin = max(2, min(h, w) // 40)
        corners = [
            result[:margin, :margin],
            result[:margin, -margin:],
            result[-margin:, :margin],
            result[-margin:, -margin:],
        ]
        background = np.median(np.concatenate([c.reshape(-1, 3) for c in corners]), axis=0)
        diff = np.abs(result.astype(np.int16) - background.astype(np.int16)).sum(axis=2)
        result[diff < config.background_tolerance] = paper_bgr
        return result

    @staticmethod
    def _hex_to_bgr(value: str) -> np.ndarray:
        rgb = tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))
        return np.array((rgb[2], rgb[1], rgb[0]), dtype=np.uint8)

    @staticmethod
    def _open_writer(path: Path, fps: int, width: int, height: int) -> cv2.VideoWriter:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(
                "Could not open MP4 writer. Install the standard opencv-python wheel "
                "with FFmpeg support or install a system FFmpeg build."
            )
        return writer

    @staticmethod
    def _mapped_index(frame_index: int, frame_count: int, point_count: int) -> int:
        if point_count <= 1 or frame_count <= 1:
            return max(0, point_count - 1)
        return min(point_count - 1, round(frame_index * (point_count - 1) / (frame_count - 1)))

    @staticmethod
    def _reveal_ink_cell(
        canvas: np.ndarray,
        ink_image: np.ndarray,
        ink_mask: np.ndarray,
        point: tuple[int, int],
        grid_size: int,
        revealed_cells: set[tuple[int, int]],
    ) -> None:
        x, y = point
        row, col = y // grid_size, x // grid_size
        key = (row, col)
        if key in revealed_cells:
            return
        revealed_cells.add(key)
        h, w = ink_mask.shape
        y0, y1 = row * grid_size, min(h, (row + 1) * grid_size)
        x0, x1 = col * grid_size, min(w, (col + 1) * grid_size)
        mask = ink_mask[y0:y1, x0:x1]
        target = canvas[y0:y1, x0:x1]
        source = ink_image[y0:y1, x0:x1]
        target[mask] = source[mask]

    @staticmethod
    def _contour_wipe(base: np.ndarray, source: np.ndarray, progress: float) -> np.ndarray:
        h, w = source.shape[:2]
        progress = float(np.clip(progress, 0.0, 1.0))
        frame = base.copy()
        xs = np.arange(w, dtype=np.float32)
        wave = np.sin(xs / max(14.0, w / 18.0)) * max(3.0, h * 0.018)
        lead = progress * (h + 20) - 10
        boundary = lead + wave
        ys = np.arange(h, dtype=np.float32)[:, None]
        reveal = ys <= boundary[None, :]
        frame[reveal] = source[reveal]
        return frame

    @staticmethod
    def _wipe_tip(width: int, height: int, progress: float, frame_index: int) -> tuple[int, int]:
        lane = 0.5 + 0.45 * np.sin(frame_index * 0.32)
        x = int(np.clip(lane * width, 0, width - 1))
        y = int(np.clip(progress * height, 0, height - 1))
        return x, y

    @staticmethod
    def _draw_hand_and_pen(frame: np.ndarray, tip: tuple[int, int]) -> None:
        """Draw a lightweight procedural pen/hand overlay anchored at the active tip."""
        h, w = frame.shape[:2]
        x, y = tip
        scale = max(0.45, min(1.0, max(h, w) / 1080.0))
        pen_end = (
            int(np.clip(x + 90 * scale, 0, w - 1)),
            int(np.clip(y + 92 * scale, 0, h - 1)),
        )
        cv2.line(frame, (x, y), pen_end, (38, 38, 38), max(2, int(8 * scale)), cv2.LINE_AA)
        hand_center = (
            int(np.clip(x + 118 * scale, 0, w - 1)),
            int(np.clip(y + 108 * scale, 0, h - 1)),
        )
        cv2.ellipse(
            frame,
            hand_center,
            (max(8, int(43 * scale)), max(7, int(31 * scale))),
            35,
            0,
            360,
            (176, 205, 226),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(frame, pen_end, max(4, int(11 * scale)), (170, 198, 220), -1, cv2.LINE_AA)

    @staticmethod
    def _check_cancel(cancel_check: CancelCheck | None) -> None:
        if cancel_check is not None and cancel_check():
            raise RenderCancelled("Rendering cancelled by user")

    @staticmethod
    def _emit(callback: ProgressCallback | None, value: float, message: str) -> None:
        if callback is not None:
            callback(float(np.clip(value, 0.0, 1.0)), message)
