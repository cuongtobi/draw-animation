from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderConfig:
    """Runtime rendering options shared by UI, services and tests."""

    duration_seconds: float = 8.0
    fps: int = 30
    max_long_edge: int = 1080
    grid_size: int = 8
    ink_path_mode: str = "grid"
    skeleton_min_points: int = 8
    skeleton_spacing: float = 2.5
    ink_reveal_radius: int = 4
    paper_color: str = "#F5EBD7"
    ink_ratio: float = 0.62
    color_ratio: float = 0.30
    hold_seconds: float = 0.50
    show_hand: bool = True
    hand_image_path: str | None = None
    hand_height: int = 420
    hand_tip_anchor_x: float = 0.0
    hand_tip_anchor_y: float = 0.0
    match_background: bool = True
    background_tolerance: int = 36

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be > 0")
        if self.fps not in {24, 25, 30, 50, 60}:
            raise ValueError("fps must be one of 24, 25, 30, 50, 60")
        if self.max_long_edge < 128:
            raise ValueError("max_long_edge must be >= 128")
        if not 4 <= self.grid_size <= 32:
            raise ValueError("grid_size must be between 4 and 32")
        if self.ink_path_mode not in {"grid", "skeleton"}:
            raise ValueError("ink_path_mode must be 'grid' or 'skeleton'")
        if self.skeleton_min_points < 2:
            raise ValueError("skeleton_min_points must be >= 2")
        if self.skeleton_spacing <= 0:
            raise ValueError("skeleton_spacing must be > 0")
        if not 1 <= self.ink_reveal_radius <= 32:
            raise ValueError("ink_reveal_radius must be between 1 and 32")
        if not 0 < self.ink_ratio < 1:
            raise ValueError("ink_ratio must be between 0 and 1")
        if not 0 < self.color_ratio < 1:
            raise ValueError("color_ratio must be between 0 and 1")
        if self.ink_ratio + self.color_ratio > 0.95:
            raise ValueError("ink_ratio + color_ratio must leave room for the ending hold")
        if self.hold_seconds < 0:
            raise ValueError("hold_seconds must be >= 0")
        if self.duration_seconds <= self.hold_seconds:
            raise ValueError("duration_seconds must be greater than hold_seconds")
        if self.hand_height < 32:
            raise ValueError("hand_height must be >= 32")
        if not 0.0 <= self.hand_tip_anchor_x <= 1.0:
            raise ValueError("hand_tip_anchor_x must be between 0 and 1")
        if not 0.0 <= self.hand_tip_anchor_y <= 1.0:
            raise ValueError("hand_tip_anchor_y must be between 0 and 1")
        if self.hand_image_path is not None and not self.hand_image_path.strip():
            raise ValueError("hand_image_path must be None or a non-empty path")
        if len(self.paper_color) != 7 or not self.paper_color.startswith("#"):
            raise ValueError("paper_color must use #RRGGBB format")
        try:
            int(self.paper_color[1:], 16)
        except ValueError as exc:
            raise ValueError("paper_color must use #RRGGBB format") from exc
