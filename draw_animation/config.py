from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RenderConfig:
    """Runtime rendering options shared by UI, services and tests."""

    duration_seconds: float = 8.0
    fps: int = 30
    max_long_edge: int = 1080
    grid_size: int = 8
    paper_color: str = "#F5EBD7"
    ink_ratio: float = 0.62
    color_ratio: float = 0.30
    hold_seconds: float = 0.50
    show_hand: bool = True
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
        if len(self.paper_color) != 7 or not self.paper_color.startswith("#"):
            raise ValueError("paper_color must use #RRGGBB format")
        try:
            int(self.paper_color[1:], 16)
        except ValueError as exc:
            raise ValueError("paper_color must use #RRGGBB format") from exc
