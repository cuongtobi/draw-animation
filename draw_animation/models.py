from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderJobResult:
    input_path: Path
    output_path: Path
    success: bool
    error: str | None = None
