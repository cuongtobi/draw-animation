from __future__ import annotations

from collections import Counter
from pathlib import Path


class ImageDiscoveryService:
    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def discover(self, folder: Path) -> list[Path]:
        folder = Path(folder)
        if not folder.exists():
            raise FileNotFoundError(f"Input folder does not exist: {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"Input path is not a folder: {folder}")

        images = sorted(
            (
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ),
            key=lambda item: item.name.casefold(),
        )
        self._validate_unique_stems(images)
        return images

    @staticmethod
    def output_path(image_path: Path, output_folder: Path) -> Path:
        return Path(output_folder) / f"{image_path.stem}.mp4"

    @staticmethod
    def _validate_unique_stems(images: list[Path]) -> None:
        counts = Counter(path.stem.casefold() for path in images)
        duplicates = sorted(stem for stem, count in counts.items() if count > 1)
        if duplicates:
            joined = ", ".join(duplicates)
            raise ValueError(
                "Multiple images would produce the same MP4 name. "
                f"Rename duplicate stems first: {joined}"
            )
