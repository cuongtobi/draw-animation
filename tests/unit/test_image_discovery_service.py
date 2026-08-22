from pathlib import Path

import pytest

from draw_animation.services.image_discovery_service import ImageDiscoveryService


def test_discover_supported_images_and_keep_stable_order(tmp_path: Path) -> None:
    (tmp_path / "b.JPG").write_bytes(b"x")
    (tmp_path / "A.png").write_bytes(b"x")
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    result = ImageDiscoveryService().discover(tmp_path)

    assert [item.name for item in result] == ["A.png", "b.JPG"]


def test_duplicate_stems_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "same.png").write_bytes(b"x")
    (tmp_path / "same.jpg").write_bytes(b"x")

    with pytest.raises(ValueError, match="same MP4 name"):
        ImageDiscoveryService().discover(tmp_path)


def test_output_name_matches_image_stem(tmp_path: Path) -> None:
    image = tmp_path / "my picture.png"
    output = ImageDiscoveryService.output_path(image, tmp_path / "out")
    assert output.name == "my picture.mp4"
