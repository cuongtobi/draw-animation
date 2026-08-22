import pytest

from draw_animation.config import RenderConfig


def test_default_config_is_valid() -> None:
    RenderConfig().validate()


def test_duration_must_be_longer_than_hold() -> None:
    with pytest.raises(ValueError, match="greater than hold_seconds"):
        RenderConfig(duration_seconds=0.4, hold_seconds=0.5).validate()
