import pytest

from draw_animation.config import RenderConfig


def test_default_config_is_valid() -> None:
    RenderConfig().validate()


def test_skeleton_config_is_valid() -> None:
    RenderConfig(
        ink_path_mode="skeleton",
        skeleton_min_points=5,
        skeleton_spacing=1.8,
        hand_tip_anchor_x=0.2,
        hand_tip_anchor_y=0.8,
    ).validate()


def test_duration_must_be_longer_than_hold() -> None:
    with pytest.raises(ValueError, match="greater than hold_seconds"):
        RenderConfig(duration_seconds=0.4, hold_seconds=0.5).validate()


def test_rejects_unknown_path_mode() -> None:
    with pytest.raises(ValueError, match="ink_path_mode"):
        RenderConfig(ink_path_mode="vector").validate()


def test_rejects_invalid_hand_anchor() -> None:
    with pytest.raises(ValueError, match="hand_tip_anchor_x"):
        RenderConfig(hand_tip_anchor_x=1.1).validate()
