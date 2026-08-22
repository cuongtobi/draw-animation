import numpy as np

from draw_animation.services.stroke_path_service import (
    GridStrokePathService,
    SkeletonStrokePathService,
    StrokePathService,
)


def test_grid_builds_strokes_for_ink_pixels() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:12, 4:12] = True
    mask[20:28, 20:28] = True

    plan = GridStrokePathService().build(mask, grid_size=4)

    assert plan.mode == "grid"
    assert plan.active_cells > 0
    assert len(plan.strokes) == 2
    assert all(stroke for stroke in plan.strokes)
    assert all(0 <= x < 32 and 0 <= y < 32 for x, y in plan.flattened)


def test_skeleton_follows_thin_line_and_marks_pen_lifts() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:50, 12:15] = True
    mask[15:52, 45:48] = True

    plan = SkeletonStrokePathService().build(mask, min_points=4, spacing=2.0)

    assert plan.mode == "skeleton"
    assert len(plan.strokes) >= 2
    assert plan.pen_lift_indices
    assert all(0 <= x < 64 and 0 <= y < 64 for x, y in plan.flattened)
    assert any(abs(x - 13) <= 2 for x, _ in plan.flattened)
    assert any(abs(x - 46) <= 2 for x, _ in plan.flattened)


def test_hybrid_falls_back_to_grid_when_skeleton_has_no_long_stroke() -> None:
    mask = np.zeros((16, 16), dtype=bool)
    mask[3, 3] = True

    plan = StrokePathService().build(
        mask,
        mode="skeleton",
        grid_size=4,
        skeleton_min_points=8,
        skeleton_spacing=2.5,
    )

    assert plan.mode == "grid"
    assert plan.strokes
