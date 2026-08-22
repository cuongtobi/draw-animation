import numpy as np

from draw_animation.services.stroke_path_service import GridStrokePathService


def test_builds_strokes_for_ink_pixels() -> None:
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:12, 4:12] = True
    mask[20:28, 20:28] = True

    plan = GridStrokePathService().build(mask, grid_size=4)

    assert plan.active_cells > 0
    assert len(plan.strokes) == 2
    assert all(stroke for stroke in plan.strokes)
    assert all(0 <= x < 32 and 0 <= y < 32 for x, y in plan.flattened)
