from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StrokePlan:
    strokes: list[list[tuple[int, int]]]
    active_cells: int

    @property
    def flattened(self) -> list[tuple[int, int]]:
        return [point for stroke in self.strokes for point in stroke]


class GridStrokePathService:
    """Build continuous grid-based paths over detected ink pixels."""

    _NEIGHBORS = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    def build(self, ink_mask: np.ndarray, grid_size: int) -> StrokePlan:
        if ink_mask.ndim != 2:
            raise ValueError("ink_mask must be a 2D array")
        if grid_size <= 0:
            raise ValueError("grid_size must be > 0")

        h, w = ink_mask.shape
        rows = (h + grid_size - 1) // grid_size
        cols = (w + grid_size - 1) // grid_size
        active = np.zeros((rows, cols), dtype=np.uint8)

        for row in range(rows):
            y0, y1 = row * grid_size, min(h, (row + 1) * grid_size)
            for col in range(cols):
                x0, x1 = col * grid_size, min(w, (col + 1) * grid_size)
                if np.any(ink_mask[y0:y1, x0:x1]):
                    active[row, col] = 1

        component_count, labels = cv2.connectedComponents(active, connectivity=8)
        components: list[tuple[int, int, list[tuple[int, int]]]] = []
        for label in range(1, component_count):
            cells = [(int(r), int(c)) for r, c in np.argwhere(labels == label)]
            if not cells:
                continue
            min_row = min(r for r, _ in cells)
            min_col = min(c for _, c in cells)
            components.append((min_row, min_col, cells))
        components.sort(key=lambda item: (item[0], item[1]))

        strokes = [self._trace_component(cells, grid_size, w, h) for _, _, cells in components]
        strokes = [stroke for stroke in strokes if stroke]
        return StrokePlan(strokes=strokes, active_cells=int(active.sum()))

    def _trace_component(
        self,
        cells: list[tuple[int, int]],
        grid_size: int,
        width: int,
        height: int,
    ) -> list[tuple[int, int]]:
        allowed = set(cells)
        start = min(cells)
        visited = {start}
        path_cells = [start]
        stack: list[tuple[tuple[int, int], int]] = [(start, 0)]

        while stack:
            (row, col), neighbor_index = stack[-1]
            if neighbor_index >= len(self._NEIGHBORS):
                stack.pop()
                if stack:
                    path_cells.append(stack[-1][0])
                continue

            stack[-1] = ((row, col), neighbor_index + 1)
            dr, dc = self._NEIGHBORS[neighbor_index]
            nxt = (row + dr, col + dc)
            if nxt not in allowed or nxt in visited:
                continue
            visited.add(nxt)
            path_cells.append(nxt)
            stack.append((nxt, 0))

        return [
            (
                min(width - 1, col * grid_size + grid_size // 2),
                min(height - 1, row * grid_size + grid_size // 2),
            )
            for row, col in path_cells
        ]
