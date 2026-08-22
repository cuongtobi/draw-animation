from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StrokePlan:
    strokes: list[list[tuple[int, int]]]
    active_cells: int
    mode: str = "grid"
    pen_lift_indices: frozenset[int] = frozenset()

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
        return StrokePlan(
            strokes=strokes,
            active_cells=int(active.sum()),
            mode="grid",
            pen_lift_indices=_pen_lift_indices(strokes),
        )

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


class SkeletonStrokePathService:
    """Trace one-pixel skeleton strokes that follow the source line art closely."""

    _NEIGHBORS_8 = (
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    )

    def build(self, ink_mask: np.ndarray, min_points: int = 8, spacing: float = 2.5) -> StrokePlan:
        if ink_mask.ndim != 2:
            raise ValueError("ink_mask must be a 2D array")
        if min_points < 2:
            raise ValueError("min_points must be >= 2")
        if spacing <= 0:
            raise ValueError("spacing must be > 0")

        skeleton = self._zhang_suen_skeleton(ink_mask)
        raw_strokes = self._trace_8connected(skeleton, min_points=min_points)
        processed: list[list[tuple[int, int]]] = []
        for stroke in raw_strokes:
            points = [(float(x), float(y)) for x, y in stroke]
            points = self._resample(points, spacing)
            points = self._chaikin(points)
            points = self._resample(points, spacing)
            rounded = self._dedupe([(int(round(x)), int(round(y))) for x, y in points])
            if len(rounded) >= 2:
                processed.append(rounded)

        processed = self._order_strokes(processed)
        return StrokePlan(
            strokes=processed,
            active_cells=int(np.count_nonzero(skeleton)),
            mode="skeleton",
            pen_lift_indices=_pen_lift_indices(processed),
        )

    @staticmethod
    def _zhang_suen_skeleton(mask: np.ndarray, max_iterations: int = 160) -> np.ndarray:
        image = np.pad(mask.astype(np.uint8), 1, mode="constant")
        for _ in range(max_iterations):
            changed = False
            for step in (0, 1):
                p2, p3, p4 = image[:-2, 1:-1], image[:-2, 2:], image[1:-1, 2:]
                p5, p6, p7 = image[2:, 2:], image[2:, 1:-1], image[2:, :-2]
                p8, p9 = image[1:-1, :-2], image[:-2, :-2]
                center = image[1:-1, 1:-1]
                neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                transitions = sum(
                    (neighbors[index] == 0) & (neighbors[(index + 1) % 8] == 1)
                    for index in range(8)
                )
                count = sum(neighbors)
                if step == 0:
                    marker = (
                        (center == 1)
                        & (count >= 2)
                        & (count <= 6)
                        & (transitions == 1)
                        & ((p2 * p4 * p6) == 0)
                        & ((p4 * p6 * p8) == 0)
                    )
                else:
                    marker = (
                        (center == 1)
                        & (count >= 2)
                        & (count <= 6)
                        & (transitions == 1)
                        & ((p2 * p4 * p8) == 0)
                        & ((p2 * p6 * p8) == 0)
                    )
                if np.any(marker):
                    center[marker] = 0
                    changed = True
            if not changed:
                break
        return image[1:-1, 1:-1].astype(bool)

    def _neighbors(self, skeleton: np.ndarray, point: tuple[int, int]) -> list[tuple[int, int]]:
        x, y = point
        h, w = skeleton.shape
        result: list[tuple[int, int]] = []
        for dx, dy in self._NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < w and 0 <= ny < h and skeleton[ny, nx]):
                continue
            if dx != 0 and dy != 0 and (skeleton[y, nx] or skeleton[ny, x]):
                continue
            result.append((nx, ny))
        return result

    @staticmethod
    def _edge_key(
        first: tuple[int, int], second: tuple[int, int]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        return (first, second) if first <= second else (second, first)

    def _choose_next(
        self,
        previous: tuple[int, int],
        current: tuple[int, int],
        candidates: list[tuple[int, int]],
        visited_edges: set[tuple[tuple[int, int], tuple[int, int]]],
    ) -> tuple[int, int] | None:
        fresh = [
            point
            for point in candidates
            if point != previous and self._edge_key(current, point) not in visited_edges
        ]
        if not fresh:
            return None
        vx, vy = current[0] - previous[0], current[1] - previous[1]
        vlen = math.hypot(vx, vy) or 1.0
        return max(
            fresh,
            key=lambda point: (
                vx * (point[0] - current[0]) + vy * (point[1] - current[1])
            )
            / (vlen * (math.hypot(point[0] - current[0], point[1] - current[1]) or 1.0)),
        )

    def _trace_8connected(self, skeleton: np.ndarray, min_points: int) -> list[list[tuple[int, int]]]:
        ys, xs = np.nonzero(skeleton)
        points = [(int(x), int(y)) for x, y in zip(xs, ys)]
        if not points:
            return []

        degrees = {point: len(self._neighbors(skeleton, point)) for point in points}
        starts = (
            [point for point in points if degrees[point] == 1]
            + [point for point in points if degrees[point] > 2]
            + points
        )
        visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        strokes: list[list[tuple[int, int]]] = []

        for start in starts:
            for neighbor in self._neighbors(skeleton, start):
                edge = self._edge_key(start, neighbor)
                if edge in visited_edges:
                    continue
                path = [start]
                previous, current = start, neighbor
                visited_edges.add(edge)
                while True:
                    path.append(current)
                    next_point = self._choose_next(
                        previous,
                        current,
                        self._neighbors(skeleton, current),
                        visited_edges,
                    )
                    if next_point is None:
                        break
                    visited_edges.add(self._edge_key(current, next_point))
                    previous, current = current, next_point
                if len(path) >= min_points:
                    strokes.append(path)
        return strokes

    @staticmethod
    def _resample(points: list[tuple[float, float]], spacing: float) -> list[tuple[float, float]]:
        if len(points) < 2:
            return points
        cumulative = [0.0]
        for first, second in zip(points, points[1:]):
            cumulative.append(
                cumulative[-1] + math.hypot(second[0] - first[0], second[1] - first[1])
            )
        total = cumulative[-1]
        if total <= spacing:
            return [points[0], points[-1]]
        samples = np.arange(0.0, total, spacing, dtype=np.float32)
        if samples.size == 0 or samples[-1] < total:
            samples = np.append(samples, total)
        cumulative_np = np.asarray(cumulative, dtype=np.float32)
        xs = np.asarray([point[0] for point in points], dtype=np.float32)
        ys = np.asarray([point[1] for point in points], dtype=np.float32)
        return list(zip(np.interp(samples, cumulative_np, xs), np.interp(samples, cumulative_np, ys)))

    @staticmethod
    def _chaikin(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if len(points) < 3:
            return points
        output = [points[0]]
        for first, second in zip(points, points[1:]):
            output.append((0.75 * first[0] + 0.25 * second[0], 0.75 * first[1] + 0.25 * second[1]))
            output.append((0.25 * first[0] + 0.75 * second[0], 0.25 * first[1] + 0.75 * second[1]))
        output.append(points[-1])
        return output

    @staticmethod
    def _dedupe(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        output: list[tuple[int, int]] = []
        for point in points:
            if not output or point != output[-1]:
                output.append(point)
        return output

    @staticmethod
    def _order_strokes(strokes: list[list[tuple[int, int]]]) -> list[list[tuple[int, int]]]:
        remaining = [list(stroke) for stroke in strokes if stroke]
        ordered: list[list[tuple[int, int]]] = []
        tail: tuple[int, int] | None = None
        while remaining:
            if tail is None:
                index = min(
                    range(len(remaining)),
                    key=lambda i: (remaining[i][0][1], remaining[i][0][0]),
                )
            else:
                def distance(i: int) -> tuple[float, bool]:
                    head = remaining[i][0]
                    end = remaining[i][-1]
                    head_distance = (head[0] - tail[0]) ** 2 + (head[1] - tail[1]) ** 2
                    end_distance = (end[0] - tail[0]) ** 2 + (end[1] - tail[1]) ** 2
                    return min(head_distance, end_distance), end_distance < head_distance

                index = min(range(len(remaining)), key=lambda i: distance(i)[0])
                _, reverse = distance(index)
                if reverse:
                    remaining[index].reverse()
            stroke = remaining.pop(index)
            ordered.append(stroke)
            tail = stroke[-1]
        return ordered


class StrokePathService:
    """Select skeleton or grid planning and fall back to grid when needed."""

    def __init__(
        self,
        grid_service: GridStrokePathService | None = None,
        skeleton_service: SkeletonStrokePathService | None = None,
    ) -> None:
        self._grid = grid_service or GridStrokePathService()
        self._skeleton = skeleton_service or SkeletonStrokePathService()

    def build(
        self,
        ink_mask: np.ndarray,
        mode: str,
        grid_size: int,
        skeleton_min_points: int,
        skeleton_spacing: float,
    ) -> StrokePlan:
        if mode == "skeleton":
            plan = self._skeleton.build(
                ink_mask,
                min_points=skeleton_min_points,
                spacing=skeleton_spacing,
            )
            if plan.strokes:
                return plan
        return self._grid.build(ink_mask, grid_size)


def _pen_lift_indices(strokes: list[list[tuple[int, int]]]) -> frozenset[int]:
    offsets: set[int] = set()
    cursor = 0
    for index, stroke in enumerate(strokes):
        if index > 0 and stroke:
            offsets.add(cursor)
        cursor += len(stroke)
    return frozenset(offsets)
