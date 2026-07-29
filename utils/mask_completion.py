from __future__ import annotations

import numpy as np


def complete_unassigned_labels(
    label_map: np.ndarray,
    max_iterations: int = 16,
    edge_map: np.ndarray | None = None,
    edge_threshold: float | None = None,
) -> np.ndarray:
    if max_iterations <= 0:
        return label_map

    completed = label_map.astype(np.int32, copy=True)
    if completed.size == 0 or not np.any(completed >= 0) or not np.any(completed < 0):
        return completed

    if edge_map is not None:
        edge = edge_map.astype(np.float32, copy=False)
        if edge.shape != completed.shape:
            edge = None
    else:
        edge = None

    height, width = completed.shape
    shifts = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for _ in range(int(max_iterations)):
        unknown = completed < 0
        if not unknown.any():
            break

        candidates = []
        for dy, dx in shifts:
            shifted = np.full_like(completed, -1)
            src_y0 = max(0, -dy)
            src_y1 = min(height, height - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(width, width - dx)
            dst_y0 = max(0, dy)
            dst_y1 = min(height, height + dy)
            dst_x0 = max(0, dx)
            dst_x1 = min(width, width + dx)
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = completed[src_y0:src_y1, src_x0:src_x1]
            candidates.append(shifted)

        stack = np.stack(candidates, axis=0)
        has_neighbor = unknown & np.any(stack >= 0, axis=0)
        if edge is not None and edge_threshold is not None:
            has_neighbor &= edge <= float(edge_threshold)
        if not has_neighbor.any():
            break

        labels = np.unique(stack[:, has_neighbor])
        labels = labels[labels >= 0]
        if labels.size == 0:
            break

        best_label = np.full_like(completed, -1)
        best_count = np.zeros_like(completed, dtype=np.int16)
        for label in labels:
            count = np.sum(stack == int(label), axis=0).astype(np.int16)
            update = has_neighbor & (count > best_count)
            best_count[update] = count[update]
            best_label[update] = int(label)

        update = has_neighbor & (best_label >= 0)
        if not update.any():
            break
        completed[update] = best_label[update]

    return completed
