from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class WalkableComponentResultV1:
    enabled: bool
    mask: list[list[bool]] | None
    meta: dict[str, Any]


def _frame_to_grid(frame_any: Any) -> list[list[int]]:
    """Normalize frame payload to a 2D int grid.

    Supports either:
    - 2D grid: list[list[int]]
    - frame-chain: list[frame], where frame is list[list[int]]
    """
    frame = frame_any
    for _ in range(3):
        if not isinstance(frame, list) or not frame:
            return []
        first = frame[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            # frame chain -> use latest frame
            frame = first
            continue
        break
    if not isinstance(frame, list) or not frame:
        return []
    out: list[list[int]] = []
    width = None
    for row in frame:
        if not isinstance(row, list):
            return []
        if width is None:
            width = len(row)
            if width <= 0:
                return []
        if len(row) < int(width):
            return []
        out.append([int(v) for v in row[: int(width)]])
    return out


def _frame_dimensions(frame: list[list[int]]) -> tuple[int, int]:
    if not frame:
        return (0, 0)
    height = int(len(frame))
    width = int(min(len(row) for row in frame if isinstance(row, list)) or 0)
    return (height, width)


def _mask_digest_v1(mask: list[list[bool]] | None) -> str:
    if mask is None:
        return "na"
    h = hashlib.sha256()
    for row in mask:
        if not isinstance(row, list):
            continue
        h.update(bytes([1 if bool(v) else 0 for v in row]))
    return h.hexdigest()[:16]


def compute_walkable_component_mask_from_anchor_v1(
    frame_any: Any,
    *,
    anchor_x: int,
    anchor_y: int,
    search_radius: int = 6,
    min_component_pixels: int = 80,
    dilation_steps: int = 3,
) -> WalkableComponentResultV1:
    meta: dict[str, Any] = {
        "schema_name": "active_inference_walkable_component_meta_v1",
        "schema_version": 1,
        "enabled": False,
        "anchor_xy": {"x": int(anchor_x), "y": int(anchor_y)},
        "selected_color": -1,
        "component_pixels": 0,
        "dilated_pixels": 0,
        "candidate_count": 0,
        "reason": "na",
        "search_radius": int(search_radius),
        "min_component_pixels": int(min_component_pixels),
        "dilation_steps": int(dilation_steps),
    }

    frame = _frame_to_grid(frame_any)
    height, width = _frame_dimensions(frame)
    if height <= 0 or width <= 0:
        meta["reason"] = "invalid_dimensions"
        return WalkableComponentResultV1(False, None, meta)
    if anchor_x < 0 or anchor_y < 0 or anchor_x >= width or anchor_y >= height:
        meta["reason"] = "invalid_anchor"
        return WalkableComponentResultV1(False, None, meta)

    def _component_mask_for_seed(seed_x: int, seed_y: int, color: int) -> list[list[bool]]:
        mask = [[False for _ in range(width)] for _ in range(height)]
        queue: list[tuple[int, int]] = [(int(seed_x), int(seed_y))]
        head = 0
        mask[int(seed_y)][int(seed_x)] = True
        while head < len(queue):
            cx, cy = queue[head]
            head += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = int(cx + dx)
                ny = int(cy + dy)
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if mask[ny][nx]:
                    continue
                if int(frame[ny][nx]) != int(color):
                    continue
                mask[ny][nx] = True
                queue.append((int(nx), int(ny)))
        return mask

    def _mask_pixels(mask: list[list[bool]]) -> int:
        return int(sum(1 for row in mask for v in row if bool(v)))

    center_color = int(frame[int(anchor_y)][int(anchor_x)])
    color_hist: dict[int, int] = {}
    nearest_xy_by_color: dict[int, tuple[int, int, int]] = {}

    y0 = max(0, int(anchor_y) - int(search_radius))
    y1 = min(height, int(anchor_y) + int(search_radius) + 1)
    x0 = max(0, int(anchor_x) - int(search_radius))
    x1 = min(width, int(anchor_x) + int(search_radius) + 1)
    for y in range(y0, y1):
        row = frame[y]
        for x in range(x0, x1):
            dist = int(abs(int(x) - int(anchor_x)) + abs(int(y) - int(anchor_y)))
            if dist < 2 or dist > int(search_radius):
                continue
            color = int(row[x])
            color_hist[color] = int(color_hist.get(color, 0) + 1)
            prev = nearest_xy_by_color.get(color)
            if prev is None or dist < int(prev[0]):
                nearest_xy_by_color[color] = (int(dist), int(x), int(y))

    ranked_colors = sorted(
        color_hist.items(),
        key=lambda item: (-int(item[1]), 0 if int(item[0]) != int(center_color) else 1, int(item[0])),
    )
    candidate_colors = [int(color) for (color, count) in ranked_colors if int(count) >= 4][:6]
    if int(center_color) not in candidate_colors:
        candidate_colors.append(int(center_color))
    meta["candidate_count"] = int(len(candidate_colors))

    best_mask: list[list[bool]] | None = None
    best_pixels = 0
    best_color = -1

    for color in candidate_colors:
        if int(color) in nearest_xy_by_color:
            _, sx, sy = nearest_xy_by_color[int(color)]
        else:
            sx, sy = int(anchor_x), int(anchor_y)
        if int(frame[int(sy)][int(sx)]) != int(color):
            found = False
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    if int(frame[yy][xx]) == int(color):
                        sx, sy = int(xx), int(yy)
                        found = True
                        break
                if found:
                    break
            if not found:
                continue

        mask = _component_mask_for_seed(int(sx), int(sy), int(color))
        pixels = _mask_pixels(mask)
        if pixels > int(best_pixels):
            best_pixels = int(pixels)
            best_mask = mask
            best_color = int(color)

    if best_mask is None or int(best_pixels) < int(min_component_pixels):
        meta["reason"] = "component_too_small"
        meta["component_pixels"] = int(best_pixels)
        return WalkableComponentResultV1(False, None, meta)

    dilated = [[bool(v) for v in row] for row in best_mask]
    for _ in range(int(max(0, dilation_steps))):
        expanded = [list(row) for row in dilated]
        for y in range(height):
            for x in range(width):
                if not dilated[y][x]:
                    continue
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = int(x + dx)
                    ny = int(y + dy)
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    expanded[ny][nx] = True
        dilated = expanded

    meta["enabled"] = True
    meta["reason"] = "ok"
    meta["selected_color"] = int(best_color)
    meta["component_pixels"] = int(best_pixels)
    meta["dilated_pixels"] = int(_mask_pixels(dilated))
    return WalkableComponentResultV1(True, dilated, meta)


def region_key_from_xy_v1(x: int, y: int, *, region_size: int) -> str:
    if region_size <= 0:
        return "NA"
    return f"{int(y)//int(region_size)}:{int(x)//int(region_size)}"


def parse_region_key_v1(region_key: str) -> tuple[int, int] | None:
    token = str(region_key)
    if ":" not in token:
        return None
    left, right = token.split(":", 1)
    try:
        row = int(left)
        col = int(right)
    except Exception:
        return None
    if row < 0 or col < 0:
        return None
    return (row, col)


def compute_region_walkable_ratio_v1(mask: list[list[bool]], *, region_size: int) -> dict[str, float]:
    height = int(len(mask))
    width = int(min(len(row) for row in mask if isinstance(row, list)) or 0)
    if height <= 0 or width <= 0 or region_size <= 0:
        return {}

    rows = int(math.ceil(float(height) / float(region_size)))
    cols = int(math.ceil(float(width) / float(region_size)))
    ratios: dict[str, float] = {}

    for rr in range(rows):
        for cc in range(cols):
            y0 = int(rr * region_size)
            y1 = int(min(height, (rr + 1) * region_size))
            x0 = int(cc * region_size)
            x1 = int(min(width, (cc + 1) * region_size))
            if y1 <= y0 or x1 <= x0:
                continue
            count = 0
            for y in range(y0, y1):
                row = mask[y]
                for x in range(x0, x1):
                    if bool(row[x]):
                        count += 1
            denom = float(max(1, (y1 - y0) * (x1 - x0)))
            ratios[f"{rr}:{cc}"] = float(count) / denom

    return ratios


def compute_region_adjacency_from_mask_v1(mask: list[list[bool]], *, region_size: int) -> dict[str, dict[str, int]]:
    height = int(len(mask))
    width = int(min(len(row) for row in mask if isinstance(row, list)) or 0)
    if height <= 0 or width <= 0 or region_size <= 0:
        return {}

    adjacency: dict[str, dict[str, int]] = {}

    def _add(a: str, b: str) -> None:
        if a == b:
            return
        row = adjacency.setdefault(a, {})
        row[b] = int(row.get(b, 0) + 1)

    for y in range(height):
        row = mask[y]
        for x in range(width):
            if not bool(row[x]):
                continue
            a = region_key_from_xy_v1(x, y, region_size=region_size)
            if x + 1 < width and bool(row[x + 1]):
                b = region_key_from_xy_v1(x + 1, y, region_size=region_size)
                if a != b:
                    _add(a, b)
                    _add(b, a)
            if y + 1 < height and bool(mask[y + 1][x]):
                b = region_key_from_xy_v1(x, y + 1, region_size=region_size)
                if a != b:
                    _add(a, b)
                    _add(b, a)

    return adjacency


def build_navigation_map_snapshot_v1(
    frame_any: Any,
    *,
    agent_pos_xy: tuple[int, int] | None,
    region_size: int = 8,
    walkable_ratio_threshold: float = 0.02,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_name": "active_inference_navigation_map_snapshot_v1",
        "schema_version": 1,
        "enabled": False,
        "region_size": int(region_size),
        "walkable_ratio_threshold": float(walkable_ratio_threshold),
        "anchor_xy": {"x": -1, "y": -1},
        "mask_digest": "na",
        "walkable_component_meta_v1": {},
        "walkable_region_ratio_v1": {},
        "walkable_region_adjacency_v1": {},
        "reason": "na",
    }

    if agent_pos_xy is None:
        payload["reason"] = "missing_agent_pos"
        return payload

    ax, ay = agent_pos_xy
    payload["anchor_xy"] = {"x": int(ax), "y": int(ay)}

    component = compute_walkable_component_mask_from_anchor_v1(
        frame_any,
        anchor_x=int(ax),
        anchor_y=int(ay),
    )
    payload["walkable_component_meta_v1"] = dict(component.meta)
    if not component.enabled or component.mask is None:
        payload["reason"] = "walkable_component_disabled"
        return payload

    payload["enabled"] = True
    payload["mask_digest"] = str(_mask_digest_v1(component.mask))
    ratios = compute_region_walkable_ratio_v1(component.mask, region_size=int(region_size))
    payload["walkable_region_ratio_v1"] = {
        str(k): float(v) for (k, v) in ratios.items()
    }
    adjacency = compute_region_adjacency_from_mask_v1(component.mask, region_size=int(region_size))
    payload["walkable_region_adjacency_v1"] = {
        str(src): {str(dst): int(w) for (dst, w) in nbrs.items()}
        for (src, nbrs) in adjacency.items()
    }
    payload["reason"] = "ok"
    return payload
