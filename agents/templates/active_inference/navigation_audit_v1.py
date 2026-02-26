from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

from .navigation_map_v1 import compute_walkable_component_mask_from_anchor_v1


PALETTE_RGB_16 = [
    (16, 18, 22),
    (138, 180, 255),
    (250, 84, 84),
    (132, 136, 144),
    (70, 74, 80),
    (8, 10, 12),
    (203, 89, 226),
    (255, 112, 193),
    (245, 62, 62),
    (44, 162, 255),
    (132, 220, 255),
    (243, 221, 54),
    (255, 156, 47),
    (140, 32, 48),
    (72, 208, 72),
    (160, 108, 216),
]


def _frame_to_grid(frame_any: Any) -> list[list[int]]:
    frame = frame_any
    for _ in range(3):
        if not isinstance(frame, list) or not frame:
            return []
        first = frame[0]
        if isinstance(first, list) and first and isinstance(first[0], list):
            frame = first
            continue
        break
    if not isinstance(frame, list) or not frame:
        return []
    width = len(frame[0]) if isinstance(frame[0], list) else 0
    if width <= 0:
        return []
    out: list[list[int]] = []
    for row in frame:
        if not isinstance(row, list) or len(row) < width:
            return []
        out.append([int(v) for v in row[:width]])
    return out


def _parse_region_key(region_key: str) -> tuple[int, int] | None:
    token = str(region_key)
    if ":" not in token:
        return None
    left, right = token.split(":", 1)
    try:
        row_x = int(left)
        col_y = int(right)
    except Exception:
        return None
    if row_x < 0 or col_y < 0:
        return None
    return (row_x, col_y)


def _as_undirected_region_graph(raw_adj: dict[str, Any] | None) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not isinstance(raw_adj, dict):
        return out
    for src_raw, nbrs_raw in raw_adj.items():
        src = str(src_raw)
        if _parse_region_key(src) is None or not isinstance(nbrs_raw, dict):
            continue
        out.setdefault(src, {})
        for dst_raw, weight_raw in nbrs_raw.items():
            dst = str(dst_raw)
            if _parse_region_key(dst) is None:
                continue
            w = int(max(0, weight_raw))
            if w <= 0:
                continue
            out.setdefault(dst, {})
            out[src][dst] = max(int(out[src].get(dst, 0)), int(w))
            out[dst][src] = max(int(out[dst].get(src, 0)), int(w))
    return out


def _bfs_distances(graph: dict[str, dict[str, int]], start: str) -> dict[str, int]:
    start_s = str(start)
    if start_s not in graph:
        return {}
    dist: dict[str, int] = {start_s: 0}
    queue: deque[str] = deque([start_s])
    while queue:
        node = str(queue.popleft())
        d = int(dist.get(node, 0))
        for nbr in graph.get(node, {}).keys():
            nbr_s = str(nbr)
            if nbr_s in dist:
                continue
            dist[nbr_s] = int(d + 1)
            queue.append(nbr_s)
    return dist


def _region_graph_diameter(graph: dict[str, dict[str, int]]) -> dict[str, Any]:
    nodes = sorted({str(src) for src in graph.keys()} | {str(dst) for nbrs in graph.values() for dst in nbrs.keys()})
    if not nodes:
        return {
            "distance_steps": -1,
            "point_a": {"region_key": "NA", "row_x": -1, "col_y": -1},
            "point_b": {"region_key": "NA", "row_x": -1, "col_y": -1},
            "node_count": 0,
        }
    best_dist = -1
    best_a = nodes[0]
    best_b = nodes[0]
    for src in nodes:
        dist = _bfs_distances(graph, src)
        for dst, d in dist.items():
            if int(d) > int(best_dist):
                best_dist = int(d)
                best_a = str(src)
                best_b = str(dst)
    pa = _parse_region_key(best_a) or (-1, -1)
    pb = _parse_region_key(best_b) or (-1, -1)
    return {
        "distance_steps": int(best_dist),
        "point_a": {"region_key": str(best_a), "row_x": int(pa[0]), "col_y": int(pa[1])},
        "point_b": {"region_key": str(best_b), "row_x": int(pb[0]), "col_y": int(pb[1])},
        "node_count": int(len(nodes)),
    }


def _pixel_graph_diameter(mask: list[list[bool]]) -> dict[str, Any]:
    rows = int(len(mask))
    cols = int(min((len(r) for r in mask if isinstance(r, list)), default=0))
    if rows <= 0 or cols <= 0:
        return {
            "distance_steps": -1,
            "point_a": {"row_x": -1, "col_y": -1},
            "point_b": {"row_x": -1, "col_y": -1},
            "node_count": 0,
        }

    walkable = [(r, c) for r in range(rows) for c in range(cols) if bool(mask[r][c])]
    if not walkable:
        return {
            "distance_steps": -1,
            "point_a": {"row_x": -1, "col_y": -1},
            "point_b": {"row_x": -1, "col_y": -1},
            "node_count": 0,
        }

    walkable_set = set(walkable)
    if len(walkable_set) > 5000:
        # Keep runtime bounded for larger maps.
        return {
            "distance_steps": -1,
            "point_a": {"row_x": -1, "col_y": -1},
            "point_b": {"row_x": -1, "col_y": -1},
            "node_count": int(len(walkable_set)),
            "reason": "too_many_nodes",
        }

    best_dist = -1
    best_a = walkable[0]
    best_b = walkable[0]

    for src in walkable:
        queue: deque[tuple[int, int]] = deque([src])
        dist: dict[tuple[int, int], int] = {src: 0}
        while queue:
            cr, cc = queue.popleft()
            d = int(dist[(cr, cc)])
            if d > best_dist:
                best_dist = int(d)
                best_a = src
                best_b = (cr, cc)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(cr + dr)
                nc = int(cc + dc)
                nxt = (nr, nc)
                if nxt in dist or nxt not in walkable_set:
                    continue
                dist[nxt] = int(d + 1)
                queue.append(nxt)

    return {
        "distance_steps": int(best_dist),
        "point_a": {"row_x": int(best_a[0]), "col_y": int(best_a[1])},
        "point_b": {"row_x": int(best_b[0]), "col_y": int(best_b[1])},
        "node_count": int(len(walkable_set)),
    }


def _render_navigation_map_check(
    *,
    frame: list[list[int]],
    walkable_mask: list[list[bool]],
    pixel_diameter: dict[str, Any],
    region_diameter: dict[str, Any],
    output_png_path: Path,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return "matplotlib_unavailable"

    rows = int(len(frame))
    cols = int(min((len(r) for r in frame if isinstance(r, list)), default=0))
    if rows <= 0 or cols <= 0:
        return "invalid_frame"

    palette = np.array(PALETTE_RGB_16, dtype=np.uint8)
    img_idx = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            img_idx[r, c] = int(frame[r][c]) % int(len(palette))
    img_rgb = palette[img_idx]

    walk = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            walk[r, c] = 1.0 if bool(walkable_mask[r][c]) else 0.0

    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.imshow(img_rgb, interpolation="nearest")
    ax.imshow(walk, cmap="Greens", alpha=0.25, interpolation="nearest")

    pa = pixel_diameter.get("point_a", {}) if isinstance(pixel_diameter, dict) else {}
    pb = pixel_diameter.get("point_b", {}) if isinstance(pixel_diameter, dict) else {}
    for point, color, label in ((pa, "lime", "A"), (pb, "red", "B")):
        rr = int(point.get("row_x", -1)) if isinstance(point, dict) else -1
        cc = int(point.get("col_y", -1)) if isinstance(point, dict) else -1
        if rr >= 0 and cc >= 0:
            ax.scatter([cc], [rr], s=28, c=[color], marker="x")
            ax.text(cc + 0.5, rr + 0.5, label, color=color, fontsize=8)

    ax.set_title(
        f"Navigation Map Check | pixel_diameter={int(pixel_diameter.get('distance_steps', -1))} | "
        f"region_diameter={int(region_diameter.get('distance_steps', -1))}"
    )
    ax.set_xlabel("col y")
    ax.set_ylabel("row x")
    ax.set_xlim(-0.5, float(cols - 0.5))
    ax.set_ylim(float(rows - 0.5), -0.5)
    fig.tight_layout()
    fig.savefig(output_png_path)
    plt.close(fig)
    return "ok"


def run_navigation_map_audit_v1(
    *,
    frame_any: Any,
    anchor_xy: tuple[int, int] | None,
    navigation_map_snapshot_v1: dict[str, Any] | None,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_name": "active_inference_navigation_map_audit_v1",
        "schema_version": 1,
        "enabled": False,
        "reason": "na",
        "run_name": str(run_name),
        "anchor": {"row_x": -1, "col_y": -1},
        "map_dimensions": {"rows": 0, "cols": 0},
        "pixel_diameter": {
            "distance_steps": -1,
            "point_a": {"row_x": -1, "col_y": -1},
            "point_b": {"row_x": -1, "col_y": -1},
            "node_count": 0,
        },
        "region_diameter": {
            "distance_steps": -1,
            "point_a": {"region_key": "NA", "row_x": -1, "col_y": -1},
            "point_b": {"region_key": "NA", "row_x": -1, "col_y": -1},
            "node_count": 0,
        },
        "walkable_component_pixels": 0,
        "selected_color_total_pixels": 0,
        "walkable_vs_selected_iou": 0.0,
        "map_png_path": "",
        "summary_json_path": "",
    }

    frame = _frame_to_grid(frame_any)
    rows = int(len(frame))
    cols = int(min((len(r) for r in frame if isinstance(r, list)), default=0))
    summary["map_dimensions"] = {"rows": int(rows), "cols": int(cols)}
    if rows <= 0 or cols <= 0:
        summary["reason"] = "invalid_frame"
        return summary
    if anchor_xy is None:
        summary["reason"] = "missing_anchor"
        return summary

    ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
    if ay < 0 or ay >= rows or ax < 0 or ax >= cols:
        summary["reason"] = "invalid_anchor"
        return summary

    summary["anchor"] = {"row_x": int(ay), "col_y": int(ax)}

    component = compute_walkable_component_mask_from_anchor_v1(
        frame,
        anchor_x=int(ax),
        anchor_y=int(ay),
    )
    summary["walkable_component_meta_v1"] = dict(component.meta)
    if not component.enabled or component.mask is None:
        summary["reason"] = "walkable_component_disabled"
        return summary

    mask = component.mask
    summary["walkable_component_pixels"] = int(
        sum(1 for row in mask for v in row if bool(v))
    )

    selected_color = int(component.meta.get("selected_color", -1))
    if selected_color >= 0:
        selected_mask = [[int(v) == int(selected_color) for v in row] for row in frame]
        selected_total = int(sum(1 for row in selected_mask for v in row if bool(v)))
        inter = int(
            sum(
                1
                for r in range(rows)
                for c in range(cols)
                if bool(mask[r][c]) and bool(selected_mask[r][c])
            )
        )
        union = int(
            sum(
                1
                for r in range(rows)
                for c in range(cols)
                if bool(mask[r][c]) or bool(selected_mask[r][c])
            )
        )
        summary["selected_color_total_pixels"] = int(selected_total)
        summary["walkable_vs_selected_iou"] = float(
            float(inter) / float(max(1, union))
        )

    nav_snapshot = navigation_map_snapshot_v1 or {}
    if not isinstance(nav_snapshot, dict):
        nav_snapshot = {}
    adjacency = _as_undirected_region_graph(
        nav_snapshot.get("walkable_region_adjacency_v1", {})
    )
    summary["region_diameter"] = _region_graph_diameter(adjacency)
    summary["pixel_diameter"] = _pixel_graph_diameter(mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_png_path = (output_dir / f"{run_name}_navigation_map_check.png").resolve()
    render_status = _render_navigation_map_check(
        frame=frame,
        walkable_mask=mask,
        pixel_diameter=summary["pixel_diameter"],
        region_diameter=summary["region_diameter"],
        output_png_path=output_png_path,
    )
    summary["render_status"] = str(render_status)
    if str(render_status) == "ok":
        summary["map_png_path"] = str(output_png_path)

    output_json_path = (
        output_dir / f"{run_name}_navigation_map_check.summary.json"
    ).resolve()
    summary["summary_json_path"] = str(output_json_path)
    summary["enabled"] = True
    summary["reason"] = "ok"
    output_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
