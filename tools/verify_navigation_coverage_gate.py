#!/usr/bin/env python3
"""
Navigation coverage gate checker.

Hard gate:
- agent map must be available
- every reachable map region must be visited at least once
- every reachable map region must be visited at least twice

Coordinate convention in output:
- 行row, 列col（region key: row:col）
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any


def _parse_region_key(region_key: str | None) -> tuple[int, int] | None:
    if not isinstance(region_key, str) or ":" not in region_key:
        return None
    left, right = region_key.split(":", 1)
    try:
        row = int(left)
        col = int(right)
    except Exception:
        return None
    if row < 0 or col < 0:
        return None
    return (row, col)


def _format_region_key_human(region_key: str) -> str:
    parsed = _parse_region_key(region_key)
    if parsed is None:
        return "行row?-列col?"
    row, col = parsed
    return f"行row{row}列col{col}"


def _find_latest_trace(repo_root: Path) -> Path:
    trace_dir = repo_root / "recordings" / "active_inference_traces"
    traces = sorted(trace_dir.glob("*.trace.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        raise FileNotFoundError(f"No trace files found under: {trace_dir}")
    return traces[0]


def _load_final_summary(trace_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("schema_name") == "active_inference_agent_summary_v1":
                summary = row
    if not summary:
        raise ValueError(f"No active_inference_agent_summary_v1 found in {trace_path}")
    return summary


def _best_navigation_map_snapshot_from_trace(trace_path: Path) -> dict[str, Any]:
    best_snapshot: dict[str, Any] = {}
    best_score: tuple[int, float, int] | None = None
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("schema_name") != "active_inference_step_trace_v1":
                continue
            selected = row.get("selected_candidate") or {}
            metadata = selected.get("metadata") if isinstance(selected, dict) else {}
            if not isinstance(metadata, dict):
                continue
            snap = metadata.get("navigation_map_snapshot_v1") or {}
            if not isinstance(snap, dict):
                continue
            if snap.get("schema_name") != "active_inference_navigation_map_snapshot_v1":
                continue
            if not bool(snap.get("enabled", False)):
                continue
            meta = snap.get("walkable_component_meta_v1") or {}
            component_pixels = int(max(0, meta.get("component_pixels", 0)))
            component_ratio = float(meta.get("component_ratio", 0.0))
            border_touch = int(max(0, meta.get("border_touch_pixels", 0)))
            border_touch_ratio = float(border_touch) / float(max(1, component_pixels))
            stable_component = bool(
                component_pixels >= 200 and 0.05 <= component_ratio <= 0.45
            )
            score = (
                0 if stable_component else 1,
                float(border_touch_ratio),
                -int(component_pixels),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_snapshot = dict(snap)
    return best_snapshot


def _normalized_adjacency(raw: Any) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if not isinstance(raw, dict):
        return out
    for src_raw, nbrs_raw in raw.items():
        src = str(src_raw)
        if _parse_region_key(src) is None or not isinstance(nbrs_raw, dict):
            continue
        out.setdefault(src, {})
        for dst_raw, w_raw in nbrs_raw.items():
            dst = str(dst_raw)
            if _parse_region_key(dst) is None:
                continue
            try:
                w = int(max(0, int(w_raw)))
            except Exception:
                continue
            if w <= 0:
                continue
            out.setdefault(dst, {})
            out[src][dst] = max(int(out[src].get(dst, 0)), int(w))
            out[dst][src] = max(int(out[dst].get(src, 0)), int(w))
    return out


def _bfs_reachable(graph: dict[str, dict[str, int]], start: str, allowed: set[str]) -> set[str]:
    start_s = str(start)
    if start_s not in allowed:
        return set()
    if start_s not in graph:
        return {start_s}
    seen: set[str] = {start_s}
    queue: deque[str] = deque([start_s])
    while queue:
        node = str(queue.popleft())
        for nbr in graph.get(node, {}).keys():
            nbr_s = str(nbr)
            if nbr_s not in allowed or nbr_s in seen:
                continue
            seen.add(nbr_s)
            queue.append(nbr_s)
    return seen


def _bfs_distance(graph: dict[str, dict[str, int]], start: str) -> dict[str, int]:
    start_s = str(start)
    dist: dict[str, int] = {start_s: 0}
    if start_s not in graph:
        return dist
    queue: deque[str] = deque([start_s])
    while queue:
        node = str(queue.popleft())
        base = int(dist.get(node, 0))
        for nbr in graph.get(node, {}).keys():
            nbr_s = str(nbr)
            if nbr_s in dist:
                continue
            dist[nbr_s] = int(base + 1)
            queue.append(nbr_s)
    return dist


def _diameter(graph: dict[str, dict[str, int]], nodes: set[str]) -> dict[str, Any]:
    best_dist = -1
    best_a = "NA"
    best_b = "NA"
    for src in sorted(nodes):
        dmap = _bfs_distance(graph, src)
        for dst in nodes:
            d = int(dmap.get(dst, -1))
            if d > best_dist:
                best_dist = d
                best_a = str(src)
                best_b = str(dst)
    pa = _parse_region_key(best_a) or (-1, -1)
    pb = _parse_region_key(best_b) or (-1, -1)
    return {
        "distance_steps": int(best_dist),
        "point_a": {
            "region_key": str(best_a),
            "row": int(pa[0]),
            "col": int(pa[1]),
            "row_x": int(pa[0]),
            "col_y": int(pa[1]),
        },
        "point_b": {
            "region_key": str(best_b),
            "row": int(pb[0]),
            "col": int(pb[1]),
            "row_x": int(pb[0]),
            "col_y": int(pb[1]),
        },
    }


def _build_report(summary: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    nav_map_trace = _best_navigation_map_snapshot_from_trace(trace_path)
    nav_map_summary = summary.get("final_navigation_map_snapshot_v1") or {}
    nav_map = dict(nav_map_trace) if nav_map_trace else dict(nav_map_summary)
    nav_audit = summary.get("final_navigation_map_audit_v1") or {}
    operability = summary.get("final_operability_diagnostics_v1") or {}

    walkable_ratio = nav_map.get("walkable_region_ratio_v1") or {}
    walkable_ratio_threshold = float(nav_map.get("walkable_ratio_threshold", 0.02) or 0.02)
    region_size = int(nav_map.get("region_size", 8) or 8)
    adjacency = _normalized_adjacency(nav_map.get("walkable_region_adjacency_v1") or {})

    region_visits_raw = operability.get("region_visit_histogram") or {}
    region_visits: dict[str, int] = {}
    if isinstance(region_visits_raw, dict):
        for key, value in region_visits_raw.items():
            k = str(key)
            if _parse_region_key(k) is None:
                continue
            try:
                region_visits[k] = int(max(0, int(value)))
            except Exception:
                continue

    required_regions: set[str] = set()
    if isinstance(walkable_ratio, dict):
        for key, value in walkable_ratio.items():
            k = str(key)
            if _parse_region_key(k) is None:
                continue
            try:
                ratio = float(value)
            except Exception:
                continue
            if ratio >= float(walkable_ratio_threshold):
                required_regions.add(k)

    anchor = nav_map.get("anchor_xy") or {}
    anchor_x = int(anchor.get("x", -1))
    anchor_y = int(anchor.get("y", -1))
    anchor_region_key = (
        f"{int(anchor_y // max(1, region_size))}:{int(anchor_x // max(1, region_size))}"
        if anchor_x >= 0 and anchor_y >= 0
        else "NA"
    )

    reachable_regions = _bfs_reachable(adjacency, anchor_region_key, required_regions)
    if not reachable_regions and required_regions:
        # Fallback: treat all required regions as target set if anchor is invalid.
        reachable_regions = set(required_regions)

    visited_once = sorted([k for k in reachable_regions if int(region_visits.get(k, 0)) >= 1])
    visited_twice = sorted([k for k in reachable_regions if int(region_visits.get(k, 0)) >= 2])
    missing_once = sorted([k for k in reachable_regions if int(region_visits.get(k, 0)) < 1])
    missing_twice = sorted([k for k in reachable_regions if int(region_visits.get(k, 0)) < 2])

    coverage_once_ratio = (
        float(len(visited_once)) / float(max(1, len(reachable_regions)))
        if reachable_regions
        else 0.0
    )
    coverage_twice_ratio = (
        float(len(visited_twice)) / float(max(1, len(reachable_regions)))
        if reachable_regions
        else 0.0
    )

    full_coverage_once = bool(reachable_regions) and len(missing_once) == 0
    full_coverage_twice = bool(reachable_regions) and len(missing_twice) == 0
    gate_pass = bool(full_coverage_once and full_coverage_twice)

    diameter = _diameter(adjacency, reachable_regions) if reachable_regions else {
        "distance_steps": -1,
        "point_a": {"region_key": "NA", "row": -1, "col": -1, "row_x": -1, "col_y": -1},
        "point_b": {"region_key": "NA", "row": -1, "col": -1, "row_x": -1, "col_y": -1},
    }

    report: dict[str, Any] = {
        "schema_name": "active_inference_navigation_coverage_gate_v1",
        "schema_version": 1,
        "trace_path": str(trace_path.resolve()),
        "gate_pass": bool(gate_pass),
        "full_coverage_once": bool(full_coverage_once),
        "full_coverage_twice": bool(full_coverage_twice),
        "anchor_region_key": str(anchor_region_key),
        "anchor_region_human": _format_region_key_human(anchor_region_key),
        "walkable_ratio_threshold": float(walkable_ratio_threshold),
        "navigation_map_snapshot_source": (
            "trace_best_snapshot" if nav_map_trace else "summary_final_snapshot"
        ),
        "required_region_count": int(len(required_regions)),
        "reachable_region_count": int(len(reachable_regions)),
        "visited_once_count": int(len(visited_once)),
        "visited_twice_count": int(len(visited_twice)),
        "coverage_once_ratio": float(coverage_once_ratio),
        "coverage_twice_ratio": float(coverage_twice_ratio),
        "missing_once_region_keys": list(missing_once),
        "missing_twice_region_keys": list(missing_twice),
        "missing_once_regions_human": [_format_region_key_human(k) for k in missing_once],
        "missing_twice_regions_human": [_format_region_key_human(k) for k in missing_twice],
        "reachable_region_diameter": dict(diameter),
        "final_navigation_map_audit_v1": dict(nav_audit) if isinstance(nav_audit, dict) else {},
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify navigation map coverage gate from trace.")
    parser.add_argument(
        "--trace",
        type=str,
        default="",
        help="Path to *.trace.jsonl. If empty, uses latest trace.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output json path. Default: recordings/navigation_checks/<trace_stem>_coverage_gate.summary.json",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    trace_path = Path(args.trace).expanduser().resolve() if args.trace else _find_latest_trace(repo_root)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")

    summary = _load_final_summary(trace_path)
    report = _build_report(summary, trace_path)

    out_path: Path
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_dir = (repo_root / "recordings" / "navigation_checks").resolve()
        out_path = (out_dir / f"{trace_path.stem}_coverage_gate.summary.json").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"trace: {trace_path}")
    print(f"coverage_gate_summary: {out_path}")
    print(f"gate_pass: {report.get('gate_pass')}")
    print(
        "coverage_once_ratio: "
        f"{float(report.get('coverage_once_ratio', 0.0)):.4f} "
        f"({int(report.get('visited_once_count', 0))}/{int(report.get('reachable_region_count', 0))})"
    )
    print(
        "coverage_twice_ratio: "
        f"{float(report.get('coverage_twice_ratio', 0.0)):.4f} "
        f"({int(report.get('visited_twice_count', 0))}/{int(report.get('reachable_region_count', 0))})"
    )
    diameter = report.get("reachable_region_diameter", {})
    point_a = (diameter.get("point_a", {}) if isinstance(diameter, dict) else {})
    point_b = (diameter.get("point_b", {}) if isinstance(diameter, dict) else {})
    print(
        "farthest_pair: "
        f"{_format_region_key_human(str(point_a.get('region_key', 'NA')))} "
        f"<-> {_format_region_key_human(str(point_b.get('region_key', 'NA')))} "
        f"(distance_steps={int(diameter.get('distance_steps', -1)) if isinstance(diameter, dict) else -1})"
    )


if __name__ == "__main__":
    main()
