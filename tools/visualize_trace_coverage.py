#!/usr/bin/env python3
"""
Render region coverage visualizations from active inference trace JSONL.

Coordinate convention used in this script:
- region key format: "x:y"
- x = row index, y = column index
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


GRID_SIZE = 8


@dataclass
class TraceCoverageData:
    trace_path: Path
    region_histogram: dict[str, int]
    region_sequence: list[tuple[int, str]]
    levels_completed_max: int


def _parse_region_key(region_key: str | None) -> tuple[int, int] | None:
    if not isinstance(region_key, str) or ":" not in region_key:
        return None
    head, tail = region_key.split(":", 1)
    try:
        row = int(head)
        col = int(tail)
    except ValueError:
        return None
    if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
        return row, col
    return None


def _load_trace(trace_path: Path) -> TraceCoverageData:
    latest_hist: dict[str, int] = {}
    region_sequence: list[tuple[int, str]] = []
    max_levels_completed = 0

    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            action_counter = row.get("action_counter")
            if action_counter is None:
                continue

            obs = ((row.get("observation_packet_summary") or {}).get("observation") or {})
            levels_completed = int(obs.get("levels_completed", 0) or 0)
            if levels_completed > max_levels_completed:
                max_levels_completed = levels_completed

            metadata = ((row.get("selected_candidate") or {}).get("metadata") or {})
            region_snapshot = (metadata.get("region_graph_snapshot_v1") or {})
            if not region_snapshot:
                region_snapshot = (
                    (row.get("operability_diagnostics_v1") or {}).get("region_graph_snapshot_v1")
                    or {}
                )

            hist = region_snapshot.get("region_visit_histogram")
            if isinstance(hist, dict) and hist:
                normalized_hist: dict[str, int] = {}
                for k, v in hist.items():
                    parsed = _parse_region_key(str(k))
                    if parsed is None:
                        continue
                    try:
                        normalized_hist[f"{parsed[0]}:{parsed[1]}"] = int(max(0, int(v)))
                    except (TypeError, ValueError):
                        continue
                if normalized_hist:
                    latest_hist = normalized_hist

            current_region = region_snapshot.get("current_region_key")
            parsed_current = _parse_region_key(current_region)
            if parsed_current is not None:
                region_sequence.append((int(action_counter), f"{parsed_current[0]}:{parsed_current[1]}"))

    if not latest_hist:
        # Fallback: build histogram directly from sequence if snapshot histogram is missing.
        seq_hist: dict[str, int] = {}
        for _, key in region_sequence:
            seq_hist[key] = int(seq_hist.get(key, 0)) + 1
        latest_hist = seq_hist

    return TraceCoverageData(
        trace_path=trace_path,
        region_histogram=latest_hist,
        region_sequence=region_sequence,
        levels_completed_max=max_levels_completed,
    )


def _find_latest_trace(repo_root: Path) -> Path:
    trace_dir = repo_root / "recordings" / "active_inference_traces"
    candidates = sorted(trace_dir.glob("*.trace.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No trace files found under: {trace_dir}")
    return candidates[0]


def _build_grid_from_histogram(hist: dict[str, int]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
    for key, count in hist.items():
        parsed = _parse_region_key(key)
        if parsed is None:
            continue
        row, col = parsed
        grid[row, col] = int(max(0, count))
    return grid


def _build_path_points(region_sequence: list[tuple[int, str]]) -> tuple[np.ndarray, list[int]]:
    coords: list[tuple[float, float]] = []
    action_steps: list[int] = []
    for action_counter, key in region_sequence:
        parsed = _parse_region_key(key)
        if parsed is None:
            continue
        row, col = parsed
        # Use cell center for plotting.
        coords.append((float(col) + 0.5, float(row) + 0.5))
        action_steps.append(action_counter)
    if not coords:
        return np.zeros((0, 2), dtype=np.float64), action_steps
    return np.array(coords, dtype=np.float64), action_steps


def _render(data: TraceCoverageData, output_path: Path) -> None:
    heatmap = _build_grid_from_histogram(data.region_histogram)
    path_points, path_steps = _build_path_points(data.region_sequence)

    fig, (ax_heat, ax_path) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # Left: coverage heatmap (visit count by region).
    im = ax_heat.imshow(heatmap, cmap="YlOrRd", origin="upper")
    ax_heat.set_title("Coverage Heatmap (row x, col y)")
    ax_heat.set_xlabel("col y")
    ax_heat.set_ylabel("row x")
    ax_heat.set_xticks(range(GRID_SIZE))
    ax_heat.set_yticks(range(GRID_SIZE))
    ax_heat.set_xticklabels([f"y{i}" for i in range(GRID_SIZE)])
    ax_heat.set_yticklabels([f"x{i}" for i in range(GRID_SIZE)])
    ax_heat.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax_heat.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax_heat.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax_heat.grid(which="minor", color="black", linestyle="-", linewidth=0.8, alpha=0.35)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            count = int(heatmap[row, col])
            if count <= 0:
                continue
            ax_heat.text(
                col,
                row,
                str(count),
                ha="center",
                va="center",
                color="white" if count > np.percentile(heatmap[heatmap > 0], 65) else "black",
                fontsize=9,
                fontweight="bold",
            )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("visit count")

    # Right: trajectory plot in region-grid space.
    ax_path.set_title("Region Path (action order)")
    ax_path.set_xlabel("col y")
    ax_path.set_ylabel("row x")
    ax_path.set_xlim(0, GRID_SIZE)
    ax_path.set_ylim(GRID_SIZE, 0)
    ax_path.set_xticks(range(GRID_SIZE + 1))
    ax_path.set_yticks(range(GRID_SIZE + 1))
    ax_path.set_xticklabels([str(i) for i in range(GRID_SIZE + 1)])
    ax_path.set_yticklabels([str(i) for i in range(GRID_SIZE + 1)])
    ax_path.grid(True, color="#666", alpha=0.3, linewidth=0.8)

    if len(path_points) >= 2:
        segments = np.stack([path_points[:-1], path_points[1:]], axis=1)
        t = np.linspace(0.0, 1.0, len(segments))
        lc = LineCollection(segments, cmap="viridis", array=t, linewidths=1.8, alpha=0.9)
        ax_path.add_collection(lc)
    if len(path_points) >= 1:
        ax_path.scatter(path_points[0, 0], path_points[0, 1], c="lime", s=90, marker="o", label="start")
        ax_path.scatter(path_points[-1, 0], path_points[-1, 1], c="red", s=90, marker="X", label="end")
        ax_path.legend(loc="upper right", framealpha=0.9)

    meta_title = (
        f"trace: {data.trace_path.name}\\n"
        f"steps={len(data.region_sequence)} | max_levels_completed={data.levels_completed_max}"
    )
    fig.suptitle(meta_title, fontsize=10)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize coverage/path from active inference trace.")
    parser.add_argument(
        "--trace",
        type=str,
        default="",
        help="Path to *.trace.jsonl. If omitted, use latest trace under recordings/active_inference_traces.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG path. Defaults to recordings/coverage_viz/<trace_stem>_coverage.png",
    )
    args = parser.parse_args()

    repo_root = Path.cwd()
    trace_path = Path(args.trace).expanduser().resolve() if args.trace else _find_latest_trace(repo_root)
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")

    data = _load_trace(trace_path)
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        out_dir = repo_root / "recordings" / "coverage_viz"
        out_path = (out_dir / f"{trace_path.stem}_coverage.png").resolve()

    _render(data, out_path)
    print(f"trace: {trace_path}")
    print(f"output: {out_path}")
    print(f"max_levels_completed: {data.levels_completed_max}")
    print(f"region_sequence_len: {len(data.region_sequence)}")
    print(f"known_regions: {len(data.region_histogram)}")


if __name__ == "__main__":
    main()
