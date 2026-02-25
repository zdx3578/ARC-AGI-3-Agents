#!/usr/bin/env python3
"""
Reconstruct pixel-level game map from recording + trace logs.

Outputs:
- modal static map (agent-masked temporal mode)
- reachable walkable mask (connected component from agent positions)
- trajectory overlay on reconstructed map

Coordinate convention:
- image index [row=y, col=x]
- UI label "row x, col y" follows project naming (x=row, y=col).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE_RGB = np.array(
    [
        (16, 18, 22),   # 0
        (138, 180, 255),  # 1
        (250, 84, 84),  # 2
        (132, 136, 144),  # 3 (walkable gray in ls20)
        (70, 74, 80),   # 4 (background gray in ls20)
        (8, 10, 12),    # 5 (void/black panel)
        (203, 89, 226),  # 6
        (255, 112, 193),  # 7
        (245, 62, 62),  # 8
        (44, 162, 255),  # 9
        (132, 220, 255),  # 10
        (243, 221, 54),  # 11
        (255, 156, 47),  # 12
        (140, 32, 48),  # 13
        (72, 208, 72),  # 14
        (160, 108, 216),  # 15
    ],
    dtype=np.uint8,
)


@dataclass
class TraceStepInfo:
    action_counter: int
    agent_pos: tuple[int, int] | None  # (x, y)
    tracked_bbox: tuple[int, int, int, int] | None  # (x0, y0, x1, y1)


@dataclass
class ReconstructionResult:
    modal_frame: np.ndarray
    walkable_mask: np.ndarray
    agent_path_xy: list[tuple[int, int]]
    floor_color: int
    action_count: int


def _load_recording_frames(recording_path: Path) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    with recording_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            data = row.get("data") or {}
            frame_chain = data.get("frame") or []
            if not isinstance(frame_chain, list) or not frame_chain:
                continue
            frame = np.asarray(frame_chain[0], dtype=np.uint8)
            if frame.ndim != 2:
                continue
            frames.append(frame)
    if not frames:
        raise ValueError(f"No usable frames found in recording: {recording_path}")
    return frames


def _load_trace_steps(trace_path: Path) -> dict[int, TraceStepInfo]:
    steps: dict[int, TraceStepInfo] = {}
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("schema_name") != "active_inference_step_trace_v1":
                continue
            action_counter = row.get("action_counter")
            if action_counter is None:
                continue
            snap = row.get("trace_object_snapshot_v1") or {}
            agent_pos_raw = snap.get("agent_pos_xy") or {}
            agent_pos: tuple[int, int] | None = None
            try:
                ax = int(agent_pos_raw.get("x", -1))
                ay = int(agent_pos_raw.get("y", -1))
                if ax >= 0 and ay >= 0:
                    agent_pos = (ax, ay)
            except (TypeError, ValueError):
                agent_pos = None

            tracked_bbox_raw = (snap.get("tracked_agent_object") or {}).get("bbox")
            tracked_bbox: tuple[int, int, int, int] | None = None
            if (
                isinstance(tracked_bbox_raw, list)
                and len(tracked_bbox_raw) == 4
                and all(isinstance(v, (int, float)) for v in tracked_bbox_raw)
            ):
                x0, y0, x1, y1 = [int(v) for v in tracked_bbox_raw]
                if x1 >= x0 and y1 >= y0:
                    tracked_bbox = (x0, y0, x1, y1)

            steps[int(action_counter)] = TraceStepInfo(
                action_counter=int(action_counter),
                agent_pos=agent_pos,
                tracked_bbox=tracked_bbox,
            )
    return steps


def _temporal_mode_excluding_agent(
    frames: list[np.ndarray],
    trace_steps: dict[int, TraceStepInfo],
) -> tuple[np.ndarray, Counter]:
    h, w = frames[0].shape
    max_color = 16
    counts = np.zeros((max_color, h, w), dtype=np.uint16)
    agent_color_counter: Counter[int] = Counter()

    for t, frame in enumerate(frames):
        mask = np.ones((h, w), dtype=bool)
        step = trace_steps.get(t)
        if step and step.tracked_bbox:
            x0, y0, x1, y1 = step.tracked_bbox
            x0 = max(0, min(w - 1, x0))
            x1 = max(0, min(w - 1, x1))
            y0 = max(0, min(h - 1, y0))
            y1 = max(0, min(h - 1, y1))
            if x1 >= x0 and y1 >= y0:
                patch = frame[y0 : y1 + 1, x0 : x1 + 1]
                vals, freqs = np.unique(patch, return_counts=True)
                for v, c in zip(vals.tolist(), freqs.tolist()):
                    if 0 <= int(v) < max_color:
                        agent_color_counter[int(v)] += int(c)
                mask[y0 : y1 + 1, x0 : x1 + 1] = False

        ys, xs = np.where(mask)
        vals = frame[ys, xs]
        np.add.at(counts, (vals, ys, xs), 1)

    modal = np.argmax(counts, axis=0).astype(np.uint8)
    unseen = np.sum(counts, axis=0) == 0
    if np.any(unseen):
        modal[unseen] = frames[0][unseen]
    return modal, agent_color_counter


def _pick_floor_color(
    modal: np.ndarray,
    agent_path_xy: list[tuple[int, int]],
    tracked_bboxes: list[tuple[int, int, int, int]],
    agent_colors: Counter[int],
) -> int:
    votes: Counter[int] = Counter()
    h, w = modal.shape

    # Primary signal: colors on the 1-pixel ring around tracked agent bbox.
    for x0, y0, x1, y1 in tracked_bboxes:
        rx0, ry0 = max(0, x0 - 1), max(0, y0 - 1)
        rx1, ry1 = min(w - 1, x1 + 1), min(h - 1, y1 + 1)
        for yy in range(ry0, ry1 + 1):
            for xx in range(rx0, rx1 + 1):
                inside = (x0 <= xx <= x1) and (y0 <= yy <= y1)
                if inside:
                    continue
                votes[int(modal[yy, xx])] += 1

    # Fallback signal: local neighborhood around tracked center points.
    if not votes:
        for x, y in agent_path_xy:
            for yy in range(max(0, y - 1), min(h, y + 2)):
                for xx in range(max(0, x - 1), min(w, x + 2)):
                    votes[int(modal[yy, xx])] += 1

    if not votes:
        vals, freqs = np.unique(modal, return_counts=True)
        return int(vals[np.argmax(freqs)])

    blocked_colors = set(v for v, _ in agent_colors.most_common(4))
    blocked_colors.update({8, 9, 10, 11, 12, 0, 1, 2, 5})  # dynamic/ui/void-like colors
    for color, _ in votes.most_common():
        if color not in blocked_colors:
            return int(color)

    return int(votes.most_common(1)[0][0])


def _flood_fill(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    sx, sy = seed
    if not (0 <= sx < w and 0 <= sy < h):
        return np.zeros_like(mask, dtype=bool)

    if not mask[sy, sx]:
        found = None
        for radius in range(1, 10):
            for yy in range(max(0, sy - radius), min(h, sy + radius + 1)):
                for xx in range(max(0, sx - radius), min(w, sx + radius + 1)):
                    if mask[yy, xx]:
                        found = (xx, yy)
                        break
                if found is not None:
                    break
            if found is not None:
                sx, sy = found
                break
        else:
            return np.zeros_like(mask, dtype=bool)

    out = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = [(sx, sy)]
    out[sy, sx] = True
    while stack:
        x, y = stack.pop()
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                stack.append((nx, ny))
    return out


def reconstruct_map(recording_path: Path, trace_path: Path) -> ReconstructionResult:
    frames = _load_recording_frames(recording_path)
    trace_steps = _load_trace_steps(trace_path)
    modal, agent_colors = _temporal_mode_excluding_agent(frames, trace_steps)

    agent_path_xy: list[tuple[int, int]] = []
    tracked_bboxes: list[tuple[int, int, int, int]] = []
    for i in range(len(frames)):
        step = trace_steps.get(i)
        if step and step.agent_pos:
            agent_path_xy.append(step.agent_pos)
        if step and step.tracked_bbox:
            tracked_bboxes.append(step.tracked_bbox)

    floor_color = _pick_floor_color(modal, agent_path_xy, tracked_bboxes, agent_colors)
    walkable_raw = modal == int(floor_color)
    seed = agent_path_xy[0] if agent_path_xy else (0, 0)
    walkable = _flood_fill(walkable_raw, seed)

    return ReconstructionResult(
        modal_frame=modal,
        walkable_mask=walkable,
        agent_path_xy=agent_path_xy,
        floor_color=int(floor_color),
        action_count=len(frames),
    )


def _to_rgb(frame: np.ndarray) -> np.ndarray:
    rgb = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
    for v in range(16):
        rgb[frame == v] = PALETTE_RGB[v]
    return rgb


def _draw_outputs(
    result: ReconstructionResult,
    out_prefix: Path,
) -> tuple[Path, Path, Path]:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    base_rgb = _to_rgb(result.modal_frame)
    walk = result.walkable_mask

    map_only_path = out_prefix.with_name(out_prefix.name + "_map.png")
    walkable_path = out_prefix.with_name(out_prefix.name + "_walkable.png")
    overlay_path = out_prefix.with_name(out_prefix.name + "_overlay.png")

    # 1) map only
    fig1, ax1 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax1.imshow(base_rgb, interpolation="nearest")
    ax1.set_title("Reconstructed Map (pixel-level)")
    ax1.set_xlabel("col y")
    ax1.set_ylabel("row x")
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig1.savefig(map_only_path, dpi=180)
    plt.close(fig1)

    # 2) walkable mask
    fig2, ax2 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    img = np.zeros((*walk.shape, 3), dtype=np.uint8)
    img[:, :, :] = (32, 35, 40)
    img[walk] = (165, 173, 184)
    ax2.imshow(img, interpolation="nearest")
    ax2.set_title(f"Walkable Mask (floor_color={result.floor_color})")
    ax2.set_xlabel("col y")
    ax2.set_ylabel("row x")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig2.savefig(walkable_path, dpi=180)
    plt.close(fig2)

    # 3) overlay trajectory
    fig3, ax3 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    ax3.imshow(base_rgb, interpolation="nearest")
    ys, xs = np.where(walk)
    ax3.scatter(xs, ys, s=1.0, c="#71d7ff", alpha=0.08)
    if result.agent_path_xy:
        xs_path = np.array([p[0] for p in result.agent_path_xy], dtype=float)
        ys_path = np.array([p[1] for p in result.agent_path_xy], dtype=float)
        t = np.linspace(0.0, 1.0, len(xs_path))
        ax3.scatter(xs_path, ys_path, c=t, cmap="autumn", s=9, alpha=0.9, linewidths=0)
        ax3.scatter(xs_path[0], ys_path[0], c="lime", s=50, marker="o", label="start")
        ax3.scatter(xs_path[-1], ys_path[-1], c="red", s=55, marker="X", label="end")
        ax3.legend(loc="upper right", framealpha=0.9)
    ax3.set_title("Reconstructed Map + Agent Trajectory")
    ax3.set_xlabel("col y")
    ax3.set_ylabel("row x")
    ax3.set_xticks([])
    ax3.set_yticks([])
    fig3.savefig(overlay_path, dpi=180)
    plt.close(fig3)

    return map_only_path, walkable_path, overlay_path


def _default_trace_path(repo_root: Path) -> Path:
    trace_dir = repo_root / "recordings" / "active_inference_traces"
    traces = sorted(trace_dir.glob("*.trace.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        raise FileNotFoundError(f"No trace found under {trace_dir}")
    return traces[0]


def _default_recording_path(repo_root: Path) -> Path:
    rec_dir = repo_root / "recordings"
    recs = sorted(rec_dir.glob("*.recording.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not recs:
        raise FileNotFoundError(f"No recording found under {rec_dir}")
    return recs[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct game map from logs.")
    parser.add_argument("--recording", type=str, default="", help="Path to *.recording.jsonl")
    parser.add_argument("--trace", type=str, default="", help="Path to *.trace.jsonl")
    parser.add_argument("--out-prefix", type=str, default="", help="Output prefix path")
    args = parser.parse_args()

    repo_root = Path.cwd()
    recording_path = (
        Path(args.recording).expanduser().resolve() if args.recording else _default_recording_path(repo_root)
    )
    trace_path = Path(args.trace).expanduser().resolve() if args.trace else _default_trace_path(repo_root)

    if not recording_path.exists():
        raise FileNotFoundError(f"Recording not found: {recording_path}")
    if not trace_path.exists():
        raise FileNotFoundError(f"Trace not found: {trace_path}")

    if args.out_prefix:
        out_prefix = Path(args.out_prefix).expanduser().resolve()
    else:
        out_dir = repo_root / "recordings" / "coverage_viz"
        out_prefix = (out_dir / f"{recording_path.stem}_reconstructed").resolve()

    result = reconstruct_map(recording_path, trace_path)
    map_only_path, walkable_path, overlay_path = _draw_outputs(result, out_prefix)

    print(f"recording: {recording_path}")
    print(f"trace: {trace_path}")
    print(f"steps: {result.action_count}")
    print(f"floor_color: {result.floor_color}")
    print(f"path_points: {len(result.agent_path_xy)}")
    print(f"map: {map_only_path}")
    print(f"walkable: {walkable_path}")
    print(f"overlay: {overlay_path}")


if __name__ == "__main__":
    main()
