from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import FreeEnergyLedgerEntryV1


@dataclass(slots=True)
class NavPrepassConfigV1:
    region_size: int = 8
    walkable_ratio_threshold: float = 0.02
    frontier_block_attempts_threshold: int = 3
    frontier_blocked_rate_threshold: float = 0.75


def _parse_region_key(region_key: str) -> tuple[int, int] | None:
    token = str(region_key)
    if ":" not in token:
        return None
    left, right = token.split(":", 1)
    try:
        rr = int(left)
        cc = int(right)
    except Exception:
        return None
    if rr < 0 or cc < 0:
        return None
    return (rr, cc)


def _neighbor_region_key_for_action(region_key: str, action_id: int) -> str:
    parsed = _parse_region_key(str(region_key))
    if parsed is None:
        return "NA"
    rr, cc = parsed
    aid = int(action_id)
    # action semantics in this project: 1=up, 2=down, 3=left, 4=right
    if aid == 1:
        rr -= 1
    elif aid == 2:
        rr += 1
    elif aid == 3:
        cc -= 1
    elif aid == 4:
        cc += 1
    if rr < 0 or cc < 0:
        return "NA"
    return f"{rr}:{cc}"


def _action_id_for_neighbor_step(current_region_key: str, next_region_key: str) -> int:
    cur = _parse_region_key(str(current_region_key))
    nxt = _parse_region_key(str(next_region_key))
    if cur is None or nxt is None:
        return 0
    rr0, cc0 = cur
    rr1, cc1 = nxt
    dr = int(rr1 - rr0)
    dc = int(cc1 - cc0)
    if dr == -1 and dc == 0:
        return 1
    if dr == 1 and dc == 0:
        return 2
    if dr == 0 and dc == -1:
        return 3
    if dr == 0 and dc == 1:
        return 4
    return 0


def _bfs_distance(adjacency: dict[str, dict[str, int]], *, start: str, goal: str) -> int | None:
    if str(start) == str(goal):
        return 0
    if str(start) not in adjacency:
        return None
    queue: list[str] = [str(start)]
    dist: dict[str, int] = {str(start): 0}
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        d = int(dist.get(node, 0))
        for nbr in adjacency.get(node, {}).keys():
            nbr = str(nbr)
            if nbr in dist:
                continue
            dist[nbr] = int(d + 1)
            if nbr == str(goal):
                return int(dist[nbr])
            queue.append(nbr)
    return None


def _bfs_next_step(adjacency: dict[str, dict[str, int]], *, start: str, goal: str) -> str | None:
    if str(start) == str(goal):
        return str(start)
    if str(start) not in adjacency:
        return None
    queue: list[str] = [str(start)]
    parent: dict[str, str] = {}
    visited: set[str] = {str(start)}
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for nbr in adjacency.get(node, {}).keys():
            nbr = str(nbr)
            if nbr in visited:
                continue
            visited.add(nbr)
            parent[nbr] = str(node)
            if nbr == str(goal):
                cur = str(goal)
                prev = parent.get(cur)
                while prev is not None and prev != str(start):
                    cur = prev
                    prev = parent.get(cur)
                return str(cur)
            queue.append(nbr)
    return None


def _extract_region_graph_snapshot(entries: list[FreeEnergyLedgerEntryV1]) -> dict[str, Any]:
    for entry in entries:
        try:
            meta = dict(entry.candidate.metadata)
            snapshot = meta.get("region_graph_snapshot_v1", {})
            if isinstance(snapshot, dict) and snapshot.get("schema_name") == "active_inference_region_graph_snapshot_v1":
                return snapshot
        except Exception:
            continue
    return {}


def _extract_navigation_map_snapshot(entries: list[FreeEnergyLedgerEntryV1]) -> dict[str, Any]:
    for entry in entries:
        try:
            meta = dict(entry.candidate.metadata)
            snap = meta.get("navigation_map_snapshot_v1", {})
            if isinstance(snap, dict) and snap.get("schema_name") == "active_inference_navigation_map_snapshot_v1":
                return snap
        except Exception:
            continue
    return {}


def select_prepass_recommended_action_v1(
    entries: list[FreeEnergyLedgerEntryV1],
    *,
    config: NavPrepassConfigV1,
) -> tuple[int, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "schema_name": "active_inference_nav_prepass_diagnostics_v1",
        "schema_version": 1,
        "enabled": False,
        "reason": "na",
        "recommended_action_id": 0,
        "mode": "na",
    }

    region_graph = _extract_region_graph_snapshot(entries)
    if not region_graph:
        diagnostics["reason"] = "missing_region_graph_snapshot"
        return 0, diagnostics

    current_region_key = str(region_graph.get("current_region_key", "NA"))
    if _parse_region_key(current_region_key) is None:
        diagnostics["reason"] = "invalid_current_region"
        return 0, diagnostics

    nav_map = _extract_navigation_map_snapshot(entries)
    walkable_ratio: dict[str, float] = {}
    adjacency_geom: dict[str, dict[str, int]] = {}
    if nav_map and bool(nav_map.get("enabled", False)):
        raw_ratio = nav_map.get("walkable_region_ratio_v1", {})
        if isinstance(raw_ratio, dict):
            walkable_ratio = {str(k): float(v) for (k, v) in raw_ratio.items()}
        raw_adj = nav_map.get("walkable_region_adjacency_v1", {})
        if isinstance(raw_adj, dict):
            adjacency_geom = {
                str(src): {str(dst): int(w) for (dst, w) in (nbrs or {}).items()}
                for (src, nbrs) in raw_adj.items()
                if isinstance(nbrs, dict)
            }

    adjacency_empirical: dict[str, dict[str, int]] = {}
    if not adjacency_geom:
        raw_edges = region_graph.get("edges", [])
        if isinstance(raw_edges, list):
            for row in raw_edges:
                if not isinstance(row, dict):
                    continue
                src = str(row.get("source_region_key", "NA"))
                dst = str(row.get("target_region_key", "NA"))
                if _parse_region_key(src) is None or _parse_region_key(dst) is None:
                    continue
                adjacency_empirical.setdefault(src, {})
                adjacency_empirical[src][dst] = int(
                    adjacency_empirical[src].get(dst, 0) + int(max(0, row.get("count", 0)))
                )
    adjacency = adjacency_geom or adjacency_empirical
    if not adjacency:
        diagnostics["reason"] = "missing_adjacency"
        return 0, diagnostics

    region_visits_raw = region_graph.get("region_visit_histogram", {})
    region_visits: dict[str, int] = {}
    if isinstance(region_visits_raw, dict):
        for k, v in region_visits_raw.items():
            if _parse_region_key(str(k)) is None:
                continue
            try:
                region_visits[str(k)] = int(v)
            except Exception:
                continue

    activity_edges = region_graph.get("activity_edges", [])
    activity_index: dict[tuple[str, int], dict[str, Any]] = {}
    if isinstance(activity_edges, list):
        for row in activity_edges:
            if not isinstance(row, dict):
                continue
            src = str(row.get("source_region_key", "NA"))
            aid = int(row.get("action_id", 0))
            if _parse_region_key(src) is None or aid not in (1, 2, 3, 4):
                continue
            activity_index[(src, aid)] = row

    known_regions: set[str] = set(region_visits.keys())
    known_regions.add(current_region_key)

    frontier_candidates: list[tuple[str, str, int]] = []
    for parent_key in sorted(known_regions):
        if _parse_region_key(parent_key) is None:
            continue
        for aid in (1, 2, 3, 4):
            nbr = _neighbor_region_key_for_action(parent_key, aid)
            if _parse_region_key(nbr) is None:
                continue
            if nbr in known_regions:
                continue

            if walkable_ratio:
                ratio = float(walkable_ratio.get(nbr, 0.0))
                if ratio < float(config.walkable_ratio_threshold):
                    continue

            edge_row = activity_index.get((parent_key, aid), {})
            attempts = int(edge_row.get("attempts", 0))
            moved = int(edge_row.get("moved_count", 0))
            blocked_rate = float(edge_row.get("blocked_rate", 0.0))
            if attempts >= int(config.frontier_block_attempts_threshold) and moved <= 0:
                continue
            if (
                attempts >= int(config.frontier_block_attempts_threshold)
                and blocked_rate >= float(config.frontier_blocked_rate_threshold)
            ):
                continue
            frontier_candidates.append((parent_key, nbr, int(aid)))

    diagnostics["enabled"] = True
    diagnostics["mode"] = "frontier"
    diagnostics["current_region_key"] = str(current_region_key)
    diagnostics["frontier_candidate_count"] = int(len(frontier_candidates))
    diagnostics["walkable_ratio_threshold"] = float(config.walkable_ratio_threshold)
    diagnostics["adjacency_source"] = "geometry" if adjacency_geom else "empirical"

    if not frontier_candidates:
        diagnostics["mode"] = "no_frontier"
        diagnostics["reason"] = "no_frontier_candidates"
        return 0, diagnostics

    best: tuple[int, int, str, str, int] | None = None
    for parent_key, nbr, aid in frontier_candidates:
        dist = _bfs_distance(adjacency, start=current_region_key, goal=parent_key)
        if dist is None:
            continue
        parent_vis = int(region_visits.get(parent_key, 0))
        score = (int(dist), int(parent_vis), str(parent_key), str(nbr), int(aid))
        if best is None or score < best:
            best = score

    if best is None:
        diagnostics["mode"] = "no_reachable_frontier_parent"
        diagnostics["reason"] = "all_frontier_parents_unreachable"
        return 0, diagnostics

    _, _, parent_key, nbr_key, enter_aid = best
    diagnostics["frontier_parent_region_key"] = str(parent_key)
    diagnostics["frontier_target_region_key"] = str(nbr_key)
    diagnostics["frontier_enter_action_id"] = int(enter_aid)

    if parent_key == current_region_key:
        diagnostics["mode"] = "enter_frontier"
        diagnostics["recommended_action_id"] = int(enter_aid)
        return int(enter_aid), diagnostics

    next_step = _bfs_next_step(adjacency, start=current_region_key, goal=parent_key)
    if next_step is None:
        diagnostics["mode"] = "route_failed"
        diagnostics["reason"] = "bfs_next_step_none"
        return 0, diagnostics

    aid = _action_id_for_neighbor_step(current_region_key, next_step)
    diagnostics["mode"] = "route_to_frontier_parent"
    diagnostics["next_region_key"] = str(next_step)
    diagnostics["recommended_action_id"] = int(aid)
    return int(aid), diagnostics
