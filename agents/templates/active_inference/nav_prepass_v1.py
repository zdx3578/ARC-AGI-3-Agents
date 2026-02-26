from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import FreeEnergyLedgerEntryV1


@dataclass(slots=True)
class NavPrepassConfigV1:
    region_size: int = 8
    walkable_ratio_threshold: float = 0.02
    # Hard rule: boundary/wall must be confirmed by repeated blocked attempts.
    frontier_block_confirm_attempts: int = 2
    frontier_blocked_rate_threshold: float = 0.85
    # Do not permanently seal a blocked edge; periodically re-probe in case map opens.
    confirmed_block_reprobe_interval_steps: int = 24
    # Coverage hard rule: known reachable regions should be revisited at least twice.
    prepass_min_region_visits: int = 2


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


def _to_undirected(adjacency: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for src, nbrs in (adjacency or {}).items():
        if _parse_region_key(str(src)) is None or not isinstance(nbrs, dict):
            continue
        src_s = str(src)
        out.setdefault(src_s, {})
        for dst, w_raw in nbrs.items():
            if _parse_region_key(str(dst)) is None:
                continue
            w = int(max(0, w_raw))
            if w <= 0:
                continue
            dst_s = str(dst)
            out.setdefault(dst_s, {})
            out[src_s][dst_s] = max(int(out[src_s].get(dst_s, 0)), int(w))
            out[dst_s][src_s] = max(int(out[dst_s].get(src_s, 0)), int(w))
    return out


def _extract_activity_index(region_graph: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    activity_index: dict[tuple[str, int], dict[str, Any]] = {}
    activity_edges = region_graph.get("activity_edges", [])
    if isinstance(activity_edges, list):
        for row in activity_edges:
            if not isinstance(row, dict):
                continue
            src = str(row.get("source_region_key", "NA"))
            aid = int(row.get("action_id", 0))
            if _parse_region_key(src) is None or aid not in (1, 2, 3, 4):
                continue
            activity_index[(src, aid)] = row
    return activity_index


def _extract_region_action_counts(region_graph: dict[str, Any]) -> dict[tuple[str, int], dict[str, int]]:
    edge_attempt_counts = region_graph.get("edge_attempt_counts", {})
    blocked_edge_counts = region_graph.get("blocked_edge_counts", {})
    if not isinstance(edge_attempt_counts, dict):
        edge_attempt_counts = {}
    if not isinstance(blocked_edge_counts, dict):
        blocked_edge_counts = {}

    result: dict[tuple[str, int], dict[str, int]] = {}

    def _parse_edge_key(token: str) -> tuple[str, int] | None:
        text = str(token)
        parts = text.split("|")
        if len(parts) != 2:
            return None
        if not parts[0].startswith("region=") or not parts[1].startswith("action="):
            return None
        region_key = str(parts[0][len("region=") :])
        if _parse_region_key(region_key) is None:
            return None
        try:
            aid = int(parts[1][len("action=") :])
        except Exception:
            return None
        if aid not in (1, 2, 3, 4):
            return None
        return (region_key, aid)

    for key_raw, attempts_raw in edge_attempt_counts.items():
        parsed = _parse_edge_key(str(key_raw))
        if parsed is None:
            continue
        attempts = int(max(0, attempts_raw))
        row = result.setdefault(parsed, {"attempts": 0, "blocked": 0, "moved": 0})
        row["attempts"] = max(int(row.get("attempts", 0)), int(attempts))

    for key_raw, blocked_raw in blocked_edge_counts.items():
        parsed = _parse_edge_key(str(key_raw))
        if parsed is None:
            continue
        blocked = int(max(0, blocked_raw))
        row = result.setdefault(parsed, {"attempts": 0, "blocked": 0, "moved": 0})
        row["blocked"] = max(int(row.get("blocked", 0)), int(blocked))

    for row in result.values():
        attempts = int(max(0, row.get("attempts", 0)))
        blocked = int(max(0, row.get("blocked", 0)))
        row["moved"] = int(max(0, attempts - blocked))

    return result


def _edge_evidence(
    *,
    source_region_key: str,
    action_id: int,
    activity_index: dict[tuple[str, int], dict[str, Any]],
    region_action_counts: dict[tuple[str, int], dict[str, int]],
) -> dict[str, float]:
    key = (str(source_region_key), int(action_id))
    activity_row = activity_index.get(key, {})
    counts_row = region_action_counts.get(key, {})

    activity_attempts = int(max(0, activity_row.get("attempts", 0))) if isinstance(activity_row, dict) else 0
    activity_moved = int(max(0, activity_row.get("moved_count", 0))) if isinstance(activity_row, dict) else 0
    activity_blocked_rate = float(activity_row.get("blocked_rate", 0.0)) if isinstance(activity_row, dict) else 0.0

    count_attempts = int(max(0, counts_row.get("attempts", 0))) if isinstance(counts_row, dict) else 0
    count_blocked = int(max(0, counts_row.get("blocked", 0))) if isinstance(counts_row, dict) else 0
    count_moved = int(max(0, counts_row.get("moved", 0))) if isinstance(counts_row, dict) else 0
    count_blocked_rate = float(count_blocked / float(max(1, count_attempts)))

    attempts = int(max(activity_attempts, count_attempts))
    moved = int(max(activity_moved, count_moved))
    blocked_rate = float(max(activity_blocked_rate, count_blocked_rate))

    return {
        "attempts": int(attempts),
        "moved": int(moved),
        "blocked_rate": float(max(0.0, min(1.0, blocked_rate))),
    }


def _choose_low_attempt_navigation_action(
    current_region_key: str,
    *,
    activity_index: dict[tuple[str, int], dict[str, Any]],
    region_action_counts: dict[tuple[str, int], dict[str, int]],
    confirm_attempts: int,
    blocked_rate_threshold: float,
) -> int:
    candidates: list[tuple[int, int, int]] = []
    for aid in (1, 2, 3, 4):
        evidence = _edge_evidence(
            source_region_key=str(current_region_key),
            action_id=int(aid),
            activity_index=activity_index,
            region_action_counts=region_action_counts,
        )
        attempts = int(evidence.get("attempts", 0))
        moved = int(evidence.get("moved", 0))
        blocked_rate = float(evidence.get("blocked_rate", 0.0))
        confirmed_blocked = bool(
            attempts >= int(confirm_attempts)
            and moved <= 0
            and blocked_rate >= float(blocked_rate_threshold)
        )
        if confirmed_blocked:
            continue
        needs_confirmation = 0 if attempts < int(confirm_attempts) else 1
        candidates.append((int(needs_confirmation), int(attempts), int(aid)))
    if not candidates:
        return 0
    candidates.sort()
    return int(candidates[0][2])


def select_prepass_recommended_action_v1(
    entries: list[FreeEnergyLedgerEntryV1],
    *,
    config: NavPrepassConfigV1,
    action_counter: int | None = None,
) -> tuple[int, dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "schema_name": "active_inference_nav_prepass_diagnostics_v1",
        "schema_version": 2,
        "enabled": False,
        "reason": "na",
        "recommended_action_id": 0,
        "mode": "na",
        "boundary_confirm_attempts_required": int(max(1, config.frontier_block_confirm_attempts)),
        "confirmed_block_reprobe_interval_steps": int(
            max(0, config.confirmed_block_reprobe_interval_steps)
        ),
        "prepass_min_region_visits": int(max(2, config.prepass_min_region_visits)),
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
                str(src): {str(dst): int(max(0, w)) for (dst, w) in (nbrs or {}).items()}
                for (src, nbrs) in raw_adj.items()
                if isinstance(nbrs, dict)
            }

    adjacency_empirical: dict[str, dict[str, int]] = {}
    raw_edges = region_graph.get("edges", [])
    if isinstance(raw_edges, list):
        for row in raw_edges:
            if not isinstance(row, dict):
                continue
            src = str(row.get("source_region_key", "NA"))
            dst = str(row.get("target_region_key", "NA"))
            if _parse_region_key(src) is None or _parse_region_key(dst) is None:
                continue
            count = int(max(0, row.get("count", 0)))
            if count <= 0:
                continue
            adjacency_empirical.setdefault(src, {})
            adjacency_empirical[src][dst] = int(
                max(int(adjacency_empirical[src].get(dst, 0)), int(count))
            )

    adjacency = _to_undirected(adjacency_geom or adjacency_empirical)
    if current_region_key not in adjacency:
        adjacency.setdefault(current_region_key, {})

    region_visits_raw = region_graph.get("region_visit_histogram", {})
    region_visits: dict[str, int] = {}
    if isinstance(region_visits_raw, dict):
        for k, v in region_visits_raw.items():
            if _parse_region_key(str(k)) is None:
                continue
            try:
                region_visits[str(k)] = int(max(0, v))
            except Exception:
                continue

    known_regions: set[str] = set(region_visits.keys())
    known_regions.add(current_region_key)
    for src, nbrs in adjacency.items():
        if _parse_region_key(str(src)) is not None:
            known_regions.add(str(src))
        for dst in (nbrs or {}).keys():
            if _parse_region_key(str(dst)) is not None:
                known_regions.add(str(dst))

    activity_index = _extract_activity_index(region_graph)
    region_action_counts = _extract_region_action_counts(region_graph)

    confirm_attempts = int(max(1, config.frontier_block_confirm_attempts))
    blocked_rate_threshold = float(max(0.0, min(1.0, config.frontier_blocked_rate_threshold)))
    reprobe_interval_steps = int(max(0, config.confirmed_block_reprobe_interval_steps))
    current_step = int(action_counter) if action_counter is not None else -1

    frontier_candidates: list[dict[str, Any]] = []
    boundary_confirmed_count = 0
    boundary_pending_count = 0

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

            evidence = _edge_evidence(
                source_region_key=str(parent_key),
                action_id=int(aid),
                activity_index=activity_index,
                region_action_counts=region_action_counts,
            )
            attempts = int(evidence.get("attempts", 0))
            moved = int(evidence.get("moved", 0))
            blocked_rate = float(evidence.get("blocked_rate", 0.0))

            confirmed_blocked = bool(
                attempts >= int(confirm_attempts)
                and moved <= 0
                and blocked_rate >= float(blocked_rate_threshold)
            )
            if confirmed_blocked:
                boundary_confirmed_count += 1
                periodic_reprobe = bool(
                    int(reprobe_interval_steps) > 0
                    and int(current_step) >= 0
                    and int(current_step % int(reprobe_interval_steps)) == 0
                )
                if not periodic_reprobe:
                    continue

            if attempts < int(confirm_attempts):
                boundary_pending_count += 1

            frontier_candidates.append(
                {
                    "parent": str(parent_key),
                    "target": str(nbr),
                    "enter_action_id": int(aid),
                    "attempts": int(attempts),
                    "moved": int(moved),
                    "blocked_rate": float(blocked_rate),
                }
            )

    diagnostics["enabled"] = True
    diagnostics["mode"] = "frontier"
    diagnostics["current_region_key"] = str(current_region_key)
    diagnostics["frontier_candidate_count"] = int(len(frontier_candidates))
    diagnostics["boundary_confirmed_edge_count"] = int(boundary_confirmed_count)
    diagnostics["boundary_pending_edge_count"] = int(boundary_pending_count)
    diagnostics["walkable_ratio_threshold"] = float(config.walkable_ratio_threshold)
    diagnostics["adjacency_source"] = "geometry" if adjacency_geom else "empirical"

    best_frontier: tuple[Any, dict[str, Any]] | None = None
    for cand in frontier_candidates:
        parent_key = str(cand["parent"])
        dist = _bfs_distance(adjacency, start=current_region_key, goal=parent_key)
        if dist is None:
            continue
        parent_visits = int(region_visits.get(parent_key, 0))
        attempts = int(cand["attempts"])
        # Priority: edges that have not met the minimum confirmation budget.
        needs_confirmation = 0 if attempts < int(confirm_attempts) else 1
        score = (
            int(needs_confirmation),
            int(dist),
            int(parent_visits),
            int(attempts),
            str(parent_key),
            str(cand["target"]),
            int(cand["enter_action_id"]),
        )
        if best_frontier is None or score < best_frontier[0]:
            best_frontier = (score, cand)

    if best_frontier is not None:
        cand = dict(best_frontier[1])
        parent_key = str(cand["parent"])
        target_key = str(cand["target"])
        enter_aid = int(cand["enter_action_id"])
        diagnostics["frontier_parent_region_key"] = str(parent_key)
        diagnostics["frontier_target_region_key"] = str(target_key)
        diagnostics["frontier_enter_action_id"] = int(enter_aid)
        diagnostics["frontier_parent_distance"] = int(best_frontier[0][1])
        diagnostics["frontier_parent_attempts"] = int(cand.get("attempts", 0))

        if parent_key == current_region_key:
            diagnostics["mode"] = "enter_frontier"
            diagnostics["recommended_action_id"] = int(enter_aid)
            return int(enter_aid), diagnostics

        next_step = _bfs_next_step(adjacency, start=current_region_key, goal=parent_key)
        if next_step is not None:
            aid = _action_id_for_neighbor_step(current_region_key, next_step)
            if aid in (1, 2, 3, 4):
                diagnostics["mode"] = "route_to_frontier_parent"
                diagnostics["next_region_key"] = str(next_step)
                diagnostics["recommended_action_id"] = int(aid)
                return int(aid), diagnostics

    # Fallback: complete coverage of known reachable map (>=2 visits each by default).
    visit_target = int(max(2, config.prepass_min_region_visits))
    under_visited: list[tuple[int, int, str]] = []
    for region_key in sorted(known_regions):
        if _parse_region_key(region_key) is None:
            continue
        dist = _bfs_distance(adjacency, start=current_region_key, goal=region_key)
        if dist is None:
            continue
        visits = int(region_visits.get(region_key, 0))
        if visits >= int(visit_target):
            continue
        under_visited.append((int(dist), int(visits), str(region_key)))

    if under_visited:
        under_visited.sort()
        _, _, goal_key = under_visited[0]
        diagnostics["mode"] = "route_under_visited"
        diagnostics["goal_region_key"] = str(goal_key)
        diagnostics["regions_under_visit_target"] = int(len(under_visited))
        if str(goal_key) == str(current_region_key):
            local_probe_action = _choose_low_attempt_navigation_action(
                str(current_region_key),
                activity_index=activity_index,
                region_action_counts=region_action_counts,
                confirm_attempts=int(confirm_attempts),
                blocked_rate_threshold=float(blocked_rate_threshold),
            )
            if local_probe_action in (1, 2, 3, 4):
                diagnostics["mode"] = "probe_local_boundary"
                diagnostics["recommended_action_id"] = int(local_probe_action)
                return int(local_probe_action), diagnostics
        next_step = _bfs_next_step(adjacency, start=current_region_key, goal=goal_key)
        if next_step is not None:
            aid = _action_id_for_neighbor_step(current_region_key, next_step)
            if aid in (1, 2, 3, 4):
                diagnostics["next_region_key"] = str(next_step)
                diagnostics["recommended_action_id"] = int(aid)
                return int(aid), diagnostics

    diagnostics["mode"] = "no_prepass_action"
    diagnostics["reason"] = "no_reachable_frontier_or_under_visited"
    return 0, diagnostics
