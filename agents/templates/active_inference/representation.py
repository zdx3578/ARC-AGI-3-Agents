from __future__ import annotations

import hashlib
from collections import Counter, deque
from typing import Any

from arcengine import FrameData

from .contracts import (
    ActionCandidateV1,
    ObservationPacketV1,
    ObjectNodeV1,
    RepresentationStateV1,
)


def _color_value(cell_any: Any) -> int:
    if isinstance(cell_any, bool):
        return int(cell_any)
    if isinstance(cell_any, int):
        return int(cell_any)
    if isinstance(cell_any, float):
        return int(cell_any)
    if isinstance(cell_any, (list, tuple)):
        if not cell_any:
            return 0
        return _color_value(cell_any[0])
    try:
        return int(cell_any)
    except Exception:
        return 0


def _normalize_frame_to_int_grid(frame_any: Any) -> list[list[int]]:
    if not isinstance(frame_any, list):
        return []
    if not frame_any:
        return []

    # ARC payloads can occasionally add one redundant wrapping level.
    if (
        len(frame_any) == 1
        and isinstance(frame_any[0], list)
        and frame_any[0]
        and isinstance(frame_any[0][0], list)
    ):
        frame_any = frame_any[0]

    rows: list[list[int]] = []
    for row_any in frame_any:
        if isinstance(row_any, list):
            rows.append([_color_value(cell_any) for cell_any in row_any])
        else:
            rows.append([_color_value(row_any)])

    max_width = max((len(row) for row in rows), default=0)
    if max_width <= 0:
        return []

    padded: list[list[int]] = []
    for row in rows:
        if len(row) < max_width:
            row = row + [0] * (max_width - len(row))
        padded.append(row)
    return padded


def build_observation_packet_v1(
    frame_data: FrameData, game_id: str, card_id: str, action_counter: int
) -> ObservationPacketV1:
    available_actions: list[int] = []
    for value_any in frame_data.available_actions:
        try:
            available_actions.append(int(value_any))
        except Exception:
            continue

    frame = _normalize_frame_to_int_grid(frame_data.frame)
    return ObservationPacketV1(
        schema_name="active_inference_observation_packet_v1",
        schema_version=1,
        game_id=game_id,
        card_id=card_id,
        action_counter=int(action_counter),
        state=frame_data.state.name,
        levels_completed=int(frame_data.levels_completed),
        win_levels=int(frame_data.win_levels),
        available_actions=sorted(set(available_actions)),
        frame=frame,
    )


def _frame_dimensions(frame: list[list[int]]) -> tuple[int, int]:
    if not frame:
        return (0, 0)
    width = len(frame[0]) if frame[0] else 0
    return (len(frame), width)


def _infer_background_color(frame: list[list[int]]) -> int:
    values: list[int] = []
    for row in frame:
        for color in row:
            values.append(_color_value(color))
    if not values:
        return 0
    return int(Counter(values).most_common(1)[0][0])


def _neighbors(
    y: int, x: int, height: int, width: int, connectivity: int
) -> list[tuple[int, int]]:
    out = []
    base = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        base.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    for dy, dx in base:
        ny = y + dy
        nx = x + dx
        if 0 <= ny < height and 0 <= nx < width:
            out.append((ny, nx))
    return out


def _extract_objects(
    frame: list[list[int]], background_color: int, connectivity: int
) -> list[ObjectNodeV1]:
    height, width = _frame_dimensions(frame)
    if height <= 0 or width <= 0:
        return []

    visited = [[False for _ in range(width)] for _ in range(height)]
    objects: list[ObjectNodeV1] = []
    object_index = 0

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue
            color = _color_value(frame[y][x])
            visited[y][x] = True
            if color == background_color:
                continue

            queue: deque[tuple[int, int]] = deque([(y, x)])
            cells: list[tuple[int, int]] = []
            min_x = x
            min_y = y
            max_x = x
            max_y = y

            while queue:
                cy, cx = queue.popleft()
                cells.append((cy, cx))
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)

                for ny, nx in _neighbors(cy, cx, height, width, connectivity):
                    if visited[ny][nx]:
                        continue
                    visited[ny][nx] = True
                    if _color_value(frame[ny][nx]) == color:
                        queue.append((ny, nx))

            if not cells:
                continue

            area = len(cells)
            centroid_x = int(round(sum(cx for (_, cx) in cells) / float(area)))
            centroid_y = int(round(sum(cy for (cy, _) in cells) / float(area)))
            touches_boundary = (
                min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1
            )
            digest_source = (
                color,
                area,
                centroid_x,
                centroid_y,
                min_x,
                min_y,
                max_x,
                max_y,
            )
            digest = hashlib.sha256(
                repr(digest_source).encode("utf-8", errors="ignore")
            ).hexdigest()[:16]

            objects.append(
                ObjectNodeV1(
                    object_id=f"obj_{object_index}",
                    color=color,
                    area=area,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    bbox_min_x=min_x,
                    bbox_min_y=min_y,
                    bbox_max_x=max_x,
                    bbox_max_y=max_y,
                    touches_boundary=touches_boundary,
                    digest=digest,
                )
            )
            object_index += 1

    objects.sort(
        key=lambda obj: (
            obj.color,
            obj.area,
            obj.bbox_min_y,
            obj.bbox_min_x,
            obj.object_id,
        )
    )
    return objects


def _clamp_coord(value: int, lower: int = 0, upper: int = 63) -> int:
    return max(lower, min(upper, int(value)))


def _build_action6_coordinate_proposals(
    object_nodes: list[ObjectNodeV1],
    frame_height: int,
    frame_width: int,
    max_points: int,
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []

    for obj in object_nodes:
        points = [
            (obj.centroid_x, obj.centroid_y),
            ((obj.bbox_min_x + obj.bbox_max_x) // 2, (obj.bbox_min_y + obj.bbox_max_y) // 2),
            (obj.bbox_min_x, obj.bbox_min_y),
            (obj.bbox_max_x, obj.bbox_min_y),
            (obj.bbox_min_x, obj.bbox_max_y),
            (obj.bbox_max_x, obj.bbox_max_y),
        ]
        candidates.extend(points)

    if frame_width > 0 and frame_height > 0:
        candidates.append((frame_width // 2, frame_height // 2))
    else:
        candidates.append((31, 31))

    dedup = sorted(
        {
            (_clamp_coord(x), _clamp_coord(y))
            for (x, y) in candidates
        },
        key=lambda p: (p[1], p[0]),
    )
    if max_points > 0:
        dedup = dedup[:max_points]
    return dedup


def build_representation_state_v1(
    packet: ObservationPacketV1, *, connectivity: int = 8, max_action6_points: int = 16
) -> RepresentationStateV1:
    frame = packet.frame
    frame_height, frame_width = _frame_dimensions(frame)
    background_color = _infer_background_color(frame)
    object_nodes = _extract_objects(frame, background_color, connectivity)
    action6_points = _build_action6_coordinate_proposals(
        object_nodes,
        frame_height,
        frame_width,
        max_action6_points,
    )

    color_hist = Counter(int(obj.color) for obj in object_nodes)
    object_areas = sorted(int(obj.area) for obj in object_nodes)
    max_proposals = max(1, (6 * len(object_nodes)) + 1)
    proposal_coverage = float(len(action6_points) / float(max_proposals))
    summary: dict[str, Any] = {
        "object_count": int(len(object_nodes)),
        "background_color": int(background_color),
        "color_histogram": {str(k): int(v) for (k, v) in sorted(color_hist.items())},
        "touch_boundary_object_count": int(
            sum(1 for obj in object_nodes if obj.touches_boundary)
        ),
        "action6_coordinate_proposal_count": int(len(action6_points)),
        "action6_coordinate_proposal_coverage": float(proposal_coverage),
        "object_area_stats": {
            "min": int(object_areas[0]) if object_areas else 0,
            "max": int(object_areas[-1]) if object_areas else 0,
            "mean": (
                float(sum(object_areas) / float(len(object_areas)))
                if object_areas
                else 0.0
            ),
        },
    }
    return RepresentationStateV1(
        schema_name="active_inference_representation_state_v1",
        schema_version=1,
        frame_height=frame_height,
        frame_width=frame_width,
        background_color=background_color,
        object_nodes=object_nodes,
        action6_coordinate_proposals=action6_points,
        summary=summary,
    )


def build_action_candidates_v1(
    packet: ObservationPacketV1,
    representation: RepresentationStateV1,
) -> list[ActionCandidateV1]:
    candidates: list[ActionCandidateV1] = []
    available = sorted(set(int(v) for v in packet.available_actions))

    for action_id in available:
        if action_id == 6:
            for idx, (x, y) in enumerate(representation.action6_coordinate_proposals):
                candidates.append(
                    ActionCandidateV1(
                        candidate_id=f"a6_p{idx}_{x}_{y}",
                        action_id=6,
                        x=int(x),
                        y=int(y),
                        source="representation/action6_proposals_v1",
                        metadata={
                            "proposal_rank": idx,
                            "proposal_pool_size": len(
                                representation.action6_coordinate_proposals
                            ),
                            "proposal_coverage": float(
                                representation.summary.get(
                                    "action6_coordinate_proposal_coverage", 0.0
                                )
                            ),
                        },
                    )
                )
        else:
            candidates.append(
                ActionCandidateV1(
                    candidate_id=f"a{action_id}",
                    action_id=int(action_id),
                    source="available_actions",
                )
            )

    return candidates
