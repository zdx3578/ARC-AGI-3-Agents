from __future__ import annotations

import hashlib
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from arcengine import FrameData

from .contracts import (
    ActionCandidateV1,
    ComponentNodeV1,
    ObservationPacketV1,
    ObjectNodeV1,
    RepresentationStateV1,
)


@dataclass(slots=True)
class _ComponentWithCells:
    node: ComponentNodeV1
    cells: list[tuple[int, int]]


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


def _frame_digest(frame: list[list[int]]) -> str:
    return hashlib.sha256(
        repr(frame).encode("utf-8", errors="ignore")
    ).hexdigest()[:24]


def _changed_pixels_and_bbox(
    previous_frame: list[list[int]],
    current_frame: list[list[int]],
) -> tuple[int, tuple[int, int, int, int] | None]:
    prev_h = len(previous_frame)
    prev_w = len(previous_frame[0]) if previous_frame else 0
    curr_h = len(current_frame)
    curr_w = len(current_frame[0]) if current_frame else 0
    union_h = max(prev_h, curr_h)
    union_w = max(prev_w, curr_w)

    changed = 0
    min_x: int | None = None
    min_y: int | None = None
    max_x: int | None = None
    max_y: int | None = None

    for y in range(union_h):
        for x in range(union_w):
            prev_value = (
                int(previous_frame[y][x]) if y < prev_h and x < prev_w else -1
            )
            curr_value = int(current_frame[y][x]) if y < curr_h and x < curr_w else -1
            if prev_value == curr_value:
                continue
            changed += 1
            if min_x is None or x < min_x:
                min_x = x
            if min_y is None or y < min_y:
                min_y = y
            if max_x is None or x > max_x:
                max_x = x
            if max_y is None or y > max_y:
                max_y = y

    if changed <= 0:
        return (0, None)
    return (int(changed), (int(min_x), int(min_y), int(max_x), int(max_y)))


def _micro_signature_for_transition(
    previous_frame: list[list[int]],
    current_frame: list[list[int]],
    from_index: int,
    to_index: int,
) -> dict[str, Any]:
    changed_count, changed_bbox = _changed_pixels_and_bbox(previous_frame, current_frame)
    frame_height = len(previous_frame)
    frame_width = len(previous_frame[0]) if previous_frame else 0
    frame_area = max(1, int(frame_height) * int(frame_width))
    changed_area_ratio = float(changed_count) / float(frame_area)

    if changed_count <= 0:
        micro_type = "NO_CHANGE"
    elif changed_area_ratio >= 0.35:
        micro_type = "GLOBAL_PATTERN_CHANGE"
    elif changed_area_ratio < 0.12:
        micro_type = "LOCAL_COLOR_CHANGE"
    else:
        micro_type = "OBSERVED_UNCLASSIFIED"

    object_micro_type, object_witness = _micro_object_change_type(
        previous_frame,
        current_frame,
        changed_pixel_count=int(changed_count),
        changed_area_ratio=float(changed_area_ratio),
    )

    digest_source = {
        "from_index": int(from_index),
        "to_index": int(to_index),
        "micro_pixel_change_type": micro_type,
        "micro_object_change_type": object_micro_type,
        "changed_pixel_count": int(changed_count),
        "changed_area_ratio": round(changed_area_ratio, 6),
        "changed_bbox": changed_bbox,
    }
    signature_digest = hashlib.sha256(
        repr(digest_source).encode("utf-8", errors="ignore")
    ).hexdigest()[:24]

    out: dict[str, Any] = {
        "from_index": int(from_index),
        "to_index": int(to_index),
        "micro_change_type": micro_type,
        "micro_pixel_change_type": micro_type,
        "micro_object_change_type": object_micro_type,
        "changed_pixel_count": int(changed_count),
        "changed_area_ratio": float(changed_area_ratio),
        "object_witness": object_witness,
        "signature_digest": signature_digest,
    }
    if changed_bbox is not None:
        out["changed_bbox"] = {
            "min_x": int(changed_bbox[0]),
            "min_y": int(changed_bbox[1]),
            "max_x": int(changed_bbox[2]),
            "max_y": int(changed_bbox[3]),
        }
    return out


def _frame_chain_summary(
    latest_frame_data: FrameData,
    frame_chain: list[FrameData] | None = None,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    chain_frames: list[list[list[int]]] = []
    if frame_chain:
        for frame_data in frame_chain:
            normalized = _normalize_frame_to_int_grid(getattr(frame_data, "frame", None))
            if normalized:
                chain_frames.append(normalized)

    latest_normalized = _normalize_frame_to_int_grid(getattr(latest_frame_data, "frame", None))
    if latest_normalized:
        chain_frames.append(latest_normalized)

    dedup_frames: list[list[list[int]]] = []
    dedup_digests: list[str] = []
    for frame in chain_frames:
        digest = _frame_digest(frame)
        if dedup_digests and dedup_digests[-1] == digest:
            continue
        dedup_frames.append(frame)
        dedup_digests.append(digest)

    micro_signatures: list[dict[str, Any]] = []
    micro_pixel_hist: dict[str, int] = {}
    micro_object_hist: dict[str, int] = {}
    total_changed_pixels = 0
    for idx in range(len(dedup_frames) - 1):
        micro = _micro_signature_for_transition(
            dedup_frames[idx],
            dedup_frames[idx + 1],
            idx,
            idx + 1,
        )
        micro_signatures.append(micro)
        pixel_type = str(micro.get("micro_pixel_change_type", "OBSERVED_UNCLASSIFIED"))
        object_type = str(micro.get("micro_object_change_type", "OBSERVED_UNCLASSIFIED"))
        micro_pixel_hist[pixel_type] = micro_pixel_hist.get(pixel_type, 0) + 1
        micro_object_hist[object_type] = micro_object_hist.get(object_type, 0) + 1
        total_changed_pixels += int(micro.get("changed_pixel_count", 0))

    dominant_micro_pixel_type = (
        max(micro_pixel_hist.items(), key=lambda item: (item[1], item[0]))[0]
        if micro_pixel_hist
        else "NO_CHANGE"
    )
    dominant_micro_object_type = (
        max(micro_object_hist.items(), key=lambda item: (item[1], item[0]))[0]
        if micro_object_hist
        else "NO_CHANGE"
    )
    macro_signature = {
        "micro_signature_count": int(len(micro_signatures)),
        "total_changed_pixels": int(total_changed_pixels),
        "non_no_change_pixel_transition_count": int(
            sum(
                1
                for micro in micro_signatures
                if str(micro.get("micro_pixel_change_type", "")) != "NO_CHANGE"
            )
        ),
        "non_no_change_object_transition_count": int(
            sum(
                1
                for micro in micro_signatures
                if str(micro.get("micro_object_change_type", "")) != "NO_CHANGE"
            )
        ),
        "dominant_micro_change_type": dominant_micro_pixel_type,
        "dominant_micro_pixel_change_type": dominant_micro_pixel_type,
        "dominant_micro_object_change_type": dominant_micro_object_type,
        "micro_pixel_histogram": {
            str(key): int(value)
            for (key, value) in sorted(micro_pixel_hist.items())
        },
        "micro_object_histogram": {
            str(key): int(value)
            for (key, value) in sorted(micro_object_hist.items())
        },
    }
    return dedup_digests, micro_signatures, macro_signature


def build_observation_packet_v1(
    frame_data: FrameData,
    game_id: str,
    card_id: str,
    action_counter: int,
    frame_chain: list[FrameData] | None = None,
) -> ObservationPacketV1:
    available_actions: list[int] = []
    for value_any in frame_data.available_actions:
        try:
            available_actions.append(int(value_any))
        except Exception:
            continue

    frame = _normalize_frame_to_int_grid(frame_data.frame)
    frame_chain_digests, frame_chain_micro, frame_chain_macro = _frame_chain_summary(
        frame_data,
        frame_chain=frame_chain,
    )
    return ObservationPacketV1(
        schema_name="active_inference_observation_packet_v1",
        schema_version=2,
        game_id=game_id,
        card_id=card_id,
        action_counter=int(action_counter),
        state=frame_data.state.name,
        levels_completed=int(frame_data.levels_completed),
        win_levels=int(frame_data.win_levels),
        available_actions=sorted(set(available_actions)),
        frame=frame,
        num_frames_received=int(len(frame_chain_digests)),
        frame_chain_digests=frame_chain_digests,
        frame_chain_micro_signatures=frame_chain_micro,
        frame_chain_macro_signature=frame_chain_macro,
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


def _extract_components(
    frame: list[list[int]],
    *,
    background_color: int,
    connectivity: int,
    same_color: bool,
    view_name: str,
) -> list[_ComponentWithCells]:
    height, width = _frame_dimensions(frame)
    if height <= 0 or width <= 0:
        return []

    visited = [[False for _ in range(width)] for _ in range(height)]
    components: list[_ComponentWithCells] = []

    for y in range(height):
        for x in range(width):
            if visited[y][x]:
                continue
            color_seed = _color_value(frame[y][x])
            if color_seed == background_color:
                continue

            visited[y][x] = True
            queue: deque[tuple[int, int]] = deque([(y, x)])
            cells: list[tuple[int, int]] = []
            color_values: list[int] = []
            min_x = x
            min_y = y
            max_x = x
            max_y = y

            while queue:
                cy, cx = queue.popleft()
                color_here = _color_value(frame[cy][cx])
                cells.append((cy, cx))
                color_values.append(color_here)
                min_x = min(min_x, cx)
                min_y = min(min_y, cy)
                max_x = max(max_x, cx)
                max_y = max(max_y, cy)

                for ny, nx in _neighbors(cy, cx, height, width, connectivity):
                    if visited[ny][nx]:
                        continue
                    neighbor_color = _color_value(frame[ny][nx])
                    if neighbor_color == background_color:
                        continue
                    if same_color and neighbor_color != color_seed:
                        continue
                    visited[ny][nx] = True
                    queue.append((ny, nx))

            if not cells:
                continue

            area = len(cells)
            centroid_x = int(round(sum(cx for (_, cx) in cells) / float(area)))
            centroid_y = int(round(sum(cy for (cy, _) in cells) / float(area)))
            touches_boundary = (
                min_x == 0 or min_y == 0 or max_x == width - 1 or max_y == height - 1
            )
            color_signature = sorted(set(int(v) for v in color_values))
            digest_source = (
                view_name,
                connectivity,
                "same" if same_color else "mixed",
                tuple(color_signature),
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
            component_id = f"{view_name}_c{len(components)}"

            node = ComponentNodeV1(
                component_id=component_id,
                connectivity=int(connectivity),
                color_connectivity="same" if same_color else "mixed",
                area=int(area),
                centroid_x=int(centroid_x),
                centroid_y=int(centroid_y),
                bbox_min_x=int(min_x),
                bbox_min_y=int(min_y),
                bbox_max_x=int(max_x),
                bbox_max_y=int(max_y),
                touches_boundary=bool(touches_boundary),
                color_signature=color_signature,
                digest=digest,
            )
            components.append(_ComponentWithCells(node=node, cells=cells))

    components.sort(
        key=lambda comp: (
            comp.node.area,
            comp.node.bbox_min_y,
            comp.node.bbox_min_x,
            comp.node.component_id,
        ),
        reverse=True,
    )
    return components


def _translation_match_count_between_component_lists(
    previous_components: list[_ComponentWithCells],
    current_components: list[_ComponentWithCells],
) -> int:
    if not previous_components or not current_components:
        return 0
    matched = 0
    used_current_ids: set[str] = set()
    for previous in previous_components:
        best_current: _ComponentWithCells | None = None
        best_score = 10**9
        for current in current_components:
            if current.node.component_id in used_current_ids:
                continue
            if current.node.color_signature != previous.node.color_signature:
                continue
            area_gap = abs(int(current.node.area) - int(previous.node.area))
            if area_gap > max(2, int(previous.node.area * 0.2)):
                continue
            centroid_gap = abs(int(current.node.centroid_x) - int(previous.node.centroid_x)) + abs(
                int(current.node.centroid_y) - int(previous.node.centroid_y)
            )
            score = area_gap * 10 + centroid_gap
            if score < best_score:
                best_score = score
                best_current = current
        if best_current is None:
            continue
        shift = abs(int(best_current.node.centroid_x) - int(previous.node.centroid_x)) + abs(
            int(best_current.node.centroid_y) - int(previous.node.centroid_y)
        )
        if shift > 0:
            matched += 1
        used_current_ids.add(best_current.node.component_id)
    return int(matched)


def _micro_object_change_type(
    previous_frame: list[list[int]],
    current_frame: list[list[int]],
    *,
    changed_pixel_count: int,
    changed_area_ratio: float,
) -> tuple[str, dict[str, Any]]:
    if int(changed_pixel_count) <= 0:
        return (
            "NO_CHANGE",
            {
                "previous_same8_count": 0,
                "current_same8_count": 0,
                "object_count_delta": 0,
                "translation_match_count": 0,
            },
        )

    previous_background = _infer_background_color(previous_frame)
    current_background = _infer_background_color(current_frame)
    previous_components = _extract_components(
        previous_frame,
        background_color=previous_background,
        connectivity=8,
        same_color=True,
        view_name="micro_prev_same8",
    )
    current_components = _extract_components(
        current_frame,
        background_color=current_background,
        connectivity=8,
        same_color=True,
        view_name="micro_curr_same8",
    )
    object_count_delta = int(len(current_components) - len(previous_components))
    translation_match_count = _translation_match_count_between_component_lists(
        previous_components,
        current_components,
    )

    if object_count_delta != 0:
        object_type = "CC_COUNT_CHANGE"
    elif translation_match_count > 0 and float(changed_area_ratio) < 0.35:
        object_type = "CC_TRANSLATION"
    elif float(changed_area_ratio) >= 0.35:
        object_type = "GLOBAL_PATTERN_CHANGE"
    elif int(changed_pixel_count) > 0 and float(changed_area_ratio) < 0.12:
        object_type = "LOCAL_COLOR_CHANGE"
    else:
        object_type = "OBSERVED_UNCLASSIFIED"

    witness = {
        "previous_same8_count": int(len(previous_components)),
        "current_same8_count": int(len(current_components)),
        "object_count_delta": int(object_count_delta),
        "translation_match_count": int(translation_match_count),
    }
    return object_type, witness


def _build_owner_map(
    components: list[_ComponentWithCells],
) -> dict[tuple[int, int], str]:
    owner: dict[tuple[int, int], str] = {}
    for comp in components:
        for cell in comp.cells:
            owner[cell] = comp.node.component_id
    return owner


def _build_hierarchy_links(
    same4: list[_ComponentWithCells],
    same8: list[_ComponentWithCells],
    mixed8: list[_ComponentWithCells],
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    same8_owner = _build_owner_map(same8)
    mixed8_owner = _build_owner_map(mixed8)

    for comp in same4:
        if not comp.cells:
            continue
        anchor = comp.cells[0]
        parent_same8 = same8_owner.get(anchor, "")
        parent_mixed8 = mixed8_owner.get(anchor, "")
        links.append(
            {
                "child_view": "same_color_4",
                "child_component_id": comp.node.component_id,
                "parent_same_color_8_component_id": parent_same8,
                "parent_mixed_color_8_component_id": parent_mixed8,
            }
        )

    for comp in same8:
        if not comp.cells:
            continue
        anchor = comp.cells[0]
        parent_mixed8 = mixed8_owner.get(anchor, "")
        links.append(
            {
                "child_view": "same_color_8",
                "child_component_id": comp.node.component_id,
                "parent_mixed_color_8_component_id": parent_mixed8,
            }
        )

    return links


def _to_object_nodes_from_same8(
    same8_components: list[_ComponentWithCells],
) -> list[ObjectNodeV1]:
    objects: list[ObjectNodeV1] = []
    for index, comp in enumerate(same8_components):
        color = int(comp.node.color_signature[0]) if comp.node.color_signature else 0
        objects.append(
            ObjectNodeV1(
                object_id=f"obj_{index}",
                color=color,
                area=int(comp.node.area),
                centroid_x=int(comp.node.centroid_x),
                centroid_y=int(comp.node.centroid_y),
                bbox_min_x=int(comp.node.bbox_min_x),
                bbox_min_y=int(comp.node.bbox_min_y),
                bbox_max_x=int(comp.node.bbox_max_x),
                bbox_max_y=int(comp.node.bbox_max_y),
                touches_boundary=bool(comp.node.touches_boundary),
                digest=comp.node.digest,
            )
        )
    return objects


def _clamp_coord(value: int, lower: int = 0, upper: int = 63) -> int:
    return max(lower, min(upper, int(value)))


def _build_action6_coordinate_proposals(
    object_nodes: list[ObjectNodeV1],
    mixed8_components: list[_ComponentWithCells],
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

    for comp in mixed8_components:
        node = comp.node
        candidates.append((node.centroid_x, node.centroid_y))
        candidates.append((node.bbox_min_x, node.bbox_min_y))
        candidates.append((node.bbox_max_x, node.bbox_max_y))

    if frame_width > 0 and frame_height > 0:
        candidates.extend(
            [
                (frame_width // 2, frame_height // 2),
                (frame_width // 2, max(0, (frame_height // 2) - 1)),
                (max(0, (frame_width // 2) - 1), frame_height // 2),
            ]
        )
    else:
        candidates.append((31, 31))

    dedup = sorted(
        {(_clamp_coord(x), _clamp_coord(y)) for (x, y) in candidates},
        key=lambda p: (p[1], p[0]),
    )
    if max_points > 0:
        dedup = dedup[:max_points]
    return dedup


def _component_view_summary(components: list[_ComponentWithCells]) -> dict[str, Any]:
    areas = sorted(int(comp.node.area) for comp in components)
    return {
        "count": int(len(components)),
        "area_stats": {
            "min": int(areas[0]) if areas else 0,
            "max": int(areas[-1]) if areas else 0,
            "mean": float(sum(areas) / float(len(areas))) if areas else 0.0,
        },
    }


def build_representation_state_v1(
    packet: ObservationPacketV1, *, connectivity: int = 8, max_action6_points: int = 16
) -> RepresentationStateV1:
    frame = packet.frame
    frame_height, frame_width = _frame_dimensions(frame)
    background_color = _infer_background_color(frame)

    same4 = _extract_components(
        frame,
        background_color=background_color,
        connectivity=4,
        same_color=True,
        view_name="same_color_4",
    )
    same8 = _extract_components(
        frame,
        background_color=background_color,
        connectivity=8,
        same_color=True,
        view_name="same_color_8",
    )
    mixed4 = _extract_components(
        frame,
        background_color=background_color,
        connectivity=4,
        same_color=False,
        view_name="mixed_color_4",
    )
    mixed8 = _extract_components(
        frame,
        background_color=background_color,
        connectivity=8,
        same_color=False,
        view_name="mixed_color_8",
    )

    object_components = same4 if int(connectivity) == 4 else same8
    object_nodes = _to_object_nodes_from_same8(object_components)
    action6_points = _build_action6_coordinate_proposals(
        object_nodes,
        mixed8,
        frame_height,
        frame_width,
        max_action6_points,
    )

    hierarchy_links = _build_hierarchy_links(same4, same8, mixed8)

    color_hist = Counter(int(obj.color) for obj in object_nodes)
    object_areas = sorted(int(obj.area) for obj in object_nodes)
    max_proposals = max(1, (9 * len(object_nodes)) + (3 * len(mixed8)) + 3)
    proposal_coverage = float(len(action6_points) / float(max_proposals))
    proposal_diagnostics = _action6_proposal_diagnostics(
        packet,
        object_nodes,
        frame_height,
        frame_width,
        background_color,
        action6_points,
    )

    summary: dict[str, Any] = {
        "object_count": int(len(object_nodes)),
        "background_color": int(background_color),
        "color_histogram": {str(k): int(v) for (k, v) in sorted(color_hist.items())},
        "touch_boundary_object_count": int(
            sum(1 for obj in object_nodes if obj.touches_boundary)
        ),
        "action6_coordinate_proposal_count": int(len(action6_points)),
        "action6_coordinate_proposal_coverage": float(proposal_coverage),
        "action6_candidate_diagnostics": proposal_diagnostics,
        "object_area_stats": {
            "min": int(object_areas[0]) if object_areas else 0,
            "max": int(object_areas[-1]) if object_areas else 0,
            "mean": (
                float(sum(object_areas) / float(len(object_areas)))
                if object_areas
                else 0.0
            ),
        },
        "component_view_summary": {
            "same_color_4": _component_view_summary(same4),
            "same_color_8": _component_view_summary(same8),
            "mixed_color_4": _component_view_summary(mixed4),
            "mixed_color_8": _component_view_summary(mixed8),
        },
        "object_primary_view": "same_color_4" if int(connectivity) == 4 else "same_color_8",
        "hierarchy_link_count": int(len(hierarchy_links)),
    }

    return RepresentationStateV1(
        schema_name="active_inference_representation_state_v1",
        schema_version=2,
        frame_height=frame_height,
        frame_width=frame_width,
        background_color=background_color,
        object_nodes=object_nodes,
        action6_coordinate_proposals=action6_points,
        summary=summary,
        component_views={
            "same_color_4": [comp.node for comp in same4],
            "same_color_8": [comp.node for comp in same8],
            "mixed_color_4": [comp.node for comp in mixed4],
            "mixed_color_8": [comp.node for comp in mixed8],
        },
        hierarchy_links=hierarchy_links,
    )


def _distance_to_nearest_object(x: int, y: int, object_nodes: list[ObjectNodeV1]) -> float:
    if not object_nodes:
        return 9999.0
    return min(
        math.sqrt((float(x) - float(obj.centroid_x)) ** 2 + (float(y) - float(obj.centroid_y)) ** 2)
        for obj in object_nodes
    )


def _coordinate_context_feature_from_geometry(
    packet: ObservationPacketV1,
    object_nodes: list[ObjectNodeV1],
    frame_height: int,
    frame_width: int,
    background_color: int,
    x: int,
    y: int,
) -> dict[str, Any]:
    in_bounds = 0 <= y < int(frame_height) and 0 <= x < int(frame_width)
    color_at_point = int(packet.frame[y][x]) if in_bounds else int(background_color)
    hit_object = bool(in_bounds and color_at_point != int(background_color))
    on_boundary = bool(
        in_bounds
        and (x == 0 or y == 0 or x == int(frame_width) - 1 or y == int(frame_height) - 1)
    )
    dist = _distance_to_nearest_object(x, y, object_nodes)
    if dist < 3.0:
        dist_bucket = "near"
    elif dist < 8.0:
        dist_bucket = "mid"
    else:
        dist_bucket = "far"

    return {
        "hit_object": int(hit_object),
        "hit_color": int(color_at_point),
        "on_boundary": int(on_boundary),
        "distance_to_nearest_object_bucket": dist_bucket,
        "coarse_region_x": int(max(0, min(7, x // 8))),
        "coarse_region_y": int(max(0, min(7, y // 8))),
    }


def _coordinate_context_feature(
    packet: ObservationPacketV1,
    representation: RepresentationStateV1,
    x: int,
    y: int,
) -> dict[str, Any]:
    return _coordinate_context_feature_from_geometry(
        packet,
        representation.object_nodes,
        int(representation.frame_height),
        int(representation.frame_width),
        int(representation.background_color),
        int(x),
        int(y),
    )


def _action6_proposal_diagnostics(
    packet: ObservationPacketV1,
    object_nodes: list[ObjectNodeV1],
    frame_height: int,
    frame_width: int,
    background_color: int,
    proposals: list[tuple[int, int]],
) -> dict[str, Any]:
    proposal_count = int(len(proposals))
    if proposal_count <= 0:
        return {
            "proposal_count": 0,
            "unique_region_count": 0,
            "region_coverage_ratio": 0.0,
            "redundancy_rate": 0.0,
            "hit_object_rate": 0.0,
            "on_boundary_rate": 0.0,
            "unique_context_feature_rate": 0.0,
            "distance_bucket_histogram": {},
        }

    coarse_regions: set[tuple[int, int]] = set()
    context_signatures: set[tuple[object, ...]] = set()
    hit_object_count = 0
    on_boundary_count = 0
    distance_hist: dict[str, int] = {}

    for (x, y) in proposals:
        feature = _coordinate_context_feature_from_geometry(
            packet,
            object_nodes,
            frame_height,
            frame_width,
            background_color,
            int(x),
            int(y),
        )
        cx = int(feature.get("coarse_region_x", -1))
        cy = int(feature.get("coarse_region_y", -1))
        coarse_regions.add((cx, cy))
        context_signature = (
            int(feature.get("hit_object", 0)),
            int(feature.get("on_boundary", 0)),
            str(feature.get("distance_to_nearest_object_bucket", "na")),
            int(feature.get("hit_color", -1)),
            cx,
            cy,
        )
        context_signatures.add(context_signature)
        if int(feature.get("hit_object", 0)) == 1:
            hit_object_count += 1
        if int(feature.get("on_boundary", 0)) == 1:
            on_boundary_count += 1
        bucket = str(feature.get("distance_to_nearest_object_bucket", "na"))
        distance_hist[bucket] = distance_hist.get(bucket, 0) + 1

    unique_region_count = int(len(coarse_regions))
    redundant_region_points = max(0, proposal_count - unique_region_count)
    return {
        "proposal_count": int(proposal_count),
        "unique_region_count": unique_region_count,
        "region_coverage_ratio": float(unique_region_count / 64.0),
        "redundancy_rate": float(redundant_region_points / float(max(1, proposal_count))),
        "hit_object_rate": float(hit_object_count / float(max(1, proposal_count))),
        "on_boundary_rate": float(on_boundary_count / float(max(1, proposal_count))),
        "unique_context_feature_rate": float(
            len(context_signatures) / float(max(1, proposal_count))
        ),
        "distance_bucket_histogram": {
            str(key): int(value) for (key, value) in sorted(distance_hist.items())
        },
    }


def build_action_candidates_v1(
    packet: ObservationPacketV1,
    representation: RepresentationStateV1,
) -> list[ActionCandidateV1]:
    candidates: list[ActionCandidateV1] = []
    available = sorted(set(int(v) for v in packet.available_actions))

    for action_id in available:
        if action_id == 6:
            for idx, (x, y) in enumerate(representation.action6_coordinate_proposals):
                context_feature = _coordinate_context_feature(packet, representation, int(x), int(y))
                candidates.append(
                    ActionCandidateV1(
                        candidate_id=f"a6_p{idx}_{x}_{y}",
                        action_id=6,
                        x=int(x),
                        y=int(y),
                        source="representation/action6_proposals_v2",
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
                            "proposal_diagnostics": dict(
                                representation.summary.get(
                                    "action6_candidate_diagnostics", {}
                                )
                            ),
                            "coordinate_context_feature": context_feature,
                        },
                    )
                )
        else:
            candidates.append(
                ActionCandidateV1(
                    candidate_id=f"a{action_id}",
                    action_id=int(action_id),
                    source="available_actions",
                    metadata={
                        "coordinate_context_feature": {
                            "hit_object": -1,
                            "hit_color": -1,
                            "on_boundary": -1,
                            "distance_to_nearest_object_bucket": "na",
                            "coarse_region_x": -1,
                            "coarse_region_y": -1,
                        }
                    },
                )
            )

    return candidates
