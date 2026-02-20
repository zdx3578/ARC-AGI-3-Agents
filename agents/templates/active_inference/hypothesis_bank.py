from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ActionCandidateV1,
    CausalEventSignatureV1,
    ObservationPacketV1,
    RepresentationStateV1,
)

OBS_CHANGE_TYPES: tuple[str, ...] = (
    "NO_CHANGE",
    "LOCAL_COLOR_CHANGE",
    "CC_TRANSLATION",
    "CC_COUNT_CHANGE",
    "GLOBAL_PATTERN_CHANGE",
    "METADATA_PROGRESS_CHANGE",
    "OBSERVED_UNCLASSIFIED",
)

EPS = 1.0e-9


@dataclass(slots=True)
class RuleFamilySpecV1:
    family_id: str
    parameter_ids: list[str]
    base_mdl_bits: float


@dataclass(slots=True)
class HypothesisInstanceV1:
    hypothesis_id: str
    mode_id: str
    family_id: str
    parameter_id: str
    mdl_bits: float


def _color_value(cell_any: object) -> int:
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
        return int(cell_any)  # type: ignore[arg-type]
    except Exception:
        return 0


def _safe_entropy(distribution: dict[str, float]) -> float:
    entropy = 0.0
    for probability in distribution.values():
        p = float(probability)
        if p <= 0.0:
            continue
        entropy -= p * math.log2(p)
    return float(entropy)


def _normalize_distribution(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in values.values())
    if total <= EPS:
        if not values:
            return {}
        uniform = 1.0 / float(len(values))
        return {key: uniform for key in values}
    return {key: max(0.0, float(v)) / total for (key, v) in values.items()}


def _base_signature_key(obs_change_type: str, progress_flag: bool) -> str:
    return f"type={obs_change_type}|progress={int(progress_flag)}"


def _signature_key(
    obs_change_type: str,
    progress_flag: bool,
    *,
    delta_bucket: str = "na",
    click_context_bucket: str = "na",
) -> str:
    return (
        f"type={obs_change_type}|progress={int(progress_flag)}"
        f"|delta={str(delta_bucket)}|click={str(click_context_bucket)}"
    )


def _signature_key_parts(signature_key: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(signature_key).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[str(key)] = str(value)
    return out


def _signature_progress_flag(signature_key: str) -> bool:
    parts = _signature_key_parts(signature_key)
    return str(parts.get("progress", "0")) == "1"


def _signature_base_from_key(signature_key: str) -> str:
    parts = _signature_key_parts(signature_key)
    return _base_signature_key(
        str(parts.get("type", "OBSERVED_UNCLASSIFIED")),
        str(parts.get("progress", "0")) == "1",
    )


def signature_key_from_event(signature: CausalEventSignatureV1) -> str:
    if str(signature.signature_key_v2):
        return str(signature.signature_key_v2)
    return _signature_key(
        str(signature.obs_change_type),
        bool(int(signature.level_delta) > 0),
        delta_bucket=str(signature.translation_delta_bucket or "na"),
        click_context_bucket=str(signature.click_context_bucket or "na"),
    )


def _state_transition(previous_state: str, current_state: str) -> str:
    return f"{previous_state}->{current_state}"


def _changed_pixels_and_bbox(
    previous_frame: list[list[int]], current_frame: list[list[int]]
) -> tuple[int, tuple[int, int, int, int] | None]:
    previous_height = len(previous_frame)
    previous_width = len(previous_frame[0]) if previous_frame else 0
    current_height = len(current_frame)
    current_width = len(current_frame[0]) if current_frame else 0
    union_height = max(previous_height, current_height)
    union_width = max(previous_width, current_width)

    changed = 0
    min_x: int | None = None
    min_y: int | None = None
    max_x: int | None = None
    max_y: int | None = None

    for y in range(union_height):
        for x in range(union_width):
            prev_value = (
                _color_value(previous_frame[y][x])
                if y < previous_height and x < previous_width
                else -1
            )
            curr_value = (
                _color_value(current_frame[y][x])
                if y < current_height and x < current_width
                else -1
            )
            if prev_value != curr_value:
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
    return (changed, (int(min_x), int(min_y), int(max_x), int(max_y)))


def _palette_delta_topk(
    previous_frame: list[list[int]],
    current_frame: list[list[int]],
    top_k: int = 5,
) -> dict[str, int]:
    prev_hist: dict[int, int] = {}
    curr_hist: dict[int, int] = {}

    for row in previous_frame:
        for value in row:
            color = _color_value(value)
            prev_hist[color] = prev_hist.get(color, 0) + 1

    for row in current_frame:
        for value in row:
            color = _color_value(value)
            curr_hist[color] = curr_hist.get(color, 0) + 1

    delta: list[tuple[int, int]] = []
    for color in set(prev_hist.keys()) | set(curr_hist.keys()):
        diff = int(curr_hist.get(color, 0) - prev_hist.get(color, 0))
        if diff != 0:
            delta.append((color, diff))

    delta.sort(key=lambda item: (abs(item[1]), item[0]), reverse=True)
    return {str(color): int(diff) for (color, diff) in delta[:top_k]}


def _delta_bucket_from_vector(dx: int, dy: int) -> str:
    dx_i = int(dx)
    dy_i = int(dy)
    if dx_i == 0 and dy_i == 0:
        return "stay"
    if abs(dx_i) >= abs(dy_i):
        if dx_i < 0:
            return "dir_l"
        if dx_i > 0:
            return "dir_r"
    if dy_i < 0:
        return "dir_u"
    if dy_i > 0:
        return "dir_d"
    return "dir_unknown"


def _click_context_bucket_from_feature(feature: dict[str, Any]) -> str:
    if not isinstance(feature, dict):
        return "na"
    if str(feature.get("click_context_bucket_v2", "")).strip():
        return str(feature.get("click_context_bucket_v2"))
    hit = int(feature.get("hit_object", -1))
    boundary = int(feature.get("on_boundary", -1))
    dist_bucket = str(feature.get("distance_to_nearest_object_bucket", "na"))
    coarse_x = int(feature.get("coarse_region_x", -1))
    coarse_y = int(feature.get("coarse_region_y", -1))
    return (
        f"hit={hit}|boundary={boundary}|dist={dist_bucket}"
        f"|region={coarse_x}:{coarse_y}"
    )


def _component_translation_summary(
    previous_representation: RepresentationStateV1,
    current_representation: RepresentationStateV1,
) -> dict[str, Any]:
    prev_nodes = previous_representation.component_views.get("same_color_8", [])
    curr_nodes = current_representation.component_views.get("same_color_8", [])
    if not prev_nodes or not curr_nodes:
        return {
            "translation_match_count": 0,
            "dominant_delta_bucket": "na",
            "delta_bucket_histogram": {},
        }

    matched = 0
    delta_histogram: dict[str, int] = {}
    used_curr: set[str] = set()
    for prev in prev_nodes:
        best = None
        best_score = 10**9
        for curr in curr_nodes:
            if curr.component_id in used_curr:
                continue
            if prev.color_signature != curr.color_signature:
                continue
            area_gap = abs(int(prev.area) - int(curr.area))
            if area_gap > max(2, int(prev.area * 0.2)):
                continue
            centroid_gap = abs(int(prev.centroid_x) - int(curr.centroid_x)) + abs(
                int(prev.centroid_y) - int(curr.centroid_y)
            )
            score = area_gap * 10 + centroid_gap
            if score < best_score:
                best_score = score
                best = curr
        if best is None:
            continue
        delta_x = int(best.centroid_x) - int(prev.centroid_x)
        delta_y = int(best.centroid_y) - int(prev.centroid_y)
        shift = abs(int(delta_x)) + abs(int(delta_y))
        if shift > 0:
            matched += 1
            delta_bucket = _delta_bucket_from_vector(delta_x, delta_y)
            delta_histogram[delta_bucket] = int(delta_histogram.get(delta_bucket, 0) + 1)
        used_curr.add(best.component_id)
    dominant_delta_bucket = "na"
    if delta_histogram:
        dominant_delta_bucket = sorted(
            delta_histogram.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[0][0]
    return {
        "translation_match_count": int(matched),
        "dominant_delta_bucket": str(dominant_delta_bucket),
        "delta_bucket_histogram": {
            str(key): int(value)
            for (key, value) in sorted(delta_histogram.items())
        },
    }


def _classify_observed_change_type(
    *,
    changed_pixel_count: int,
    changed_area_ratio: float,
    level_delta: int,
    state_transition: str,
    changed_object_count: int,
    object_count_delta: int,
    translation_match_count: int,
    palette_delta_total: int,
) -> str:
    if changed_pixel_count <= 0 and level_delta <= 0:
        return "NO_CHANGE"

    if level_delta > 0 or state_transition.endswith("->WIN"):
        return "METADATA_PROGRESS_CHANGE"

    if object_count_delta != 0:
        return "CC_COUNT_CHANGE"

    if translation_match_count > 0 and changed_area_ratio < 0.35:
        return "CC_TRANSLATION"

    if changed_area_ratio >= 0.35:
        return "GLOBAL_PATTERN_CHANGE"

    if changed_pixel_count > 0 and changed_area_ratio < 0.12 and palette_delta_total > 0:
        return "LOCAL_COLOR_CHANGE"

    return "OBSERVED_UNCLASSIFIED"


def build_causal_event_signature_v1(
    previous_packet: ObservationPacketV1,
    current_packet: ObservationPacketV1,
    previous_representation: RepresentationStateV1,
    current_representation: RepresentationStateV1,
    executed_action: ActionCandidateV1,
) -> CausalEventSignatureV1:
    changed_pixels, changed_bbox = _changed_pixels_and_bbox(
        previous_packet.frame, current_packet.frame
    )

    frame_area = max(1, int(previous_representation.frame_height) * int(previous_representation.frame_width))
    changed_area_ratio = float(changed_pixels) / float(frame_area)

    level_delta = int(current_packet.levels_completed - previous_packet.levels_completed)
    state_transition = _state_transition(previous_packet.state, current_packet.state)

    previous_digests = {obj.digest for obj in previous_representation.object_nodes}
    current_digests = {obj.digest for obj in current_representation.object_nodes}
    changed_object_count = len(previous_digests.symmetric_difference(current_digests))
    object_count_delta = int(
        len(current_representation.object_nodes) - len(previous_representation.object_nodes)
    )

    palette_delta = _palette_delta_topk(previous_packet.frame, current_packet.frame)
    palette_delta_total = sum(abs(int(v)) for v in palette_delta.values())

    translation_summary = _component_translation_summary(
        previous_representation,
        current_representation,
    )
    translation_match_count = int(translation_summary.get("translation_match_count", 0))
    dominant_delta_bucket = str(translation_summary.get("dominant_delta_bucket", "na"))

    click_context_feature = executed_action.metadata.get("coordinate_context_feature", {})
    click_context_bucket = "na"
    if int(executed_action.action_id) == 6:
        click_context_bucket = str(
            click_context_feature.get(
                "click_context_subcluster_v1",
                _click_context_bucket_from_feature(click_context_feature),
            )
        )

    obs_change_type = _classify_observed_change_type(
        changed_pixel_count=changed_pixels,
        changed_area_ratio=changed_area_ratio,
        level_delta=level_delta,
        state_transition=state_transition,
        changed_object_count=changed_object_count,
        object_count_delta=object_count_delta,
        translation_match_count=translation_match_count,
        palette_delta_total=palette_delta_total,
    )
    frame_chain_macro = (
        current_packet.frame_chain_macro_signature
        if isinstance(current_packet.frame_chain_macro_signature, dict)
        else {}
    )
    non_no_change_object_transition_count = int(
        frame_chain_macro.get("non_no_change_object_transition_count", 0)
    )
    micro_object_histogram = frame_chain_macro.get("micro_object_histogram", {})
    micro_translation_count = int(
        micro_object_histogram.get("CC_TRANSLATION", 0)
        if isinstance(micro_object_histogram, dict)
        else 0
    )
    is_navigation_action = int(executed_action.action_id) in (1, 2, 3, 4)
    is_navigation_blocked = bool(
        is_navigation_action
        and int(level_delta) <= 0
        and int(translation_match_count) <= 0
        and str(obs_change_type) in ("NO_CHANGE", "LOCAL_COLOR_CHANGE", "OBSERVED_UNCLASSIFIED")
    )
    bounce_like_blocked = bool(
        is_navigation_blocked
        and str(obs_change_type) == "NO_CHANGE"
        and (
            int(non_no_change_object_transition_count) > 0
            or int(micro_translation_count) > 0
        )
    )
    if is_navigation_blocked:
        dominant_delta_bucket = "blocked"

    tags: list[str] = [str(obs_change_type).lower()]
    if level_delta > 0:
        tags.append("progress")
    if executed_action.action_id == 6:
        tags.append("action6_intervention")
    if changed_object_count > 0:
        tags.append("object_change")
    if is_navigation_blocked:
        tags.append("navigation_blocked")
    if bounce_like_blocked:
        tags.append("navigation_bounce")

    cc_match_summary = {
        "previous_same8_count": int(
            len(previous_representation.component_views.get("same_color_8", []))
        ),
        "current_same8_count": int(
            len(current_representation.component_views.get("same_color_8", []))
        ),
        "translation_match_count": int(translation_match_count),
        "dominant_delta_bucket": str(dominant_delta_bucket),
        "delta_bucket_histogram": dict(
            translation_summary.get("delta_bucket_histogram", {})
        ),
        "object_count_delta": int(object_count_delta),
        "navigation_blocked": bool(is_navigation_blocked),
        "navigation_bounce": bool(bounce_like_blocked),
        "frame_chain_non_no_change_object_transition_count": int(
            non_no_change_object_transition_count
        ),
        "frame_chain_micro_translation_count": int(micro_translation_count),
    }

    signature_key_v2 = _signature_key(
        str(obs_change_type),
        bool(level_delta > 0),
        delta_bucket=(str(dominant_delta_bucket) if str(dominant_delta_bucket) != "na" else "na"),
        click_context_bucket=str(click_context_bucket),
    )

    digest_source = {
        "obs_change_type": obs_change_type,
        "changed_pixels": changed_pixels,
        "changed_bbox": changed_bbox,
        "changed_area_ratio": round(changed_area_ratio, 6),
        "level_delta": level_delta,
        "state_transition": state_transition,
        "changed_object_count": changed_object_count,
        "palette_delta": palette_delta,
        "cc_match_summary": cc_match_summary,
        "translation_delta_bucket": str(dominant_delta_bucket),
        "click_context_bucket": str(click_context_bucket),
        "signature_key_v2": str(signature_key_v2),
        "executed_action_id": int(executed_action.action_id),
        "tags": sorted(tags),
    }
    signature_digest = hashlib.sha256(
        repr(digest_source).encode("utf-8", errors="ignore")
    ).hexdigest()[:24]

    return CausalEventSignatureV1(
        schema_name="active_inference_causal_event_signature_v2",
        schema_version=2,
        obs_change_type=str(obs_change_type),
        changed_pixel_count=int(changed_pixels),
        changed_bbox=changed_bbox,
        changed_area_ratio=float(changed_area_ratio),
        level_delta=int(level_delta),
        state_transition=state_transition,
        changed_object_count=int(changed_object_count),
        event_tags=sorted(tags),
        palette_delta_topk=palette_delta,
        cc_match_summary=cc_match_summary,
        translation_delta_bucket=str(dominant_delta_bucket),
        click_context_bucket=str(click_context_bucket),
        signature_key_v2=str(signature_key_v2),
        executed_action_id=int(executed_action.action_id),
        signature_digest=signature_digest,
    )


def _rule_family_specs() -> list[RuleFamilySpecV1]:
    return [
        RuleFamilySpecV1("no_op_guard", ["strict_noop"], 10.0),
        RuleFamilySpecV1("navigation", ["axis_shift", "cursor_step"], 14.0),
        RuleFamilySpecV1("local_transform", ["require_hit_object", "allow_any_click"], 16.0),
        RuleFamilySpecV1("global_transform", ["wide_change"], 20.0),
        RuleFamilySpecV1("progress_trigger", ["on_hit", "on_confirm"], 18.0),
    ]


def _all_signature_keys() -> list[str]:
    out: list[str] = []
    for obs_type in OBS_CHANGE_TYPES:
        for progress in (0, 1):
            delta_buckets: list[str]
            if obs_type == "CC_TRANSLATION":
                delta_buckets = ["dir_l", "dir_r", "dir_u", "dir_d", "stay", "dir_unknown"]
            elif obs_type == "NO_CHANGE":
                delta_buckets = ["na", "blocked"]
            else:
                delta_buckets = ["na"]
            for delta_bucket in delta_buckets:
                out.append(
                    _signature_key(
                        obs_type,
                        bool(progress),
                        delta_bucket=delta_bucket,
                        click_context_bucket="na",
                    )
                )
    return out


def _context_features(
    packet: ObservationPacketV1,
    candidate: ActionCandidateV1,
    representation: RepresentationStateV1,
) -> dict[str, Any]:
    coord_context = candidate.metadata.get("coordinate_context_feature", {})
    control_schema_observed_posterior = candidate.metadata.get(
        "control_schema_observed_posterior",
        {},
    )
    if not isinstance(control_schema_observed_posterior, dict):
        control_schema_observed_posterior = {}
    click_bucket_observed_stats = candidate.metadata.get(
        "click_bucket_observed_stats",
        {},
    )
    if not isinstance(click_bucket_observed_stats, dict):
        click_bucket_observed_stats = {}
    click_subcluster_observed_stats = candidate.metadata.get(
        "click_subcluster_observed_stats",
        {},
    )
    if not isinstance(click_subcluster_observed_stats, dict):
        click_subcluster_observed_stats = {}
    blocked_edge_observed_stats = candidate.metadata.get(
        "blocked_edge_observed_stats",
        {},
    )
    if not isinstance(blocked_edge_observed_stats, dict):
        blocked_edge_observed_stats = {}
    transition_exploration_stats = candidate.metadata.get(
        "transition_exploration_stats",
        {},
    )
    if not isinstance(transition_exploration_stats, dict):
        transition_exploration_stats = {}
    click_context_bucket = _click_context_bucket_from_feature(coord_context)
    click_context_subcluster = str(
        coord_context.get(
            "click_context_subcluster_v1",
            f"{click_context_bucket}|fr=NA_NA|sub=lpNA",
        )
    )
    hit_object = int(coord_context.get("hit_object", -1))
    on_boundary = int(coord_context.get("on_boundary", -1))
    dist_bucket = str(coord_context.get("distance_to_nearest_object_bucket", "na"))

    object_count = int(representation.summary.get("object_count", 0))
    if object_count <= 0:
        object_count_bucket = "zero"
    elif object_count <= 3:
        object_count_bucket = "few"
    elif object_count <= 9:
        object_count_bucket = "mid"
    else:
        object_count_bucket = "many"

    mixed8_count = int(
        representation.summary.get("component_view_summary", {})
        .get("mixed_color_8", {})
        .get("count", 0)
    )

    observed_delta_bucket = "dir_unknown"
    observed_delta_confidence = 0.0
    if control_schema_observed_posterior:
        ranked = sorted(
            (
                (_delta_bucket_from_control_key(str(key)), float(value))
                for (key, value) in control_schema_observed_posterior.items()
            ),
            key=lambda item: (-float(item[1]), item[0]),
        )
        if ranked:
            observed_delta_bucket = str(ranked[0][0])
            observed_delta_confidence = float(max(0.0, min(1.0, ranked[0][1])))

    click_attempts = int(click_bucket_observed_stats.get("attempts", 0))
    click_non_no_change = int(click_bucket_observed_stats.get("non_no_change", 0))
    click_progress = int(click_bucket_observed_stats.get("progress", 0))
    click_subcluster_attempts = int(click_subcluster_observed_stats.get("attempts", 0))
    click_subcluster_non_no_change = int(
        click_subcluster_observed_stats.get("non_no_change", 0)
    )
    click_subcluster_progress = int(click_subcluster_observed_stats.get("progress", 0))
    navigation_action_attempts = int(blocked_edge_observed_stats.get("action_attempts", 0))
    navigation_action_blocked_rate = float(
        max(0.0, min(1.0, float(blocked_edge_observed_stats.get("action_blocked_rate", 0.0))))
    )
    navigation_edge_attempts = int(blocked_edge_observed_stats.get("edge_attempts", 0))
    navigation_edge_blocked_rate = float(
        max(0.0, min(1.0, float(blocked_edge_observed_stats.get("edge_blocked_rate", 0.0))))
    )
    region_revisit_count_current = int(
        blocked_edge_observed_stats.get("region_revisit_count_current", 0)
    )
    transition_state_visit_count = int(
        transition_exploration_stats.get("state_visit_count", 0)
    )
    transition_state_action_visit_count = int(
        transition_exploration_stats.get("state_action_visit_count", 0)
    )
    transition_state_outgoing_edge_count = int(
        transition_exploration_stats.get("state_outgoing_edge_count", 0)
    )

    return {
        "action_id": int(candidate.action_id),
        "is_action6": int(candidate.action_id == 6),
        "is_action5": int(candidate.action_id == 5),
        "is_action7": int(candidate.action_id == 7),
        "hit_object": int(hit_object),
        "on_boundary": int(on_boundary),
        "distance_bucket": dist_bucket,
        "click_context_bucket": str(click_context_bucket),
        "click_context_subcluster": str(click_context_subcluster),
        "observed_delta_bucket": str(observed_delta_bucket),
        "observed_delta_confidence": float(observed_delta_confidence),
        "object_count_bucket": object_count_bucket,
        "mixed8_count": int(mixed8_count),
        "levels_completed": int(packet.levels_completed),
        "progress_gap": int(max(0, packet.win_levels - packet.levels_completed)),
        "navigation_action_attempts": int(navigation_action_attempts),
        "navigation_action_blocked_rate": float(navigation_action_blocked_rate),
        "navigation_edge_attempts": int(navigation_edge_attempts),
        "navigation_edge_blocked_rate": float(navigation_edge_blocked_rate),
        "region_revisit_count_current": int(region_revisit_count_current),
        "transition_state_visit_count": int(transition_state_visit_count),
        "transition_state_action_visit_count": int(
            transition_state_action_visit_count
        ),
        "transition_state_outgoing_edge_count": int(
            transition_state_outgoing_edge_count
        ),
        "click_attempts": int(click_attempts),
        "click_non_no_change_rate": float(
            click_non_no_change / float(max(1, click_attempts))
        ),
        "click_progress_rate": float(click_progress / float(max(1, click_attempts))),
        "click_subcluster_attempts": int(click_subcluster_attempts),
        "click_subcluster_non_no_change_rate": float(
            click_subcluster_non_no_change / float(max(1, click_subcluster_attempts))
        ),
        "click_subcluster_progress_rate": float(
            click_subcluster_progress / float(max(1, click_subcluster_attempts))
        ),
    }


def _action_space_compatibility_reason(
    hypothesis: HypothesisInstanceV1,
    available_actions: set[int],
) -> str | None:
    family_id = str(hypothesis.family_id)
    mode_id = str(hypothesis.mode_id)

    if family_id == "navigation" and not any(v in available_actions for v in (1, 2, 3, 4, 7)):
        return "family_requires_navigation_action"
    if family_id == "local_transform" and 6 not in available_actions:
        return "family_requires_action6"
    if family_id == "global_transform" and not any(v in available_actions for v in (5, 6)):
        return "family_requires_action5_or_action6"
    if family_id == "progress_trigger" and not any(v in available_actions for v in (5, 6)):
        return "family_requires_progress_action"

    if mode_id == "interact" and not any(v in available_actions for v in (5, 6)):
        return "mode_requires_interact_action"
    if mode_id == "confirm" and not any(v in available_actions for v in (5, 6, 7)):
        return "mode_requires_confirm_action"
    if mode_id == "navigate" and not any(v in available_actions for v in (1, 2, 3, 4, 7)):
        return "mode_requires_navigation_action"
    return None


def _impossible_signature_keys(
    family_id: str,
    parameter_id: str,
    features: dict[str, Any],
) -> set[str]:
    impossible: set[str] = set()

    if family_id == "no_op_guard":
        impossible.add(_base_signature_key("METADATA_PROGRESS_CHANGE", True))

    if family_id == "local_transform" and parameter_id == "require_hit_object":
        if int(features.get("hit_object", -1)) == 0:
            impossible.add(_base_signature_key("LOCAL_COLOR_CHANGE", False))
            impossible.add(_base_signature_key("CC_COUNT_CHANGE", False))

    if family_id == "navigation":
        impossible.add(_base_signature_key("METADATA_PROGRESS_CHANGE", True))

    return impossible


def _predicted_delta_bucket(action_id: int, parameter_id: str) -> str:
    action = int(action_id)
    if action not in (1, 2, 3, 4, 7):
        return "dir_unknown"
    if action == 7:
        return "stay"
    if parameter_id == "axis_shift":
        mapping = {1: "dir_l", 2: "dir_r", 3: "dir_u", 4: "dir_d"}
    else:
        mapping = {1: "dir_u", 2: "dir_d", 3: "dir_l", 4: "dir_r"}
    return str(mapping.get(action, "dir_unknown"))


def _delta_bucket_from_control_key(delta_key: str) -> str:
    dx = 0
    dy = 0
    for part in str(delta_key).split("|"):
        if part.startswith("dx="):
            try:
                dx = int(part.split("=", 1)[1])
            except Exception:
                dx = 0
        elif part.startswith("dy="):
            try:
                dy = int(part.split("=", 1)[1])
            except Exception:
                dy = 0
    return _delta_bucket_from_vector(dx, dy)


def _signature_key_for_features(
    *,
    obs_change_type: str,
    progress_flag: bool,
    features: dict[str, Any],
    delta_bucket: str = "na",
) -> str:
    click_context_bucket = (
        str(features.get("click_context_subcluster", "na"))
        if int(features.get("is_action6", 0)) == 1
        else "na"
    )
    return _signature_key(
        obs_change_type,
        progress_flag,
        delta_bucket=delta_bucket,
        click_context_bucket=click_context_bucket,
    )


def _predict_distribution_for_hypothesis(
    *,
    family_id: str,
    parameter_id: str,
    mode_id: str,
    features: dict[str, Any],
) -> dict[str, float]:
    action_id = int(features.get("action_id", -1))
    is_action6 = bool(int(features.get("is_action6", 0)) == 1)
    is_action5 = bool(int(features.get("is_action5", 0)) == 1)
    is_action7 = bool(int(features.get("is_action7", 0)) == 1)
    hit_object = int(features.get("hit_object", -1))
    progress_gap = int(features.get("progress_gap", 0))
    navigation_action_blocked_rate = float(
        max(0.0, min(1.0, float(features.get("navigation_action_blocked_rate", 0.0))))
    )
    navigation_edge_blocked_rate = float(
        max(0.0, min(1.0, float(features.get("navigation_edge_blocked_rate", 0.0))))
    )
    region_revisit_count_current = int(features.get("region_revisit_count_current", 0))
    observed_delta_bucket = str(features.get("observed_delta_bucket", "dir_unknown"))
    observed_delta_confidence = float(
        max(0.0, min(1.0, float(features.get("observed_delta_confidence", 0.0))))
    )
    predicted_delta_bucket = _predicted_delta_bucket(action_id, parameter_id)
    if observed_delta_bucket != "dir_unknown":
        blend = 0.20 + (0.60 * observed_delta_confidence)
        if parameter_id == "axis_shift":
            predicted_delta_bucket = (
                observed_delta_bucket
                if blend >= 0.35
                else predicted_delta_bucket
            )

    distribution: dict[str, float] = {
        _signature_key_for_features(
            obs_change_type="OBSERVED_UNCLASSIFIED",
            progress_flag=False,
            features=features,
        ): 1.0
    }

    if family_id == "no_op_guard":
        distribution = {
            _signature_key_for_features(
                obs_change_type="NO_CHANGE",
                progress_flag=False,
                features=features,
            ): 0.86,
            _signature_key_for_features(
                obs_change_type="LOCAL_COLOR_CHANGE",
                progress_flag=False,
                features=features,
            ): 0.12,
            _signature_key_for_features(
                obs_change_type="OBSERVED_UNCLASSIFIED",
                progress_flag=False,
                features=features,
            ): 0.02,
        }

    elif family_id == "navigation":
        if action_id in (1, 2, 3, 4):
            blocked_rate = max(
                float(navigation_action_blocked_rate),
                float(navigation_edge_blocked_rate),
            )
            loop_pressure = float(
                max(0.0, min(1.0, float(region_revisit_count_current) / 4.0))
            )
            blocked_mass = min(0.72, 0.08 + (0.48 * blocked_rate) + (0.16 * loop_pressure))
            translation_mass = max(
                0.12,
                (0.42 + (0.34 * observed_delta_confidence)) * (1.0 - blocked_mass),
            )
            no_change_mass = max(
                0.04,
                (0.30 - (0.18 * observed_delta_confidence)) * (1.0 - blocked_mass),
            )
            local_mass = max(
                0.03,
                (0.12 - (0.05 * observed_delta_confidence)) * (1.0 - blocked_mass),
            )
            translation_unknown_mass = min(
                0.10,
                max(
                    0.0,
                    (
                        1.0
                        - (
                            blocked_mass
                            + translation_mass
                            + no_change_mass
                            + local_mass
                        )
                    )
                    * 0.45,
                ),
            )
            unknown_mass = max(
                0.04,
                1.0
                - (
                    blocked_mass
                    + translation_mass
                    + translation_unknown_mass
                    + no_change_mass
                    + local_mass
                ),
            )
            distribution = {
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                    delta_bucket="blocked",
                ): blocked_mass,
                _signature_key_for_features(
                    obs_change_type="CC_TRANSLATION",
                    progress_flag=False,
                    features=features,
                    delta_bucket=predicted_delta_bucket,
                ): translation_mass,
                _signature_key_for_features(
                    obs_change_type="CC_TRANSLATION",
                    progress_flag=False,
                    features=features,
                    delta_bucket="dir_unknown",
                ): translation_unknown_mass,
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                ): no_change_mass,
                _signature_key_for_features(
                    obs_change_type="LOCAL_COLOR_CHANGE",
                    progress_flag=False,
                    features=features,
                ): local_mass,
                _signature_key_for_features(
                    obs_change_type="OBSERVED_UNCLASSIFIED",
                    progress_flag=False,
                    features=features,
                ): unknown_mass,
            }
        elif is_action7:
            distribution = {
                _signature_key_for_features(
                    obs_change_type="CC_TRANSLATION",
                    progress_flag=False,
                    features=features,
                    delta_bucket="stay",
                ): 0.34,
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                    delta_bucket="blocked",
                ): 0.24,
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                ): 0.24,
                _signature_key_for_features(
                    obs_change_type="OBSERVED_UNCLASSIFIED",
                    progress_flag=False,
                    features=features,
                ): 0.18,
            }

    elif family_id == "local_transform":
        click_non_no_change_rate = float(
            max(0.0, min(1.0, float(features.get("click_non_no_change_rate", 0.0))))
        )
        click_progress_rate = float(
            max(0.0, min(1.0, float(features.get("click_progress_rate", 0.0))))
        )
        click_subcluster_attempts = int(features.get("click_subcluster_attempts", 0))
        click_subcluster_non_no_change_rate = float(
            max(
                0.0,
                min(
                    1.0,
                    float(features.get("click_subcluster_non_no_change_rate", 0.0)),
                ),
            )
        )
        click_subcluster_progress_rate = float(
            max(
                0.0,
                min(
                    1.0,
                    float(features.get("click_subcluster_progress_rate", 0.0)),
                ),
            )
        )
        subcluster_conf = float(
            click_subcluster_attempts / float(max(1, click_subcluster_attempts + 3))
        )
        effective_non_no_change_rate = (
            (subcluster_conf * click_subcluster_non_no_change_rate)
            + ((1.0 - subcluster_conf) * click_non_no_change_rate)
        )
        effective_progress_rate = (
            (subcluster_conf * click_subcluster_progress_rate)
            + ((1.0 - subcluster_conf) * click_progress_rate)
        )
        if is_action6 and hit_object == 1:
            local_mass = 0.40 + (0.32 * effective_non_no_change_rate)
            count_mass = 0.10 + (0.12 * effective_non_no_change_rate)
            progress_mass = 0.05 + (
                0.25 * effective_progress_rate if progress_gap > 0 else 0.0
            )
            remaining = max(0.0, 1.0 - (local_mass + count_mass + progress_mass))
            translation_mass = min(0.16, remaining * 0.44)
            no_change_mass = min(0.20, remaining * 0.32)
            uncertainty_bonus = (
                0.12
                if int(click_subcluster_attempts) <= 0
                else (0.06 if int(click_subcluster_attempts) <= 1 else 0.0)
            )
            unknown_mass = max(0.0, remaining - (translation_mass + no_change_mass))
            unknown_mass = min(0.35, unknown_mass + uncertainty_bonus)
            no_change_mass = max(0.04, no_change_mass - uncertainty_bonus)
            distribution = {
                _signature_key_for_features(
                    obs_change_type="LOCAL_COLOR_CHANGE",
                    progress_flag=False,
                    features=features,
                ): local_mass,
                _signature_key_for_features(
                    obs_change_type="CC_COUNT_CHANGE",
                    progress_flag=False,
                    features=features,
                ): count_mass,
                _signature_key_for_features(
                    obs_change_type="CC_TRANSLATION",
                    progress_flag=False,
                    features=features,
                    delta_bucket="dir_unknown",
                ): translation_mass,
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                ): no_change_mass,
                _signature_key_for_features(
                    obs_change_type="OBSERVED_UNCLASSIFIED",
                    progress_flag=False,
                    features=features,
                ): unknown_mass,
                _signature_key_for_features(
                    obs_change_type="METADATA_PROGRESS_CHANGE",
                    progress_flag=True,
                    features=features,
                ): progress_mass,
            }
        elif is_action6 and hit_object == 0:
            if parameter_id == "require_hit_object":
                distribution = {
                    _signature_key_for_features(
                        obs_change_type="NO_CHANGE",
                        progress_flag=False,
                        features=features,
                    ): 0.76,
                    _signature_key_for_features(
                        obs_change_type="OBSERVED_UNCLASSIFIED",
                        progress_flag=False,
                        features=features,
                    ): 0.24,
                }
            else:
                distribution = {
                    _signature_key_for_features(
                        obs_change_type="LOCAL_COLOR_CHANGE",
                        progress_flag=False,
                        features=features,
                    ): 0.26,
                    _signature_key_for_features(
                        obs_change_type="NO_CHANGE",
                        progress_flag=False,
                        features=features,
                    ): 0.56,
                    _signature_key_for_features(
                        obs_change_type="OBSERVED_UNCLASSIFIED",
                        progress_flag=False,
                        features=features,
                    ): 0.18,
                }

    elif family_id == "global_transform":
        if is_action5 or is_action6:
            distribution = {
                _signature_key_for_features(
                    obs_change_type="GLOBAL_PATTERN_CHANGE",
                    progress_flag=False,
                    features=features,
                ): 0.46,
                _signature_key_for_features(
                    obs_change_type="CC_COUNT_CHANGE",
                    progress_flag=False,
                    features=features,
                ): 0.22,
                _signature_key_for_features(
                    obs_change_type="LOCAL_COLOR_CHANGE",
                    progress_flag=False,
                    features=features,
                ): 0.10,
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                ): 0.10,
                _signature_key_for_features(
                    obs_change_type="OBSERVED_UNCLASSIFIED",
                    progress_flag=False,
                    features=features,
                ): 0.12,
            }

    elif family_id == "progress_trigger":
        click_non_no_change_rate = float(
            max(0.0, min(1.0, float(features.get("click_non_no_change_rate", 0.0))))
        )
        click_progress_rate = float(
            max(0.0, min(1.0, float(features.get("click_progress_rate", 0.0))))
        )
        click_subcluster_attempts = int(features.get("click_subcluster_attempts", 0))
        click_subcluster_non_no_change_rate = float(
            max(
                0.0,
                min(
                    1.0,
                    float(features.get("click_subcluster_non_no_change_rate", 0.0)),
                ),
            )
        )
        click_subcluster_progress_rate = float(
            max(
                0.0,
                min(
                    1.0,
                    float(features.get("click_subcluster_progress_rate", 0.0)),
                ),
            )
        )
        subcluster_conf = float(
            click_subcluster_attempts / float(max(1, click_subcluster_attempts + 3))
        )
        effective_non_no_change_rate = (
            (subcluster_conf * click_subcluster_non_no_change_rate)
            + ((1.0 - subcluster_conf) * click_non_no_change_rate)
        )
        effective_progress_rate = (
            (subcluster_conf * click_subcluster_progress_rate)
            + ((1.0 - subcluster_conf) * click_progress_rate)
        )
        if (is_action6 and hit_object == 1 and progress_gap > 0) or (
            is_action5 and parameter_id == "on_confirm" and progress_gap > 0
        ):
            progress_mass = 0.22 + (0.45 * effective_progress_rate)
            local_mass = 0.14 + (0.12 * effective_non_no_change_rate)
            global_mass = 0.10
            no_change_mass = max(0.04, 0.24 - (0.18 * effective_non_no_change_rate))
            unknown_mass = max(0.04, 1.0 - (progress_mass + local_mass + global_mass + no_change_mass))
            if int(click_subcluster_attempts) <= 0 and is_action6:
                unknown_mass = min(0.40, unknown_mass + 0.10)
                no_change_mass = max(0.04, no_change_mass - 0.10)
            distribution = {
                _signature_key_for_features(
                    obs_change_type="METADATA_PROGRESS_CHANGE",
                    progress_flag=True,
                    features=features,
                ): progress_mass,
                _signature_key_for_features(
                    obs_change_type="LOCAL_COLOR_CHANGE",
                    progress_flag=False,
                    features=features,
                ): local_mass,
                _signature_key_for_features(
                    obs_change_type="GLOBAL_PATTERN_CHANGE",
                    progress_flag=False,
                    features=features,
                ): global_mass,
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                ): no_change_mass,
                _signature_key_for_features(
                    obs_change_type="OBSERVED_UNCLASSIFIED",
                    progress_flag=False,
                    features=features,
                ): unknown_mass,
            }
        else:
            local_mass = 0.16 + (0.26 * effective_non_no_change_rate)
            no_change_mass = 0.58 - (0.30 * effective_non_no_change_rate)
            unknown_mass = max(0.06, 1.0 - (local_mass + no_change_mass))
            if int(click_subcluster_attempts) <= 0 and is_action6:
                unknown_mass = min(0.44, unknown_mass + 0.12)
                no_change_mass = max(0.04, no_change_mass - 0.12)
            distribution = {
                _signature_key_for_features(
                    obs_change_type="NO_CHANGE",
                    progress_flag=False,
                    features=features,
                ): no_change_mass,
                _signature_key_for_features(
                    obs_change_type="LOCAL_COLOR_CHANGE",
                    progress_flag=False,
                    features=features,
                ): local_mass,
                _signature_key_for_features(
                    obs_change_type="OBSERVED_UNCLASSIFIED",
                    progress_flag=False,
                    features=features,
                ): unknown_mass,
            }

    if mode_id == "confirm":
        progress_key = _signature_key_for_features(
            obs_change_type="METADATA_PROGRESS_CHANGE",
            progress_flag=True,
            features=features,
        )
        distribution[progress_key] = (
            distribution.get(progress_key, 0.0) + 0.06
        )

    impossible_base_keys = _impossible_signature_keys(family_id, parameter_id, features)
    for key in list(distribution.keys()):
        if _signature_base_from_key(key) in impossible_base_keys:
            distribution[key] = 0.0

    return _normalize_distribution(distribution)


def _transition_mode(mode_id: str, action_id: int, observed_signature_key: str) -> str:
    if _signature_progress_flag(observed_signature_key):
        return "confirm"
    if action_id in (1, 2, 3, 4):
        return "navigate"
    if action_id in (5, 6):
        return "interact"
    if action_id == 7:
        return "navigate"
    return mode_id


def _mode_transition_soft_confidence(action_id: int, observed_signature_key: str) -> float:
    if _signature_progress_flag(observed_signature_key):
        return 0.95
    if int(action_id) in (1, 2, 3, 4, 7):
        return 0.72
    if int(action_id) in (5, 6):
        return 0.78
    return 0.60


class ActiveInferenceHypothesisBankV1:
    def __init__(self) -> None:
        self.modes: list[str] = ["unknown", "navigate", "interact", "confirm"]
        self.family_specs: list[RuleFamilySpecV1] = _rule_family_specs()
        self.signature_space: list[str] = _all_signature_keys()

        self.hypotheses: list[HypothesisInstanceV1] = []
        self._hypothesis_index_by_triplet: dict[tuple[str, str, str], str] = {}

        for mode_id in self.modes:
            for family_spec in self.family_specs:
                for parameter_id in family_spec.parameter_ids:
                    hypothesis_id = (
                        f"h::{mode_id}::{family_spec.family_id}::{parameter_id}"
                    )
                    mdl_bits = float(
                        family_spec.base_mdl_bits
                        + (0.75 * len(parameter_id))
                        + (0.25 * len(mode_id))
                    )
                    hypothesis = HypothesisInstanceV1(
                        hypothesis_id=hypothesis_id,
                        mode_id=mode_id,
                        family_id=family_spec.family_id,
                        parameter_id=parameter_id,
                        mdl_bits=mdl_bits,
                    )
                    self.hypotheses.append(hypothesis)
                    self._hypothesis_index_by_triplet[
                        (mode_id, family_spec.family_id, parameter_id)
                    ] = hypothesis_id

        prior = 1.0 / float(max(1, len(self.hypotheses)))
        self.posterior_by_hypothesis_id: dict[str, float] = {
            hypothesis.hypothesis_id: prior for hypothesis in self.hypotheses
        }
        self.last_update: dict[str, float] = {}
        self.last_vfe_bits: float = 0.0
        self.last_observed_signature: str = ""
        self.last_posterior_delta_report: dict[str, Any] = {}
        self.last_action_space_signature: str = ""
        self.action_space_constraint_report: dict[str, Any] = {}

    def _renormalize(self, posterior: dict[str, float]) -> dict[str, float]:
        total = sum(max(0.0, weight) for weight in posterior.values())
        if total <= EPS:
            uniform = 1.0 / float(max(1, len(self.hypotheses)))
            return {hypothesis.hypothesis_id: uniform for hypothesis in self.hypotheses}
        return {
            hid: float(max(0.0, posterior.get(hid, 0.0)) / total)
            for hid in posterior
        }

    def posterior_entropy(self) -> float:
        return _safe_entropy(self.posterior_by_hypothesis_id)

    def current_vfe_bits(self) -> float:
        return float(self.last_vfe_bits)

    def partition_distribution(
        self,
        partition_key: str,
        *,
        posterior: dict[str, float] | None = None,
    ) -> dict[str, float]:
        posterior_used = posterior or self.posterior_by_hypothesis_id
        out: dict[str, float] = {}

        for hypothesis in self.hypotheses:
            weight = float(posterior_used.get(hypothesis.hypothesis_id, 0.0))
            if weight <= 0.0:
                continue
            if partition_key == "action_semantics":
                key = hypothesis.family_id
            elif partition_key == "mechanism_dynamics":
                key = f"{hypothesis.family_id}:{hypothesis.parameter_id}"
            elif partition_key == "causal_mapping":
                key = hypothesis.mode_id
            else:
                key = "unknown"
            out[key] = out.get(key, 0.0) + weight

        return _normalize_distribution(out)

    def partition_entropy(
        self,
        partition_key: str,
        *,
        posterior: dict[str, float] | None = None,
    ) -> float:
        return _safe_entropy(self.partition_distribution(partition_key, posterior=posterior))

    def summary(self) -> dict[str, object]:
        ranked = sorted(
            self.posterior_by_hypothesis_id.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return {
            "schema_name": "active_inference_hypothesis_summary_v2",
            "schema_version": 2,
            "posterior_entropy_bits": float(self.posterior_entropy()),
            "active_hypothesis_count": int(len(self.hypotheses)),
            "partition_entropy": {
                "action_semantics": float(self.partition_entropy("action_semantics")),
                "mechanism_dynamics": float(
                    self.partition_entropy("mechanism_dynamics")
                ),
                "causal_mapping": float(self.partition_entropy("causal_mapping")),
            },
            "posterior_by_hypothesis_id": {
                hypothesis_id: float(weight) for (hypothesis_id, weight) in ranked
            },
            "last_update": {
                hypothesis_id: float(weight)
                for (hypothesis_id, weight) in sorted(self.last_update.items())
            },
            "last_vfe_bits": float(self.last_vfe_bits),
            "last_observed_signature": self.last_observed_signature,
            "last_posterior_delta_report": dict(self.last_posterior_delta_report),
            "action_space_constraint_report": dict(self.action_space_constraint_report),
        }

    def apply_action_space_constraints(
        self,
        available_actions: list[int],
    ) -> dict[str, Any]:
        available_set = {int(v) for v in available_actions}
        signature = ",".join(str(v) for v in sorted(available_set))
        if (
            signature == self.last_action_space_signature
            and self.action_space_constraint_report
        ):
            return dict(self.action_space_constraint_report)

        compatible_ids: set[str] = set()
        elimination_by_reason: dict[str, int] = {}
        posterior_before = dict(self.posterior_by_hypothesis_id)
        for hypothesis in self.hypotheses:
            reason = _action_space_compatibility_reason(hypothesis, available_set)
            if reason is None:
                compatible_ids.add(hypothesis.hypothesis_id)
                continue
            elimination_by_reason[reason] = elimination_by_reason.get(reason, 0) + 1

        if compatible_ids:
            total = sum(
                float(posterior_before.get(hid, 0.0))
                for hid in compatible_ids
            )
            constrained: dict[str, float] = {
                hid: 0.0 for hid in posterior_before
            }
            if total <= EPS:
                uniform = 1.0 / float(max(1, len(compatible_ids)))
                for hid in compatible_ids:
                    constrained[hid] = uniform
            else:
                for hid in compatible_ids:
                    constrained[hid] = float(posterior_before.get(hid, 0.0) / total)
            self.posterior_by_hypothesis_id = constrained
        else:
            # Keep posterior unchanged if all hypotheses were deemed incompatible.
            self.posterior_by_hypothesis_id = dict(posterior_before)

        report = {
            "schema_name": "active_inference_action_space_constraint_report_v1",
            "schema_version": 1,
            "available_actions": [int(v) for v in sorted(available_set)],
            "active_hypothesis_count_before": int(
                sum(1 for v in posterior_before.values() if float(v) > 1.0e-4)
            ),
            "active_hypothesis_count_after": int(
                sum(1 for v in self.posterior_by_hypothesis_id.values() if float(v) > 1.0e-4)
            ),
            "compatible_hypothesis_count": int(len(compatible_ids)),
            "eliminated_count_by_action_space_reason": {
                str(key): int(value)
                for (key, value) in sorted(elimination_by_reason.items())
            },
            "mode_elimination_due_to_action_space_incompatibility": int(
                sum(
                    value
                    for (key, value) in elimination_by_reason.items()
                    if str(key).startswith("mode_")
                )
            ),
            "family_elimination_due_to_action_space_incompatibility": int(
                sum(
                    value
                    for (key, value) in elimination_by_reason.items()
                    if str(key).startswith("family_")
                )
            ),
            "applied": bool(bool(compatible_ids)),
        }
        self.last_action_space_signature = signature
        self.action_space_constraint_report = report
        return dict(report)

    def _predict_distribution_for_hypothesis(
        self,
        hypothesis: HypothesisInstanceV1,
        features: dict[str, Any],
    ) -> dict[str, float]:
        return _predict_distribution_for_hypothesis(
            family_id=hypothesis.family_id,
            parameter_id=hypothesis.parameter_id,
            mode_id=hypothesis.mode_id,
            features=features,
        )

    def predictive_statistics(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
        *,
        posterior: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        posterior_used = posterior or self.posterior_by_hypothesis_id
        features = _context_features(packet, candidate, representation)

        predictive_distribution: dict[str, float] = {}
        supports: dict[str, list[str]] = {}
        hypothesis_signature: dict[str, str] = {}
        expected_mdl_bits = 0.0
        per_hypothesis_distribution: dict[str, dict[str, float]] = {}

        for hypothesis in self.hypotheses:
            weight = float(posterior_used.get(hypothesis.hypothesis_id, 0.0))
            if weight <= 0.0:
                continue

            distribution_h = self._predict_distribution_for_hypothesis(hypothesis, features)
            per_hypothesis_distribution[hypothesis.hypothesis_id] = distribution_h

            best_key = max(distribution_h.items(), key=lambda item: item[1])[0]
            hypothesis_signature[hypothesis.hypothesis_id] = best_key
            expected_mdl_bits += weight * float(hypothesis.mdl_bits)

            for signature_key, probability in distribution_h.items():
                p = float(probability)
                if p <= 0.0:
                    continue
                predictive_distribution[signature_key] = (
                    predictive_distribution.get(signature_key, 0.0) + (weight * p)
                )
                supports.setdefault(signature_key, []).append(hypothesis.hypothesis_id)

        predictive_distribution = _normalize_distribution(predictive_distribution)

        return {
            "features": features,
            "predictive_distribution": predictive_distribution,
            "supports_by_signature": supports,
            "expected_mdl_bits": float(expected_mdl_bits),
            "hypothesis_signature": hypothesis_signature,
            "per_hypothesis_distribution": per_hypothesis_distribution,
        }

    def predictive_distribution(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
    ) -> tuple[dict[str, float], dict[str, list[str]], float, dict[str, str]]:
        stats = self.predictive_statistics(packet, candidate, representation)
        return (
            stats["predictive_distribution"],
            stats["supports_by_signature"],
            float(stats["expected_mdl_bits"]),
            stats["hypothesis_signature"],
        )

    def expected_ambiguity(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
        *,
        posterior: dict[str, float] | None = None,
    ) -> float:
        posterior_used = posterior or self.posterior_by_hypothesis_id
        stats = self.predictive_statistics(
            packet,
            candidate,
            representation,
            posterior=posterior_used,
        )
        per_h = stats["per_hypothesis_distribution"]
        ambiguity = 0.0
        for hypothesis_id, distribution in per_h.items():
            weight = float(posterior_used.get(hypothesis_id, 0.0))
            if weight <= 0.0:
                continue
            ambiguity += weight * _safe_entropy(distribution)
        return float(max(0.0, ambiguity))

    def posterior_after_signature(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
        observed_signature_key: str,
        *,
        posterior: dict[str, float] | None = None,
    ) -> dict[str, float]:
        posterior_after, _ = self.posterior_after_signature_with_report(
            packet,
            candidate,
            representation,
            observed_signature_key,
            posterior=posterior,
        )
        return posterior_after

    def posterior_after_signature_with_report(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
        observed_signature_key: str,
        *,
        posterior: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        posterior_used = posterior or self.posterior_by_hypothesis_id
        features = _context_features(packet, candidate, representation)
        mdl_lambda = 0.015
        active_threshold = 1.0e-4

        updated: dict[str, float] = {}
        primary_reason_by_hypothesis: dict[str, str] = {}
        likelihood_by_hypothesis: dict[str, float] = {}
        for hypothesis in self.hypotheses:
            prior = float(posterior_used.get(hypothesis.hypothesis_id, 0.0))
            if prior <= 0.0:
                updated[hypothesis.hypothesis_id] = 0.0
                primary_reason_by_hypothesis[hypothesis.hypothesis_id] = "inactive_prior"
                likelihood_by_hypothesis[hypothesis.hypothesis_id] = 0.0
                continue

            distribution_h = self._predict_distribution_for_hypothesis(hypothesis, features)
            likelihood = float(distribution_h.get(observed_signature_key, 0.0))
            impossible_keys = _impossible_signature_keys(
                hypothesis.family_id,
                hypothesis.parameter_id,
                features,
            )
            if _signature_base_from_key(observed_signature_key) in impossible_keys:
                reason = "impossible_constraint"
            elif likelihood <= EPS:
                reason = "zero_likelihood"
            else:
                reason = "supported"
            penalized_prior = prior * math.exp(-mdl_lambda * (float(hypothesis.mdl_bits) / 64.0))
            updated[hypothesis.hypothesis_id] = penalized_prior * max(EPS, likelihood)
            primary_reason_by_hypothesis[hypothesis.hypothesis_id] = reason
            likelihood_by_hypothesis[hypothesis.hypothesis_id] = likelihood

        updated = self._renormalize(updated)

        transitioned: dict[str, float] = {
            hypothesis.hypothesis_id: 0.0 for hypothesis in self.hypotheses
        }
        transitioned_out_hypothesis_ids: set[str] = set()
        transition_count = 0
        soft_confidence_weighted_sum = 0.0
        soft_confidence_weight_mass = 0.0
        for hypothesis in self.hypotheses:
            weight = float(updated.get(hypothesis.hypothesis_id, 0.0))
            if weight <= 0.0:
                continue
            next_mode = _transition_mode(
                hypothesis.mode_id,
                int(candidate.action_id),
                observed_signature_key,
            )
            target_id = self._hypothesis_index_by_triplet.get(
                (next_mode, hypothesis.family_id, hypothesis.parameter_id),
                hypothesis.hypothesis_id,
            )
            if target_id == hypothesis.hypothesis_id:
                transitioned[target_id] = transitioned.get(target_id, 0.0) + weight
                continue

            soft_confidence = _mode_transition_soft_confidence(
                int(candidate.action_id),
                observed_signature_key,
            )
            transition_mass = weight * float(max(0.0, min(1.0, soft_confidence)))
            stay_mass = weight - transition_mass
            transitioned[target_id] = transitioned.get(target_id, 0.0) + transition_mass
            transitioned[hypothesis.hypothesis_id] = (
                transitioned.get(hypothesis.hypothesis_id, 0.0) + stay_mass
            )
            transitioned_out_hypothesis_ids.add(hypothesis.hypothesis_id)
            transition_count += 1
            soft_confidence_weighted_sum += float(weight) * float(soft_confidence)
            soft_confidence_weight_mass += float(weight)

        posterior_after = self._renormalize(transitioned)

        eliminated_count_by_reason: dict[str, int] = {}
        eliminated_mass_by_reason: dict[str, float] = {}
        eliminated_examples: list[dict[str, Any]] = []
        survivor_family_histogram: dict[str, float] = {}
        survivor_mode_histogram: dict[str, float] = {}

        active_before = 0
        active_after = 0
        for hypothesis in self.hypotheses:
            hypothesis_id = hypothesis.hypothesis_id
            prior = float(posterior_used.get(hypothesis_id, 0.0))
            after = float(posterior_after.get(hypothesis_id, 0.0))
            if prior > active_threshold:
                active_before += 1
            if after > active_threshold:
                active_after += 1
                survivor_family_histogram[hypothesis.family_id] = (
                    survivor_family_histogram.get(hypothesis.family_id, 0.0) + after
                )
                survivor_mode_histogram[hypothesis.mode_id] = (
                    survivor_mode_histogram.get(hypothesis.mode_id, 0.0) + after
                )
            if prior > active_threshold and after <= active_threshold:
                reason = primary_reason_by_hypothesis.get(hypothesis_id, "posterior_collapse")
                if hypothesis_id in transitioned_out_hypothesis_ids:
                    reason = "mode_transition"
                eliminated_count_by_reason[reason] = eliminated_count_by_reason.get(reason, 0) + 1
                eliminated_mass_by_reason[reason] = (
                    eliminated_mass_by_reason.get(reason, 0.0) + prior
                )
                if len(eliminated_examples) < 12:
                    eliminated_examples.append(
                        {
                            "hypothesis_id": hypothesis_id,
                            "reason": reason,
                            "prior_weight": float(prior),
                            "posterior_weight": float(after),
                            "likelihood": float(likelihood_by_hypothesis.get(hypothesis_id, 0.0)),
                        }
                    )

        report = {
            "schema_name": "active_inference_posterior_delta_report_v1",
            "schema_version": 1,
            "observed_signature_key": observed_signature_key,
            "active_hypothesis_count_before": int(active_before),
            "active_hypothesis_count_after": int(active_after),
            "eliminated_count_by_reason": {
                str(key): int(value)
                for (key, value) in sorted(eliminated_count_by_reason.items())
            },
            "eliminated_mass_by_reason": {
                str(key): float(value)
                for (key, value) in sorted(eliminated_mass_by_reason.items())
            },
            "survivor_family_histogram": {
                str(key): float(value)
                for (key, value) in sorted(survivor_family_histogram.items())
            },
            "survivor_mode_histogram": {
                str(key): float(value)
                for (key, value) in sorted(survivor_mode_histogram.items())
            },
            "mode_transition_count": int(transition_count),
            "mode_transition_soft_confidence": float(
                soft_confidence_weighted_sum / max(EPS, soft_confidence_weight_mass)
            ),
            "example_eliminated_hypotheses": eliminated_examples,
            "mode_elimination_due_to_action_space_incompatibility": int(
                self.action_space_constraint_report.get(
                    "mode_elimination_due_to_action_space_incompatibility",
                    0,
                )
            ),
        }
        return posterior_after, report

    def split_information_gain(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
        *,
        posterior: dict[str, float] | None = None,
    ) -> dict[str, float]:
        posterior_used = posterior or self.posterior_by_hypothesis_id
        stats = self.predictive_statistics(
            packet,
            candidate,
            representation,
            posterior=posterior_used,
        )
        predictive_distribution = stats["predictive_distribution"]

        out: dict[str, float] = {}
        for partition_key in (
            "action_semantics",
            "mechanism_dynamics",
            "causal_mapping",
        ):
            prior_entropy = self.partition_entropy(
                partition_key,
                posterior=posterior_used,
            )
            expected_posterior_entropy = 0.0
            for signature_key, probability in predictive_distribution.items():
                p = float(probability)
                if p <= 0.0:
                    continue
                posterior_after = self.posterior_after_signature(
                    packet,
                    candidate,
                    representation,
                    signature_key,
                    posterior=posterior_used,
                )
                expected_posterior_entropy += p * self.partition_entropy(
                    partition_key,
                    posterior=posterior_after,
                )

            out[partition_key] = float(max(0.0, prior_entropy - expected_posterior_entropy))

        return out

    def with_posterior(
        self,
        posterior: dict[str, float],
    ) -> ActiveInferenceHypothesisBankV1:
        cloned = ActiveInferenceHypothesisBankV1()
        cloned.posterior_by_hypothesis_id = cloned._renormalize(dict(posterior))
        cloned.last_update = dict(self.last_update)
        cloned.last_vfe_bits = float(self.last_vfe_bits)
        cloned.last_observed_signature = str(self.last_observed_signature)
        cloned.last_posterior_delta_report = dict(self.last_posterior_delta_report)
        cloned.last_action_space_signature = str(self.last_action_space_signature)
        cloned.action_space_constraint_report = dict(self.action_space_constraint_report)
        return cloned

    def update_with_observation(
        self,
        previous_packet: ObservationPacketV1,
        current_packet: ObservationPacketV1,
        executed_candidate: ActionCandidateV1,
        previous_representation: RepresentationStateV1,
        observed_signature: CausalEventSignatureV1,
    ) -> None:
        observed_signature_key = signature_key_from_event(observed_signature)

        stats_before = self.predictive_statistics(
            previous_packet,
            executed_candidate,
            previous_representation,
            posterior=self.posterior_by_hypothesis_id,
        )
        predictive_distribution = stats_before["predictive_distribution"]
        p_obs = float(predictive_distribution.get(observed_signature_key, EPS))
        self.last_vfe_bits = float(-math.log2(max(EPS, p_obs)))
        self.last_observed_signature = observed_signature_key

        previous_posterior = dict(self.posterior_by_hypothesis_id)
        posterior_after, report = self.posterior_after_signature_with_report(
            previous_packet,
            executed_candidate,
            previous_representation,
            observed_signature_key,
            posterior=previous_posterior,
        )
        report["action_id"] = int(executed_candidate.action_id)
        report["candidate_id"] = str(executed_candidate.candidate_id)
        report["action_counter"] = int(current_packet.action_counter)
        report["state_transition"] = str(observed_signature.state_transition)
        report["level_delta"] = int(observed_signature.level_delta)
        report["signature_key_v2"] = str(observed_signature_key)
        report["translation_delta_bucket"] = str(observed_signature.translation_delta_bucket)
        report["click_context_bucket"] = str(observed_signature.click_context_bucket)
        self.posterior_by_hypothesis_id = posterior_after
        self.last_update = dict(self.posterior_by_hypothesis_id)
        self.last_posterior_delta_report = report
