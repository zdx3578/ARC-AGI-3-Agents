from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Callable

from .contracts import (
    ActionCandidateV1,
    CausalEventSignatureV1,
    ObservationPacketV1,
    RepresentationStateV1,
)

_MAGNITUDE_ORDER = {
    "none": 0,
    "small": 1,
    "medium": 2,
    "large": 3,
}


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


def _magnitude_from_changed_pixels(
    changed_pixel_count: int, frame_height: int, frame_width: int
) -> str:
    total = max(1, int(frame_height) * int(frame_width))
    ratio = float(changed_pixel_count) / float(total)
    if changed_pixel_count <= 0:
        return "none"
    if ratio < 0.01:
        return "small"
    if ratio < 0.08:
        return "medium"
    return "large"


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
    level_delta = int(current_packet.levels_completed - previous_packet.levels_completed)
    state_transition = _state_transition(previous_packet.state, current_packet.state)
    previous_digests = {obj.digest for obj in previous_representation.object_nodes}
    current_digests = {obj.digest for obj in current_representation.object_nodes}
    changed_object_count = len(previous_digests.symmetric_difference(current_digests))

    tags: list[str] = []
    if changed_pixels == 0:
        tags.append("no_change")
    else:
        tags.append("pixel_change")
    if level_delta > 0:
        tags.append("progress")
    if executed_action.action_id == 6:
        tags.append("action6_intervention")
    if changed_object_count > 0:
        tags.append("object_change")

    digest_source = {
        "changed_pixels": changed_pixels,
        "changed_bbox": changed_bbox,
        "level_delta": level_delta,
        "state_transition": state_transition,
        "changed_object_count": changed_object_count,
        "tags": sorted(tags),
    }
    signature_digest = hashlib.sha256(
        repr(digest_source).encode("utf-8", errors="ignore")
    ).hexdigest()[:24]

    return CausalEventSignatureV1(
        schema_name="active_inference_causal_event_signature_v1",
        schema_version=1,
        changed_pixel_count=int(changed_pixels),
        changed_bbox=changed_bbox,
        level_delta=int(level_delta),
        state_transition=state_transition,
        changed_object_count=int(changed_object_count),
        event_tags=sorted(tags),
        signature_digest=signature_digest,
    )


@dataclass(slots=True)
class TransitionHypothesisV1:
    hypothesis_id: str
    family: str
    mdl_bits: float
    predict_fn: Callable[
        [ObservationPacketV1, ActionCandidateV1, RepresentationStateV1], tuple[str, bool]
    ]

    def predict(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
    ) -> tuple[str, bool]:
        return self.predict_fn(packet, candidate, representation)


def _pred_sparse(
    packet: ObservationPacketV1,
    candidate: ActionCandidateV1,
    representation: RepresentationStateV1,
) -> tuple[str, bool]:
    if candidate.action_id == 0:
        return ("large", False)
    if candidate.action_id == 6:
        return ("small", False)
    return ("small", False)


def _pred_navigation(
    packet: ObservationPacketV1,
    candidate: ActionCandidateV1,
    representation: RepresentationStateV1,
) -> tuple[str, bool]:
    if candidate.action_id in (1, 2, 3, 4):
        return ("small", False)
    if candidate.action_id in (5, 7):
        return ("medium", False)
    if candidate.action_id == 6:
        return ("medium", False)
    return ("large", False)


def _pred_trigger(
    packet: ObservationPacketV1,
    candidate: ActionCandidateV1,
    representation: RepresentationStateV1,
) -> tuple[str, bool]:
    if candidate.action_id in (5, 6):
        return ("large", True)
    if candidate.action_id == 7:
        return ("medium", False)
    return ("small", False)


def _pred_action6_focused(
    packet: ObservationPacketV1,
    candidate: ActionCandidateV1,
    representation: RepresentationStateV1,
) -> tuple[str, bool]:
    if candidate.action_id == 6:
        return ("medium", False)
    return ("small", False)


class ActiveInferenceHypothesisBankV1:
    def __init__(self) -> None:
        self.hypotheses: list[TransitionHypothesisV1] = [
            TransitionHypothesisV1(
                hypothesis_id="h_sparse_v1",
                family="sparse_dynamics",
                mdl_bits=12.0,
                predict_fn=_pred_sparse,
            ),
            TransitionHypothesisV1(
                hypothesis_id="h_navigation_v1",
                family="navigation_dynamics",
                mdl_bits=20.0,
                predict_fn=_pred_navigation,
            ),
            TransitionHypothesisV1(
                hypothesis_id="h_trigger_v1",
                family="trigger_dynamics",
                mdl_bits=26.0,
                predict_fn=_pred_trigger,
            ),
            TransitionHypothesisV1(
                hypothesis_id="h_action6_focused_v1",
                family="action6_focus",
                mdl_bits=18.0,
                predict_fn=_pred_action6_focused,
            ),
        ]
        prior = 1.0 / float(max(1, len(self.hypotheses)))
        self.posterior_by_hypothesis_id: dict[str, float] = {
            h.hypothesis_id: prior for h in self.hypotheses
        }
        self.last_update: dict[str, float] = {}

    def _renormalize(self) -> None:
        total = sum(max(0.0, w) for w in self.posterior_by_hypothesis_id.values())
        if total <= 1.0e-12:
            uniform = 1.0 / float(max(1, len(self.hypotheses)))
            for h in self.hypotheses:
                self.posterior_by_hypothesis_id[h.hypothesis_id] = uniform
            return
        for hid in list(self.posterior_by_hypothesis_id.keys()):
            self.posterior_by_hypothesis_id[hid] = float(
                max(0.0, self.posterior_by_hypothesis_id[hid]) / total
            )

    def posterior_entropy(self) -> float:
        return _safe_entropy(
            {
                hypothesis_id: weight
                for (hypothesis_id, weight) in self.posterior_by_hypothesis_id.items()
            }
        )

    def summary(self) -> dict[str, object]:
        ranked = sorted(
            self.posterior_by_hypothesis_id.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return {
            "schema_name": "active_inference_hypothesis_summary_v1",
            "schema_version": 1,
            "posterior_entropy_bits": float(self.posterior_entropy()),
            "active_hypothesis_count": int(len(self.hypotheses)),
            "posterior_by_hypothesis_id": {
                hypothesis_id: float(weight) for (hypothesis_id, weight) in ranked
            },
            "last_update": {
                hypothesis_id: float(weight)
                for (hypothesis_id, weight) in sorted(self.last_update.items())
            },
        }

    def predictive_distribution(
        self,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        representation: RepresentationStateV1,
    ) -> tuple[dict[str, float], dict[str, list[str]], float, dict[str, str]]:
        distribution: dict[str, float] = {}
        supports: dict[str, list[str]] = {}
        hypothesis_signature: dict[str, str] = {}
        expected_mdl = 0.0

        for hypothesis in self.hypotheses:
            weight = float(self.posterior_by_hypothesis_id.get(hypothesis.hypothesis_id, 0.0))
            signature = hypothesis.predict(packet, candidate, representation)
            signature_key = f"magnitude={signature[0]}|progress={int(signature[1])}"
            distribution[signature_key] = distribution.get(signature_key, 0.0) + weight
            supports.setdefault(signature_key, []).append(hypothesis.hypothesis_id)
            hypothesis_signature[hypothesis.hypothesis_id] = signature_key
            expected_mdl += weight * float(hypothesis.mdl_bits)

        total = sum(distribution.values())
        if total > 1.0e-12:
            for key in list(distribution.keys()):
                distribution[key] = float(distribution[key] / total)
        return distribution, supports, float(expected_mdl), hypothesis_signature

    def update_with_observation(
        self,
        previous_packet: ObservationPacketV1,
        current_packet: ObservationPacketV1,
        executed_candidate: ActionCandidateV1,
        previous_representation: RepresentationStateV1,
        observed_signature: CausalEventSignatureV1,
    ) -> None:
        observed_magnitude = _magnitude_from_changed_pixels(
            observed_signature.changed_pixel_count,
            previous_representation.frame_height,
            previous_representation.frame_width,
        )
        observed_progress = bool(observed_signature.level_delta > 0)
        observed_order = _MAGNITUDE_ORDER[observed_magnitude]

        beta = 1.35
        mdl_lambda = 0.01

        updated: dict[str, float] = {}
        for hypothesis in self.hypotheses:
            predicted_magnitude, predicted_progress = hypothesis.predict(
                previous_packet, executed_candidate, previous_representation
            )
            predicted_order = _MAGNITUDE_ORDER[predicted_magnitude]
            mismatch_cost = 0.75 * abs(predicted_order - observed_order)
            if bool(predicted_progress) != observed_progress:
                mismatch_cost += 1.25
            old_weight = float(
                self.posterior_by_hypothesis_id.get(hypothesis.hypothesis_id, 0.0)
            )
            new_weight = old_weight * math.exp(-beta * mismatch_cost)
            new_weight *= math.exp(-mdl_lambda * (float(hypothesis.mdl_bits) / 64.0))
            updated[hypothesis.hypothesis_id] = float(new_weight)

        self.posterior_by_hypothesis_id = updated
        self._renormalize()
        self.last_update = dict(self.posterior_by_hypothesis_id)
