from __future__ import annotations

import math
from dataclasses import dataclass

from .contracts import ActionCandidateV1, ObservationPacketV1, RepresentationStateV1


@dataclass(slots=True, frozen=True)
class EFEWeightsV1:
    risk: float
    ambiguity: float
    information_gain: float
    action_cost: float
    complexity: float


def entropy_bits(distribution: dict[str, float]) -> float:
    out = 0.0
    for probability in distribution.values():
        p = float(probability)
        if p <= 0.0:
            continue
        out -= p * math.log2(p)
    return float(out)


def determine_phase_v1(
    *,
    action_counter: int,
    posterior_entropy_bits: float,
    explore_steps: int,
    exploit_entropy_threshold: float,
) -> str:
    if action_counter < explore_steps:
        return "explore"
    if posterior_entropy_bits > exploit_entropy_threshold:
        return "explain"
    return "exploit"


def weights_for_phase_v1(phase: str) -> EFEWeightsV1:
    if phase == "explore":
        return EFEWeightsV1(
            risk=0.65,
            ambiguity=0.45,
            information_gain=1.25,
            action_cost=0.15,
            complexity=0.15,
        )
    if phase == "explain":
        return EFEWeightsV1(
            risk=0.85,
            ambiguity=0.40,
            information_gain=0.95,
            action_cost=0.25,
            complexity=0.20,
        )
    return EFEWeightsV1(
        risk=1.15,
        ambiguity=0.25,
        information_gain=0.45,
        action_cost=0.80,
        complexity=0.30,
    )


def compute_risk_v1(
    packet: ObservationPacketV1, candidate: ActionCandidateV1, representation: RepresentationStateV1
) -> tuple[float, dict[str, float]]:
    progress_gap = max(0.0, float(packet.win_levels - packet.levels_completed))
    normalizer = float(max(1, packet.win_levels))
    normalized_gap = progress_gap / normalizer

    state_penalty = 0.0
    if packet.state == "GAME_OVER":
        state_penalty = 1.5
    elif packet.state == "NOT_PLAYED":
        state_penalty = 0.7
    elif packet.state == "WIN":
        state_penalty = 0.0
    else:
        state_penalty = 0.2

    action_penalty = 0.0
    if candidate.action_id == 6:
        action_penalty = 0.20
    elif candidate.action_id == 7:
        action_penalty = 0.10

    availability_penalty = (
        0.0 if candidate.action_id in packet.available_actions or candidate.action_id == 0 else 3.0
    )

    object_sparsity_penalty = 0.0
    if representation.summary.get("object_count", 0) <= 0 and candidate.action_id == 6:
        object_sparsity_penalty = 0.4

    total_risk = (
        normalized_gap
        + state_penalty
        + action_penalty
        + availability_penalty
        + object_sparsity_penalty
    )
    return float(total_risk), {
        "normalized_gap": float(normalized_gap),
        "state_penalty": float(state_penalty),
        "action_penalty": float(action_penalty),
        "availability_penalty": float(availability_penalty),
        "object_sparsity_penalty": float(object_sparsity_penalty),
    }


def compute_information_gain_v1(
    *,
    posterior_by_hypothesis_id: dict[str, float],
    hypothesis_signature: dict[str, str],
    predictive_distribution: dict[str, float],
) -> float:
    if not predictive_distribution:
        return 0.0

    prior_entropy = entropy_bits(
        {
            hypothesis_id: weight
            for (hypothesis_id, weight) in posterior_by_hypothesis_id.items()
            if weight > 0.0
        }
    )

    expected_posterior_entropy = 0.0
    for signature_key, signature_probability in predictive_distribution.items():
        p = float(signature_probability)
        if p <= 1.0e-12:
            continue
        mass_by_hypothesis: dict[str, float] = {}
        mass = 0.0
        for hypothesis_id, signature in hypothesis_signature.items():
            if signature != signature_key:
                continue
            weight = float(posterior_by_hypothesis_id.get(hypothesis_id, 0.0))
            if weight <= 0.0:
                continue
            mass_by_hypothesis[hypothesis_id] = weight
            mass += weight
        if mass <= 1.0e-12:
            continue
        normalized = {hid: w / mass for (hid, w) in mass_by_hypothesis.items()}
        posterior_entropy_for_signature = entropy_bits(normalized)
        expected_posterior_entropy += p * posterior_entropy_for_signature

    information_gain = max(0.0, prior_entropy - expected_posterior_entropy)
    return float(information_gain)
