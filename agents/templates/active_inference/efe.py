from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .contracts import ObservationPacketV1

EPS = 1.0e-9


@dataclass(slots=True, frozen=True)
class EFEWeightsV1:
    risk: float
    ambiguity: float
    information_gain_action_semantics: float
    information_gain_mechanism_dynamics: float
    information_gain_causal_mapping: float
    action_cost: float
    complexity: float
    vfe: float


def entropy_bits(distribution: dict[str, float]) -> float:
    out = 0.0
    for probability in distribution.values():
        p = float(probability)
        if p <= 0.0:
            continue
        out -= p * math.log2(p)
    return float(out)


def normalize_distribution(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in values.values())
    if total <= EPS:
        if not values:
            return {}
        uniform = 1.0 / float(len(values))
        return {key: uniform for key in values}
    return {
        key: float(max(0.0, value) / total)
        for (key, value) in values.items()
    }


def _base_signature_key(signature_key: str) -> str:
    sig_type = "OBSERVED_UNCLASSIFIED"
    progress = "0"
    for part in str(signature_key).split("|"):
        if part.startswith("type="):
            sig_type = part.split("=", 1)[1]
        elif part.startswith("progress="):
            progress = part.split("=", 1)[1]
    return f"type={sig_type}|progress={progress}"


def _signature_key_with_delta(signature_key: str) -> str:
    sig_type = "OBSERVED_UNCLASSIFIED"
    progress = "0"
    delta = "na"
    for part in str(signature_key).split("|"):
        if part.startswith("type="):
            sig_type = part.split("=", 1)[1]
        elif part.startswith("progress="):
            progress = part.split("=", 1)[1]
        elif part.startswith("delta="):
            delta = part.split("=", 1)[1]
    return f"type={sig_type}|progress={progress}|delta={delta}"


def determine_phase_v1(
    *,
    action_counter: int,
    posterior_entropy_bits: float,
    explore_steps: int,
    exploit_entropy_threshold: float,
    remaining_budget: int,
) -> str:
    if remaining_budget <= 4:
        return "exploit"
    if action_counter < explore_steps:
        return "explore"
    if posterior_entropy_bits > exploit_entropy_threshold:
        return "explain"
    return "exploit"


def weights_for_phase_v1(phase: str) -> EFEWeightsV1:
    if phase == "explore":
        return EFEWeightsV1(
            risk=0.70,
            ambiguity=0.45,
            information_gain_action_semantics=1.15,
            information_gain_mechanism_dynamics=1.35,
            information_gain_causal_mapping=1.05,
            action_cost=0.10,
            complexity=0.15,
            vfe=0.05,
        )
    if phase == "explain":
        return EFEWeightsV1(
            risk=0.95,
            ambiguity=0.40,
            information_gain_action_semantics=0.85,
            information_gain_mechanism_dynamics=1.05,
            information_gain_causal_mapping=0.80,
            action_cost=0.22,
            complexity=0.20,
            vfe=0.08,
        )
    return EFEWeightsV1(
        risk=1.25,
        ambiguity=0.25,
        information_gain_action_semantics=0.35,
        information_gain_mechanism_dynamics=0.45,
        information_gain_causal_mapping=0.25,
        action_cost=0.90,
        complexity=0.32,
        vfe=0.12,
    )


def _opposite_direction_bucket(direction_bucket: str) -> str:
    mapping = {
        "dir_l": "dir_r",
        "dir_r": "dir_l",
        "dir_u": "dir_d",
        "dir_d": "dir_u",
    }
    return str(mapping.get(str(direction_bucket), "dir_unknown"))


def preference_distribution_v1(
    packet: ObservationPacketV1,
    phase: str,
    *,
    navigation_target: dict[str, Any] | None = None,
) -> dict[str, float]:
    progress_gap = int(max(0, packet.win_levels - packet.levels_completed))
    progress_weight = 0.45 if progress_gap > 0 else 0.20
    if phase == "exploit":
        progress_weight = 0.58

    no_change_penalty_mass = 0.12 if phase == "explore" else 0.05

    pref = {
        "type=METADATA_PROGRESS_CHANGE|progress=1": progress_weight,
        "type=LOCAL_COLOR_CHANGE|progress=0": 0.18,
        "type=CC_TRANSLATION|progress=0": 0.12,
        "type=CC_COUNT_CHANGE|progress=0": 0.10,
        "type=GLOBAL_PATTERN_CHANGE|progress=0": 0.08,
        "type=NO_CHANGE|progress=0": no_change_penalty_mass,
        "type=NO_CHANGE|progress=0|delta=blocked": 0.01,
        "type=NO_CHANGE|progress=0|delta=na": 0.06 if phase != "explore" else 0.08,
        "type=OBSERVED_UNCLASSIFIED|progress=0": 0.10,
        "type=METADATA_PROGRESS_CHANGE|progress=0": 0.05,
    }

    target = navigation_target if isinstance(navigation_target, dict) else {}
    target_enabled = bool(target.get("enabled", False))
    target_direction = str(target.get("target_direction_bucket", "dir_unknown"))
    target_salience = max(0.0, min(1.0, float(target.get("target_salience", 0.0))))
    if (
        target_enabled
        and target_direction in ("dir_l", "dir_r", "dir_u", "dir_d")
        and target_salience > 0.0
    ):
        toward_key = f"type=CC_TRANSLATION|progress=0|delta={target_direction}"
        away_direction = _opposite_direction_bucket(target_direction)
        away_key = f"type=CC_TRANSLATION|progress=0|delta={away_direction}"
        toward_bonus = 0.22 * target_salience
        away_scale = max(0.10, 1.0 - (0.60 * target_salience))
        pref[toward_key] = float(pref.get(toward_key, 0.0) + toward_bonus)
        pref[away_key] = float(max(0.005, pref.get(away_key, 0.02) * away_scale))

    return normalize_distribution(pref)


def compute_risk_kl_v1(
    predictive_distribution: dict[str, float],
    preference_distribution: dict[str, float],
) -> tuple[float, dict[str, float]]:
    pred = normalize_distribution(dict(predictive_distribution))
    pref = normalize_distribution(dict(preference_distribution))

    kl = 0.0
    terms: dict[str, float] = {}
    for signature_key, pred_probability in pred.items():
        p = float(pred_probability)
        if p <= 0.0:
            continue
        base_key = _base_signature_key(signature_key)
        delta_key = _signature_key_with_delta(signature_key)
        q = float(
            pref.get(
                signature_key,
                pref.get(delta_key, pref.get(base_key, EPS)),
            )
        )
        term = p * math.log2(max(EPS, p) / max(EPS, q))
        terms[signature_key] = float(term)
        kl += term

    return float(max(0.0, kl)), terms
