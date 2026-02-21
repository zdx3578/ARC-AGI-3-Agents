from __future__ import annotations

import math
from typing import Any

from .contracts import (
    ActionCandidateV1,
    FreeEnergyLedgerEntryV1,
    ObservationPacketV1,
    RepresentationStateV1,
)
from .efe import (
    compute_risk_kl_v1,
    determine_phase_v1,
    preference_distribution_v1,
    weights_for_phase_v1,
)
from .hypothesis_bank import ActiveInferenceHypothesisBankV1


class ActiveInferencePolicyEvaluatorV1:
    def __init__(
        self,
        *,
        explore_steps: int = 20,
        exploit_entropy_threshold: float = 0.9,
        top_k_reasoning: int = 5,
        rollout_horizon: int = 2,
        rollout_discount: float = 0.55,
        tie_epsilon: float = 1.0e-6,
        ignore_action_cost: bool = True,
        weight_overrides: dict[str, dict[str, float]] | None = None,
        action6_bucket_probe_min_attempts: int = 3,
        action6_subcluster_probe_min_attempts: int = 2,
        action6_probe_score_margin: float = 0.06,
        action6_explore_probe_score_margin: float = 0.12,
        action6_stagnation_step_threshold: int = 12,
        stagnation_probe_trigger_steps: int = 24,
        stagnation_probe_score_margin: float = 0.22,
        stagnation_probe_min_action_usage_gap: int = 8,
        rollout_max_candidates: int = 8,
        rollout_only_in_exploit: bool = True,
        region_revisit_hard_threshold: int = 24,
        sequence_rollout_frontier_weight: float = 0.35,
        sequence_rollout_direction_weight: float = 0.25,
        sequence_probe_score_margin: float = 0.28,
        sequence_probe_trigger_steps: int = 20,
        coverage_sweep_target_regions: int = 24,
        coverage_sweep_score_margin: float = 0.42,
        coverage_resweep_interval: int = 96,
        coverage_resweep_span: int = 24,
        coverage_sweep_direction_retry_limit: int = 6,
        coverage_matrix_sweep_enabled: bool = False,
        coverage_sweep_force_in_exploit: bool = True,
        hierarchy_weight_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.explore_steps = int(max(1, explore_steps))
        self.exploit_entropy_threshold = float(max(0.0, exploit_entropy_threshold))
        self.top_k_reasoning = int(max(1, top_k_reasoning))
        self.rollout_horizon = int(max(1, rollout_horizon))
        self.rollout_discount = float(max(0.0, min(1.0, rollout_discount)))
        self.tie_epsilon = float(max(0.0, tie_epsilon))
        self.ignore_action_cost = bool(ignore_action_cost)
        self.weight_overrides = weight_overrides or {}
        self.action6_bucket_probe_min_attempts = int(
            max(1, action6_bucket_probe_min_attempts)
        )
        self.action6_subcluster_probe_min_attempts = int(
            max(1, action6_subcluster_probe_min_attempts)
        )
        self.action6_probe_score_margin = float(max(0.0, action6_probe_score_margin))
        self.action6_explore_probe_score_margin = float(
            max(0.0, action6_explore_probe_score_margin)
        )
        self.action6_stagnation_step_threshold = int(
            max(1, action6_stagnation_step_threshold)
        )
        self.stagnation_probe_trigger_steps = int(max(1, stagnation_probe_trigger_steps))
        self.stagnation_probe_score_margin = float(max(0.0, stagnation_probe_score_margin))
        self.stagnation_probe_min_action_usage_gap = int(
            max(1, stagnation_probe_min_action_usage_gap)
        )
        self.rollout_max_candidates = int(max(1, rollout_max_candidates))
        self.rollout_only_in_exploit = bool(rollout_only_in_exploit)
        self.region_revisit_hard_threshold = int(max(4, region_revisit_hard_threshold))
        self.sequence_rollout_frontier_weight = float(
            max(0.0, sequence_rollout_frontier_weight)
        )
        self.sequence_rollout_direction_weight = float(
            max(0.0, sequence_rollout_direction_weight)
        )
        self.sequence_probe_score_margin = float(max(0.0, sequence_probe_score_margin))
        self.sequence_probe_trigger_steps = int(max(1, sequence_probe_trigger_steps))
        self.coverage_sweep_target_regions = int(max(1, coverage_sweep_target_regions))
        self.coverage_sweep_score_margin = float(max(0.0, coverage_sweep_score_margin))
        self.coverage_resweep_interval = int(max(0, coverage_resweep_interval))
        self.coverage_resweep_span = int(max(0, coverage_resweep_span))
        self.coverage_sweep_direction_retry_limit = int(
            max(1, coverage_sweep_direction_retry_limit)
        )
        self.coverage_matrix_sweep_enabled = bool(coverage_matrix_sweep_enabled)
        self.coverage_sweep_force_in_exploit = bool(coverage_sweep_force_in_exploit)
        self.hierarchy_weight_overrides = hierarchy_weight_overrides or {}

    @staticmethod
    def _clamp01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _scaled_weight(base_value: float, factor: float) -> float:
        return float(max(0.0, float(base_value) * max(0.0, float(factor))))

    @staticmethod
    def _opposite_direction_bucket(direction_bucket: str) -> str:
        mapping = {
            "dir_l": "dir_r",
            "dir_r": "dir_l",
            "dir_u": "dir_d",
            "dir_d": "dir_u",
        }
        return str(mapping.get(str(direction_bucket), "dir_unknown"))

    def _hierarchy_weights_for_phase(self, phase: str) -> dict[str, float]:
        # Layerwise EFE shaping terms:
        # - progress_risk: L2 task-level pressure (no progress under repeated interactions)
        # - operability_risk: L1 control-level pressure (blocked / uncontrollable outcomes)
        # - habit_risk: L2 anti-attractor pressure (over-used action under no progress)
        # - progress_information_gain: L2 epistemic term (reward mechanisms that reduce
        #   progress uncertainty)
        defaults: dict[str, float]
        if phase == "explore":
            defaults = {
                "progress_risk": 0.16,
                "operability_risk": 0.10,
                "habit_risk": 0.20,
                "progress_information_gain": 0.10,
            }
        elif phase == "explain":
            defaults = {
                "progress_risk": 0.30,
                "operability_risk": 0.14,
                "habit_risk": 0.32,
                "progress_information_gain": 0.12,
            }
        else:
            defaults = {
                "progress_risk": 0.55,
                "operability_risk": 0.20,
                "habit_risk": 0.40,
                "progress_information_gain": 0.12,
            }
        out = dict(defaults)
        override = self.hierarchy_weight_overrides.get(phase, {})
        for key in out:
            if key not in override:
                continue
            try:
                out[key] = float(override[key])
            except Exception:
                continue
        return out

    def _signature_probability_mass(
        self,
        predictive_distribution: dict[str, float],
        *,
        token_in_signature: str,
    ) -> float:
        mass = 0.0
        for signature_key, probability in predictive_distribution.items():
            if token_in_signature not in str(signature_key):
                continue
            try:
                mass += float(probability)
            except Exception:
                continue
        return float(max(0.0, min(1.0, mass)))

    def _hierarchical_efe_terms(
        self,
        *,
        packet: ObservationPacketV1,
        candidate: ActionCandidateV1,
        predictive_distribution: dict[str, float],
        risk_value: float,
        ambiguity: float,
        ig_action_semantics: float,
        ig_mechanism_dynamics: float,
        ig_causal_mapping: float,
    ) -> dict[str, Any]:
        # Level 0 (perception/outcome-shape): map to existing risk/ambiguity and
        # action-semantics information gain.
        level0 = {
            "risk": float(max(0.0, risk_value)),
            "ambiguity": float(max(0.0, ambiguity)),
            "information_gain": float(max(0.0, ig_action_semantics)),
        }

        blocked_probability = self._signature_probability_mass(
            predictive_distribution,
            token_in_signature="|delta=blocked",
        )
        translation_probability = self._signature_probability_mass(
            predictive_distribution,
            token_in_signature="type=CC_TRANSLATION|progress=0",
        )
        no_change_probability = self._signature_probability_mass(
            predictive_distribution,
            token_in_signature="type=NO_CHANGE|progress=0",
        )
        progress_probability = self._signature_probability_mass(
            predictive_distribution,
            token_in_signature="progress=1",
        )
        progress_gap = int(max(0, int(packet.win_levels) - int(packet.levels_completed)))
        progress_gap_ratio = float(progress_gap / float(max(1, int(packet.win_levels))))

        blocked_edge_stats = candidate.metadata.get("blocked_edge_observed_stats", {})
        if not isinstance(blocked_edge_stats, dict):
            blocked_edge_stats = {}
        transition_stats = candidate.metadata.get("transition_exploration_stats", {})
        if not isinstance(transition_stats, dict):
            transition_stats = {}

        action_attempts = int(max(0, blocked_edge_stats.get("action_attempts", 0)))
        edge_attempts = int(max(0, blocked_edge_stats.get("edge_attempts", 0)))
        action_blocked_rate = float(
            max(0.0, min(1.0, blocked_edge_stats.get("action_blocked_rate", 0.0)))
        )
        action_moved_rate = float(
            max(0.0, min(1.0, blocked_edge_stats.get("action_moved_rate", 0.0)))
        )
        edge_blocked_rate = float(
            max(0.0, min(1.0, blocked_edge_stats.get("edge_blocked_rate", 0.0)))
        )
        region_revisit_count = int(max(0, blocked_edge_stats.get("region_revisit_count_current", 0)))
        state_visit_count = int(max(0, transition_stats.get("state_visit_count", 0)))
        state_action_visit_count = int(max(0, transition_stats.get("state_action_visit_count", 0)))
        state_outgoing_edge_count = int(
            max(0, transition_stats.get("state_outgoing_edge_count", 0))
        )
        region_revisit_ratio = float(
            min(
                1.0,
                float(region_revisit_count)
                / float(max(1, self.region_revisit_hard_threshold)),
            )
        )
        region_novelty = float(max(0.0, 1.0 - region_revisit_ratio))
        action_frontier_novelty = float(
            max(0.0, 1.0 - min(1.0, float(action_attempts) / 10.0))
        )
        edge_frontier_novelty = float(
            max(0.0, 1.0 - min(1.0, float(edge_attempts) / 8.0))
        )
        state_action_frontier_novelty = float(
            max(0.0, 1.0 - min(1.0, float(state_action_visit_count) / 4.0))
        )
        state_outgoing_frontier_novelty = float(
            max(0.0, 1.0 - min(1.0, float(state_outgoing_edge_count) / 6.0))
        )
        frontier_novelty = float(
            max(
                0.0,
                min(
                    1.0,
                    (0.35 * action_frontier_novelty)
                    + (0.20 * edge_frontier_novelty)
                    + (0.30 * state_action_frontier_novelty)
                    + (0.15 * state_outgoing_frontier_novelty),
                ),
            )
        )
        region_revisit_hard_penalty_raw = float(
            max(
                0.0,
                (
                    float(region_revisit_count - int(self.region_revisit_hard_threshold) + 1)
                    / float(max(1, int(self.region_revisit_hard_threshold)))
                ),
            )
        )
        # Saturating penalty keeps anti-loop pressure without exploding magnitudes.
        region_revisit_hard_penalty = float(
            region_revisit_hard_penalty_raw / (1.0 + region_revisit_hard_penalty_raw)
        )
        evidence_count_effective = int(max(action_attempts, state_action_visit_count))
        evidence_confidence = float(
            1.0 - math.exp(-float(evidence_count_effective) / 10.0)
        )
        evidence_confidence = self._clamp01(evidence_confidence)
        evidence_novelty = float(max(0.0, 1.0 - evidence_confidence))
        if evidence_count_effective < 4:
            evidence_regime = "few_shot"
        elif evidence_count_effective < 12:
            evidence_regime = "forming"
        elif evidence_count_effective < 24:
            evidence_regime = "stable"
        else:
            evidence_regime = "saturated"

        predicted_region_stats = self._candidate_predicted_region_stats(candidate)
        coverage_enabled = bool(predicted_region_stats.get("enabled", False))
        coverage_confidence = self._clamp01(float(predicted_region_stats.get("confidence", 0.0)))
        coverage_next_region_key = str(
            predicted_region_stats.get("predicted_region_key", "NA")
        )
        coverage_next_region_visit_count = int(
            max(0, predicted_region_stats.get("predicted_region_visit_count", 0))
        )
        coverage_region_novelty = 0.0
        coverage_frontier_bonus = 0.0
        coverage_repeat_penalty = 0.0
        if coverage_enabled:
            coverage_region_novelty = float(
                max(0.0, 1.0 - min(1.0, float(coverage_next_region_visit_count) / 6.0))
            )
            coverage_frontier_bonus = float(
                max(
                    0.0,
                    translation_probability
                    * coverage_region_novelty
                    * (0.35 + (0.65 * coverage_confidence)),
                )
            )
            coverage_repeat_penalty = float(
                max(
                    0.0,
                    translation_probability
                    * (1.0 - coverage_region_novelty)
                    * (0.25 + (0.75 * coverage_confidence)),
                )
            )

        navigation_target = candidate.metadata.get("navigation_target_features_v1", {})
        if not isinstance(navigation_target, dict):
            navigation_target = {}
        key_target_enabled = bool(navigation_target.get("enabled", False))
        key_target_direction_bucket = str(
            navigation_target.get("target_direction_bucket", "dir_unknown")
        )
        key_target_salience = self._clamp01(
            float(navigation_target.get("target_salience", 0.0))
        )
        key_target_kind = str(navigation_target.get("target_kind", "unknown"))
        key_target_distance_before_raw = float(
            navigation_target.get("distance_before", -1.0)
        )
        key_target_distance_before = float(
            key_target_distance_before_raw if key_target_distance_before_raw >= 0.0 else -1.0
        )
        key_target_valid_direction = bool(
            key_target_direction_bucket in ("dir_l", "dir_r", "dir_u", "dir_d")
        )
        key_target_toward_probability = 0.0
        key_target_away_probability = 0.0
        key_target_expected_distance_delta = 0.0
        key_target_expected_distance_after = float(
            key_target_distance_before if key_target_distance_before >= 0.0 else -1.0
        )
        key_target_direction_alignment = 0.0
        key_target_escape_bonus = 0.0
        key_target_away_penalty = 0.0
        if key_target_enabled and key_target_valid_direction and key_target_salience > 0.0:
            key_target_toward_probability = self._signature_probability_mass(
                predictive_distribution,
                token_in_signature=f"|delta={key_target_direction_bucket}",
            )
            opposite_bucket = self._opposite_direction_bucket(key_target_direction_bucket)
            key_target_away_probability = self._signature_probability_mass(
                predictive_distribution,
                token_in_signature=f"|delta={opposite_bucket}",
            )
            key_target_expected_distance_delta = float(
                key_target_away_probability - key_target_toward_probability
            )
            if key_target_distance_before >= 0.0:
                key_target_expected_distance_after = float(
                    max(0.0, key_target_distance_before + key_target_expected_distance_delta)
                )
            key_target_direction_alignment = float(
                max(-1.0, min(1.0, key_target_toward_probability - key_target_away_probability))
            )
            key_target_escape_bonus = float(
                max(0.0, -key_target_expected_distance_delta) * key_target_salience
            )
            key_target_away_penalty = float(
                max(0.0, key_target_expected_distance_delta) * key_target_salience
            )

        # Level 1 (operability/control): risk captures blocked tendencies and
        # control uncertainty in movement outcomes.
        same_region_translation_penalty = float(
            max(
                0.0,
                translation_probability
                * region_revisit_ratio
                * (0.45 + (0.55 * action_moved_rate)),
            )
        )
        operability_risk = float(
            max(
                0.0,
                (0.60 * blocked_probability)
                + (0.25 * action_blocked_rate)
                + (0.15 * edge_blocked_rate),
            )
        )
        operability_risk = float(
            max(
                0.0,
                operability_risk + (0.70 * same_region_translation_penalty),
            )
        )
        operability_risk = float(
            max(
                0.0,
                operability_risk
                + (0.25 * key_target_away_penalty)
                + (0.10 * coverage_repeat_penalty)
                - (
                    0.30
                    * key_target_escape_bonus
                    * max(0.25, float(translation_probability))
                ),
            )
        )
        operability_risk = float(
            max(0.0, operability_risk - (0.12 * coverage_frontier_bonus))
        )
        level1 = {
            "risk": float(operability_risk),
            "ambiguity": float(max(0.0, ambiguity)),
            "information_gain": float(max(0.0, ig_mechanism_dynamics)),
            "blocked_probability": float(blocked_probability),
            "action_blocked_rate": float(action_blocked_rate),
            "action_moved_rate": float(action_moved_rate),
            "edge_blocked_rate": float(edge_blocked_rate),
            "same_region_translation_penalty": float(same_region_translation_penalty),
            "key_target_enabled": bool(key_target_enabled),
            "key_target_kind": str(key_target_kind),
            "key_target_salience": float(key_target_salience),
            "key_target_direction_bucket": str(key_target_direction_bucket),
            "key_target_toward_probability": float(key_target_toward_probability),
            "key_target_away_probability": float(key_target_away_probability),
            "key_target_direction_alignment": float(key_target_direction_alignment),
            "key_target_expected_distance_delta": float(key_target_expected_distance_delta),
            "key_target_expected_distance_after": float(key_target_expected_distance_after),
        }

        # Level 2 (task/progress): penalize repetitive no-progress attractors.
        no_progress_probability = float(max(0.0, min(1.0, 1.0 - progress_probability)))
        translation_no_progress_probability_amplified = float(
            max(
                0.0,
                min(
                    1.0,
                    translation_probability * (1.0 + (0.90 * region_revisit_ratio)),
                ),
            )
        )
        stagnation_context = float(
            max(
                0.0,
                min(
                    1.0,
                    (
                        0.30 * min(1.0, float(region_revisit_count) / 24.0)
                        + 0.50 * min(1.0, float(state_action_visit_count) / 8.0)
                        + 0.20 * (1.0 - frontier_novelty)
                    ),
                ),
            )
        )
        action_repeat_ratio = float(
            min(1.0, float(action_attempts) / 40.0)
        )
        repeated_no_progress_penalty = float(
            max(
                0.0,
                action_repeat_ratio
                * max(
                    no_change_probability,
                    translation_no_progress_probability_amplified,
                )
                * (0.35 + (0.65 * region_revisit_ratio)),
            )
        )
        habit_pressure = float(
            max(
                0.0,
                min(
                    1.0,
                    (
                        0.60
                        * (
                            math.log1p(float(action_attempts))
                            / max(1.0, math.log1p(80.0))
                        )
                        + 0.25 * stagnation_context
                        + 0.15 * region_revisit_ratio
                    ),
                ),
            )
        )
        leave_high_revisit_potential = float(
            max(
                0.0,
                translation_probability
                * frontier_novelty
                * (0.35 + (0.65 * region_revisit_ratio)),
            )
        )
        progress_risk = float(
            max(
                0.0,
                progress_gap_ratio
                * (
                    (0.35 * no_progress_probability * (0.5 + (0.5 * region_revisit_ratio)))
                    + (0.35 * translation_no_progress_probability_amplified)
                    + (0.18 * (1.0 - frontier_novelty))
                    + (0.12 * region_revisit_hard_penalty)
                    + (0.20 * repeated_no_progress_penalty)
                    + (0.18 * key_target_away_penalty)
                    + (0.20 * coverage_repeat_penalty)
                    - (0.25 * leave_high_revisit_potential)
                    - (0.30 * key_target_escape_bonus)
                    - (0.35 * coverage_frontier_bonus)
                ),
            )
        )
        habit_risk = float(
            max(
                0.0,
                progress_gap_ratio
                * no_progress_probability
                * ((0.65 * habit_pressure) + (0.35 * region_revisit_ratio)),
            )
        )
        habit_risk += float(
            max(
                0.0,
                progress_gap_ratio * no_progress_probability * region_revisit_hard_penalty,
            )
        )
        habit_risk += float(
            max(0.0, 0.70 * repeated_no_progress_penalty)
        )
        habit_risk += float(max(0.0, 0.25 * key_target_away_penalty))
        habit_risk += float(max(0.0, 0.30 * coverage_repeat_penalty))
        habit_risk = float(max(0.0, habit_risk - (0.22 * key_target_escape_bonus)))
        habit_risk = float(max(0.0, habit_risk - (0.26 * coverage_frontier_bonus)))
        progress_information_gain = float(
            max(
                0.0,
                ig_causal_mapping
                * (0.5 + (0.5 * progress_gap_ratio))
                * (
                    0.30
                    + (0.45 * frontier_novelty)
                    + (0.25 * evidence_novelty)
                    + (0.20 * key_target_escape_bonus)
                    + (0.28 * coverage_frontier_bonus)
                ),
            )
        )
        level2 = {
            "risk": float(progress_risk),
            "habit_risk": float(habit_risk),
            "ambiguity": float(no_progress_probability),
            "information_gain": float(progress_information_gain),
            "progress_probability": float(progress_probability),
            "translation_no_progress_probability": float(translation_probability),
            "translation_no_progress_probability_amplified": float(
                translation_no_progress_probability_amplified
            ),
            "no_change_probability": float(no_change_probability),
            "progress_gap_ratio": float(progress_gap_ratio),
            "action_attempts": int(action_attempts),
            "edge_attempts": int(edge_attempts),
            "state_visit_count": int(state_visit_count),
            "state_action_visit_count": int(state_action_visit_count),
            "state_outgoing_edge_count": int(state_outgoing_edge_count),
            "region_revisit_count_current": int(region_revisit_count),
            "region_revisit_ratio": float(region_revisit_ratio),
            "region_novelty": float(region_novelty),
            "frontier_novelty": float(frontier_novelty),
            "region_revisit_hard_penalty": float(region_revisit_hard_penalty),
            "region_revisit_hard_penalty_raw": float(region_revisit_hard_penalty_raw),
            "leave_high_revisit_potential": float(leave_high_revisit_potential),
            "stagnation_context": float(stagnation_context),
            "habit_pressure": float(habit_pressure),
            "action_repeat_ratio": float(action_repeat_ratio),
            "repeated_no_progress_penalty": float(repeated_no_progress_penalty),
            "evidence_count_effective": int(evidence_count_effective),
            "evidence_confidence": float(evidence_confidence),
            "evidence_novelty": float(evidence_novelty),
            "evidence_regime": str(evidence_regime),
            "key_target_enabled": bool(key_target_enabled),
            "key_target_kind": str(key_target_kind),
            "key_target_salience": float(key_target_salience),
            "key_target_direction_bucket": str(key_target_direction_bucket),
            "key_target_distance_before": float(key_target_distance_before),
            "key_target_toward_probability": float(key_target_toward_probability),
            "key_target_away_probability": float(key_target_away_probability),
            "key_target_direction_alignment": float(key_target_direction_alignment),
            "key_target_expected_distance_delta": float(key_target_expected_distance_delta),
            "key_target_expected_distance_after": float(key_target_expected_distance_after),
            "key_target_escape_bonus": float(key_target_escape_bonus),
            "key_target_away_penalty": float(key_target_away_penalty),
            "coverage_enabled": bool(coverage_enabled),
            "coverage_confidence": float(coverage_confidence),
            "coverage_next_region_key": str(coverage_next_region_key),
            "coverage_next_region_visit_count": int(coverage_next_region_visit_count),
            "coverage_region_novelty": float(coverage_region_novelty),
            "coverage_frontier_bonus": float(coverage_frontier_bonus),
            "coverage_repeat_penalty": float(coverage_repeat_penalty),
        }

        return {
            "schema_name": "active_inference_hierarchical_efe_terms_v1",
            "schema_version": 1,
            "levels": {
                "level0_perception": level0,
                "level1_operability": level1,
                "level2_progress": level2,
            },
        }

    def _adaptive_objective_profile(
        self,
        *,
        phase: str,
        base_weights: dict[str, float],
        base_hierarchy_weights: dict[str, float],
        level1_terms: dict[str, Any],
        level2_terms: dict[str, Any],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
        weights = {str(k): float(v) for (k, v) in base_weights.items()}
        hierarchy_weights = {
            str(k): float(v) for (k, v) in base_hierarchy_weights.items()
        }
        progress_probability = self._clamp01(float(level2_terms.get("progress_probability", 0.0)))
        no_progress_signal = float(max(0.0, 1.0 - progress_probability))
        blocked_probability = self._clamp01(float(level1_terms.get("blocked_probability", 0.0)))
        region_revisit_ratio = self._clamp01(float(level2_terms.get("region_revisit_ratio", 0.0)))
        stagnation_context = self._clamp01(float(level2_terms.get("stagnation_context", 0.0)))
        habit_pressure = self._clamp01(float(level2_terms.get("habit_pressure", 0.0)))
        translation_no_progress_amplified = self._clamp01(
            float(level2_terms.get("translation_no_progress_probability_amplified", 0.0))
        )
        evidence_novelty = self._clamp01(float(level2_terms.get("evidence_novelty", 0.0)))
        evidence_confidence = self._clamp01(float(level2_terms.get("evidence_confidence", 0.0)))
        stuck_score = self._clamp01(
            (0.30 * stagnation_context)
            + (0.30 * region_revisit_ratio)
            + (0.25 * translation_no_progress_amplified)
            + (0.15 * habit_pressure)
        )

        phase_focus = "balanced"
        explain_escape_pressure = 0.0
        if phase == "explore":
            phase_focus = "epistemic_first"
            weights["risk"] = self._scaled_weight(
                weights["risk"],
                max(0.70, 1.0 - (0.25 * evidence_novelty)),
            )
            weights["information_gain_mechanism_dynamics"] = self._scaled_weight(
                weights["information_gain_mechanism_dynamics"],
                1.0 + (0.85 * evidence_novelty) + (0.35 * stuck_score),
            )
            weights["information_gain_causal_mapping"] = self._scaled_weight(
                weights["information_gain_causal_mapping"],
                1.0 + (0.55 * evidence_novelty) + (0.35 * stuck_score),
            )
            hierarchy_weights["progress_information_gain"] = self._scaled_weight(
                hierarchy_weights["progress_information_gain"],
                1.0 + (0.60 * evidence_novelty) + (0.25 * stuck_score),
            )
            hierarchy_weights["progress_risk"] = self._scaled_weight(
                hierarchy_weights["progress_risk"],
                1.0 + (0.25 * stuck_score),
            )
        elif phase == "explain":
            phase_focus = "mechanism_disambiguation"
            explain_escape_pressure = self._clamp01(
                (0.55 * stagnation_context) + (0.45 * region_revisit_ratio)
            )
            weights["ambiguity"] = self._scaled_weight(
                weights["ambiguity"],
                1.0 + (0.20 * evidence_novelty),
            )
            weights["information_gain_mechanism_dynamics"] = self._scaled_weight(
                weights["information_gain_mechanism_dynamics"],
                max(
                    0.60,
                    1.0
                    + (0.20 * evidence_novelty)
                    - (0.55 * evidence_confidence)
                    - (0.45 * explain_escape_pressure),
                ),
            )
            weights["information_gain_causal_mapping"] = self._scaled_weight(
                weights["information_gain_causal_mapping"],
                max(
                    0.65,
                    1.0
                    + (0.15 * evidence_novelty)
                    - (0.35 * evidence_confidence)
                    - (0.25 * explain_escape_pressure),
                ),
            )
            hierarchy_weights["operability_risk"] = self._scaled_weight(
                hierarchy_weights["operability_risk"],
                1.0 + (0.50 * blocked_probability),
            )
            hierarchy_weights["progress_risk"] = self._scaled_weight(
                hierarchy_weights["progress_risk"],
                1.0 + (0.45 * explain_escape_pressure),
            )
            hierarchy_weights["habit_risk"] = self._scaled_weight(
                hierarchy_weights["habit_risk"],
                1.0 + (0.55 * explain_escape_pressure),
            )
            hierarchy_weights["progress_information_gain"] = self._scaled_weight(
                hierarchy_weights["progress_information_gain"],
                max(
                    0.70,
                    1.0
                    + (0.20 * evidence_novelty)
                    - (0.30 * evidence_confidence),
                ),
            )
        else:
            phase_focus = "progress_with_escape"
            # When stuck, reduce rigid risk pressure to allow escape rollouts.
            if stuck_score > 0.55:
                weights["risk"] = self._scaled_weight(weights["risk"], 0.88)
            else:
                weights["risk"] = self._scaled_weight(
                    weights["risk"],
                    1.0 + (0.20 * (1.0 - stuck_score)),
                )
            weights["information_gain_mechanism_dynamics"] = self._scaled_weight(
                weights["information_gain_mechanism_dynamics"],
                1.0 + (0.55 * stuck_score * no_progress_signal),
            )
            weights["information_gain_causal_mapping"] = self._scaled_weight(
                weights["information_gain_causal_mapping"],
                1.0 + (0.45 * stuck_score * no_progress_signal),
            )
            hierarchy_weights["progress_risk"] = self._scaled_weight(
                hierarchy_weights["progress_risk"],
                1.0 + (0.95 * stuck_score),
            )
            hierarchy_weights["habit_risk"] = self._scaled_weight(
                hierarchy_weights["habit_risk"],
                1.0 + (1.35 * stuck_score),
            )
            hierarchy_weights["operability_risk"] = self._scaled_weight(
                hierarchy_weights["operability_risk"],
                1.0 + (0.30 * blocked_probability),
            )
            hierarchy_weights["progress_information_gain"] = self._scaled_weight(
                hierarchy_weights["progress_information_gain"],
                1.0 + (0.80 * stuck_score * no_progress_signal),
            )

        diagnostics = {
            "phase_focus": str(phase_focus),
            "stuck_score": float(stuck_score),
            "progress_probability": float(progress_probability),
            "no_progress_signal": float(no_progress_signal),
            "blocked_probability": float(blocked_probability),
            "region_revisit_ratio": float(region_revisit_ratio),
            "stagnation_context": float(stagnation_context),
            "habit_pressure": float(habit_pressure),
            "translation_no_progress_probability_amplified": float(
                translation_no_progress_amplified
            ),
            "explain_escape_pressure": float(explain_escape_pressure),
            "evidence_confidence": float(evidence_confidence),
            "evidence_novelty": float(evidence_novelty),
            "evidence_count_effective": int(level2_terms.get("evidence_count_effective", 0)),
            "evidence_regime": str(level2_terms.get("evidence_regime", "unknown")),
            "effect_thresholds": {
                "few_shot_min": 4,
                "forming_min": 12,
                "stable_min": 24,
            },
        }
        return weights, hierarchy_weights, diagnostics

    def _candidate_cluster_id(self, candidate: ActionCandidateV1) -> str:
        metadata_cluster = str(candidate.metadata.get("candidate_cluster_id", "")).strip()
        if metadata_cluster:
            return metadata_cluster
        action_id = int(candidate.action_id)
        if action_id != 6:
            return f"a{action_id}"
        feature = candidate.metadata.get("coordinate_context_feature", {})
        if isinstance(feature, dict):
            bucket_v2 = str(feature.get("click_context_bucket_v2", "")).strip()
            if bucket_v2:
                return f"a6|{bucket_v2}"
            bucket_v1 = str(feature.get("click_context_bucket", "")).strip()
            if bucket_v1:
                return f"a6|{bucket_v1}"
        return "a6|na"

    def _candidate_subcluster_id(self, candidate: ActionCandidateV1) -> str:
        metadata_subcluster = str(
            candidate.metadata.get("candidate_subcluster_id", "")
        ).strip()
        if metadata_subcluster:
            return metadata_subcluster
        return self._candidate_cluster_id(candidate)

    def _candidate_transition_stats(
        self,
        candidate: ActionCandidateV1,
    ) -> dict[str, int]:
        raw = candidate.metadata.get("transition_exploration_stats", {})
        if not isinstance(raw, dict):
            raw = {}
        return {
            "state_visit_count": int(raw.get("state_visit_count", 0)),
            "state_action_visit_count": int(raw.get("state_action_visit_count", 0)),
            "state_outgoing_edge_count": int(raw.get("state_outgoing_edge_count", 0)),
        }

    def _candidate_blocked_edge_stats(
        self,
        candidate: ActionCandidateV1,
    ) -> dict[str, float]:
        raw = candidate.metadata.get("blocked_edge_observed_stats", {})
        if not isinstance(raw, dict):
            raw = {}
        return {
            "action_attempts": int(raw.get("action_attempts", 0)),
            "action_blocked_rate": float(raw.get("action_blocked_rate", 0.0)),
            "action_moved_rate": float(raw.get("action_moved_rate", 0.0)),
            "edge_attempts": int(raw.get("edge_attempts", 0)),
            "edge_blocked_rate": float(raw.get("edge_blocked_rate", 0.0)),
            "region_revisit_count_current": int(raw.get("region_revisit_count_current", 0)),
        }

    def _candidate_predicted_region_stats(
        self,
        candidate: ActionCandidateV1,
    ) -> dict[str, Any]:
        raw = candidate.metadata.get("predicted_region_features_v1", {})
        if not isinstance(raw, dict):
            raw = {}
        predicted_region = raw.get("predicted_region", {})
        if not isinstance(predicted_region, dict):
            predicted_region = {}
        current_region = raw.get("current_region", {})
        if not isinstance(current_region, dict):
            current_region = {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "predicted_region_key": str(raw.get("predicted_region_key", "NA")),
            "predicted_region_source": str(
                raw.get("predicted_region_source", "posterior_expected_delta")
            ),
            "predicted_region_visit_count": int(
                max(0, raw.get("predicted_region_visit_count", 0))
            ),
            "current_region_key": str(raw.get("current_region_key", "NA")),
            "current_region_source": str(raw.get("current_region_source", "unknown")),
            "current_region_visit_count": int(
                max(0, raw.get("current_region_visit_count", 0))
            ),
            "known_region_count": int(max(0, raw.get("known_region_count", 0))),
            "region_visit_total": int(max(0, raw.get("region_visit_total", 0))),
            "max_region_visit_count": int(max(0, raw.get("max_region_visit_count", 0))),
            "empirical_transition_total": int(
                max(0, raw.get("empirical_transition_total", 0))
            ),
            "empirical_transition_target_key": str(
                raw.get("empirical_transition_target_key", "NA")
            ),
            "empirical_transition_confidence": self._clamp01(
                float(raw.get("empirical_transition_confidence", 0.0))
            ),
            "empirical_transition_override_applied": bool(
                raw.get("empirical_transition_override_applied", False)
            ),
            "predicted_region_x": int(predicted_region.get("x", -1)),
            "predicted_region_y": int(predicted_region.get("y", -1)),
            "current_region_x": int(current_region.get("x", -1)),
            "current_region_y": int(current_region.get("y", -1)),
            "confidence": self._clamp01(float(raw.get("confidence", 0.0))),
            "edge_attempts": int(max(0, raw.get("edge_attempts", 0))),
            "edge_blocked_rate": self._clamp01(float(raw.get("edge_blocked_rate", 0.0))),
            "dominant_delta_key": str(raw.get("dominant_delta_key", "")),
        }

    def _candidate_frontier_graph_cost(self, candidate: ActionCandidateV1) -> float:
        transition_stats = self._candidate_transition_stats(candidate)
        blocked_stats = self._candidate_blocked_edge_stats(candidate)
        state_visit_count = int(max(0, transition_stats.get("state_visit_count", 0)))
        state_action_visit_count = int(
            max(0, transition_stats.get("state_action_visit_count", 0))
        )
        state_outgoing_edge_count = int(
            max(0, transition_stats.get("state_outgoing_edge_count", 0))
        )
        edge_attempts = int(max(0, blocked_stats.get("edge_attempts", 0)))
        region_revisit_count = int(
            max(0, blocked_stats.get("region_revisit_count_current", 0))
        )
        state_repeat = min(1.0, float(state_visit_count) / 6.0)
        state_action_repeat = min(1.0, float(state_action_visit_count) / 3.0)
        edge_repeat = min(1.0, float(edge_attempts) / 8.0)
        region_repeat = min(
            1.0,
            float(region_revisit_count) / float(max(1, self.region_revisit_hard_threshold)),
        )
        outgoing_repeat = min(1.0, float(state_outgoing_edge_count) / 6.0)
        frontier_cost = (
            0.30 * state_repeat
            + 0.32 * state_action_repeat
            + 0.14 * edge_repeat
            + 0.14 * region_repeat
            + 0.10 * outgoing_repeat
        )
        return float(max(0.0, min(1.0, frontier_cost)))

    def _predicted_direction_distribution(
        self,
        entry: FreeEnergyLedgerEntryV1,
    ) -> dict[str, float]:
        predictive = entry.predictive_signature_distribution
        if not isinstance(predictive, dict):
            return {}
        out: dict[str, float] = {
            "dir_l": 0.0,
            "dir_r": 0.0,
            "dir_u": 0.0,
            "dir_d": 0.0,
        }
        for signature_key, probability in predictive.items():
            signature = str(signature_key)
            if "type=CC_TRANSLATION" not in signature:
                continue
            p = float(probability)
            if p <= 0.0:
                continue
            for direction in ("dir_l", "dir_r", "dir_u", "dir_d"):
                if f"|delta={direction}" in signature:
                    out[direction] = float(out.get(direction, 0.0) + p)
                    break
        return out

    def _dominant_predicted_direction(self, entry: FreeEnergyLedgerEntryV1) -> str:
        distribution = self._predicted_direction_distribution(entry)
        if not distribution:
            return "na"
        direction, mass = max(distribution.items(), key=lambda item: float(item[1]))
        if float(mass) <= 0.0:
            return "na"
        return str(direction)

    def _last_valid_direction(self, recent_navigation_directions: list[str] | None) -> str:
        if not recent_navigation_directions:
            return "na"
        for value in reversed(recent_navigation_directions):
            direction = str(value)
            if direction in ("dir_l", "dir_r", "dir_u", "dir_d"):
                return direction
        return "na"

    def _direction_sequence_key(self, previous_direction: str, next_direction: str) -> str:
        return f"{str(previous_direction)}->{str(next_direction)}"

    def _action6_bucket_probe_needed(self, candidate: ActionCandidateV1) -> bool:
        if int(candidate.action_id) != 6:
            return False
        stats = candidate.metadata.get("click_bucket_observed_stats", {})
        if not isinstance(stats, dict):
            stats = {}
        attempts = int(stats.get("attempts", 0))
        bucket_needed = attempts < int(self.action6_bucket_probe_min_attempts)
        sub_stats = candidate.metadata.get("click_subcluster_observed_stats", {})
        if not isinstance(sub_stats, dict):
            sub_stats = {}
        sub_attempts = int(sub_stats.get("attempts", 0))
        subcluster_needed = (
            sub_attempts < int(self.action6_subcluster_probe_min_attempts)
        )
        return bool(bucket_needed or subcluster_needed)

    def _predicted_probability_mass(
        self,
        entry: FreeEnergyLedgerEntryV1,
        *,
        signature_filter: str,
    ) -> float:
        predictive = entry.predictive_signature_distribution
        if not isinstance(predictive, dict):
            return 0.0
        mass = 0.0
        for signature_key, probability in predictive.items():
            if signature_filter in str(signature_key):
                try:
                    mass += float(probability)
                except Exception:
                    continue
        return float(max(0.0, min(1.0, mass)))

    def _candidate_object_interaction_rank(
        self,
        candidate: ActionCandidateV1,
    ) -> tuple[int, int, int]:
        if int(candidate.action_id) != 6:
            return (2, 2, 2)
        feature = candidate.metadata.get("coordinate_context_feature", {})
        if not isinstance(feature, dict):
            return (1, 1, 1)
        hit_object = int(feature.get("hit_object", -1))
        on_object_boundary_raw = feature.get("on_object_boundary", feature.get("on_boundary", 0))
        on_object_boundary = str(on_object_boundary_raw).strip() in ("1", "true", "True")
        distance_bucket = str(
            feature.get(
                "distance_to_nearest_object_bucket",
                "na",
            )
        )
        distance_rank = {
            "near": 0,
            "mid": 1,
            "far": 2,
        }.get(distance_bucket, 3)
        # Lower is better in sort ordering.
        return (
            0 if hit_object == 1 else 1,
            0 if on_object_boundary else 1,
            int(distance_rank),
        )

    def _weights_for_phase(self, phase: str) -> dict[str, float]:
        defaults = weights_for_phase_v1(phase)
        out = {
            "risk": float(defaults.risk),
            "ambiguity": float(defaults.ambiguity),
            "information_gain_action_semantics": float(
                defaults.information_gain_action_semantics
            ),
            "information_gain_mechanism_dynamics": float(
                defaults.information_gain_mechanism_dynamics
            ),
            "information_gain_causal_mapping": float(
                defaults.information_gain_causal_mapping
            ),
            "action_cost": float(defaults.action_cost),
            "complexity": float(defaults.complexity),
            "vfe": float(defaults.vfe),
        }
        override = self.weight_overrides.get(phase, {})
        for key in out:
            if key not in override:
                continue
            value_any: Any = override[key]
            try:
                out[key] = float(value_any)
            except Exception:
                continue
        if self.ignore_action_cost:
            out["action_cost"] = 0.0
        return out

    def determine_phase(
        self,
        *,
        action_counter: int,
        remaining_budget: int,
        hypothesis_bank: ActiveInferenceHypothesisBankV1,
        explore_steps_override: int | None = None,
    ) -> str:
        explore_steps = int(self.explore_steps)
        if explore_steps_override is not None:
            explore_steps = int(max(1, explore_steps_override))
        return determine_phase_v1(
            action_counter=action_counter,
            posterior_entropy_bits=hypothesis_bank.posterior_entropy(),
            explore_steps=explore_steps,
            exploit_entropy_threshold=self.exploit_entropy_threshold,
            remaining_budget=max(0, int(remaining_budget)),
        )

    def evaluate_candidates(
        self,
        *,
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
        candidates: list[ActionCandidateV1],
        hypothesis_bank: ActiveInferenceHypothesisBankV1,
        phase: str,
    ) -> list[FreeEnergyLedgerEntryV1]:
        base_weights = self._weights_for_phase(phase)
        base_hierarchy_weights = self._hierarchy_weights_for_phase(phase)
        entries: list[FreeEnergyLedgerEntryV1] = []

        for candidate in candidates:
            stats = hypothesis_bank.predictive_statistics(packet, candidate, representation)
            predictive_distribution = stats["predictive_distribution"]
            supports_by_signature = stats["supports_by_signature"]
            expected_mdl_bits = float(stats["expected_mdl_bits"])
            navigation_target = candidate.metadata.get("navigation_target_features_v1", {})
            if not isinstance(navigation_target, dict):
                navigation_target = {}
            preference_distribution = preference_distribution_v1(
                packet,
                phase,
                navigation_target=navigation_target,
            )

            risk_value, risk_terms = compute_risk_kl_v1(
                predictive_distribution,
                preference_distribution,
            )
            ambiguity = hypothesis_bank.expected_ambiguity_from_stats(
                stats,
            )

            ig_split = hypothesis_bank.split_information_gain_from_stats(
                packet,
                candidate,
                representation,
                stats,
            )
            ig_action_semantics = float(ig_split.get("action_semantics", 0.0))
            ig_mechanism_dynamics = float(ig_split.get("mechanism_dynamics", 0.0))
            ig_causal_mapping = float(ig_split.get("causal_mapping", 0.0))
            information_gain_total = (
                ig_action_semantics + ig_mechanism_dynamics + ig_causal_mapping
            )

            action_cost = float(packet.action_cost_per_step)
            complexity_penalty = float(expected_mdl_bits / 64.0)
            vfe_current = float(hypothesis_bank.current_vfe_bits())

            hierarchical_terms = self._hierarchical_efe_terms(
                packet=packet,
                candidate=candidate,
                predictive_distribution=predictive_distribution,
                risk_value=risk_value,
                ambiguity=ambiguity,
                ig_action_semantics=ig_action_semantics,
                ig_mechanism_dynamics=ig_mechanism_dynamics,
                ig_causal_mapping=ig_causal_mapping,
            )
            level1_terms = (
                hierarchical_terms.get("levels", {}).get("level1_operability", {})
            )
            level2_terms = hierarchical_terms.get("levels", {}).get("level2_progress", {})
            weights, hierarchy_weights, adaptive_objective = self._adaptive_objective_profile(
                phase=phase,
                base_weights=base_weights,
                base_hierarchy_weights=base_hierarchy_weights,
                level1_terms=level1_terms,
                level2_terms=level2_terms,
            )
            total_efe = (
                weights["risk"] * risk_value
                + weights["ambiguity"] * ambiguity
                - weights["information_gain_action_semantics"] * ig_action_semantics
                - weights["information_gain_mechanism_dynamics"] * ig_mechanism_dynamics
                - weights["information_gain_causal_mapping"] * ig_causal_mapping
                + weights["action_cost"] * action_cost
                + weights["complexity"] * complexity_penalty
                + weights["vfe"] * vfe_current
            )
            hierarchy_adjustment = (
                hierarchy_weights["progress_risk"] * float(level2_terms.get("risk", 0.0))
                + hierarchy_weights["operability_risk"] * float(level1_terms.get("risk", 0.0))
                + hierarchy_weights["habit_risk"] * float(level2_terms.get("habit_risk", 0.0))
                - hierarchy_weights["progress_information_gain"]
                * float(level2_terms.get("information_gain", 0.0))
            )
            total_efe += float(hierarchy_adjustment)

            witness = {
                "weights": {str(k): float(v) for (k, v) in weights.items()},
                "weights_base_v1": {
                    str(k): float(v) for (k, v) in base_weights.items()
                },
                "hierarchy_weights_v1": {
                    str(k): float(v) for (k, v) in hierarchy_weights.items()
                },
                "hierarchy_weights_base_v1": {
                    str(k): float(v) for (k, v) in base_hierarchy_weights.items()
                },
                "objective_policy_v1": {
                    "ignore_action_cost": bool(self.ignore_action_cost),
                    "applied_action_cost_weight": float(weights["action_cost"]),
                },
                "adaptive_objective_v1": adaptive_objective,
                "risk_terms": {str(k): float(v) for (k, v) in risk_terms.items()},
                "preference_distribution": {
                    str(k): float(v) for (k, v) in preference_distribution.items()
                },
                "supports_by_signature": {
                    str(signature_key): list(hypothesis_ids)
                    for (signature_key, hypothesis_ids) in supports_by_signature.items()
                },
                "expected_mdl_bits": float(expected_mdl_bits),
                "posterior_entropy_bits": float(hypothesis_bank.posterior_entropy()),
                "vfe_current_bits": float(vfe_current),
                "hierarchical_efe_v1": hierarchical_terms,
                "hierarchy_adjustment_v1": {
                    "applied_adjustment_to_total_efe": float(hierarchy_adjustment),
                    "progress_risk_component": float(
                        hierarchy_weights["progress_risk"]
                        * float(level2_terms.get("risk", 0.0))
                    ),
                    "operability_risk_component": float(
                        hierarchy_weights["operability_risk"]
                        * float(level1_terms.get("risk", 0.0))
                    ),
                    "habit_risk_component": float(
                        hierarchy_weights["habit_risk"]
                        * float(level2_terms.get("habit_risk", 0.0))
                    ),
                    "progress_information_gain_component": float(
                        hierarchy_weights["progress_information_gain"]
                        * float(level2_terms.get("information_gain", 0.0))
                    ),
                },
            }

            entries.append(
                FreeEnergyLedgerEntryV1(
                    schema_name="active_inference_free_energy_ledger_entry_v2",
                    schema_version=2,
                    phase=phase,
                    candidate=candidate,
                    risk=float(risk_value),
                    ambiguity=float(ambiguity),
                    information_gain=float(information_gain_total),
                    information_gain_action_semantics=float(ig_action_semantics),
                    information_gain_mechanism_dynamics=float(ig_mechanism_dynamics),
                    information_gain_causal_mapping=float(ig_causal_mapping),
                    action_cost=float(action_cost),
                    complexity_penalty=float(complexity_penalty),
                    vfe_current=float(vfe_current),
                    total_efe=float(total_efe),
                    predictive_signature_distribution={
                        str(k): float(v)
                        for (k, v) in predictive_distribution.items()
                    },
                    witness=witness,
                )
            )

        entries.sort(
            key=lambda entry: (
                float(entry.total_efe),
                int(entry.candidate.action_id),
                int(entry.candidate.y) if entry.candidate.y is not None else -1,
                int(entry.candidate.x) if entry.candidate.x is not None else -1,
                entry.candidate.candidate_id,
            )
        )
        return entries

    def _rollout_score_by_candidate(
        self,
        *,
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
        candidates: list[ActionCandidateV1],
        hypothesis_bank: ActiveInferenceHypothesisBankV1,
        phase: str,
        entries: list[FreeEnergyLedgerEntryV1],
        direction_sequence_visit_count: dict[str, int] | None = None,
        recent_navigation_directions: list[str] | None = None,
    ) -> dict[str, float]:
        rollout_scores: dict[str, float] = {}
        sequence_visit_map = {
            str(key): int(value)
            for (key, value) in (direction_sequence_visit_count or {}).items()
        }
        previous_direction = self._last_valid_direction(recent_navigation_directions)
        if self.rollout_horizon < 2:
            for entry in entries:
                frontier_cost = self._candidate_frontier_graph_cost(entry.candidate)
                current_direction = self._dominant_predicted_direction(entry)
                current_sequence_key = ""
                current_sequence_count = 0
                if (
                    previous_direction in ("dir_l", "dir_r", "dir_u", "dir_d")
                    and current_direction in ("dir_l", "dir_r", "dir_u", "dir_d")
                ):
                    current_sequence_key = self._direction_sequence_key(
                        previous_direction,
                        current_direction,
                    )
                    current_sequence_count = int(
                        sequence_visit_map.get(current_sequence_key, 0)
                    )
                current_sequence_cost = float(
                    min(1.0, float(current_sequence_count) / 12.0)
                )
                rollout_total = float(entry.total_efe) + (
                    self.sequence_rollout_frontier_weight * float(frontier_cost)
                ) + (self.sequence_rollout_direction_weight * float(current_sequence_cost))
                entry.witness["rollout_horizon"] = int(self.rollout_horizon)
                entry.witness["rollout_expected_future_efe"] = 0.0
                entry.witness["rollout_expected_future_frontier_cost"] = 0.0
                entry.witness["rollout_expected_future_sequence_cost"] = 0.0
                entry.witness["rollout_current_frontier_cost"] = float(frontier_cost)
                entry.witness["rollout_current_sequence_key"] = str(current_sequence_key)
                entry.witness["rollout_current_sequence_count"] = int(current_sequence_count)
                entry.witness["rollout_current_sequence_cost"] = float(
                    current_sequence_cost
                )
                entry.witness["rollout_total_efe"] = float(rollout_total)
                rollout_scores[entry.candidate.candidate_id] = float(rollout_total)
            return rollout_scores

        for entry in entries:
            current_frontier_cost = self._candidate_frontier_graph_cost(entry.candidate)
            current_direction = self._dominant_predicted_direction(entry)
            current_sequence_key = ""
            current_sequence_count = 0
            if (
                previous_direction in ("dir_l", "dir_r", "dir_u", "dir_d")
                and current_direction in ("dir_l", "dir_r", "dir_u", "dir_d")
            ):
                current_sequence_key = self._direction_sequence_key(
                    previous_direction,
                    current_direction,
                )
                current_sequence_count = int(sequence_visit_map.get(current_sequence_key, 0))
            current_sequence_cost = float(min(1.0, float(current_sequence_count) / 12.0))
            expected_future_composite = 0.0
            expected_future_frontier_cost = 0.0
            expected_future_sequence_cost = 0.0
            predictive = entry.predictive_signature_distribution
            if predictive:
                for signature_key, probability in predictive.items():
                    p = float(probability)
                    if p <= 0.0:
                        continue
                    next_posterior = hypothesis_bank.posterior_after_signature(
                        packet,
                        entry.candidate,
                        representation,
                        signature_key,
                    )
                    simulated_bank = hypothesis_bank.with_posterior(next_posterior)
                    future_entries = self.evaluate_candidates(
                        packet=packet,
                        representation=representation,
                        candidates=candidates,
                        hypothesis_bank=simulated_bank,
                        phase=phase,
                    )
                    best_future_composite = 0.0
                    best_future_frontier_cost = 0.0
                    best_future_sequence_cost = 0.0
                    if future_entries:
                        best_future_composite = float("inf")
                        for future_entry in future_entries:
                            future_frontier_cost = self._candidate_frontier_graph_cost(
                                future_entry.candidate
                            )
                            future_direction = self._dominant_predicted_direction(future_entry)
                            sequence_pair_cost = 0.0
                            if (
                                current_direction in ("dir_l", "dir_r", "dir_u", "dir_d")
                                and future_direction
                                in ("dir_l", "dir_r", "dir_u", "dir_d")
                            ):
                                sequence_pair_key = self._direction_sequence_key(
                                    current_direction,
                                    future_direction,
                                )
                                sequence_pair_count = int(
                                    sequence_visit_map.get(sequence_pair_key, 0)
                                )
                                sequence_pair_cost = float(
                                    min(1.0, float(sequence_pair_count) / 12.0)
                                )
                            future_composite = (
                                float(future_entry.total_efe)
                                + (
                                    self.sequence_rollout_frontier_weight
                                    * float(future_frontier_cost)
                                )
                                + (
                                    self.sequence_rollout_direction_weight
                                    * float(sequence_pair_cost)
                                )
                            )
                            if future_composite < best_future_composite:
                                best_future_composite = float(future_composite)
                                best_future_frontier_cost = float(future_frontier_cost)
                                best_future_sequence_cost = float(sequence_pair_cost)
                        if best_future_composite == float("inf"):
                            best_future_composite = 0.0
                    expected_future_composite += p * float(best_future_composite)
                    expected_future_frontier_cost += p * float(best_future_frontier_cost)
                    expected_future_sequence_cost += p * float(best_future_sequence_cost)

            rollout_total = (
                float(entry.total_efe)
                + (self.sequence_rollout_frontier_weight * float(current_frontier_cost))
                + (self.sequence_rollout_direction_weight * float(current_sequence_cost))
                + (self.rollout_discount * float(expected_future_composite))
            )
            entry.witness["rollout_horizon"] = int(self.rollout_horizon)
            entry.witness["rollout_expected_future_efe"] = float(
                expected_future_composite
            )
            entry.witness["rollout_expected_future_frontier_cost"] = float(
                expected_future_frontier_cost
            )
            entry.witness["rollout_expected_future_sequence_cost"] = float(
                expected_future_sequence_cost
            )
            entry.witness["rollout_current_frontier_cost"] = float(current_frontier_cost)
            entry.witness["rollout_current_sequence_key"] = str(current_sequence_key)
            entry.witness["rollout_current_sequence_count"] = int(current_sequence_count)
            entry.witness["rollout_current_sequence_cost"] = float(
                current_sequence_cost
            )
            entry.witness["rollout_total_efe"] = float(rollout_total)
            rollout_scores[entry.candidate.candidate_id] = float(rollout_total)

        return rollout_scores

    def select_action(
        self,
        *,
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
        candidates: list[ActionCandidateV1],
        hypothesis_bank: ActiveInferenceHypothesisBankV1,
        phase: str,
        remaining_budget: int,
        action_select_count: dict[int, int] | None = None,
        candidate_select_count: dict[str, int] | None = None,
        cluster_select_count: dict[str, int] | None = None,
        subcluster_select_count: dict[str, int] | None = None,
        early_probe_budget_remaining: int = 0,
        no_change_streak: int = 0,
        stagnation_streak: int = 0,
        direction_sequence_visit_count: dict[str, int] | None = None,
        direction_visit_count: dict[str, int] | None = None,
        recent_navigation_directions: list[str] | None = None,
    ) -> tuple[ActionCandidateV1, list[FreeEnergyLedgerEntryV1]]:
        entries = self.evaluate_candidates(
            packet=packet,
            representation=representation,
            candidates=candidates,
            hypothesis_bank=hypothesis_bank,
            phase=phase,
        )
        if not entries:
            fallback = ActionCandidateV1(
                candidate_id="fallback_reset",
                action_id=0,
                source="policy/no_candidates",
            )
            return fallback, []

        selection_metric = "total_efe"
        rollout_applied = False
        rollout_skip_reason = "disabled"
        selection_score_by_candidate: dict[str, float] = {
            entry.candidate.candidate_id: float(entry.total_efe) for entry in entries
        }
        rollout_eligible = True
        if int(remaining_budget) < 2:
            rollout_eligible = False
            rollout_skip_reason = "remaining_budget_lt_2"
        elif int(self.rollout_horizon) < 2:
            rollout_eligible = False
            rollout_skip_reason = "rollout_horizon_lt_2"
        elif self.rollout_only_in_exploit and str(phase) != "exploit":
            rollout_eligible = False
            rollout_skip_reason = "phase_not_exploit"
        elif len(entries) > int(self.rollout_max_candidates):
            rollout_eligible = False
            rollout_skip_reason = "candidate_count_exceeds_limit"
        if rollout_eligible:
            rollout_scores = self._rollout_score_by_candidate(
                packet=packet,
                representation=representation,
                candidates=candidates,
                hypothesis_bank=hypothesis_bank,
                phase=phase,
                entries=entries,
                direction_sequence_visit_count=direction_sequence_visit_count,
                recent_navigation_directions=recent_navigation_directions,
            )
            selection_metric = "rollout_total_efe"
            rollout_applied = True
            rollout_skip_reason = ""
            selection_score_by_candidate = {
                entry.candidate.candidate_id: float(
                    rollout_scores.get(entry.candidate.candidate_id, entry.total_efe)
                )
                for entry in entries
            }
            entries.sort(
                key=lambda entry: (
                    float(selection_score_by_candidate.get(entry.candidate.candidate_id, entry.total_efe)),
                    float(entry.total_efe),
                    int(entry.candidate.action_id),
                    entry.candidate.candidate_id,
                )
            )

        action_count_map = action_select_count or {}
        candidate_count_map = candidate_select_count or {}
        cluster_count_map = cluster_select_count or {}
        subcluster_count_map = subcluster_select_count or {}
        direction_sequence_count_map = direction_sequence_visit_count or {}
        direction_count_map = direction_visit_count or {}
        previous_navigation_direction = self._last_valid_direction(
            recent_navigation_directions
        )
        cluster_id_by_candidate_id: dict[str, str] = {
            str(entry.candidate.candidate_id): self._candidate_cluster_id(entry.candidate)
            for entry in entries
        }
        subcluster_id_by_candidate_id: dict[str, str] = {
            str(entry.candidate.candidate_id): self._candidate_subcluster_id(
                entry.candidate
            )
            for entry in entries
        }
        best_score = float(
            selection_score_by_candidate.get(
                entries[0].candidate.candidate_id,
                entries[0].total_efe,
            )
        )
        second_score = (
            float(
                selection_score_by_candidate.get(
                    entries[1].candidate.candidate_id,
                    entries[1].total_efe,
                )
            )
            if len(entries) >= 2
            else best_score
        )
        tie_group_size = int(
            sum(
                1
                for entry in entries
                if abs(
                    float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    )
                    - best_score
                )
                <= self.tie_epsilon
            )
        )
        selected_entry = entries[0]
        least_tried_probe_applied = False
        exploit_action6_bucket_probe_applied = False
        near_tie_probe_applied = False
        tie_probe_candidates = []
        near_tie_probe_candidates = []
        early_probe_applied = False
        early_probe_target_action_ids: list[int] = []
        early_probe_candidate_pool: list[FreeEnergyLedgerEntryV1] = []
        early_probe_min_action_usage = 0
        coverage_region_probe_applied = False
        coverage_region_probe_candidates: list[dict[str, Any]] = []
        coverage_sweep_active = False
        coverage_sweep_reason = "inactive"
        coverage_known_region_count = 0
        coverage_target_region_count = int(self.coverage_sweep_target_regions)
        coverage_periodic_resweep_active = False
        coverage_score_margin_used = 0.0
        coverage_sweep_pattern = "disabled"
        coverage_sweep_target_region = {"x": -1, "y": -1}
        coverage_sweep_target_direction = "na"
        direction_sequence_probe_applied = False
        direction_sequence_probe_candidates: list[dict[str, Any]] = []

        early_probe_active = bool(
            int(early_probe_budget_remaining) > 0 and phase in ("explore", "explain")
        )
        if early_probe_active:
            available_action_ids = sorted(
                {
                    int(entry.candidate.action_id)
                    for entry in entries
                    if int(entry.candidate.action_id) in set(int(v) for v in packet.available_actions)
                }
            )
            if available_action_ids:
                early_probe_min_action_usage = min(
                    int(action_count_map.get(action_id, 0))
                    for action_id in available_action_ids
                )
                early_probe_target_action_ids = [
                    int(action_id)
                    for action_id in available_action_ids
                    if int(action_count_map.get(action_id, 0)) == early_probe_min_action_usage
                ]
                early_probe_candidate_pool = [
                    entry
                    for entry in entries
                    if int(entry.candidate.action_id) in set(early_probe_target_action_ids)
                ]
                if early_probe_candidate_pool:
                    early_probe_candidate_pool.sort(
                        key=lambda entry: (
                            float(
                                selection_score_by_candidate.get(
                                    entry.candidate.candidate_id,
                                    entry.total_efe,
                                )
                            ),
                            int(
                                self._candidate_transition_stats(entry.candidate).get(
                                    "state_action_visit_count",
                                    0,
                                )
                            ),
                            int(
                                self._candidate_transition_stats(entry.candidate).get(
                                    "state_visit_count",
                                    0,
                                )
                            ),
                            int(
                                self._candidate_transition_stats(entry.candidate).get(
                                    "state_outgoing_edge_count",
                                    0,
                                )
                            ),
                            int(action_count_map.get(int(entry.candidate.action_id), 0)),
                            int(
                                cluster_count_map.get(
                                    str(
                                        cluster_id_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            self._candidate_cluster_id(entry.candidate),
                                        )
                                    ),
                                    0,
                                )
                            ),
                            int(
                                subcluster_count_map.get(
                                    str(
                                        subcluster_id_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            self._candidate_subcluster_id(entry.candidate),
                                        )
                                    ),
                                    0,
                                )
                            ),
                            int(self._candidate_object_interaction_rank(entry.candidate)[0]),
                            int(self._candidate_object_interaction_rank(entry.candidate)[1]),
                            int(self._candidate_object_interaction_rank(entry.candidate)[2]),
                            int(candidate_count_map.get(str(entry.candidate.candidate_id), 0)),
                            int(entry.candidate.action_id),
                            str(entry.candidate.candidate_id),
                        )
                    )
                    selected_entry = early_probe_candidate_pool[0]
                    early_probe_applied = True
                    if selected_entry is not entries[0]:
                        entries.remove(selected_entry)
                        entries.insert(0, selected_entry)

        coverage_phase_allowed = bool(
            phase in ("explore", "explain")
            or (self.coverage_sweep_force_in_exploit and phase == "exploit")
        )
        if (
            not early_probe_applied
            and coverage_phase_allowed
            and int(packet.levels_completed) <= 0
            and len(entries) >= 2
        ):
            navigation_entries = [
                entry
                for entry in entries
                if int(entry.candidate.action_id) in (1, 2, 3, 4)
                and bool(
                    self._candidate_predicted_region_stats(entry.candidate).get(
                        "enabled",
                        False,
                    )
                )
            ]
            if len(navigation_entries) >= 2:
                predicted_stats_by_candidate_id: dict[str, dict[str, Any]] = {
                    str(entry.candidate.candidate_id): self._candidate_predicted_region_stats(
                        entry.candidate
                    )
                    for entry in navigation_entries
                }
                coverage_known_region_count = int(
                    max(
                        (
                            int(stats.get("known_region_count", 0))
                            for stats in predicted_stats_by_candidate_id.values()
                        ),
                        default=0,
                    )
                )
                coverage_goal_unmet = bool(
                    coverage_known_region_count < int(self.coverage_sweep_target_regions)
                )
                periodic_window_active = False
                if (
                    int(self.coverage_resweep_interval) > 0
                    and int(self.coverage_resweep_span) > 0
                ):
                    periodic_slot = int(packet.action_counter) % int(
                        self.coverage_resweep_interval
                    )
                    periodic_window_active = bool(
                        periodic_slot
                        < int(
                            min(
                                self.coverage_resweep_interval,
                                self.coverage_resweep_span,
                            )
                        )
                    )
                coverage_periodic_resweep_active = bool(periodic_window_active)
                coverage_sweep_active = bool(coverage_goal_unmet or periodic_window_active)
                if coverage_goal_unmet:
                    coverage_sweep_reason = "goal_unmet"
                elif periodic_window_active:
                    coverage_sweep_reason = "periodic_resweep"
                else:
                    coverage_sweep_reason = "goal_reached"
                if coverage_sweep_active:
                    target_rx = -1
                    target_ry = -1
                    target_direction = "na"
                    if self.coverage_matrix_sweep_enabled:
                        coverage_sweep_pattern = "row_serpentine_up"
                        sample_stats = predicted_stats_by_candidate_id.get(
                            str(navigation_entries[0].candidate.candidate_id),
                            {},
                        )
                        current_rx = int(sample_stats.get("current_region_x", -1))
                        current_ry = int(sample_stats.get("current_region_y", -1))
                        if current_rx >= 0 and current_ry >= 0:
                            target_rx = int(current_rx)
                            target_ry = int(current_ry)
                            row_sweeps_left = bool(current_ry % 2 == 1)
                            if row_sweeps_left:
                                if current_rx > 0:
                                    target_rx = int(current_rx - 1)
                                    target_direction = "dir_l"
                                elif current_ry > 0:
                                    target_ry = int(current_ry - 1)
                                    target_direction = "dir_u"
                            else:
                                if current_rx < 7:
                                    target_rx = int(current_rx + 1)
                                    target_direction = "dir_r"
                                elif current_ry > 0:
                                    target_ry = int(current_ry - 1)
                                    target_direction = "dir_u"
                    coverage_sweep_target_region = {
                        "x": int(target_rx),
                        "y": int(target_ry),
                    }
                    coverage_sweep_target_direction = str(target_direction)
                    best_navigation_score = min(
                        float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        )
                        for entry in navigation_entries
                    )
                    coverage_margin = float(
                        max(
                            self.coverage_sweep_score_margin,
                            self.stagnation_probe_score_margin,
                        )
                    )
                    coverage_score_margin_used = float(coverage_margin)
                    coverage_pool = [
                        entry
                        for entry in navigation_entries
                        if (
                            float(
                                selection_score_by_candidate.get(
                                    entry.candidate.candidate_id,
                                    entry.total_efe,
                                )
                            )
                            - best_navigation_score
                        )
                        <= coverage_margin
                    ]
                    if len(coverage_pool) < 2:
                        coverage_pool = sorted(
                            navigation_entries,
                            key=lambda entry: (
                                float(
                                    selection_score_by_candidate.get(
                                        entry.candidate.candidate_id,
                                        entry.total_efe,
                                    )
                                ),
                                int(entry.candidate.action_id),
                                str(entry.candidate.candidate_id),
                            ),
                        )[: min(4, len(navigation_entries))]
                    if coverage_pool:
                        coverage_pool.sort(
                            key=lambda entry: (
                                0
                                if (
                                    str(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("predicted_region_key", "NA")
                                    )
                                    != str(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("current_region_key", "NA")
                                    )
                                    and str(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("predicted_region_key", "NA")
                                    )
                                    != "NA"
                                )
                                else 1,
                                0
                                if (
                                    str(coverage_sweep_target_direction)
                                    in ("dir_l", "dir_r", "dir_u", "dir_d")
                                    and self._dominant_predicted_direction(entry)
                                    == str(coverage_sweep_target_direction)
                                    and int(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("edge_attempts", 0)
                                    )
                                    < int(self.coverage_sweep_direction_retry_limit)
                                )
                                else 1,
                                int(
                                    (
                                        abs(
                                            int(
                                                predicted_stats_by_candidate_id.get(
                                                    str(entry.candidate.candidate_id),
                                                    {},
                                                ).get("predicted_region_x", -1)
                                            )
                                            - int(target_rx)
                                        )
                                        + abs(
                                            int(
                                                predicted_stats_by_candidate_id.get(
                                                    str(entry.candidate.candidate_id),
                                                    {},
                                                ).get("predicted_region_y", -1)
                                            )
                                            - int(target_ry)
                                        )
                                    )
                                    if (
                                        int(target_rx) >= 0
                                        and int(target_ry) >= 0
                                        and int(
                                            predicted_stats_by_candidate_id.get(
                                                str(entry.candidate.candidate_id),
                                                {},
                                            ).get("predicted_region_x", -1)
                                        )
                                        >= 0
                                        and int(
                                            predicted_stats_by_candidate_id.get(
                                                str(entry.candidate.candidate_id),
                                                {},
                                            ).get("predicted_region_y", -1)
                                        )
                                        >= 0
                                    )
                                    else 10**6
                                ),
                                int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("predicted_region_visit_count", 10**6)
                                ),
                                int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("current_region_visit_count", 10**6)
                                ),
                                0
                                if str(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("predicted_region_source", "")
                                )
                                == "empirical_transition"
                                else 1,
                                -float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_confidence", 0.0)
                                ),
                                -int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_total", 0)
                                ),
                                int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("edge_attempts", 10**6)
                                ),
                                float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("edge_blocked_rate", 1.0)
                                ),
                                -float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("confidence", 0.0)
                                ),
                                int(action_count_map.get(int(entry.candidate.action_id), 0)),
                                float(
                                    selection_score_by_candidate.get(
                                        entry.candidate.candidate_id,
                                        entry.total_efe,
                                    )
                                ),
                                int(entry.candidate.action_id),
                                str(entry.candidate.candidate_id),
                            )
                        )
                        selected_entry = coverage_pool[0]
                        least_tried_probe_applied = True
                        coverage_region_probe_applied = True
                        if selected_entry is not entries[0]:
                            entries.remove(selected_entry)
                            entries.insert(0, selected_entry)
                        coverage_region_probe_candidates = [
                            {
                                "candidate_id": str(entry.candidate.candidate_id),
                                "action_id": int(entry.candidate.action_id),
                                "score": float(
                                    selection_score_by_candidate.get(
                                        entry.candidate.candidate_id,
                                        entry.total_efe,
                                    )
                                ),
                                "current_region_key": str(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("current_region_key", "NA")
                                ),
                                "predicted_region_key": str(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("predicted_region_key", "NA")
                                ),
                                "predicted_region_source": str(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get(
                                        "predicted_region_source",
                                        "posterior_expected_delta",
                                    )
                                ),
                                "predicted_region_visit_count": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("predicted_region_visit_count", 0)
                                ),
                                "current_region_visit_count": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("current_region_visit_count", 0)
                                ),
                                "known_region_count": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("known_region_count", 0)
                                ),
                                "empirical_transition_target_key": str(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_target_key", "NA")
                                ),
                                "empirical_transition_total": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_total", 0)
                                ),
                                "empirical_transition_confidence": float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_confidence", 0.0)
                                ),
                                "empirical_transition_override_applied": bool(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_override_applied", False)
                                ),
                                "current_region_source": str(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("current_region_source", "unknown")
                                ),
                                "confidence": float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("confidence", 0.0)
                                ),
                                "predicted_direction": str(
                                    self._dominant_predicted_direction(entry)
                                ),
                                "target_direction_match": bool(
                                    str(coverage_sweep_target_direction)
                                    in ("dir_l", "dir_r", "dir_u", "dir_d")
                                    and self._dominant_predicted_direction(entry)
                                    == str(coverage_sweep_target_direction)
                                ),
                                "target_direction_retry_exceeded": bool(
                                    int(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("edge_attempts", 0)
                                    )
                                    >= int(self.coverage_sweep_direction_retry_limit)
                                ),
                                "coverage_sweep_direction_retry_limit": int(
                                    self.coverage_sweep_direction_retry_limit
                                ),
                                "target_region_distance": int(
                                    (
                                        abs(
                                            int(
                                                predicted_stats_by_candidate_id.get(
                                                    str(entry.candidate.candidate_id),
                                                    {},
                                                ).get("predicted_region_x", -1)
                                            )
                                            - int(target_rx)
                                        )
                                        + abs(
                                            int(
                                                predicted_stats_by_candidate_id.get(
                                                    str(entry.candidate.candidate_id),
                                                    {},
                                                ).get("predicted_region_y", -1)
                                            )
                                            - int(target_ry)
                                        )
                                    )
                                    if (
                                        int(target_rx) >= 0
                                        and int(target_ry) >= 0
                                        and int(
                                            predicted_stats_by_candidate_id.get(
                                                str(entry.candidate.candidate_id),
                                                {},
                                            ).get("predicted_region_x", -1)
                                        )
                                        >= 0
                                        and int(
                                            predicted_stats_by_candidate_id.get(
                                                str(entry.candidate.candidate_id),
                                                {},
                                            ).get("predicted_region_y", -1)
                                        )
                                        >= 0
                                    )
                                    else -1
                                ),
                                "edge_attempts": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("edge_attempts", 0)
                                ),
                                "edge_blocked_rate": float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("edge_blocked_rate", 0.0)
                                ),
                            }
                            for entry in coverage_pool[:10]
                        ]

        if tie_group_size > 1 and not early_probe_applied and not least_tried_probe_applied:
            tie_probe_candidates = entries[:tie_group_size]
            # In exploration/explanation phases, break score ties by probing least-tried actions.
            exploit_action6_probe_needed = bool(
                phase == "exploit"
                and any(
                    self._action6_bucket_probe_needed(entry.candidate)
                    for entry in tie_probe_candidates
                )
            )
            if phase in ("explore", "explain") or exploit_action6_probe_needed:
                tie_probe_candidates.sort(
                    key=lambda entry: (
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_action_visit_count",
                                0,
                            )
                        ),
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_visit_count",
                                0,
                            )
                        ),
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_outgoing_edge_count",
                                0,
                            )
                        ),
                        int(action_count_map.get(int(entry.candidate.action_id), 0)),
                        int(
                            cluster_count_map.get(
                                str(
                                    cluster_id_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        self._candidate_cluster_id(entry.candidate),
                                    )
                                ),
                                0,
                            )
                        ),
                        int(
                            subcluster_count_map.get(
                                str(
                                    subcluster_id_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        self._candidate_subcluster_id(entry.candidate),
                                    )
                                ),
                                0,
                            )
                        ),
                        int(self._candidate_object_interaction_rank(entry.candidate)[0]),
                        int(self._candidate_object_interaction_rank(entry.candidate)[1]),
                        int(self._candidate_object_interaction_rank(entry.candidate)[2]),
                        int(candidate_count_map.get(str(entry.candidate.candidate_id), 0)),
                        int(entry.candidate.action_id),
                        str(entry.candidate.candidate_id),
                    )
                )
                selected_entry = tie_probe_candidates[0]
                least_tried_probe_applied = True
                exploit_action6_bucket_probe_applied = bool(exploit_action6_probe_needed)
                if selected_entry is not entries[0]:
                    entries.remove(selected_entry)
                    entries.insert(0, selected_entry)
        elif phase == "exploit":
            best_entry = entries[0]
            best_entry_score = float(
                selection_score_by_candidate.get(
                    best_entry.candidate.candidate_id,
                    best_entry.total_efe,
                )
            )
            near_tie_probe_candidates = [
                entry
                for entry in entries
                if int(entry.candidate.action_id) == 6
                and (
                    float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    )
                    - best_entry_score
                )
                <= float(self.action6_probe_score_margin)
            ]
            need_probe = any(
                self._action6_bucket_probe_needed(entry.candidate)
                for entry in near_tie_probe_candidates
            )
            if len(near_tie_probe_candidates) > 1 and need_probe:
                near_tie_probe_candidates.sort(
                    key=lambda entry: (
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_action_visit_count",
                                0,
                            )
                        ),
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_visit_count",
                                0,
                            )
                        ),
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_outgoing_edge_count",
                                0,
                            )
                        ),
                        int(action_count_map.get(int(entry.candidate.action_id), 0)),
                        int(
                            cluster_count_map.get(
                                str(
                                    cluster_id_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        self._candidate_cluster_id(entry.candidate),
                                    )
                                ),
                                0,
                            )
                        ),
                        int(
                            subcluster_count_map.get(
                                str(
                                    subcluster_id_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        self._candidate_subcluster_id(entry.candidate),
                                    )
                                ),
                                0,
                            )
                        ),
                        int(self._candidate_object_interaction_rank(entry.candidate)[0]),
                        int(self._candidate_object_interaction_rank(entry.candidate)[1]),
                        int(self._candidate_object_interaction_rank(entry.candidate)[2]),
                        int(candidate_count_map.get(str(entry.candidate.candidate_id), 0)),
                        float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        ),
                        str(entry.candidate.candidate_id),
                    )
                )
                selected_entry = near_tie_probe_candidates[0]
                near_tie_probe_applied = True
                least_tried_probe_applied = True
                exploit_action6_bucket_probe_applied = True
                if selected_entry is not entries[0]:
                    entries.remove(selected_entry)
                    entries.insert(0, selected_entry)

        action6_only_stagnation_probe_applied = False
        action6_only_probe_candidates: list[FreeEnergyLedgerEntryV1] = []
        available_action_set = {int(v) for v in packet.available_actions}
        action6_only_space = bool(available_action_set) and available_action_set == {6}
        action6_stagnated = bool(
            int(packet.levels_completed) <= 0
            and (
                int(no_change_streak) >= 2
                or int(packet.action_counter) >= int(self.action6_stagnation_step_threshold)
            )
        )
        if (
            not early_probe_applied
            and not least_tried_probe_applied
            and action6_only_space
            and action6_stagnated
            and int(early_probe_budget_remaining) <= 0
            and len(entries) > 1
        ):
            best_entry = entries[0]
            best_entry_score = float(
                selection_score_by_candidate.get(
                    best_entry.candidate.candidate_id,
                    best_entry.total_efe,
                )
            )
            probe_margin = max(
                float(self.action6_probe_score_margin),
                float(self.action6_explore_probe_score_margin),
            )
            action6_only_probe_candidates = [
                entry
                for entry in entries
                if int(entry.candidate.action_id) == 6
                and (
                    float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    )
                    - best_entry_score
                )
                <= probe_margin
            ]
            need_probe = any(
                self._action6_bucket_probe_needed(entry.candidate)
                for entry in action6_only_probe_candidates
            )
            if len(action6_only_probe_candidates) > 1 and need_probe:
                action6_only_probe_candidates.sort(
                    key=lambda entry: (
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_action_visit_count",
                                0,
                            )
                        ),
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_visit_count",
                                0,
                            )
                        ),
                        int(
                            self._candidate_transition_stats(entry.candidate).get(
                                "state_outgoing_edge_count",
                                0,
                            )
                        ),
                        int(self._candidate_object_interaction_rank(entry.candidate)[0]),
                        int(self._candidate_object_interaction_rank(entry.candidate)[1]),
                        int(self._candidate_object_interaction_rank(entry.candidate)[2]),
                        int(
                            cluster_count_map.get(
                                str(
                                    cluster_id_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        self._candidate_cluster_id(entry.candidate),
                                    )
                                ),
                                0,
                            )
                        ),
                        int(
                            subcluster_count_map.get(
                                str(
                                    subcluster_id_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        self._candidate_subcluster_id(entry.candidate),
                                    )
                                ),
                                0,
                            )
                        ),
                        int(candidate_count_map.get(str(entry.candidate.candidate_id), 0)),
                        float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        ),
                        str(entry.candidate.candidate_id),
                    )
                )
                selected_entry = action6_only_probe_candidates[0]
                action6_only_stagnation_probe_applied = True
                least_tried_probe_applied = True
                if selected_entry is not entries[0]:
                    entries.remove(selected_entry)
                    entries.insert(0, selected_entry)

        if (
            not early_probe_applied
            and not least_tried_probe_applied
            and str(phase) == "exploit"
            and int(packet.levels_completed) <= 0
            and int(stagnation_streak) >= int(self.sequence_probe_trigger_steps)
            and len(entries) >= 2
        ):
            navigation_sequence_pool = [
                entry
                for entry in entries
                if int(entry.candidate.action_id) in (1, 2, 3, 4)
            ]
            if len(navigation_sequence_pool) >= 2:
                best_navigation_score = min(
                    float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    )
                    for entry in navigation_sequence_pool
                )
                sequence_pool = [
                    entry
                    for entry in navigation_sequence_pool
                    if (
                        float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        )
                        - best_navigation_score
                    )
                    <= float(
                        max(
                            self.stagnation_probe_score_margin,
                            self.sequence_probe_score_margin,
                        )
                    )
                ]
                if len(sequence_pool) < 2:
                    sequence_pool = sorted(
                        navigation_sequence_pool,
                        key=lambda entry: (
                            float(
                                selection_score_by_candidate.get(
                                    entry.candidate.candidate_id,
                                    entry.total_efe,
                                )
                            ),
                            int(entry.candidate.action_id),
                            str(entry.candidate.candidate_id),
                        ),
                    )[: min(4, len(navigation_sequence_pool))]
                scored_sequence_pool: list[dict[str, Any]] = []
                for entry in sequence_pool:
                    predicted_direction = self._dominant_predicted_direction(entry)
                    if predicted_direction not in ("dir_l", "dir_r", "dir_u", "dir_d"):
                        continue
                    transition_stats = self._candidate_transition_stats(entry.candidate)
                    predicted_sequence_key = ""
                    predicted_sequence_count = 0
                    if previous_navigation_direction in (
                        "dir_l",
                        "dir_r",
                        "dir_u",
                        "dir_d",
                    ):
                        predicted_sequence_key = self._direction_sequence_key(
                            previous_navigation_direction,
                            predicted_direction,
                        )
                        predicted_sequence_count = int(
                            direction_sequence_count_map.get(predicted_sequence_key, 0)
                        )
                    predicted_direction_count = int(
                        direction_count_map.get(predicted_direction, 0)
                    )
                    frontier_cost = float(
                        self._candidate_frontier_graph_cost(entry.candidate)
                    )
                    scored_sequence_pool.append(
                        {
                            "entry": entry,
                            "predicted_direction": str(predicted_direction),
                            "predicted_sequence_key": str(predicted_sequence_key),
                            "predicted_sequence_count": int(predicted_sequence_count),
                            "predicted_direction_count": int(predicted_direction_count),
                            "frontier_graph_cost": float(frontier_cost),
                            "state_action_visit_count": int(
                                transition_stats.get("state_action_visit_count", 0)
                            ),
                            "score": float(
                                selection_score_by_candidate.get(
                                    entry.candidate.candidate_id,
                                    entry.total_efe,
                                )
                            ),
                            "action_usage_count_before": int(
                                action_count_map.get(int(entry.candidate.action_id), 0)
                            ),
                        }
                    )
                if scored_sequence_pool:
                    scored_sequence_pool.sort(
                        key=lambda row: (
                            int(row.get("predicted_sequence_count", 0)),
                            int(row.get("predicted_direction_count", 0)),
                            int(row.get("state_action_visit_count", 0)),
                            int(row.get("action_usage_count_before", 0)),
                            float(row.get("frontier_graph_cost", 1.0)),
                            float(row.get("score", 0.0)),
                            int(
                                (row.get("entry") or entries[0]).candidate.action_id
                            ),
                            str((row.get("entry") or entries[0]).candidate.candidate_id),
                        )
                    )
                    best_row = scored_sequence_pool[0]
                    selected_entry = best_row["entry"]
                    direction_sequence_probe_applied = True
                    least_tried_probe_applied = True
                    if selected_entry is not entries[0]:
                        entries.remove(selected_entry)
                        entries.insert(0, selected_entry)
                    direction_sequence_probe_candidates = [
                        {
                            "candidate_id": str(
                                row["entry"].candidate.candidate_id
                            ),
                            "action_id": int(row["entry"].candidate.action_id),
                            "predicted_direction": str(
                                row.get("predicted_direction", "na")
                            ),
                            "predicted_sequence_key": str(
                                row.get("predicted_sequence_key", "")
                            ),
                            "predicted_sequence_count": int(
                                row.get("predicted_sequence_count", 0)
                            ),
                            "predicted_direction_count": int(
                                row.get("predicted_direction_count", 0)
                            ),
                            "frontier_graph_cost": float(
                                row.get("frontier_graph_cost", 0.0)
                            ),
                            "state_action_visit_count": int(
                                row.get("state_action_visit_count", 0)
                            ),
                            "action_usage_count_before": int(
                                row.get("action_usage_count_before", 0)
                            ),
                            "score": float(row.get("score", 0.0)),
                        }
                        for row in scored_sequence_pool[:12]
                    ]

        navigation_stagnation_probe_applied = False
        navigation_probe_candidates: list[FreeEnergyLedgerEntryV1] = []
        navigation_stagnated = bool(
            int(stagnation_streak) >= int(self.stagnation_probe_trigger_steps)
            and int(packet.levels_completed) <= 0
        )
        navigation_actions_available = sorted(
            {int(v) for v in packet.available_actions if int(v) in (1, 2, 3, 4)}
        )
        navigation_usage_gap = 0
        if navigation_actions_available:
            usage_values = [
                int(action_count_map.get(int(action_id), 0))
                for action_id in navigation_actions_available
            ]
            if usage_values:
                navigation_usage_gap = int(max(usage_values) - min(usage_values))
        if (
            not early_probe_applied
            and not least_tried_probe_applied
            and str(phase) == "exploit"
            and navigation_stagnated
            and len(navigation_actions_available) >= 2
            and (
                int(navigation_usage_gap) >= int(self.stagnation_probe_min_action_usage_gap)
                or int(stagnation_streak)
                >= int(self.stagnation_probe_trigger_steps + 10)
            )
        ):
            navigation_candidates_all = [
                entry
                for entry in entries
                if int(entry.candidate.action_id) in (1, 2, 3, 4)
            ]
            if len(navigation_candidates_all) >= 2:
                best_navigation_score = min(
                    float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    )
                    for entry in navigation_candidates_all
                )
                navigation_probe_candidates = [
                    entry
                    for entry in navigation_candidates_all
                    if (
                        float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        )
                        - best_navigation_score
                    )
                    <= float(self.stagnation_probe_score_margin)
                ]
                if len(navigation_probe_candidates) < 2:
                    navigation_probe_candidates = sorted(
                        navigation_candidates_all,
                        key=lambda entry: (
                            float(
                                selection_score_by_candidate.get(
                                    entry.candidate.candidate_id,
                                    entry.total_efe,
                                )
                            ),
                            int(entry.candidate.action_id),
                            str(entry.candidate.candidate_id),
                        ),
                    )[: min(3, len(navigation_candidates_all))]
                if len(navigation_probe_candidates) >= 2:
                    navigation_probe_candidates.sort(
                        key=lambda entry: (
                            int(action_count_map.get(int(entry.candidate.action_id), 0)),
                            int(
                                self._candidate_transition_stats(entry.candidate).get(
                                    "state_action_visit_count",
                                    0,
                                )
                            ),
                            int(
                                self._candidate_transition_stats(entry.candidate).get(
                                    "state_visit_count",
                                    0,
                                )
                            ),
                            int(
                                self._candidate_transition_stats(entry.candidate).get(
                                    "state_outgoing_edge_count",
                                    0,
                                )
                            ),
                            float(
                                self._predicted_probability_mass(
                                    entry,
                                    signature_filter="|delta=blocked",
                                )
                            ),
                            float(
                                self._predicted_probability_mass(
                                    entry,
                                    signature_filter="type=NO_CHANGE|progress=0",
                                )
                            ),
                            float(
                                selection_score_by_candidate.get(
                                    entry.candidate.candidate_id,
                                    entry.total_efe,
                                )
                            ),
                            int(entry.candidate.action_id),
                            str(entry.candidate.candidate_id),
                        )
                    )
                    selected_entry = navigation_probe_candidates[0]
                    navigation_stagnation_probe_applied = True
                    least_tried_probe_applied = True
                    if selected_entry is not entries[0]:
                        entries.remove(selected_entry)
                        entries.insert(0, selected_entry)

        selected_action_usage_count = int(
            action_count_map.get(int(entries[0].candidate.action_id), 0)
        )
        selected_cluster_id = str(
            cluster_id_by_candidate_id.get(
                str(entries[0].candidate.candidate_id),
                self._candidate_cluster_id(entries[0].candidate),
            )
        )
        selected_cluster_usage_count = int(cluster_count_map.get(selected_cluster_id, 0))
        selected_subcluster_id = str(
            subcluster_id_by_candidate_id.get(
                str(entries[0].candidate.candidate_id),
                self._candidate_subcluster_id(entries[0].candidate),
            )
        )
        selected_subcluster_usage_count = int(
            subcluster_count_map.get(selected_subcluster_id, 0)
        )
        selected_candidate_usage_count = int(
            candidate_count_map.get(str(entries[0].candidate.candidate_id), 0)
        )
        selected_transition_stats = self._candidate_transition_stats(entries[0].candidate)
        if early_probe_applied:
            tie_breaker_rule_applied = "early_probe_budget_least_tried"
        elif coverage_region_probe_applied:
            tie_breaker_rule_applied = (
                "coverage_region_probe_least_visited(predicted_next_region,edge_attempts,confidence,action_usage,score)"
            )
        elif direction_sequence_probe_applied:
            tie_breaker_rule_applied = (
                "exploit_direction_sequence_probe_least_visited(sequence_count,direction_count,state_action,action_usage,frontier,score)"
            )
        elif navigation_stagnation_probe_applied:
            tie_breaker_rule_applied = (
                "navigation_stagnation_probe_least_tried(action_usage,state_action,state,edge,blocked,nochange,score)"
            )
        elif action6_only_stagnation_probe_applied:
            tie_breaker_rule_applied = (
                "action6_only_stagnation_probe_least_tried(state_action,state,edge,cluster,subcluster,candidate,score)"
            )
        elif tie_group_size > 1:
            if least_tried_probe_applied:
                if exploit_action6_bucket_probe_applied:
                    tie_breaker_rule_applied = (
                        "exploit_action6_bucket_probe_least_tried(action_usage,cluster_usage,subcluster_usage,candidate_usage)"
                    )
                else:
                    tie_breaker_rule_applied = (
                        "least_tried_probe(action_usage,cluster_usage,subcluster_usage,candidate_usage,action_id,candidate_id)"
                    )
            else:
                tie_breaker_rule_applied = "fixed_order(action_id,candidate_id)"
        elif near_tie_probe_applied:
            tie_breaker_rule_applied = (
                "exploit_action6_near_tie_probe_least_tried(action_usage,cluster_usage,subcluster_usage,candidate_usage,score)"
            )
        else:
            tie_breaker_rule_applied = "argmin_unique"

        entries[0].witness["selection_diagnostics_v1"] = {
            "selection_metric": selection_metric,
            "best_score": float(best_score),
            "second_best_score": float(second_score),
            "selected_score": float(
                selection_score_by_candidate.get(
                    entries[0].candidate.candidate_id,
                    entries[0].total_efe,
                )
            ),
            "best_vs_second_best_delta_total_efe": float(second_score - best_score),
            "tie_group_size": tie_group_size,
            "tie_epsilon": float(self.tie_epsilon),
            "tie_breaker_rule_applied": tie_breaker_rule_applied,
            "least_tried_probe_applied": bool(least_tried_probe_applied),
            "exploit_action6_bucket_probe_applied": bool(
                exploit_action6_bucket_probe_applied
            ),
            "near_tie_probe_applied": bool(near_tie_probe_applied),
            "action6_probe_score_margin": float(self.action6_probe_score_margin),
            "action6_explore_probe_score_margin": float(
                self.action6_explore_probe_score_margin
            ),
            "action6_stagnation_step_threshold": int(
                self.action6_stagnation_step_threshold
            ),
            "stagnation_streak": int(stagnation_streak),
            "stagnation_probe_trigger_steps": int(self.stagnation_probe_trigger_steps),
            "stagnation_probe_score_margin": float(self.stagnation_probe_score_margin),
            "stagnation_probe_min_action_usage_gap": int(
                self.stagnation_probe_min_action_usage_gap
            ),
            "sequence_probe_score_margin": float(self.sequence_probe_score_margin),
            "sequence_probe_trigger_steps": int(self.sequence_probe_trigger_steps),
            "previous_navigation_direction": str(previous_navigation_direction),
            "direction_sequence_probe_applied": bool(direction_sequence_probe_applied),
            "sequence_rollout_frontier_weight": float(
                self.sequence_rollout_frontier_weight
            ),
            "sequence_rollout_direction_weight": float(
                self.sequence_rollout_direction_weight
            ),
            "rollout_applied": bool(rollout_applied),
            "rollout_skip_reason": str(rollout_skip_reason),
            "rollout_max_candidates": int(self.rollout_max_candidates),
            "rollout_only_in_exploit": bool(self.rollout_only_in_exploit),
            "early_probe_budget_remaining": int(max(0, int(early_probe_budget_remaining))),
            "early_probe_applied": bool(early_probe_applied),
            "early_probe_target_action_ids": [int(v) for v in early_probe_target_action_ids],
            "early_probe_min_action_usage": int(early_probe_min_action_usage),
            "early_probe_candidate_pool_size": int(len(early_probe_candidate_pool)),
            "coverage_region_probe_applied": bool(coverage_region_probe_applied),
            "coverage_sweep_active": bool(coverage_sweep_active),
            "coverage_sweep_reason": str(coverage_sweep_reason),
            "coverage_known_region_count": int(coverage_known_region_count),
            "coverage_target_region_count": int(coverage_target_region_count),
            "coverage_periodic_resweep_active": bool(
                coverage_periodic_resweep_active
            ),
            "coverage_sweep_score_margin": float(coverage_score_margin_used),
            "coverage_sweep_pattern": str(coverage_sweep_pattern),
            "coverage_sweep_target_region": {
                "x": int(coverage_sweep_target_region.get("x", -1)),
                "y": int(coverage_sweep_target_region.get("y", -1)),
            },
            "coverage_sweep_target_direction": str(coverage_sweep_target_direction),
            "coverage_sweep_direction_retry_limit": int(
                self.coverage_sweep_direction_retry_limit
            ),
            "coverage_matrix_sweep_enabled": bool(self.coverage_matrix_sweep_enabled),
            "coverage_sweep_force_in_exploit": bool(
                self.coverage_sweep_force_in_exploit
            ),
            "coverage_resweep_interval": int(self.coverage_resweep_interval),
            "coverage_resweep_span": int(self.coverage_resweep_span),
            "action6_only_space": bool(action6_only_space),
            "action6_stagnated": bool(action6_stagnated),
            "action6_only_stagnation_probe_applied": bool(
                action6_only_stagnation_probe_applied
            ),
            "navigation_actions_available": [int(v) for v in navigation_actions_available],
            "navigation_usage_gap": int(navigation_usage_gap),
            "navigation_stagnated": bool(navigation_stagnated),
            "navigation_stagnation_probe_applied": bool(
                navigation_stagnation_probe_applied
            ),
            "selected_action_usage_count_before": int(selected_action_usage_count),
            "selected_cluster_id": selected_cluster_id,
            "selected_cluster_usage_count_before": int(selected_cluster_usage_count),
            "selected_subcluster_id": selected_subcluster_id,
            "selected_subcluster_usage_count_before": int(
                selected_subcluster_usage_count
            ),
            "selected_candidate_usage_count_before": int(selected_candidate_usage_count),
            "selected_state_visit_count_before": int(
                selected_transition_stats.get("state_visit_count", 0)
            ),
            "selected_state_action_visit_count_before": int(
                selected_transition_stats.get("state_action_visit_count", 0)
            ),
            "selected_state_outgoing_edge_count_before": int(
                selected_transition_stats.get("state_outgoing_edge_count", 0)
            ),
            "tie_probe_candidates": [
                {
                    "candidate_id": str(entry.candidate.candidate_id),
                    "action_id": int(entry.candidate.action_id),
                    "cluster_id": str(
                        cluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_cluster_id(entry.candidate),
                        )
                    ),
                    "subcluster_id": str(
                        subcluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_subcluster_id(entry.candidate),
                        )
                    ),
                    "action_usage_count_before": int(
                        action_count_map.get(int(entry.candidate.action_id), 0)
                    ),
                    "cluster_usage_count_before": int(
                        cluster_count_map.get(
                            str(
                                cluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_cluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "subcluster_usage_count_before": int(
                        subcluster_count_map.get(
                            str(
                                subcluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_subcluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "candidate_usage_count_before": int(
                        candidate_count_map.get(
                            str(entry.candidate.candidate_id),
                            0,
                        )
                    ),
                    "state_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_visit_count",
                            0,
                        )
                    ),
                    "state_action_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_action_visit_count",
                            0,
                        )
                    ),
                    "state_outgoing_edge_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_outgoing_edge_count",
                            0,
                        )
                    ),
                }
                for entry in tie_probe_candidates
            ],
            "near_tie_probe_candidates": [
                {
                    "candidate_id": str(entry.candidate.candidate_id),
                    "action_id": int(entry.candidate.action_id),
                    "cluster_id": str(
                        cluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_cluster_id(entry.candidate),
                        )
                    ),
                    "subcluster_id": str(
                        subcluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_subcluster_id(entry.candidate),
                        )
                    ),
                    "score": float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    ),
                    "action_usage_count_before": int(
                        action_count_map.get(int(entry.candidate.action_id), 0)
                    ),
                    "cluster_usage_count_before": int(
                        cluster_count_map.get(
                            str(
                                cluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_cluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "subcluster_usage_count_before": int(
                        subcluster_count_map.get(
                            str(
                                subcluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_subcluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "candidate_usage_count_before": int(
                        candidate_count_map.get(str(entry.candidate.candidate_id), 0)
                    ),
                    "state_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_visit_count",
                            0,
                        )
                    ),
                    "state_action_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_action_visit_count",
                            0,
                        )
                    ),
                    "state_outgoing_edge_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_outgoing_edge_count",
                            0,
                        )
                    ),
                }
                for entry in near_tie_probe_candidates[:12]
            ],
            "action6_only_probe_candidates": [
                {
                    "candidate_id": str(entry.candidate.candidate_id),
                    "action_id": int(entry.candidate.action_id),
                    "cluster_id": str(
                        cluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_cluster_id(entry.candidate),
                        )
                    ),
                    "subcluster_id": str(
                        subcluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_subcluster_id(entry.candidate),
                        )
                    ),
                    "score": float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    ),
                    "action_usage_count_before": int(
                        action_count_map.get(int(entry.candidate.action_id), 0)
                    ),
                    "cluster_usage_count_before": int(
                        cluster_count_map.get(
                            str(
                                cluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_cluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "subcluster_usage_count_before": int(
                        subcluster_count_map.get(
                            str(
                                subcluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_subcluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "candidate_usage_count_before": int(
                        candidate_count_map.get(str(entry.candidate.candidate_id), 0)
                    ),
                    "state_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_visit_count",
                            0,
                        )
                    ),
                    "state_action_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_action_visit_count",
                            0,
                        )
                    ),
                    "state_outgoing_edge_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_outgoing_edge_count",
                            0,
                        )
                    ),
                }
                for entry in action6_only_probe_candidates[:12]
            ],
            "navigation_probe_candidates": [
                {
                    "candidate_id": str(entry.candidate.candidate_id),
                    "action_id": int(entry.candidate.action_id),
                    "score": float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    ),
                    "predicted_blocked_probability": float(
                        self._predicted_probability_mass(
                            entry,
                            signature_filter="|delta=blocked",
                        )
                    ),
                    "predicted_no_change_probability": float(
                        self._predicted_probability_mass(
                            entry,
                            signature_filter="type=NO_CHANGE|progress=0",
                        )
                    ),
                    "action_usage_count_before": int(
                        action_count_map.get(int(entry.candidate.action_id), 0)
                    ),
                    "state_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_visit_count",
                            0,
                        )
                    ),
                    "state_action_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_action_visit_count",
                            0,
                        )
                    ),
                    "state_outgoing_edge_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_outgoing_edge_count",
                            0,
                        )
                    ),
                }
                for entry in navigation_probe_candidates[:12]
            ],
            "direction_sequence_probe_candidates": [
                {
                    "candidate_id": str(row.get("candidate_id", "")),
                    "action_id": int(row.get("action_id", -1)),
                    "predicted_direction": str(row.get("predicted_direction", "na")),
                    "predicted_sequence_key": str(
                        row.get("predicted_sequence_key", "")
                    ),
                    "predicted_sequence_count": int(
                        row.get("predicted_sequence_count", 0)
                    ),
                    "predicted_direction_count": int(
                        row.get("predicted_direction_count", 0)
                    ),
                    "frontier_graph_cost": float(
                        row.get("frontier_graph_cost", 0.0)
                    ),
                    "state_action_visit_count": int(
                        row.get("state_action_visit_count", 0)
                    ),
                    "action_usage_count_before": int(
                        row.get("action_usage_count_before", 0)
                    ),
                    "score": float(row.get("score", 0.0)),
                }
                for row in direction_sequence_probe_candidates[:12]
            ],
            "coverage_region_probe_candidates": [
                {
                    "candidate_id": str(row.get("candidate_id", "")),
                    "action_id": int(row.get("action_id", -1)),
                    "score": float(row.get("score", 0.0)),
                    "current_region_key": str(row.get("current_region_key", "NA")),
                    "predicted_region_key": str(row.get("predicted_region_key", "NA")),
                    "predicted_region_source": str(
                        row.get("predicted_region_source", "posterior_expected_delta")
                    ),
                    "predicted_region_visit_count": int(
                        row.get("predicted_region_visit_count", 0)
                    ),
                    "current_region_visit_count": int(
                        row.get("current_region_visit_count", 0)
                    ),
                    "known_region_count": int(row.get("known_region_count", 0)),
                    "empirical_transition_target_key": str(
                        row.get("empirical_transition_target_key", "NA")
                    ),
                    "empirical_transition_total": int(
                        row.get("empirical_transition_total", 0)
                    ),
                    "empirical_transition_confidence": float(
                        row.get("empirical_transition_confidence", 0.0)
                    ),
                    "empirical_transition_override_applied": bool(
                        row.get("empirical_transition_override_applied", False)
                    ),
                    "current_region_source": str(
                        row.get("current_region_source", "unknown")
                    ),
                    "confidence": float(row.get("confidence", 0.0)),
                    "predicted_direction": str(row.get("predicted_direction", "na")),
                    "target_direction_match": bool(
                        row.get("target_direction_match", False)
                    ),
                    "target_direction_retry_exceeded": bool(
                        row.get("target_direction_retry_exceeded", False)
                    ),
                    "coverage_sweep_direction_retry_limit": int(
                        row.get("coverage_sweep_direction_retry_limit", 0)
                    ),
                    "target_region_distance": int(row.get("target_region_distance", -1)),
                    "edge_attempts": int(row.get("edge_attempts", 0)),
                    "edge_blocked_rate": float(row.get("edge_blocked_rate", 0.0)),
                }
                for row in coverage_region_probe_candidates[:12]
            ],
            "early_probe_candidates": [
                {
                    "candidate_id": str(entry.candidate.candidate_id),
                    "action_id": int(entry.candidate.action_id),
                    "cluster_id": str(
                        cluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_cluster_id(entry.candidate),
                        )
                    ),
                    "subcluster_id": str(
                        subcluster_id_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            self._candidate_subcluster_id(entry.candidate),
                        )
                    ),
                    "score": float(
                        selection_score_by_candidate.get(
                            entry.candidate.candidate_id,
                            entry.total_efe,
                        )
                    ),
                    "action_usage_count_before": int(
                        action_count_map.get(int(entry.candidate.action_id), 0)
                    ),
                    "cluster_usage_count_before": int(
                        cluster_count_map.get(
                            str(
                                cluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_cluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "subcluster_usage_count_before": int(
                        subcluster_count_map.get(
                            str(
                                subcluster_id_by_candidate_id.get(
                                    str(entry.candidate.candidate_id),
                                    self._candidate_subcluster_id(entry.candidate),
                                )
                            ),
                            0,
                        )
                    ),
                    "state_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_visit_count",
                            0,
                        )
                    ),
                    "state_action_visit_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_action_visit_count",
                            0,
                        )
                    ),
                    "state_outgoing_edge_count": int(
                        self._candidate_transition_stats(entry.candidate).get(
                            "state_outgoing_edge_count",
                            0,
                        )
                    ),
                }
                for entry in early_probe_candidate_pool[:8]
            ],
        }
        return entries[0].candidate, entries
