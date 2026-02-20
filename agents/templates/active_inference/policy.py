from __future__ import annotations

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
        weights = self._weights_for_phase(phase)
        preference_distribution = preference_distribution_v1(packet, phase)
        entries: list[FreeEnergyLedgerEntryV1] = []

        for candidate in candidates:
            stats = hypothesis_bank.predictive_statistics(packet, candidate, representation)
            predictive_distribution = stats["predictive_distribution"]
            supports_by_signature = stats["supports_by_signature"]
            expected_mdl_bits = float(stats["expected_mdl_bits"])

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

            witness = {
                "weights": {str(k): float(v) for (k, v) in weights.items()},
                "objective_policy_v1": {
                    "ignore_action_cost": bool(self.ignore_action_cost),
                    "applied_action_cost_weight": float(weights["action_cost"]),
                },
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
    ) -> dict[str, float]:
        rollout_scores: dict[str, float] = {}
        if self.rollout_horizon < 2:
            for entry in entries:
                rollout_scores[entry.candidate.candidate_id] = float(entry.total_efe)
            return rollout_scores

        for entry in entries:
            expected_future_efe = 0.0
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
                    best_future = float(future_entries[0].total_efe) if future_entries else 0.0
                    expected_future_efe += p * best_future

            rollout_total = float(entry.total_efe) + (
                self.rollout_discount * float(expected_future_efe)
            )
            entry.witness["rollout_horizon"] = int(self.rollout_horizon)
            entry.witness["rollout_expected_future_efe"] = float(expected_future_efe)
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

        if tie_group_size > 1:
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
            "rollout_applied": bool(rollout_applied),
            "rollout_skip_reason": str(rollout_skip_reason),
            "rollout_max_candidates": int(self.rollout_max_candidates),
            "rollout_only_in_exploit": bool(self.rollout_only_in_exploit),
            "early_probe_budget_remaining": int(max(0, int(early_probe_budget_remaining))),
            "early_probe_applied": bool(early_probe_applied),
            "early_probe_target_action_ids": [int(v) for v in early_probe_target_action_ids],
            "early_probe_min_action_usage": int(early_probe_min_action_usage),
            "early_probe_candidate_pool_size": int(len(early_probe_candidate_pool)),
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
