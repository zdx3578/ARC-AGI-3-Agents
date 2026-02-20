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
    ) -> None:
        self.explore_steps = int(max(1, explore_steps))
        self.exploit_entropy_threshold = float(max(0.0, exploit_entropy_threshold))
        self.top_k_reasoning = int(max(1, top_k_reasoning))
        self.rollout_horizon = int(max(1, rollout_horizon))
        self.rollout_discount = float(max(0.0, min(1.0, rollout_discount)))
        self.tie_epsilon = float(max(0.0, tie_epsilon))
        self.ignore_action_cost = bool(ignore_action_cost)
        self.weight_overrides = weight_overrides or {}

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
            ambiguity = hypothesis_bank.expected_ambiguity(
                packet,
                candidate,
                representation,
            )

            ig_split = hypothesis_bank.split_information_gain(
                packet,
                candidate,
                representation,
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
        early_probe_budget_remaining: int = 0,
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
        selection_score_by_candidate: dict[str, float] = {
            entry.candidate.candidate_id: float(entry.total_efe) for entry in entries
        }
        if int(remaining_budget) >= 2:
            rollout_scores = self._rollout_score_by_candidate(
                packet=packet,
                representation=representation,
                candidates=candidates,
                hypothesis_bank=hypothesis_bank,
                phase=phase,
                entries=entries,
            )
            selection_metric = "rollout_total_efe"
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
        cluster_id_by_candidate_id: dict[str, str] = {
            str(entry.candidate.candidate_id): self._candidate_cluster_id(entry.candidate)
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
        tie_probe_candidates = []
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
            if phase in ("explore", "explain"):
                tie_probe_candidates.sort(
                    key=lambda entry: (
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
                        int(candidate_count_map.get(str(entry.candidate.candidate_id), 0)),
                        int(entry.candidate.action_id),
                        str(entry.candidate.candidate_id),
                    )
                )
                selected_entry = tie_probe_candidates[0]
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
        selected_candidate_usage_count = int(
            candidate_count_map.get(str(entries[0].candidate.candidate_id), 0)
        )
        if early_probe_applied:
            tie_breaker_rule_applied = "early_probe_budget_least_tried"
        elif tie_group_size > 1:
            if least_tried_probe_applied:
                tie_breaker_rule_applied = (
                    "least_tried_probe(action_usage,cluster_usage,candidate_usage,action_id,candidate_id)"
                )
            else:
                tie_breaker_rule_applied = "fixed_order(action_id,candidate_id)"
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
            "early_probe_budget_remaining": int(max(0, int(early_probe_budget_remaining))),
            "early_probe_applied": bool(early_probe_applied),
            "early_probe_target_action_ids": [int(v) for v in early_probe_target_action_ids],
            "early_probe_min_action_usage": int(early_probe_min_action_usage),
            "early_probe_candidate_pool_size": int(len(early_probe_candidate_pool)),
            "selected_action_usage_count_before": int(selected_action_usage_count),
            "selected_cluster_id": selected_cluster_id,
            "selected_cluster_usage_count_before": int(selected_cluster_usage_count),
            "selected_candidate_usage_count_before": int(selected_candidate_usage_count),
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
                    "candidate_usage_count_before": int(
                        candidate_count_map.get(
                            str(entry.candidate.candidate_id),
                            0,
                        )
                    ),
                }
                for entry in tie_probe_candidates
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
                }
                for entry in early_probe_candidate_pool[:8]
            ],
        }
        return entries[0].candidate, entries
