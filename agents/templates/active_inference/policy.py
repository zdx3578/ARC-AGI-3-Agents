from __future__ import annotations

from typing import Any

from .contracts import (
    ActionCandidateV1,
    FreeEnergyLedgerEntryV1,
    ObservationPacketV1,
    RepresentationStateV1,
)
from .efe import (
    compute_information_gain_v1,
    compute_risk_v1,
    determine_phase_v1,
    entropy_bits,
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
        weight_overrides: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.explore_steps = int(max(1, explore_steps))
        self.exploit_entropy_threshold = float(max(0.0, exploit_entropy_threshold))
        self.top_k_reasoning = int(max(1, top_k_reasoning))
        self.weight_overrides = weight_overrides or {}

    def _weights_for_phase(self, phase: str) -> dict[str, float]:
        defaults = weights_for_phase_v1(phase)
        out = {
            "risk": float(defaults.risk),
            "ambiguity": float(defaults.ambiguity),
            "information_gain": float(defaults.information_gain),
            "action_cost": float(defaults.action_cost),
            "complexity": float(defaults.complexity),
        }
        override = self.weight_overrides.get(phase, {})
        for key in ("risk", "ambiguity", "information_gain", "action_cost", "complexity"):
            if key not in override:
                continue
            value_any: Any = override[key]
            try:
                out[key] = float(value_any)
            except Exception:
                continue
        return out

    def determine_phase(
        self,
        *,
        action_counter: int,
        hypothesis_bank: ActiveInferenceHypothesisBankV1,
    ) -> str:
        return determine_phase_v1(
            action_counter=action_counter,
            posterior_entropy_bits=hypothesis_bank.posterior_entropy(),
            explore_steps=self.explore_steps,
            exploit_entropy_threshold=self.exploit_entropy_threshold,
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
        entries: list[FreeEnergyLedgerEntryV1] = []

        for candidate in candidates:
            (
                predictive_distribution,
                supports_by_signature,
                expected_mdl_bits,
                hypothesis_signature,
            ) = hypothesis_bank.predictive_distribution(packet, candidate, representation)

            risk_value, risk_components = compute_risk_v1(packet, candidate, representation)
            ambiguity = 0.0
            if len(predictive_distribution) > 1:
                entropy = entropy_bits(predictive_distribution)
                ambiguity = entropy / max(
                    1.0, entropy_bits({k: 1.0 / len(predictive_distribution) for k in predictive_distribution})
                )
            information_gain = compute_information_gain_v1(
                posterior_by_hypothesis_id=hypothesis_bank.posterior_by_hypothesis_id,
                hypothesis_signature=hypothesis_signature,
                predictive_distribution=predictive_distribution,
            )
            action_cost = float(packet.action_cost_per_step)
            complexity_penalty = float(expected_mdl_bits / 64.0)

            total_efe = (
                weights["risk"] * risk_value
                + weights["ambiguity"] * ambiguity
                - weights["information_gain"] * information_gain
                + weights["action_cost"] * action_cost
                + weights["complexity"] * complexity_penalty
            )

            witness = {
                "weights": {
                    "risk": float(weights["risk"]),
                    "ambiguity": float(weights["ambiguity"]),
                    "information_gain": float(weights["information_gain"]),
                    "action_cost": float(weights["action_cost"]),
                    "complexity": float(weights["complexity"]),
                },
                "risk_components": risk_components,
                "supports_by_signature": supports_by_signature,
                "expected_mdl_bits": float(expected_mdl_bits),
            }

            entries.append(
                FreeEnergyLedgerEntryV1(
                    schema_name="active_inference_free_energy_ledger_entry_v1",
                    schema_version=1,
                    phase=phase,
                    candidate=candidate,
                    risk=float(risk_value),
                    ambiguity=float(ambiguity),
                    information_gain=float(information_gain),
                    action_cost=float(action_cost),
                    complexity_penalty=float(complexity_penalty),
                    total_efe=float(total_efe),
                    predictive_signature_distribution=predictive_distribution,
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

    def select_action(
        self,
        *,
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
        candidates: list[ActionCandidateV1],
        hypothesis_bank: ActiveInferenceHypothesisBankV1,
        phase: str,
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
        return entries[0].candidate, entries
