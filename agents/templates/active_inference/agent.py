from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ...agent import Agent
from .contracts import (
    ActionCandidateV1,
    ObservationPacketV1,
    RepresentationStateV1,
)
from .diagnostics import StageDiagnosticsCollectorV1
from .hypothesis_bank import (
    ActiveInferenceHypothesisBankV1,
    build_causal_event_signature_v1,
)
from .policy import ActiveInferencePolicyEvaluatorV1
from .representation import (
    build_action_candidates_v1,
    build_observation_packet_v1,
    build_representation_state_v1,
)
from .trace import ActiveInferenceTraceRecorderV1


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return bool(default)
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    return bool(default)


def _env_weight_overrides() -> dict[str, dict[str, float]]:
    raw = os.getenv("ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}

    out: dict[str, dict[str, float]] = {}
    for phase_any, values_any in parsed.items():
        phase = str(phase_any).strip()
        if phase not in ("explore", "explain", "exploit"):
            continue
        if not isinstance(values_any, dict):
            continue
        normalized: dict[str, float] = {}
        for key in ("risk", "ambiguity", "information_gain", "action_cost", "complexity"):
            if key not in values_any:
                continue
            try:
                normalized[key] = float(values_any[key])
            except Exception:
                continue
        if normalized:
            out[phase] = normalized
    return out


class ActiveInferenceEFE(Agent):
    """Active Inference / EFE-driven ARC-AGI-3 agent framework.

    This class intentionally prioritizes contract clarity and auditability over
    benchmark performance:
      - observation contract (A1)
      - representation/objectization contract (A2)
      - hypothesis bank + MDL-aware posterior updates (A3)
      - EFE decomposition ledger per candidate (A4)
      - Action6 coordinate proposer from object representation (A5)
      - one-step deterministic policy evaluation (A6)
      - causal event signature extraction for intervention traces (A7)
      - explicit trace and reasoning schema (A8)
    """

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.MAX_ACTIONS = max(1, _env_int("ACTIVE_INFERENCE_MAX_ACTIONS", 80))
        self.component_connectivity = (
            4 if _env_int("ACTIVE_INFERENCE_COMPONENT_CONNECTIVITY", 8) == 4 else 8
        )
        self.max_action6_points = max(1, _env_int("ACTIVE_INFERENCE_MAX_ACTION6_POINTS", 16))
        self.top_k_reasoning = max(1, _env_int("ACTIVE_INFERENCE_TOP_K_REASONING", 5))
        self.trace_candidate_limit = max(1, _env_int("ACTIVE_INFERENCE_TRACE_CANDIDATE_LIMIT", 30))
        self.trace_include_full_representation = _env_bool(
            "ACTIVE_INFERENCE_TRACE_INCLUDE_FULL_REPRESENTATION",
            False,
        )

        explore_steps = max(1, _env_int("ACTIVE_INFERENCE_EXPLORE_STEPS", 20))
        exploit_entropy_threshold = max(
            0.0, _env_float("ACTIVE_INFERENCE_EXPLOIT_ENTROPY_THRESHOLD", 0.9)
        )
        weight_overrides = _env_weight_overrides()
        self.policy = ActiveInferencePolicyEvaluatorV1(
            explore_steps=explore_steps,
            exploit_entropy_threshold=exploit_entropy_threshold,
            top_k_reasoning=self.top_k_reasoning,
            weight_overrides=weight_overrides,
        )
        self.hypothesis_bank = ActiveInferenceHypothesisBankV1()

        self._previous_packet: ObservationPacketV1 | None = None
        self._previous_representation: RepresentationStateV1 | None = None
        self._previous_action_candidate: ActionCandidateV1 | None = None

        self.trace_enabled = _env_bool("ACTIVE_INFERENCE_TRACE_ENABLED", True)
        self.trace_recorder: ActiveInferenceTraceRecorderV1 | None = None
        self._trace_closed = False
        if self.trace_enabled:
            trace_root = os.getenv("RECORDINGS_DIR", "recordings")
            self.trace_recorder = ActiveInferenceTraceRecorderV1(
                root_dir=trace_root,
                game_id=self.game_id,
                agent_name=self.__class__.__name__.lower(),
                card_id=self.card_id,
            )

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def _candidate_to_game_action(self, candidate: ActionCandidateV1) -> GameAction:
        action = GameAction.from_id(int(candidate.action_id))
        if action == GameAction.ACTION6:
            action.set_data(
                {
                    "x": int(candidate.x if candidate.x is not None else 31),
                    "y": int(candidate.y if candidate.y is not None else 31),
                }
            )
        return action

    def _observation_summary_for_trace(
        self, packet: ObservationPacketV1
    ) -> dict[str, Any]:
        frame_height = len(packet.frame)
        frame_width = len(packet.frame[0]) if packet.frame else 0
        frame_digest = hashlib.sha256(
            repr(packet.frame).encode("utf-8", errors="ignore")
        ).hexdigest()[:24]
        return {
            "schema_name": packet.schema_name,
            "schema_version": int(packet.schema_version),
            "task": {
                "game_id": packet.game_id,
                "card_id": packet.card_id,
                "action_counter": int(packet.action_counter),
            },
            "observation": {
                "state": packet.state,
                "levels_completed": int(packet.levels_completed),
                "win_levels": int(packet.win_levels),
                "available_actions": [int(v) for v in packet.available_actions],
                "frame_height": int(frame_height),
                "frame_width": int(frame_width),
                "frame_digest": frame_digest,
            },
            "constraints": {
                "action_cost_per_step": int(packet.action_cost_per_step),
                "action6_coordinate_min": int(packet.action6_coordinate_min),
                "action6_coordinate_max": int(packet.action6_coordinate_max),
            },
        }

    def _reasoning_for_forced_reset(self, packet: ObservationPacketV1) -> dict[str, Any]:
        return {
            "schema_name": "active_inference_reasoning_v1",
            "schema_version": 1,
            "phase": "control",
            "selected_candidate": {
                "candidate_id": "reset_forced_by_state",
                "action_id": 0,
                "source": "state_guard",
            },
            "state_guard": {
                "state": packet.state,
                "reason": "state_requires_reset",
            },
            "hypothesis_summary": self.hypothesis_bank.summary(),
        }

    def _reasoning_for_failure(
        self,
        packet: ObservationPacketV1 | None,
        diagnostics: StageDiagnosticsCollectorV1,
        failure_code: str,
        failure_message: str,
    ) -> dict[str, Any]:
        packet_summary = (
            self._observation_summary_for_trace(packet) if packet is not None else None
        )
        return {
            "schema_name": "active_inference_reasoning_v1",
            "schema_version": 1,
            "phase": "failure_fallback",
            "failure_taxonomy_v1": {
                "failure_code": failure_code,
                "failure_message": failure_message,
            },
            "stage_diagnostics_v1": diagnostics.to_dicts(),
            "bottleneck_stage_v1": diagnostics.bottleneck_stage(),
            "observation_packet_summary": packet_summary,
            "hypothesis_summary": self.hypothesis_bank.summary(),
        }

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        diagnostics = StageDiagnosticsCollectorV1()
        packet: ObservationPacketV1 | None = None
        representation: RepresentationStateV1 | None = None
        selected_candidate = ActionCandidateV1(
            candidate_id="reset_failure_bootstrap",
            action_id=0,
            source="bootstrap",
        )
        candidates: list[ActionCandidateV1] = []
        ranked_entries = []
        causal_signature = None
        phase = "control"

        diagnostics.start("observation_contract")
        try:
            packet = build_observation_packet_v1(
                latest_frame,
                game_id=self.game_id,
                card_id=self.card_id,
                action_counter=self.action_counter,
            )
            diagnostics.finish_ok(
                "observation_contract",
                {
                    "state": packet.state,
                    "levels_completed": int(packet.levels_completed),
                    "win_levels": int(packet.win_levels),
                    "available_action_count": int(len(packet.available_actions)),
                    "frame_height": int(len(packet.frame)),
                    "frame_width": int(len(packet.frame[0]) if packet.frame else 0),
                },
            )
        except Exception as exc:
            diagnostics.finish_rejected(
                "observation_contract",
                f"observation_packet_build_error::{type(exc).__name__}",
            )
            fallback = GameAction.RESET
            fallback.reasoning = self._reasoning_for_failure(
                packet=None,
                diagnostics=diagnostics,
                failure_code="A1_OBSERVATION_CONTRACT_FAILURE",
                failure_message=str(exc),
            )
            return fallback

        diagnostics.start("representation_build")
        try:
            representation = build_representation_state_v1(
                packet,
                connectivity=self.component_connectivity,
                max_action6_points=self.max_action6_points,
            )
            diagnostics.finish_ok(
                "representation_build",
                {
                    "object_count": int(representation.summary.get("object_count", 0)),
                    "action6_proposal_count": int(
                        representation.summary.get(
                            "action6_coordinate_proposal_count",
                            0,
                        )
                    ),
                    "action6_proposal_coverage": float(
                        representation.summary.get(
                            "action6_coordinate_proposal_coverage",
                            0.0,
                        )
                    ),
                },
            )
        except Exception as exc:
            diagnostics.finish_rejected(
                "representation_build",
                f"representation_build_error::{type(exc).__name__}",
            )
            fallback = GameAction.RESET
            fallback.reasoning = self._reasoning_for_failure(
                packet=packet,
                diagnostics=diagnostics,
                failure_code="A2_REPRESENTATION_BUILD_FAILURE",
                failure_message=str(exc),
            )
            return fallback

        diagnostics.start("causal_update")
        try:
            if (
                self._previous_packet is not None
                and self._previous_representation is not None
                and self._previous_action_candidate is not None
            ):
                causal_signature = build_causal_event_signature_v1(
                    self._previous_packet,
                    packet,
                    self._previous_representation,
                    representation,
                    self._previous_action_candidate,
                )
                self.hypothesis_bank.update_with_observation(
                    previous_packet=self._previous_packet,
                    current_packet=packet,
                    executed_candidate=self._previous_action_candidate,
                    previous_representation=self._previous_representation,
                    observed_signature=causal_signature,
                )
                diagnostics.finish_ok(
                    "causal_update",
                    {
                        "updated": True,
                        "signature_digest": causal_signature.signature_digest,
                        "event_tags": list(causal_signature.event_tags),
                    },
                )
            else:
                diagnostics.finish_ok(
                    "causal_update",
                    {"updated": False, "reason": "insufficient_history"},
                )
        except Exception as exc:
            diagnostics.finish_rejected(
                "causal_update",
                f"causal_update_error::{type(exc).__name__}",
            )

        diagnostics.start("phase_determination")
        try:
            phase = self.policy.determine_phase(
                action_counter=self.action_counter,
                hypothesis_bank=self.hypothesis_bank,
            )
            diagnostics.finish_ok(
                "phase_determination",
                {
                    "phase": phase,
                    "posterior_entropy_bits": float(self.hypothesis_bank.posterior_entropy()),
                },
            )
        except Exception as exc:
            diagnostics.finish_rejected(
                "phase_determination",
                f"phase_determination_error::{type(exc).__name__}",
            )
            phase = "control"

        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            diagnostics.start("state_guard")
            selected_candidate = ActionCandidateV1(
                candidate_id="reset_forced_by_state",
                action_id=0,
                source="state_guard",
            )
            action = GameAction.RESET
            action.reasoning = self._reasoning_for_forced_reset(packet)
            diagnostics.finish_ok(
                "state_guard",
                {
                    "selected_action_id": 0,
                    "reason": "state_requires_reset",
                    "state": latest_frame.state.name,
                },
            )
        else:
            diagnostics.start("candidate_generation")
            try:
                candidates = build_action_candidates_v1(packet, representation)
                if not candidates:
                    diagnostics.finish_rejected(
                        "candidate_generation",
                        "no_candidates_generated",
                    )
                    selected_candidate = ActionCandidateV1(
                        candidate_id="reset_no_candidates",
                        action_id=0,
                        source="candidate_guard",
                    )
                    action = GameAction.RESET
                    action.reasoning = {
                        "schema_name": "active_inference_reasoning_v1",
                        "schema_version": 1,
                        "phase": phase,
                        "selected_candidate": selected_candidate.to_dict(),
                        "reason": "no_candidates",
                        "hypothesis_summary": self.hypothesis_bank.summary(),
                        "stage_diagnostics_v1": diagnostics.to_dicts(),
                        "bottleneck_stage_v1": diagnostics.bottleneck_stage(),
                    }
                else:
                    diagnostics.finish_ok(
                        "candidate_generation",
                        {
                            "candidate_count": int(len(candidates)),
                            "action6_candidate_count": int(
                                sum(1 for candidate in candidates if candidate.action_id == 6)
                            ),
                        },
                    )

                    diagnostics.start("policy_selection")
                    selected_candidate, ranked_entries = self.policy.select_action(
                        packet=packet,
                        representation=representation,
                        candidates=candidates,
                        hypothesis_bank=self.hypothesis_bank,
                        phase=phase,
                    )
                    diagnostics.finish_ok(
                        "policy_selection",
                        {
                            "selected_candidate_id": selected_candidate.candidate_id,
                            "selected_action_id": int(selected_candidate.action_id),
                            "ranked_count": int(len(ranked_entries)),
                        },
                    )

                    diagnostics.start("action_materialization")
                    try:
                        action = self._candidate_to_game_action(selected_candidate)
                        diagnostics.finish_ok(
                            "action_materialization",
                            {
                                "selected_action_id": int(selected_candidate.action_id),
                                "is_action6": bool(selected_candidate.action_id == 6),
                            },
                        )
                    except Exception as exc:
                        diagnostics.finish_rejected(
                            "action_materialization",
                            f"action_materialization_error::{type(exc).__name__}",
                        )
                        action = GameAction.RESET
                        selected_candidate = ActionCandidateV1(
                            candidate_id="reset_action_materialization_failure",
                            action_id=0,
                            source="action_materialization_guard",
                        )

                    top_entries = [
                        entry.to_dict() for entry in ranked_entries[: self.top_k_reasoning]
                    ]
                    action.reasoning = {
                        "schema_name": "active_inference_reasoning_v1",
                        "schema_version": 1,
                        "phase": phase,
                        "selected_candidate": selected_candidate.to_dict(),
                        "selected_free_energy": (
                            ranked_entries[0].to_dict() if ranked_entries else None
                        ),
                        "top_k_candidates_by_efe": top_entries,
                        "hypothesis_summary": self.hypothesis_bank.summary(),
                        "representation_summary": representation.summary,
                        "causal_event_signature_previous_step": (
                            causal_signature.to_dict() if causal_signature else None
                        ),
                        "stage_diagnostics_v1": diagnostics.to_dicts(),
                        "bottleneck_stage_v1": diagnostics.bottleneck_stage(),
                    }
            except Exception as exc:
                diagnostics.finish_rejected(
                    "candidate_generation",
                    f"candidate_generation_error::{type(exc).__name__}",
                )
                action = GameAction.RESET
                action.reasoning = self._reasoning_for_failure(
                    packet=packet,
                    diagnostics=diagnostics,
                    failure_code="A5_A6_POLICY_PIPELINE_FAILURE",
                    failure_message=str(exc),
                )

        if isinstance(action.reasoning, dict):
            action.reasoning.setdefault("stage_diagnostics_v1", diagnostics.to_dicts())
            action.reasoning.setdefault("bottleneck_stage_v1", diagnostics.bottleneck_stage())

        if self.trace_recorder is not None:
            representation_payload = (
                representation.to_dict()
                if self.trace_include_full_representation
                else {
                    "schema_name": representation.schema_name,
                    "schema_version": int(representation.schema_version),
                    "frame_height": int(representation.frame_height),
                    "frame_width": int(representation.frame_width),
                    "summary": representation.summary,
                }
            )
            self.trace_recorder.write(
                {
                    "schema_name": "active_inference_step_trace_v1",
                    "schema_version": 1,
                    "game_id": self.game_id,
                    "card_id": self.card_id,
                    "action_counter": int(self.action_counter),
                    "phase": phase,
                    "observation_packet_summary": self._observation_summary_for_trace(packet),
                    "representation_state": representation_payload,
                    "candidate_count": int(len(candidates)),
                    "selected_candidate": selected_candidate.to_dict(),
                    "ranked_candidates_by_efe": [
                        entry.to_dict()
                        for entry in ranked_entries[: self.trace_candidate_limit]
                    ],
                    "hypothesis_summary": self.hypothesis_bank.summary(),
                    "causal_event_signature_previous_step": (
                        causal_signature.to_dict() if causal_signature else None
                    ),
                    "stage_diagnostics_v1": diagnostics.to_dicts(),
                    "bottleneck_stage_v1": diagnostics.bottleneck_stage(),
                }
            )

        self._previous_packet = packet
        self._previous_representation = representation
        self._previous_action_candidate = selected_candidate

        return action

    def cleanup(self, scorecard: Any = None) -> None:
        if self.trace_recorder is not None and not self._trace_closed:
            self.trace_recorder.write(
                {
                    "schema_name": "active_inference_agent_summary_v1",
                    "schema_version": 1,
                    "game_id": self.game_id,
                    "card_id": self.card_id,
                    "total_actions_taken": int(self.action_counter),
                    "final_hypothesis_summary": self.hypothesis_bank.summary(),
                }
            )
            self.trace_recorder.close()
            self._trace_closed = True
        super().cleanup(scorecard)
