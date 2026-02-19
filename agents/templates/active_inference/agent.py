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
        for key in (
            "risk",
            "ambiguity",
            "information_gain_action_semantics",
            "information_gain_mechanism_dynamics",
            "information_gain_causal_mapping",
            "action_cost",
            "complexity",
            "vfe",
        ):
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
        self.frame_chain_window = max(1, _env_int("ACTIVE_INFERENCE_FRAME_CHAIN_WINDOW", 8))
        self.available_actions_history_window = max(
            1,
            _env_int("ACTIVE_INFERENCE_ACTION_SPACE_HISTORY_WINDOW", 24),
        )
        self.rollout_horizon = max(1, _env_int("ACTIVE_INFERENCE_ROLLOUT_HORIZON", 2))
        self.rollout_discount = max(
            0.0,
            min(1.0, _env_float("ACTIVE_INFERENCE_ROLLOUT_DISCOUNT", 0.55)),
        )
        self.early_probe_budget = max(
            0,
            _env_int("ACTIVE_INFERENCE_EARLY_PROBE_BUDGET", 8),
        )
        self.no_change_stop_loss_steps = max(
            1,
            _env_int("ACTIVE_INFERENCE_NO_CHANGE_STOP_LOSS_STEPS", 3),
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
            rollout_horizon=self.rollout_horizon,
            rollout_discount=self.rollout_discount,
            weight_overrides=weight_overrides,
        )
        self.hypothesis_bank = ActiveInferenceHypothesisBankV1()

        self._previous_packet: ObservationPacketV1 | None = None
        self._previous_representation: RepresentationStateV1 | None = None
        self._previous_action_candidate: ActionCandidateV1 | None = None
        self._no_change_streak = 0
        self._available_actions_history: list[list[int]] = []
        self._control_schema_counts: dict[str, dict[str, int]] = {}
        self._tracked_agent_token_digest: str | None = None
        self._latest_navigation_state_estimate: dict[str, Any] = {}
        self._action_select_count: dict[int, int] = {}
        self._candidate_select_count: dict[str, int] = {}

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
                "num_frames_received": int(packet.num_frames_received),
                "frame_chain_digests": [str(v) for v in packet.frame_chain_digests],
                "frame_chain_micro_signatures": [
                    dict(v) for v in packet.frame_chain_micro_signatures
                ],
                "frame_chain_macro_signature": dict(packet.frame_chain_macro_signature),
            },
            "constraints": {
                "action_cost_per_step": int(packet.action_cost_per_step),
                "action6_coordinate_min": int(packet.action6_coordinate_min),
                "action6_coordinate_max": int(packet.action6_coordinate_max),
            },
        }

    def _available_actions_trajectory_summary(self) -> dict[str, Any]:
        if not self._available_actions_history:
            return {
                "window_size": int(self.available_actions_history_window),
                "history_length": 0,
                "last_actions": [],
                "toggle_count": 0,
                "action_presence_counts": {},
            }
        action_presence_counts: dict[int, int] = {}
        toggle_count = 0
        previous: set[int] | None = None
        for available in self._available_actions_history:
            current = set(int(v) for v in available)
            for action_id in current:
                action_presence_counts[action_id] = action_presence_counts.get(action_id, 0) + 1
            if previous is not None and current != previous:
                toggle_count += 1
            previous = current
        return {
            "window_size": int(self.available_actions_history_window),
            "history_length": int(len(self._available_actions_history)),
            "last_actions": [int(v) for v in self._available_actions_history[-1]],
            "toggle_count": int(toggle_count),
            "action_presence_counts": {
                str(key): int(value)
                for (key, value) in sorted(action_presence_counts.items())
            },
        }

    def _control_schema_posterior(self) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for action_id, counts in sorted(self._control_schema_counts.items()):
            total = float(sum(int(v) for v in counts.values()))
            if total <= 0.0:
                continue
            out[str(action_id)] = {
                str(delta_key): float(int(count) / total)
                for (delta_key, count) in sorted(counts.items())
            }
        return out

    def _estimate_navigation_state(
        self,
        previous_representation: RepresentationStateV1,
        current_representation: RepresentationStateV1,
        executed_candidate: ActionCandidateV1,
    ) -> dict[str, Any]:
        previous_nodes = list(previous_representation.object_nodes)
        current_nodes = list(current_representation.object_nodes)
        if not previous_nodes or not current_nodes:
            return {
                "schema_name": "active_inference_navigation_state_estimate_v1",
                "schema_version": 1,
                "matched": False,
                "reason": "missing_object_nodes",
                "control_schema_posterior": self._control_schema_posterior(),
            }

        best_pair: tuple[Any, Any] | None = None
        best_score = 10**9
        for previous in previous_nodes:
            for current in current_nodes:
                if int(previous.color) != int(current.color):
                    continue
                area_gap = abs(int(previous.area) - int(current.area))
                if area_gap > max(2, int(previous.area * 0.3)):
                    continue
                delta_x = int(current.centroid_x) - int(previous.centroid_x)
                delta_y = int(current.centroid_y) - int(previous.centroid_y)
                shift = abs(delta_x) + abs(delta_y)
                if shift <= 0:
                    continue
                score = (area_gap * 10) + shift
                if self._tracked_agent_token_digest is not None:
                    if str(previous.digest) != self._tracked_agent_token_digest:
                        score += 5
                if score < best_score:
                    best_score = score
                    best_pair = (previous, current)

        if best_pair is None:
            return {
                "schema_name": "active_inference_navigation_state_estimate_v1",
                "schema_version": 1,
                "matched": False,
                "reason": "no_translation_match",
                "control_schema_posterior": self._control_schema_posterior(),
            }

        previous, current = best_pair
        delta_x = int(current.centroid_x) - int(previous.centroid_x)
        delta_y = int(current.centroid_y) - int(previous.centroid_y)
        delta_key = f"dx={delta_x}|dy={delta_y}"
        action_key = str(int(executed_candidate.action_id))
        per_action = self._control_schema_counts.setdefault(action_key, {})
        per_action[delta_key] = int(per_action.get(delta_key, 0) + 1)
        self._tracked_agent_token_digest = str(current.digest)

        return {
            "schema_name": "active_inference_navigation_state_estimate_v1",
            "schema_version": 1,
            "matched": True,
            "tracked_agent_token_id": str(current.digest),
            "tracked_agent_token_pair": {
                "previous_digest": str(previous.digest),
                "current_digest": str(current.digest),
            },
            "agent_pos_xy": {
                "x": int(current.centroid_x),
                "y": int(current.centroid_y),
            },
            "agent_pos_region": {
                "x": int(max(0, min(7, int(current.centroid_x) // 8))),
                "y": int(max(0, min(7, int(current.centroid_y) // 8))),
            },
            "delta_pos_xy": {"dx": int(delta_x), "dy": int(delta_y)},
            "action_id": int(executed_candidate.action_id),
            "control_schema_posterior": self._control_schema_posterior(),
        }

    def _reasoning_for_forced_reset(self, packet: ObservationPacketV1) -> dict[str, Any]:
        return {
            "schema_name": "active_inference_reasoning_v2",
            "schema_version": 2,
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
            "posterior_delta_report_previous_step": dict(
                self.hypothesis_bank.last_posterior_delta_report
            ),
            "action_space_constraint_report_v1": dict(
                self.hypothesis_bank.action_space_constraint_report
            ),
            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
            "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
            "no_change_streak": int(self._no_change_streak),
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
            "schema_name": "active_inference_reasoning_v2",
            "schema_version": 2,
            "phase": "failure_fallback",
            "failure_taxonomy_v1": {
                "failure_code": failure_code,
                "failure_message": failure_message,
            },
            "stage_diagnostics_v1": diagnostics.to_dicts(),
            "bottleneck_stage_v1": diagnostics.bottleneck_stage(),
            "observation_packet_summary": packet_summary,
            "hypothesis_summary": self.hypothesis_bank.summary(),
            "posterior_delta_report_previous_step": dict(
                self.hypothesis_bank.last_posterior_delta_report
            ),
            "action_space_constraint_report_v1": dict(
                self.hypothesis_bank.action_space_constraint_report
            ),
            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
            "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
            "no_change_streak": int(self._no_change_streak),
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
        selection_diagnostics: dict[str, Any] = {}
        action_space_constraint_report: dict[str, Any] = {}
        navigation_state_estimate: dict[str, Any] = {}
        phase = "control"

        diagnostics.start("observation_contract")
        try:
            frame_chain = list(frames[-self.frame_chain_window :]) if frames else []
            packet = build_observation_packet_v1(
                latest_frame,
                game_id=self.game_id,
                card_id=self.card_id,
                action_counter=self.action_counter,
                frame_chain=frame_chain,
            )
            self._available_actions_history.append([int(v) for v in packet.available_actions])
            if len(self._available_actions_history) > self.available_actions_history_window:
                self._available_actions_history = self._available_actions_history[
                    -self.available_actions_history_window :
                ]
            diagnostics.finish_ok(
                "observation_contract",
                {
                    "state": packet.state,
                    "levels_completed": int(packet.levels_completed),
                    "win_levels": int(packet.win_levels),
                    "available_action_count": int(len(packet.available_actions)),
                    "frame_height": int(len(packet.frame)),
                    "frame_width": int(len(packet.frame[0]) if packet.frame else 0),
                    "num_frames_received": int(packet.num_frames_received),
                    "frame_chain_micro_signature_count": int(
                        len(packet.frame_chain_micro_signatures)
                    ),
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

        diagnostics.start("action_space_constraint")
        try:
            action_space_constraint_report = self.hypothesis_bank.apply_action_space_constraints(
                packet.available_actions
            )
            diagnostics.finish_ok(
                "action_space_constraint",
                {
                    "active_hypothesis_count_before": int(
                        action_space_constraint_report.get(
                            "active_hypothesis_count_before",
                            0,
                        )
                    ),
                    "active_hypothesis_count_after": int(
                        action_space_constraint_report.get(
                            "active_hypothesis_count_after",
                            0,
                        )
                    ),
                    "mode_elimination_due_to_action_space_incompatibility": int(
                        action_space_constraint_report.get(
                            "mode_elimination_due_to_action_space_incompatibility",
                            0,
                        )
                    ),
                },
            )
        except Exception as exc:
            diagnostics.finish_rejected(
                "action_space_constraint",
                f"action_space_constraint_error::{type(exc).__name__}",
            )

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
                navigation_state_estimate = self._estimate_navigation_state(
                    self._previous_representation,
                    representation,
                    self._previous_action_candidate,
                )
                self._latest_navigation_state_estimate = dict(navigation_state_estimate)
                if str(causal_signature.obs_change_type) == "NO_CHANGE":
                    self._no_change_streak += 1
                else:
                    self._no_change_streak = 0
                posterior_report = dict(self.hypothesis_bank.last_posterior_delta_report)
                diagnostics.finish_ok(
                    "causal_update",
                    {
                        "updated": True,
                        "signature_digest": causal_signature.signature_digest,
                        "event_tags": list(causal_signature.event_tags),
                        "obs_change_type": str(causal_signature.obs_change_type),
                        "no_change_streak": int(self._no_change_streak),
                        "active_hypothesis_count_before": int(
                            posterior_report.get("active_hypothesis_count_before", 0)
                        ),
                        "active_hypothesis_count_after": int(
                            posterior_report.get("active_hypothesis_count_after", 0)
                        ),
                        "eliminated_count_by_reason": dict(
                            posterior_report.get("eliminated_count_by_reason", {})
                        ),
                        "mode_transition_soft_confidence": float(
                            posterior_report.get("mode_transition_soft_confidence", 0.0)
                        ),
                        "navigation_state_estimate_v1": dict(navigation_state_estimate),
                    },
                )
            else:
                self._no_change_streak = 0
                self._latest_navigation_state_estimate = {}
                diagnostics.finish_ok(
                    "causal_update",
                    {"updated": False, "reason": "insufficient_history"},
                )
        except Exception as exc:
            diagnostics.finish_rejected(
                "causal_update",
                f"causal_update_error::{type(exc).__name__}",
            )
            self._latest_navigation_state_estimate = {}

        remaining_budget = max(0, int(self.MAX_ACTIONS - int(self.action_counter)))
        early_probe_budget_remaining = max(
            0,
            int(self.early_probe_budget - int(self.action_counter)),
        )
        diagnostics.start("phase_determination")
        try:
            phase = self.policy.determine_phase(
                action_counter=self.action_counter,
                remaining_budget=remaining_budget,
                hypothesis_bank=self.hypothesis_bank,
            )
            diagnostics.finish_ok(
                "phase_determination",
                {
                    "phase": phase,
                    "posterior_entropy_bits": float(self.hypothesis_bank.posterior_entropy()),
                    "remaining_budget": int(remaining_budget),
                    "early_probe_budget_remaining": int(early_probe_budget_remaining),
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
            diagnostics.start("stop_loss_guard")
            stop_loss_applied = False
            stop_loss_candidate: ActionCandidateV1 | None = None
            if self._no_change_streak >= self.no_change_stop_loss_steps:
                if 7 in packet.available_actions:
                    stop_loss_candidate = ActionCandidateV1(
                        candidate_id="a7_stop_loss",
                        action_id=7,
                        source="stop_loss_guard",
                    )
                elif 0 in packet.available_actions:
                    stop_loss_candidate = ActionCandidateV1(
                        candidate_id="reset_stop_loss",
                        action_id=0,
                        source="stop_loss_guard",
                    )
                else:
                    stop_loss_candidate = ActionCandidateV1(
                        candidate_id="reset_stop_loss",
                        action_id=0,
                        source="stop_loss_guard",
                    )

            if stop_loss_candidate is not None:
                stop_loss_applied = True
                selected_candidate = stop_loss_candidate
                if int(stop_loss_candidate.action_id) == 0:
                    action = GameAction.RESET
                else:
                    action = self._candidate_to_game_action(stop_loss_candidate)
                diagnostics.finish_ok(
                    "stop_loss_guard",
                    {
                        "triggered": True,
                        "no_change_streak": int(self._no_change_streak),
                        "threshold": int(self.no_change_stop_loss_steps),
                        "selected_action_id": int(selected_candidate.action_id),
                    },
                )
                self._no_change_streak = 0
                action.reasoning = {
                    "schema_name": "active_inference_reasoning_v2",
                    "schema_version": 2,
                    "phase": phase,
                    "selected_candidate": selected_candidate.to_dict(),
                    "reason": "stop_loss_guard_triggered",
                    "hypothesis_summary": self.hypothesis_bank.summary(),
                    "posterior_delta_report_previous_step": dict(
                        self.hypothesis_bank.last_posterior_delta_report
                    ),
                    "action_space_constraint_report_v1": dict(action_space_constraint_report),
                    "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
                    "representation_summary": representation.summary,
                    "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                    "remaining_budget": int(remaining_budget),
                    "early_probe_budget_remaining": int(early_probe_budget_remaining),
                }
            else:
                diagnostics.finish_ok(
                    "stop_loss_guard",
                    {
                        "triggered": False,
                        "no_change_streak": int(self._no_change_streak),
                        "threshold": int(self.no_change_stop_loss_steps),
                    },
                )

            if not stop_loss_applied:
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
                            "schema_name": "active_inference_reasoning_v2",
                            "schema_version": 2,
                            "phase": phase,
                            "selected_candidate": selected_candidate.to_dict(),
                            "reason": "no_candidates",
                            "hypothesis_summary": self.hypothesis_bank.summary(),
                            "posterior_delta_report_previous_step": dict(
                                self.hypothesis_bank.last_posterior_delta_report
                            ),
                            "action_space_constraint_report_v1": dict(action_space_constraint_report),
                            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
                            "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                            "remaining_budget": int(remaining_budget),
                            "early_probe_budget_remaining": int(early_probe_budget_remaining),
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
                                "action6_candidate_diagnostics": dict(
                                    representation.summary.get(
                                        "action6_candidate_diagnostics", {}
                                    )
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
                            remaining_budget=remaining_budget,
                            action_select_count=self._action_select_count,
                            candidate_select_count=self._candidate_select_count,
                            early_probe_budget_remaining=early_probe_budget_remaining,
                        )
                        if ranked_entries:
                            selection_diagnostics = dict(
                                ranked_entries[0].witness.get(
                                    "selection_diagnostics_v1",
                                    {},
                                )
                            )
                        diagnostics.finish_ok(
                            "policy_selection",
                            {
                                "selected_candidate_id": selected_candidate.candidate_id,
                                "selected_action_id": int(selected_candidate.action_id),
                                "ranked_count": int(len(ranked_entries)),
                                "selection_diagnostics_v1": dict(selection_diagnostics),
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
                            "schema_name": "active_inference_reasoning_v2",
                            "schema_version": 2,
                            "phase": phase,
                            "selected_candidate": selected_candidate.to_dict(),
                            "selected_free_energy": (
                                ranked_entries[0].to_dict() if ranked_entries else None
                            ),
                            "top_k_candidates_by_efe": top_entries,
                            "hypothesis_summary": self.hypothesis_bank.summary(),
                            "posterior_delta_report_previous_step": dict(
                                self.hypothesis_bank.last_posterior_delta_report
                            ),
                            "action_space_constraint_report_v1": dict(action_space_constraint_report),
                            "selection_diagnostics_v1": dict(selection_diagnostics),
                            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
                            "representation_summary": representation.summary,
                            "causal_event_signature_previous_step": (
                                causal_signature.to_dict() if causal_signature else None
                            ),
                            "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                            "remaining_budget": int(remaining_budget),
                            "early_probe_budget_remaining": int(early_probe_budget_remaining),
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
                    "remaining_budget": int(remaining_budget),
                    "early_probe_budget_remaining": int(early_probe_budget_remaining),
                    "no_change_streak": int(self._no_change_streak),
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
                    "posterior_delta_report_previous_step": dict(
                        self.hypothesis_bank.last_posterior_delta_report
                    ),
                    "action_space_constraint_report_v1": dict(action_space_constraint_report),
                    "selection_diagnostics_v1": dict(selection_diagnostics),
                    "causal_event_signature_previous_step": (
                        causal_signature.to_dict() if causal_signature else None
                    ),
                    "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
                    "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                    "stage_diagnostics_v1": diagnostics.to_dicts(),
                    "bottleneck_stage_v1": diagnostics.bottleneck_stage(),
                }
            )

        selected_action_id = int(selected_candidate.action_id)
        selected_candidate_id = str(selected_candidate.candidate_id)
        self._action_select_count[selected_action_id] = (
            int(self._action_select_count.get(selected_action_id, 0)) + 1
        )
        self._candidate_select_count[selected_candidate_id] = (
            int(self._candidate_select_count.get(selected_candidate_id, 0)) + 1
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
                    "early_probe_budget_config": int(self.early_probe_budget),
                    "final_hypothesis_summary": self.hypothesis_bank.summary(),
                    "final_posterior_delta_report": dict(
                        self.hypothesis_bank.last_posterior_delta_report
                    ),
                    "final_action_space_constraint_report": dict(
                        self.hypothesis_bank.action_space_constraint_report
                    ),
                    "final_navigation_state_estimate_v1": dict(
                        self._latest_navigation_state_estimate
                    ),
                    "final_control_schema_posterior": self._control_schema_posterior(),
                    "final_action_select_count": {
                        str(key): int(value)
                        for (key, value) in sorted(self._action_select_count.items())
                    },
                    "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                    "final_no_change_streak": int(self._no_change_streak),
                }
            )
            self.trace_recorder.close()
            self._trace_closed = True
        super().cleanup(scorecard)
