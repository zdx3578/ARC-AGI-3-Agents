from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ...agent import Agent
from .contracts import (
    ActionCandidateV1,
    ObservationPacketV1,
    RepresentationStateV1,
    TransitionRecordV1,
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
        self.episode_session_id = uuid.uuid4().hex[:12]
        self.cross_episode_memory_hard_off = True
        self.cross_episode_memory_enable_requested = _env_bool(
            "ACTIVE_INFERENCE_ENABLE_CROSS_EPISODE_MEMORY",
            False,
        )
        self.cross_episode_memory_override_blocked = bool(
            self.cross_episode_memory_enable_requested
        )
        self.action_cost_objective_hard_off = True
        self.action_cost_objective_enable_requested = _env_bool(
            "ACTIVE_INFERENCE_ENABLE_ACTION_COST_OBJECTIVE",
            False,
        )
        self.action_cost_objective_override_blocked = bool(
            self.action_cost_objective_enable_requested
        )
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
        self.rollout_max_candidates = max(
            1,
            _env_int("ACTIVE_INFERENCE_ROLLOUT_MAX_CANDIDATES", 8),
        )
        self.rollout_only_in_exploit = _env_bool(
            "ACTIVE_INFERENCE_ROLLOUT_ONLY_IN_EXPLOIT",
            True,
        )
        self.region_revisit_hard_threshold = max(
            4,
            _env_int("ACTIVE_INFERENCE_REGION_REVISIT_HARD_THRESHOLD", 24),
        )
        self.sequence_rollout_frontier_weight = max(
            0.0,
            _env_float("ACTIVE_INFERENCE_SEQUENCE_ROLLOUT_FRONTIER_WEIGHT", 0.35),
        )
        self.sequence_rollout_direction_weight = max(
            0.0,
            _env_float("ACTIVE_INFERENCE_SEQUENCE_ROLLOUT_DIRECTION_WEIGHT", 0.25),
        )
        self.sequence_probe_score_margin = max(
            0.0,
            _env_float("ACTIVE_INFERENCE_SEQUENCE_PROBE_SCORE_MARGIN", 0.28),
        )
        self.sequence_probe_trigger_steps = max(
            1,
            _env_int("ACTIVE_INFERENCE_SEQUENCE_PROBE_TRIGGER_STEPS", 20),
        )
        self.coverage_sweep_target_regions = max(
            1,
            _env_int("ACTIVE_INFERENCE_COVERAGE_SWEEP_TARGET_REGIONS", 24),
        )
        self.coverage_sweep_score_margin = max(
            0.0,
            _env_float("ACTIVE_INFERENCE_COVERAGE_SWEEP_SCORE_MARGIN", 0.42),
        )
        self.coverage_resweep_interval = max(
            0,
            _env_int("ACTIVE_INFERENCE_COVERAGE_RESWEEP_INTERVAL", 96),
        )
        self.coverage_resweep_span = max(
            0,
            _env_int("ACTIVE_INFERENCE_COVERAGE_RESWEEP_SPAN", 24),
        )
        self.coverage_sweep_direction_retry_limit = max(
            1,
            _env_int("ACTIVE_INFERENCE_COVERAGE_SWEEP_DIRECTION_RETRY_LIMIT", 8),
        )
        self.coverage_sweep_min_region_visits = max(
            1,
            _env_int("ACTIVE_INFERENCE_COVERAGE_SWEEP_MIN_REGION_VISITS", 2),
        )
        self.coverage_prepass_steps = max(
            0,
            _env_int(
                "ACTIVE_INFERENCE_COVERAGE_PREPASS_STEPS",
                min(300, int(self.MAX_ACTIONS)),
            ),
        )
        self.coverage_matrix_sweep_enabled = _env_bool(
            "ACTIVE_INFERENCE_COVERAGE_MATRIX_SWEEP_ENABLED",
            True,
        )
        self.coverage_sweep_force_in_exploit = _env_bool(
            "ACTIVE_INFERENCE_COVERAGE_SWEEP_FORCE_IN_EXPLOIT",
            True,
        )
        self.enable_empirical_region_override = _env_bool(
            "ACTIVE_INFERENCE_ENABLE_EMPIRICAL_REGION_OVERRIDE",
            True,
        )
        self.early_probe_budget = max(
            0,
            _env_int(
                "ACTIVE_INFERENCE_EARLY_PROBE_BUDGET",
                max(8, min(512, int(round(float(self.MAX_ACTIONS) * 0.08)))),
            ),
        )
        self.action6_bucket_probe_min_attempts = max(
            1,
            _env_int("ACTIVE_INFERENCE_ACTION6_BUCKET_PROBE_MIN_ATTEMPTS", 3),
        )
        self.action6_subcluster_probe_min_attempts = max(
            1,
            _env_int("ACTIVE_INFERENCE_ACTION6_SUBCLUSTER_PROBE_MIN_ATTEMPTS", 2),
        )
        self.action6_probe_score_margin = max(
            0.0,
            _env_float("ACTIVE_INFERENCE_ACTION6_PROBE_SCORE_MARGIN", 0.06),
        )
        self.action6_explore_probe_score_margin = max(
            0.0,
            _env_float("ACTIVE_INFERENCE_ACTION6_EXPLORE_PROBE_SCORE_MARGIN", 0.12),
        )
        self.action6_stagnation_step_threshold = max(
            1,
            _env_int("ACTIVE_INFERENCE_ACTION6_STAGNATION_STEP_THRESHOLD", 12),
        )
        self.stagnation_probe_trigger_steps = max(
            1,
            _env_int("ACTIVE_INFERENCE_STAGNATION_PROBE_TRIGGER_STEPS", 24),
        )
        self.stagnation_probe_score_margin = max(
            0.0,
            _env_float("ACTIVE_INFERENCE_STAGNATION_PROBE_SCORE_MARGIN", 0.22),
        )
        self.stagnation_probe_min_action_usage_gap = max(
            1,
            _env_int("ACTIVE_INFERENCE_STAGNATION_PROBE_MIN_ACTION_USAGE_GAP", 8),
        )
        self.stagnation_stop_loss_steps = max(
            1,
            _env_int(
                "ACTIVE_INFERENCE_STAGNATION_STOP_LOSS_STEPS",
                max(80, int(round(float(self.MAX_ACTIONS) * 0.45))),
            ),
        )
        self.no_change_stop_loss_steps = max(
            1,
            _env_int("ACTIVE_INFERENCE_NO_CHANGE_STOP_LOSS_STEPS", 3),
        )

        explore_steps = max(1, _env_int("ACTIVE_INFERENCE_EXPLORE_STEPS", 20))
        self.exploration_base_steps = int(explore_steps)
        self.exploration_min_steps = max(
            1,
            _env_int("ACTIVE_INFERENCE_EXPLORATION_MIN_STEPS", 20),
        )
        self.exploration_max_steps = max(
            self.exploration_min_steps,
            _env_int(
                "ACTIVE_INFERENCE_EXPLORATION_MAX_STEPS",
                max(120, int(round(float(self.MAX_ACTIONS) * 0.70))),
            ),
        )
        self.exploration_fraction = max(
            0.0,
            min(1.0, _env_float("ACTIVE_INFERENCE_EXPLORATION_FRACTION", 0.35)),
        )
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
            rollout_max_candidates=self.rollout_max_candidates,
            rollout_only_in_exploit=self.rollout_only_in_exploit,
            ignore_action_cost=True,
            weight_overrides=weight_overrides,
            action6_bucket_probe_min_attempts=self.action6_bucket_probe_min_attempts,
            action6_subcluster_probe_min_attempts=self.action6_subcluster_probe_min_attempts,
            action6_probe_score_margin=self.action6_probe_score_margin,
            action6_explore_probe_score_margin=self.action6_explore_probe_score_margin,
            action6_stagnation_step_threshold=self.action6_stagnation_step_threshold,
            stagnation_probe_trigger_steps=self.stagnation_probe_trigger_steps,
            stagnation_probe_score_margin=self.stagnation_probe_score_margin,
            stagnation_probe_min_action_usage_gap=self.stagnation_probe_min_action_usage_gap,
            region_revisit_hard_threshold=self.region_revisit_hard_threshold,
            sequence_rollout_frontier_weight=self.sequence_rollout_frontier_weight,
            sequence_rollout_direction_weight=self.sequence_rollout_direction_weight,
            sequence_probe_score_margin=self.sequence_probe_score_margin,
            sequence_probe_trigger_steps=self.sequence_probe_trigger_steps,
            coverage_sweep_target_regions=self.coverage_sweep_target_regions,
            coverage_sweep_min_region_visits=self.coverage_sweep_min_region_visits,
            coverage_prepass_steps=self.coverage_prepass_steps,
            coverage_sweep_score_margin=self.coverage_sweep_score_margin,
            coverage_resweep_interval=self.coverage_resweep_interval,
            coverage_resweep_span=self.coverage_resweep_span,
            coverage_sweep_direction_retry_limit=self.coverage_sweep_direction_retry_limit,
            coverage_matrix_sweep_enabled=self.coverage_matrix_sweep_enabled,
            coverage_sweep_force_in_exploit=self.coverage_sweep_force_in_exploit,
        )
        self.hypothesis_bank = ActiveInferenceHypothesisBankV1()

        self._previous_packet: ObservationPacketV1 | None = None
        self._previous_representation: RepresentationStateV1 | None = None
        self._previous_action_candidate: ActionCandidateV1 | None = None
        self._no_change_streak = 0
        self._stagnation_streak = 0
        self._available_actions_history: list[list[int]] = []
        self._control_schema_counts: dict[str, dict[str, int]] = {}
        self._tracked_agent_token_digest: str | None = None
        self._last_known_agent_pos_region: tuple[int, int] | None = None
        self._latest_navigation_state_estimate: dict[str, Any] = {}
        self._action_select_count: dict[int, int] = {}
        self._candidate_select_count: dict[str, int] = {}
        self._cluster_select_count: dict[str, int] = {}
        self._subcluster_select_count: dict[str, int] = {}
        self._navigation_direction_history_window = max(
            8,
            _env_int("ACTIVE_INFERENCE_DIRECTION_HISTORY_WINDOW", 256),
        )
        self._recent_navigation_directions: list[str] = []
        self._navigation_direction_visit_count: dict[str, int] = {}
        self._navigation_direction_sequence_visit_count: dict[str, int] = {}
        self._navigation_attempt_count = 0
        self._navigation_blocked_count = 0
        self._navigation_moved_count = 0
        self._navigation_action_stats: dict[str, dict[str, int]] = {}
        self._blocked_edge_counts: dict[str, int] = {}
        self._edge_attempt_counts: dict[str, int] = {}
        self._region_visit_counts: dict[str, int] = {}
        self._region_action_transition_counts: dict[str, dict[str, int]] = {}
        self._click_bucket_stats: dict[str, dict[str, int]] = {}
        self._click_subcluster_stats: dict[str, dict[str, int]] = {}
        self._state_visit_count: dict[str, int] = {}
        self._state_action_visit_count: dict[str, int] = {}
        self._transition_edge_visit_count: dict[str, int] = {}
        self._state_outgoing_edges: dict[str, set[str]] = {}
        self._latest_transition_record: dict[str, Any] = {}

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

    def _stable_digest_payload_v1(self, payload: Any, *, prefix: str = "") -> str:
        try:
            canonical = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
        except Exception:
            canonical = repr(payload)
        digest = hashlib.sha256(canonical.encode("utf-8", errors="ignore")).hexdigest()[:24]
        return f"{prefix}{digest}" if prefix else digest

    def _frame_digest_v1(self, packet: ObservationPacketV1) -> str:
        if packet.frame_chain_digests:
            return str(packet.frame_chain_digests[-1])
        return hashlib.sha256(
            repr(packet.frame).encode("utf-8", errors="ignore")
        ).hexdigest()[:24]

    def _state_digest_v1(
        self,
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
    ) -> str:
        payload = {
            "state": str(packet.state),
            "levels_completed": int(packet.levels_completed),
            "win_levels": int(packet.win_levels),
            "available_actions": [int(v) for v in sorted(packet.available_actions)],
            "frame_digest": self._frame_digest_v1(packet),
            "object_count": int(representation.summary.get("object_count", 0)),
            "background_color": int(representation.summary.get("background_color", 0)),
            "color_histogram": dict(representation.summary.get("color_histogram", {})),
        }
        return self._stable_digest_payload_v1(payload, prefix="st:")

    def _action_token_from_candidate_v1(self, candidate: ActionCandidateV1) -> str:
        action_id = int(candidate.action_id)
        if action_id != 6:
            return f"a{action_id}"
        x = int(candidate.x if candidate.x is not None else -1)
        y = int(candidate.y if candidate.y is not None else -1)
        feature = candidate.metadata.get("coordinate_context_feature", {})
        if isinstance(feature, dict):
            subcluster = str(feature.get("click_context_subcluster_v1", "cv2:NA|fr=NA_NA|sub=lpNA"))
            return f"a6|{subcluster}|x={x}|y={y}"
        return f"a6|na|x={x}|y={y}"

    def _patch_digest_v1(
        self,
        frame: list[list[int]],
        *,
        x: int,
        y: int,
        radius: int = 2,
    ) -> str:
        height = len(frame)
        width = len(frame[0]) if frame else 0
        values: list[int] = []
        for dy in range(-int(radius), int(radius) + 1):
            for dx in range(-int(radius), int(radius) + 1):
                ny = int(y) + int(dy)
                nx = int(x) + int(dx)
                if ny < 0 or nx < 0 or ny >= height or nx >= width:
                    values.append(-1)
                else:
                    values.append(int(frame[ny][nx]))
        return self._stable_digest_payload_v1(values, prefix="patch:")

    def _find_object_by_digest_v1(
        self,
        representation: RepresentationStateV1,
        digest: str,
    ) -> dict[str, Any] | None:
        target = str(digest).strip()
        if not target:
            return None
        for obj in representation.object_nodes:
            if str(obj.digest) == target:
                return {
                    "object_id": str(obj.object_id),
                    "digest": str(obj.digest),
                    "color": int(obj.color),
                    "area": int(obj.area),
                    "centroid_x": int(obj.centroid_x),
                    "centroid_y": int(obj.centroid_y),
                    "bbox": [
                        int(obj.bbox_min_x),
                        int(obj.bbox_min_y),
                        int(obj.bbox_max_x),
                        int(obj.bbox_max_y),
                    ],
                }
        return None

    @staticmethod
    def _direction_bucket_from_delta_v1(dx: int, dy: int) -> str:
        if int(dx) == 0 and int(dy) == 0:
            return "dir_none"
        if abs(int(dx)) >= abs(int(dy)):
            if int(dx) > 0:
                return "dir_r"
            if int(dx) < 0:
                return "dir_l"
        if int(dy) > 0:
            return "dir_d"
        if int(dy) < 0:
            return "dir_u"
        return "dir_unknown"

    def _navigation_direction_bucket_from_estimate_v1(
        self,
        navigation_state_estimate: dict[str, Any],
    ) -> str:
        if not isinstance(navigation_state_estimate, dict):
            return "na"
        if not bool(navigation_state_estimate.get("matched", False)):
            return "na"
        delta = navigation_state_estimate.get("delta_pos_xy", {})
        if not isinstance(delta, dict):
            return "na"
        dx = int(delta.get("dx", 0))
        dy = int(delta.get("dy", 0))
        bucket = self._direction_bucket_from_delta_v1(dx, dy)
        if bucket in ("dir_l", "dir_r", "dir_u", "dir_d"):
            return str(bucket)
        return "na"

    @staticmethod
    def _parse_delta_key_v1(delta_key: str) -> tuple[int, int]:
        dx = 0
        dy = 0
        for part in str(delta_key).split("|"):
            token = str(part).strip()
            if token.startswith("dx="):
                try:
                    dx = int(token.split("=", 1)[1])
                except Exception:
                    dx = 0
            elif token.startswith("dy="):
                try:
                    dy = int(token.split("=", 1)[1])
                except Exception:
                    dy = 0
        return (int(dx), int(dy))

    def _current_agent_position_xy_v1(
        self,
        representation: RepresentationStateV1,
    ) -> tuple[int, int] | None:
        latest = self._latest_navigation_state_estimate
        if isinstance(latest, dict):
            pos = latest.get("agent_pos_xy", {})
            if isinstance(pos, dict):
                x = int(pos.get("x", -1))
                y = int(pos.get("y", -1))
                if x >= 0 and y >= 0:
                    return (int(x), int(y))
        if self._tracked_agent_token_digest:
            matched = self._find_object_by_digest_v1(
                representation,
                self._tracked_agent_token_digest,
            )
            if matched is not None:
                return (
                    int(matched.get("centroid_x", -1)),
                    int(matched.get("centroid_y", -1)),
                )
        return None

    def _navigation_key_targets_v1(
        self,
        representation: RepresentationStateV1,
        *,
        agent_pos_xy: tuple[int, int] | None,
    ) -> list[dict[str, Any]]:
        frame_height = int(max(1, representation.frame_height))
        frame_width = int(max(1, representation.frame_width))
        frame_area = float(max(1, frame_height * frame_width))
        if agent_pos_xy is None:
            ref_x = int(frame_width // 2)
            ref_y = int(frame_height // 2)
        else:
            ref_x = int(agent_pos_xy[0])
            ref_y = int(agent_pos_xy[1])

        color_counts: dict[int, int] = {}
        for obj in representation.object_nodes:
            color = int(obj.color)
            color_counts[color] = int(color_counts.get(color, 0) + 1)

        targets: list[dict[str, Any]] = []
        for obj in representation.object_nodes:
            if self._tracked_agent_token_digest and str(obj.digest) == str(
                self._tracked_agent_token_digest
            ):
                continue
            area = int(max(1, obj.area))
            if bool(obj.touches_boundary) and area >= int(0.45 * frame_area):
                continue
            bbox_w = int(max(1, int(obj.bbox_max_x) - int(obj.bbox_min_x) + 1))
            bbox_h = int(max(1, int(obj.bbox_max_y) - int(obj.bbox_min_y) + 1))
            bbox_area = int(max(1, bbox_w * bbox_h))
            fill_ratio = float(max(0.0, min(1.0, float(area) / float(bbox_area))))

            color_rarity = float(1.0 / float(max(1, color_counts.get(int(obj.color), 1))))
            smallness = float(max(0.0, 1.0 - min(1.0, float(area) / 120.0)))
            topness = float(
                max(
                    0.0,
                    1.0
                    - min(
                        1.0,
                        float(int(obj.centroid_y)) / float(max(1, frame_height - 1)),
                    ),
                )
            )
            center_y_bias = float(
                max(
                    0.0,
                    1.0
                    - (
                        abs(float(int(obj.centroid_y)) - (0.5 * float(frame_height)))
                        / max(1.0, 0.5 * float(frame_height))
                    ),
                )
            )
            center_x_bias = float(
                max(
                    0.0,
                    1.0
                    - (
                        abs(float(int(obj.centroid_x)) - (0.5 * float(frame_width)))
                        / max(1.0, 0.5 * float(frame_width))
                    ),
                )
            )
            interior_bonus = 0.0 if bool(obj.touches_boundary) else 1.0
            cross_like = (
                1.0
                if (
                    area <= 48
                    and bbox_w >= 3
                    and bbox_h >= 3
                    and 0.35 <= fill_ratio <= 0.75
                )
                else 0.0
            )
            gate_like = (
                1.0
                if (
                    int(obj.centroid_y) <= int(round(0.38 * float(frame_height)))
                    and area >= 12
                    and area <= 320
                    and bbox_w >= 3
                    and bbox_h >= 3
                    and fill_ratio <= 0.78
                )
                else 0.0
            )
            activation_cross_like = (
                1.0
                if (
                    not bool(obj.touches_boundary)
                    and area <= 24
                    and bbox_w <= 7
                    and bbox_h <= 7
                    and 0.18 <= fill_ratio <= 0.85
                    and int(obj.centroid_y)
                    >= int(round(0.30 * float(frame_height)))
                    and int(obj.centroid_y)
                    <= int(round(0.74 * float(frame_height)))
                )
                else 0.0
            )
            ui_band_penalty = (
                1.0
                if (
                    int(obj.centroid_y) <= int(round(0.24 * float(frame_height)))
                    or int(obj.centroid_y) >= int(round(0.92 * float(frame_height)))
                    or (
                        int(obj.centroid_x) <= int(round(0.14 * float(frame_width)))
                        and int(obj.centroid_y) >= int(round(0.74 * float(frame_height)))
                    )
                )
                else 0.0
            )
            salience = float(
                (0.95 * cross_like)
                + (0.82 * gate_like)
                + (0.36 * color_rarity)
                + (0.22 * smallness)
                + (0.18 * topness)
                + (0.16 * interior_bonus)
            )
            target_priority = float(
                salience
                + (0.88 * activation_cross_like)
                + (0.36 * center_y_bias)
                + (0.12 * center_x_bias)
                - (0.48 * ui_band_penalty)
            )
            if salience < 0.35:
                continue

            distance_from_agent = int(
                abs(int(obj.centroid_x) - int(ref_x)) + abs(int(obj.centroid_y) - int(ref_y))
            )
            kind = "salient"
            if cross_like >= gate_like and cross_like > 0.0:
                kind = "cross_like"
            elif gate_like > 0.0:
                kind = "gate_like"

            targets.append(
                {
                    "digest": str(obj.digest),
                    "object_id": str(obj.object_id),
                    "kind": str(kind),
                    "color": int(obj.color),
                    "area": int(area),
                    "centroid_x": int(obj.centroid_x),
                    "centroid_y": int(obj.centroid_y),
                    "bbox_w": int(bbox_w),
                    "bbox_h": int(bbox_h),
                    "fill_ratio": float(fill_ratio),
                    "touches_boundary": bool(obj.touches_boundary),
                    "salience": float(salience),
                    "target_priority": float(target_priority),
                    "center_y_bias": float(center_y_bias),
                    "center_x_bias": float(center_x_bias),
                    "activation_cross_like": float(activation_cross_like),
                    "ui_band_penalty": float(ui_band_penalty),
                    "distance_from_agent": int(distance_from_agent),
                }
            )

        targets.sort(
            key=lambda row: (
                -float(row.get("target_priority", row.get("salience", 0.0))),
                -float(row.get("salience", 0.0)),
                int(row.get("distance_from_agent", 10**9)),
                int(row.get("area", 10**9)),
                str(row.get("digest", "")),
            )
        )
        return targets[:4]

    def _navigation_target_features_v1(
        self,
        representation: RepresentationStateV1,
    ) -> dict[str, Any]:
        agent_pos_xy = self._current_agent_position_xy_v1(representation)
        targets = self._navigation_key_targets_v1(
            representation,
            agent_pos_xy=agent_pos_xy,
        )
        if not targets:
            return {
                "schema_name": "active_inference_navigation_target_features_v1",
                "schema_version": 1,
                "enabled": False,
                "agent_pos_xy": {"x": -1, "y": -1},
                "target_count": 0,
                "target_direction_bucket": "dir_unknown",
                "distance_before": -1.0,
                "target_salience": 0.0,
                "cross_like_enabled": False,
                "cross_like_target_region": {"x": -1, "y": -1},
                "cross_like_target_region_visit_count": 0,
                "targets": [],
            }

        primary = dict(targets[0])
        if agent_pos_xy is None:
            agent_x = -1
            agent_y = -1
            dx = 0
            dy = 0
            distance_before = -1.0
            direction_bucket = "dir_unknown"
            enabled = False
        else:
            agent_x = int(agent_pos_xy[0])
            agent_y = int(agent_pos_xy[1])
            dx = int(primary.get("centroid_x", 0)) - int(agent_x)
            dy = int(primary.get("centroid_y", 0)) - int(agent_y)
            distance_before = float(abs(int(dx)) + abs(int(dy)))
            direction_bucket = self._direction_bucket_from_delta_v1(dx, dy)
            enabled = bool(
                direction_bucket in ("dir_l", "dir_r", "dir_u", "dir_d")
                and float(distance_before) > 0.0
            )

        cross_target = next(
            (row for row in targets if str(row.get("kind", "")) == "cross_like"),
            None,
        )
        if isinstance(cross_target, dict):
            cross_rx = int(max(0, min(7, int(cross_target.get("centroid_x", -1)) // 8)))
            cross_ry = int(max(0, min(7, int(cross_target.get("centroid_y", -1)) // 8)))
            cross_region_key = f"{cross_rx}:{cross_ry}"
            cross_like_enabled = True
            cross_like_target_region = {"x": int(cross_rx), "y": int(cross_ry)}
            cross_like_target_region_visit_count = int(
                self._region_visit_counts.get(cross_region_key, 0)
            )
        else:
            cross_like_enabled = False
            cross_like_target_region = {"x": -1, "y": -1}
            cross_like_target_region_visit_count = 0

        return {
            "schema_name": "active_inference_navigation_target_features_v1",
            "schema_version": 1,
            "enabled": bool(enabled),
            "agent_pos_xy": {"x": int(agent_x), "y": int(agent_y)},
            "target_count": int(len(targets)),
            "target_digest": str(primary.get("digest", "NA")),
            "target_object_id": str(primary.get("object_id", "NA")),
            "target_kind": str(primary.get("kind", "salient")),
            "target_color": int(primary.get("color", -1)),
            "target_area": int(primary.get("area", 0)),
            "target_salience": float(primary.get("salience", 0.0)),
            "target_pos_xy": {
                "x": int(primary.get("centroid_x", -1)),
                "y": int(primary.get("centroid_y", -1)),
            },
            "target_region": {
                "x": int(max(0, min(7, int(primary.get("centroid_x", -1)) // 8))),
                "y": int(max(0, min(7, int(primary.get("centroid_y", -1)) // 8))),
            },
            "cross_like_enabled": bool(cross_like_enabled),
            "cross_like_target_region": dict(cross_like_target_region),
            "cross_like_target_region_visit_count": int(cross_like_target_region_visit_count),
            "distance_before": float(distance_before),
            "target_direction_bucket": str(direction_bucket),
            "targets": [
                {
                    "digest": str(row.get("digest", "NA")),
                    "kind": str(row.get("kind", "salient")),
                    "salience": float(row.get("salience", 0.0)),
                    "target_priority": float(row.get("target_priority", 0.0)),
                    "distance_from_agent": int(row.get("distance_from_agent", -1)),
                    "x": int(row.get("centroid_x", -1)),
                    "y": int(row.get("centroid_y", -1)),
                    "area": int(row.get("area", 0)),
                    "color": int(row.get("color", -1)),
                }
                for row in targets[:3]
            ],
        }

    def _region_graph_snapshot_v1(self, *, max_edges: int = 256) -> dict[str, Any]:
        current_region_key = "NA"
        if self._last_known_agent_pos_region is not None:
            rx, ry = self._last_known_agent_pos_region
            current_region_key = f"{int(rx)}:{int(ry)}"

        edges: list[dict[str, Any]] = []
        for region_action_key, target_histogram in self._region_action_transition_counts.items():
            if not isinstance(target_histogram, dict):
                continue
            try:
                source_key, action_token = str(region_action_key).split("|a", 1)
                action_id = int(action_token)
            except Exception:
                continue
            for target_key, count_raw in target_histogram.items():
                count = int(max(0, count_raw))
                if count <= 0:
                    continue
                target_region_key = str(target_key)
                edges.append(
                    {
                        "source_region_key": str(source_key),
                        "target_region_key": str(target_region_key),
                        "action_id": int(action_id),
                        "count": int(count),
                    }
                )
        edges.sort(
            key=lambda row: (
                -int(row.get("count", 0)),
                int(row.get("action_id", 0)),
                str(row.get("source_region_key", "")),
                str(row.get("target_region_key", "")),
            )
        )
        if len(edges) > int(max_edges):
            edges = edges[: int(max_edges)]

        return {
            "schema_name": "active_inference_region_graph_snapshot_v1",
            "schema_version": 1,
            "current_region_key": str(current_region_key),
            "region_visit_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._region_visit_counts.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
            "edge_count": int(len(edges)),
            "edges": list(edges),
        }

    def _action_context_payload_v1(
        self,
        *,
        candidate: ActionCandidateV1,
        packet_before: ObservationPacketV1,
        representation_before: RepresentationStateV1,
        tracked_token_before: str | None,
    ) -> dict[str, Any]:
        action_id = int(candidate.action_id)
        payload: dict[str, Any] = {
            "action_id": int(action_id),
            "action_token": self._action_token_from_candidate_v1(candidate),
            "scope": "global",
        }
        if action_id == 6:
            x = int(candidate.x if candidate.x is not None else 31)
            y = int(candidate.y if candidate.y is not None else 31)
            feature = candidate.metadata.get("coordinate_context_feature", {})
            if not isinstance(feature, dict):
                feature = {}
            payload.update(
                {
                    "scope": "global+click_local",
                    "x": int(x),
                    "y": int(y),
                    "click_patch_digest": self._patch_digest_v1(
                        packet_before.frame,
                        x=int(x),
                        y=int(y),
                        radius=2,
                    ),
                    "click_context_bucket_v2": str(
                        feature.get("click_context_bucket_v2", "cv2:NA")
                    ),
                    "click_context_subcluster_v1": str(
                        feature.get(
                            "click_context_subcluster_v1",
                            "cv2:NA|fr=NA_NA|sub=lpNA",
                        )
                    ),
                    "hit_object": int(feature.get("hit_object", -1)),
                    "hit_type": str(feature.get("hit_type", "none")),
                    "object_digest_bucket": str(
                        feature.get("object_digest_bucket", "NA")
                    ),
                    "rel_pos_bucket": str(feature.get("rel_pos_bucket", "NA")),
                    "on_object_boundary": str(
                        feature.get("on_object_boundary", "NA")
                    ),
                }
            )
            return payload

        if action_id in (1, 2, 3, 4, 7):
            payload["scope"] = "global+navigate_local"
            payload["tracked_token_before"] = str(tracked_token_before or "NA")
            payload["action_region_before"] = (
                f"{self._last_known_agent_pos_region[0]}:{self._last_known_agent_pos_region[1]}"
                if self._last_known_agent_pos_region is not None
                else "NA"
            )
            if tracked_token_before:
                matched_object = self._find_object_by_digest_v1(
                    representation_before,
                    tracked_token_before,
                )
                if matched_object is not None:
                    payload["tracked_object_before"] = dict(matched_object)
                    payload["tracked_patch_digest"] = self._patch_digest_v1(
                        packet_before.frame,
                        x=int(matched_object.get("centroid_x", 31)),
                        y=int(matched_object.get("centroid_y", 31)),
                        radius=2,
                    )
            return payload

        payload["scope"] = "global+action"
        payload["available_actions_before"] = [
            int(v) for v in sorted(packet_before.available_actions)
        ]
        return payload

    def _build_transition_record_v1(
        self,
        *,
        previous_packet: ObservationPacketV1,
        current_packet: ObservationPacketV1,
        previous_representation: RepresentationStateV1,
        current_representation: RepresentationStateV1,
        executed_candidate: ActionCandidateV1,
        observed_signature: Any,
        navigation_state_estimate: dict[str, Any],
        tracked_token_before: str | None,
    ) -> TransitionRecordV1:
        state_before_digest = self._state_digest_v1(
            previous_packet,
            previous_representation,
        )
        state_after_digest = self._state_digest_v1(
            current_packet,
            current_representation,
        )
        action_token = self._action_token_from_candidate_v1(executed_candidate)
        action_context = self._action_context_payload_v1(
            candidate=executed_candidate,
            packet_before=previous_packet,
            representation_before=previous_representation,
            tracked_token_before=tracked_token_before,
        )
        action_context_digest = self._stable_digest_payload_v1(
            action_context,
            prefix="ctx:",
        )
        effect_signature_key_v2 = str(
            getattr(observed_signature, "signature_key_v2", "")
        )
        effect_obs_change_type = str(
            getattr(observed_signature, "obs_change_type", "OBSERVED_UNCLASSIFIED")
        )
        effect_translation_delta_bucket = str(
            getattr(observed_signature, "translation_delta_bucket", "na")
        )
        if int(executed_candidate.action_id) in (1, 2, 3, 4):
            navigation_direction_bucket = self._navigation_direction_bucket_from_estimate_v1(
                navigation_state_estimate
            )
            if navigation_direction_bucket in ("dir_l", "dir_r", "dir_u", "dir_d"):
                effect_translation_delta_bucket = str(navigation_direction_bucket)
        state_action_key = f"{state_before_digest}|{action_token}"
        transition_edge_key = f"{state_before_digest}|{action_token}|{state_after_digest}"
        env_delta = {
            "state_transition": str(
                f"{previous_packet.state}->{current_packet.state}"
            ),
            "levels_completed_delta": int(
                int(current_packet.levels_completed)
                - int(previous_packet.levels_completed)
            ),
            "available_actions_before": [
                int(v) for v in sorted(previous_packet.available_actions)
            ],
            "available_actions_after": [
                int(v) for v in sorted(current_packet.available_actions)
            ],
            "available_actions_changed": bool(
                sorted(previous_packet.available_actions)
                != sorted(current_packet.available_actions)
            ),
        }
        effect_summary = {
            "signature_digest": str(getattr(observed_signature, "signature_digest", "")),
            "changed_pixel_count": int(
                getattr(observed_signature, "changed_pixel_count", 0)
            ),
            "changed_object_count": int(
                getattr(observed_signature, "changed_object_count", 0)
            ),
            "changed_area_ratio": float(
                getattr(observed_signature, "changed_area_ratio", 0.0)
            ),
            "event_tags": list(getattr(observed_signature, "event_tags", [])),
            "navigation_state_estimate_v1": dict(navigation_state_estimate),
        }
        return TransitionRecordV1(
            schema_name="active_inference_transition_record_v1",
            schema_version=1,
            action_counter=int(previous_packet.action_counter),
            state_before_digest=state_before_digest,
            state_after_digest=state_after_digest,
            action_token=action_token,
            action_context_digest=action_context_digest,
            effect_signature_key_v2=effect_signature_key_v2,
            effect_obs_change_type=effect_obs_change_type,
            effect_translation_delta_bucket=effect_translation_delta_bucket,
            state_action_key=state_action_key,
            transition_edge_key=transition_edge_key,
            env_delta=env_delta,
            action_context=action_context,
            effect_summary=effect_summary,
        )

    def _update_transition_graph_v1(self, transition: TransitionRecordV1) -> None:
        before = str(transition.state_before_digest)
        after = str(transition.state_after_digest)
        state_action_key = str(transition.state_action_key)
        edge_key = str(transition.transition_edge_key)
        self._state_visit_count[before] = int(self._state_visit_count.get(before, 0) + 1)
        self._state_visit_count[after] = int(self._state_visit_count.get(after, 0) + 1)
        self._state_action_visit_count[state_action_key] = int(
            self._state_action_visit_count.get(state_action_key, 0) + 1
        )
        self._transition_edge_visit_count[edge_key] = int(
            self._transition_edge_visit_count.get(edge_key, 0) + 1
        )
        outgoing = self._state_outgoing_edges.setdefault(before, set())
        outgoing.add(edge_key)
        self._latest_transition_record = transition.to_dict()

    def _transition_graph_summary_v1(self) -> dict[str, Any]:
        return {
            "schema_name": "active_inference_transition_graph_summary_v1",
            "schema_version": 1,
            "state_count": int(len(self._state_visit_count)),
            "state_action_count": int(len(self._state_action_visit_count)),
            "edge_count": int(len(self._transition_edge_visit_count)),
            "state_visit_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._state_visit_count.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )
            },
            "state_action_visit_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._state_action_visit_count.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
            "edge_visit_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._transition_edge_visit_count.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
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

    def _memory_policy_v1(self) -> dict[str, Any]:
        return {
            "schema_name": "active_inference_memory_policy_v1",
            "schema_version": 1,
            "episode_session_id": str(self.episode_session_id),
            "cross_episode_memory": "off_hard",
            "persistent_learning_store_used": False,
            "enable_requested": bool(self.cross_episode_memory_enable_requested),
            "override_blocked": bool(self.cross_episode_memory_override_blocked),
        }

    def _reasoning_policy_payloads_v1(
        self,
        packet: ObservationPacketV1 | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        remaining_budget = max(0, int(self.MAX_ACTIONS - int(self.action_counter)))
        early_probe_budget_remaining = max(
            0,
            int(self.early_probe_budget - int(self.action_counter)),
        )
        effective_explore_steps = self._effective_explore_steps(packet)
        exploration_policy_payload = self._exploration_policy_v1(
            packet=packet,
            effective_explore_steps=effective_explore_steps,
            remaining_budget=remaining_budget,
            early_probe_budget_remaining=early_probe_budget_remaining,
        )
        return self._memory_policy_v1(), exploration_policy_payload

    def _effective_explore_steps(self, packet: ObservationPacketV1 | None) -> int:
        available_action_count = 1
        if packet is not None:
            available_action_count = max(1, int(len(packet.available_actions)))
        action_space_factor = available_action_count * 4
        budget_factor = int(round(float(self.MAX_ACTIONS) * float(self.exploration_fraction)))
        effective = max(
            int(self.exploration_base_steps),
            int(self.exploration_min_steps),
            int(action_space_factor),
            int(budget_factor),
        )
        effective = min(
            int(self.exploration_max_steps),
            int(self.MAX_ACTIONS),
            int(effective),
        )
        return int(max(1, effective))

    def _exploration_policy_v1(
        self,
        *,
        packet: ObservationPacketV1 | None,
        effective_explore_steps: int,
        remaining_budget: int,
        early_probe_budget_remaining: int,
    ) -> dict[str, Any]:
        available_action_count = int(len(packet.available_actions)) if packet is not None else 0
        exploration_budget_remaining = max(
            0,
            int(effective_explore_steps - int(self.action_counter)),
        )
        return {
            "schema_name": "active_inference_exploration_policy_v1",
            "schema_version": 1,
            "base_explore_steps": int(self.exploration_base_steps),
            "effective_explore_steps": int(effective_explore_steps),
            "exploration_min_steps": int(self.exploration_min_steps),
            "exploration_max_steps": int(self.exploration_max_steps),
            "exploration_fraction": float(self.exploration_fraction),
            "available_action_count": int(available_action_count),
            "remaining_budget": int(remaining_budget),
            "exploration_budget_remaining": int(exploration_budget_remaining),
            "early_probe_budget_config": int(self.early_probe_budget),
            "early_probe_budget_remaining": int(early_probe_budget_remaining),
            "action6_bucket_probe_min_attempts": int(
                self.action6_bucket_probe_min_attempts
            ),
            "action6_subcluster_probe_min_attempts": int(
                self.action6_subcluster_probe_min_attempts
            ),
            "action6_probe_score_margin": float(self.action6_probe_score_margin),
            "stagnation_probe_trigger_steps": int(self.stagnation_probe_trigger_steps),
            "stagnation_probe_score_margin": float(self.stagnation_probe_score_margin),
            "stagnation_probe_min_action_usage_gap": int(
                self.stagnation_probe_min_action_usage_gap
            ),
            "stagnation_stop_loss_steps": int(self.stagnation_stop_loss_steps),
            "region_revisit_hard_threshold": int(self.region_revisit_hard_threshold),
            "sequence_probe_score_margin": float(self.sequence_probe_score_margin),
            "sequence_probe_trigger_steps": int(self.sequence_probe_trigger_steps),
            "sequence_rollout_frontier_weight": float(
                self.sequence_rollout_frontier_weight
            ),
            "sequence_rollout_direction_weight": float(
                self.sequence_rollout_direction_weight
            ),
            "navigation_direction_history_window": int(
                self._navigation_direction_history_window
            ),
            "action_cost_in_objective": "off_hard",
            "action_cost_enable_requested": bool(self.action_cost_objective_enable_requested),
            "action_cost_override_blocked": bool(
                self.action_cost_objective_override_blocked
            ),
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

    def _click_context_bucket_from_candidate(
        self,
        candidate: ActionCandidateV1 | None,
    ) -> str:
        if candidate is None:
            return "na"
        feature = candidate.metadata.get("coordinate_context_feature", {})
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
            f"hit={hit}|boundary={boundary}|dist={dist_bucket}|region={coarse_x}:{coarse_y}"
        )

    def _candidate_cluster_id(self, candidate: ActionCandidateV1 | None) -> str:
        if candidate is None:
            return "na"
        action_id = int(candidate.action_id)
        if action_id != 6:
            return f"a{action_id}"
        return f"a6|{self._click_context_bucket_from_candidate(candidate)}"

    def _click_context_subcluster_from_candidate(
        self,
        candidate: ActionCandidateV1 | None,
    ) -> str:
        if candidate is None:
            return "cv2:NA|fr=NA_NA|sub=lpNA"
        feature = candidate.metadata.get("coordinate_context_feature", {})
        if isinstance(feature, dict):
            subcluster = str(feature.get("click_context_subcluster_v1", "")).strip()
            if subcluster:
                return subcluster
        return "cv2:NA|fr=NA_NA|sub=lpNA"

    def _candidate_subcluster_id(self, candidate: ActionCandidateV1 | None) -> str:
        if candidate is None:
            return "na"
        action_id = int(candidate.action_id)
        if action_id != 6:
            return f"a{action_id}"
        return f"a6|{self._click_context_subcluster_from_candidate(candidate)}"

    def _navigation_candidate_stats(self, action_id: int) -> dict[str, Any]:
        action_key = str(int(action_id))
        action_stats = self._navigation_action_stats.get(action_key, {})
        attempts = int(action_stats.get("attempts", 0))
        blocked = int(action_stats.get("blocked", 0))
        moved = int(action_stats.get("moved", 0))
        blocked_rate = float(blocked / float(max(1, attempts)))
        moved_rate = float(moved / float(max(1, attempts)))

        edge_key = "NA"
        edge_attempts = 0
        edge_blocked = 0
        edge_blocked_rate = 0.0
        revisit_count_current = 0
        if self._last_known_agent_pos_region is not None:
            rx, ry = self._last_known_agent_pos_region
            region_key = f"{rx}:{ry}"
            revisit_count_current = int(self._region_visit_counts.get(region_key, 0))
            edge_key = f"region={rx}:{ry}|action={int(action_id)}"
            edge_attempts = int(self._edge_attempt_counts.get(edge_key, 0))
            edge_blocked = int(self._blocked_edge_counts.get(edge_key, 0))
            edge_blocked_rate = float(edge_blocked / float(max(1, edge_attempts)))

        return {
            "action_attempts": int(attempts),
            "action_blocked": int(blocked),
            "action_moved": int(moved),
            "action_blocked_rate": float(blocked_rate),
            "action_moved_rate": float(moved_rate),
            "edge_key": str(edge_key),
            "edge_attempts": int(edge_attempts),
            "edge_blocked": int(edge_blocked),
            "edge_blocked_rate": float(edge_blocked_rate),
            "region_revisit_count_current": int(revisit_count_current),
        }

    def _predicted_region_features_v1(
        self,
        *,
        action_id: int,
        action_posterior: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": "active_inference_predicted_region_features_v1",
            "schema_version": 1,
            "enabled": False,
            "action_id": int(action_id),
            "current_region": {"x": -1, "y": -1},
            "predicted_region": {"x": -1, "y": -1},
            "current_region_key": "NA",
            "current_region_source": "unknown",
            "predicted_region_key": "NA",
            "predicted_region_source": "posterior_expected_delta",
            "predicted_region_visit_count": 0,
            "current_region_visit_count": 0,
            "known_region_count": 0,
            "min_region_visit_count": 0,
            "regions_visited_at_least_twice": 0,
            "region_visit_total": 0,
            "max_region_visit_count": 0,
            "empirical_transition_total": 0,
            "empirical_transition_target_key": "NA",
            "empirical_transition_target": {"x": -1, "y": -1},
            "empirical_transition_confidence": 0.0,
            "empirical_transition_override_applied": False,
            "empirical_transition_override_enabled": bool(
                self.enable_empirical_region_override
            ),
            "empirical_transition_frontier_key": "NA",
            "empirical_transition_frontier": {"x": -1, "y": -1},
            "empirical_transition_frontier_visit_count": 0,
            "empirical_transition_frontier_confidence": 0.0,
            "dominant_delta_key": "",
            "dominant_delta_pos_xy": {"dx": 0, "dy": 0},
            "expected_delta_pos_xy": {"dx": 0.0, "dy": 0.0},
            "confidence": 0.0,
            "edge_attempts": 0,
            "edge_blocked_rate": 0.0,
            "edge_key": "NA",
        }
        if int(action_id) not in (1, 2, 3, 4):
            payload["reason"] = "non_navigation_action"
            return payload
        current_region: tuple[int, int] | None = None
        latest_navigation = (
            self._latest_navigation_state_estimate
            if isinstance(self._latest_navigation_state_estimate, dict)
            else {}
        )
        latest_region = latest_navigation.get("agent_pos_region", {})
        if (
            bool(latest_navigation.get("matched", False))
            and isinstance(latest_region, dict)
        ):
            latest_rx = int(latest_region.get("x", -1))
            latest_ry = int(latest_region.get("y", -1))
            if latest_rx >= 0 and latest_ry >= 0:
                current_region = (int(latest_rx), int(latest_ry))
                payload["current_region_source"] = "latest_navigation_state"
        if current_region is None and self._last_known_agent_pos_region is not None:
            current_region = (
                int(self._last_known_agent_pos_region[0]),
                int(self._last_known_agent_pos_region[1]),
            )
            payload["current_region_source"] = "last_known_region"
        if current_region is None:
            payload["reason"] = "unknown_current_region"
            return payload

        posterior_raw = action_posterior if isinstance(action_posterior, dict) else {}
        weights: list[tuple[str, float]] = []
        total = 0.0
        for (delta_key, value) in posterior_raw.items():
            try:
                p = float(value)
            except Exception:
                p = 0.0
            if p <= 0.0:
                continue
            weights.append((str(delta_key), float(p)))
            total += float(p)
        if total <= 1.0e-9:
            payload["reason"] = "no_posterior_mass"
            return payload

        current_rx, current_ry = current_region
        payload["current_region"] = {"x": int(current_rx), "y": int(current_ry)}
        current_region_key = f"{current_rx}:{current_ry}"
        payload["current_region_key"] = str(current_region_key)
        payload["current_region_visit_count"] = int(
            self._region_visit_counts.get(current_region_key, 0)
        )
        payload["known_region_count"] = int(len(self._region_visit_counts))
        payload["min_region_visit_count"] = int(
            min(self._region_visit_counts.values(), default=0)
        )
        payload["regions_visited_at_least_twice"] = int(
            sum(1 for count in self._region_visit_counts.values() if int(count) >= 2)
        )
        payload["region_visit_total"] = int(sum(self._region_visit_counts.values()))
        payload["max_region_visit_count"] = int(
            max(self._region_visit_counts.values(), default=0)
        )
        empirical_transition_counts = self._region_action_transition_counts.get(
            f"{current_region_key}|a{int(action_id)}",
            {},
        )
        if isinstance(empirical_transition_counts, dict):
            empirical_total = int(
                sum(
                    int(max(0, count))
                    for count in empirical_transition_counts.values()
                )
            )
            payload["empirical_transition_total"] = int(empirical_total)
            if empirical_total > 0:
                empirical_target_key, empirical_target_count = max(
                    (
                        (str(region_key), int(max(0, count)))
                        for (region_key, count) in empirical_transition_counts.items()
                    ),
                    key=lambda item: item[1],
                )
                payload["empirical_transition_target_key"] = str(empirical_target_key)
                confidence = float(
                    float(empirical_target_count) / float(max(1, empirical_total))
                )
                payload["empirical_transition_confidence"] = float(
                    max(0.0, min(1.0, confidence))
                )
                try:
                    target_rx_raw, target_ry_raw = str(empirical_target_key).split(":", 1)
                    target_rx = int(target_rx_raw)
                    target_ry = int(target_ry_raw)
                except Exception:
                    target_rx = -1
                    target_ry = -1
                payload["empirical_transition_target"] = {
                    "x": int(target_rx),
                    "y": int(target_ry),
                }
                frontier_key = "NA"
                frontier_count = 0
                frontier_visit = 10**9
                for (candidate_region_key, candidate_count_raw) in empirical_transition_counts.items():
                    candidate_count = int(max(0, candidate_count_raw))
                    if candidate_count <= 0:
                        continue
                    candidate_key = str(candidate_region_key)
                    candidate_visit = int(self._region_visit_counts.get(candidate_key, 0))
                    if (
                        candidate_visit < frontier_visit
                        or (
                            candidate_visit == frontier_visit
                            and candidate_count > frontier_count
                        )
                    ):
                        frontier_key = str(candidate_key)
                        frontier_count = int(candidate_count)
                        frontier_visit = int(candidate_visit)
                if frontier_key != "NA":
                    try:
                        frontier_rx_raw, frontier_ry_raw = str(frontier_key).split(":", 1)
                        frontier_rx = int(frontier_rx_raw)
                        frontier_ry = int(frontier_ry_raw)
                    except Exception:
                        frontier_rx = -1
                        frontier_ry = -1
                    payload["empirical_transition_frontier_key"] = str(frontier_key)
                    payload["empirical_transition_frontier"] = {
                        "x": int(frontier_rx),
                        "y": int(frontier_ry),
                    }
                    payload["empirical_transition_frontier_visit_count"] = int(
                        max(0, frontier_visit if frontier_visit < 10**9 else 0)
                    )
                    payload["empirical_transition_frontier_confidence"] = float(
                        float(frontier_count) / float(max(1, empirical_total))
                    )

        expected_dx = 0.0
        expected_dy = 0.0
        dominant_delta_key = ""
        dominant_prob = -1.0
        dominant_dx = 0
        dominant_dy = 0
        for (delta_key, raw_p) in weights:
            p = float(raw_p / total)
            dx, dy = self._parse_delta_key_v1(delta_key)
            expected_dx += p * float(dx)
            expected_dy += p * float(dy)
            if p > dominant_prob:
                dominant_prob = float(p)
                dominant_delta_key = str(delta_key)
                dominant_dx = int(dx)
                dominant_dy = int(dy)

        center_x = int((current_rx * 8) + 4)
        center_y = int((current_ry * 8) + 4)
        predicted_center_x = int(round(float(center_x) + float(expected_dx)))
        predicted_center_y = int(round(float(center_y) + float(expected_dy)))
        predicted_rx = int(max(0, min(7, predicted_center_x // 8)))
        predicted_ry = int(max(0, min(7, predicted_center_y // 8)))
        empirical_total = int(payload.get("empirical_transition_total", 0))
        empirical_target = payload.get("empirical_transition_target", {})
        if not isinstance(empirical_target, dict):
            empirical_target = {}
        empirical_target_rx = int(empirical_target.get("x", -1))
        empirical_target_ry = int(empirical_target.get("y", -1))
        empirical_confidence = float(
            payload.get("empirical_transition_confidence", 0.0)
        )
        if (
            bool(self.enable_empirical_region_override)
            and empirical_total >= 2
            and empirical_target_rx >= 0
            and empirical_target_ry >= 0
        ):
            predicted_rx = int(max(0, min(7, empirical_target_rx)))
            predicted_ry = int(max(0, min(7, empirical_target_ry)))
            payload["predicted_region_source"] = "empirical_transition"
            payload["empirical_transition_override_applied"] = True
        elif (
            empirical_total >= 2
            and empirical_target_rx >= 0
            and empirical_target_ry >= 0
        ):
            payload["predicted_region_source"] = "posterior_expected_delta"
        predicted_region_key = f"{predicted_rx}:{predicted_ry}"
        payload["predicted_region"] = {"x": int(predicted_rx), "y": int(predicted_ry)}
        payload["predicted_region_key"] = str(predicted_region_key)
        payload["predicted_region_visit_count"] = int(
            self._region_visit_counts.get(predicted_region_key, 0)
        )
        payload["dominant_delta_key"] = str(dominant_delta_key)
        payload["dominant_delta_pos_xy"] = {
            "dx": int(dominant_dx),
            "dy": int(dominant_dy),
        }
        payload["expected_delta_pos_xy"] = {
            "dx": float(expected_dx),
            "dy": float(expected_dy),
        }
        payload["confidence"] = float(
            max(0.0, min(1.0, max(float(dominant_prob), float(empirical_confidence))))
        )
        edge_key = f"region={current_rx}:{current_ry}|action={int(action_id)}"
        edge_attempts = int(self._edge_attempt_counts.get(edge_key, 0))
        edge_blocked = int(self._blocked_edge_counts.get(edge_key, 0))
        payload["edge_attempts"] = int(edge_attempts)
        payload["edge_blocked_rate"] = float(
            float(edge_blocked) / float(max(1, edge_attempts))
        )
        payload["edge_key"] = str(edge_key)
        payload["enabled"] = True
        return payload

    def _transition_exploration_stats_v1(
        self,
        *,
        state_digest_current: str,
        candidate: ActionCandidateV1,
    ) -> dict[str, Any]:
        action_token = self._action_token_from_candidate_v1(candidate)
        state_action_key = f"{state_digest_current}|{action_token}"
        outgoing_edges = self._state_outgoing_edges.get(str(state_digest_current), set())
        return {
            "state_digest_current": str(state_digest_current),
            "action_token": str(action_token),
            "state_action_key": str(state_action_key),
            "state_visit_count": int(
                self._state_visit_count.get(str(state_digest_current), 0)
            ),
            "state_action_visit_count": int(
                self._state_action_visit_count.get(str(state_action_key), 0)
            ),
            "state_outgoing_edge_count": int(len(outgoing_edges)),
        }

    def _update_operability_stats_v1(
        self,
        *,
        executed_candidate: ActionCandidateV1,
        causal_signature: Any,
        navigation_state_estimate: dict[str, Any],
    ) -> None:
        action_id = int(executed_candidate.action_id)
        obs_change_type = str(getattr(causal_signature, "obs_change_type", "OBSERVED_UNCLASSIFIED"))
        level_delta = int(getattr(causal_signature, "level_delta", 0))
        translation_delta_bucket = str(
            getattr(causal_signature, "translation_delta_bucket", "na")
        )
        if action_id in (1, 2, 3, 4):
            self._navigation_attempt_count += 1
            action_key = str(action_id)
            action_stats = self._navigation_action_stats.setdefault(
                action_key,
                {"attempts": 0, "blocked": 0, "moved": 0},
            )
            action_stats["attempts"] = int(action_stats.get("attempts", 0) + 1)
            edge_key = None
            source_region_key = "NA"
            if self._last_known_agent_pos_region is not None:
                rx, ry = self._last_known_agent_pos_region
                source_region_key = f"{rx}:{ry}"
                edge_key = f"region={rx}:{ry}|action={action_id}"
                self._edge_attempt_counts[edge_key] = int(
                    self._edge_attempt_counts.get(edge_key, 0) + 1
                )
            nav_matched = bool(navigation_state_estimate.get("matched", False))
            nav_delta = navigation_state_estimate.get("delta_pos_xy", {})
            if not isinstance(nav_delta, dict):
                nav_delta = {}
            nav_dx = int(nav_delta.get("dx", 0))
            nav_dy = int(nav_delta.get("dy", 0))
            nav_has_motion = bool(nav_matched and (abs(nav_dx) + abs(nav_dy) > 0))
            moved = bool(nav_has_motion)
            if moved:
                self._navigation_moved_count += 1
                action_stats["moved"] = int(action_stats.get("moved", 0) + 1)
                movement_direction_bucket = str(translation_delta_bucket)
                navigation_direction_bucket = self._navigation_direction_bucket_from_estimate_v1(
                    navigation_state_estimate
                )
                if navigation_direction_bucket in ("dir_l", "dir_r", "dir_u", "dir_d"):
                    movement_direction_bucket = str(navigation_direction_bucket)
                if movement_direction_bucket in ("dir_l", "dir_r", "dir_u", "dir_d"):
                    self._navigation_direction_visit_count[movement_direction_bucket] = int(
                        self._navigation_direction_visit_count.get(
                            movement_direction_bucket,
                            0,
                        )
                        + 1
                    )
                    if self._recent_navigation_directions:
                        previous_direction = str(self._recent_navigation_directions[-1])
                        if previous_direction in ("dir_l", "dir_r", "dir_u", "dir_d"):
                            sequence_key = (
                                f"{previous_direction}->{movement_direction_bucket}"
                            )
                            self._navigation_direction_sequence_visit_count[sequence_key] = int(
                                self._navigation_direction_sequence_visit_count.get(
                                    sequence_key,
                                    0,
                                )
                                + 1
                            )
                    self._recent_navigation_directions.append(movement_direction_bucket)
                    if len(self._recent_navigation_directions) > int(
                        self._navigation_direction_history_window
                    ):
                        self._recent_navigation_directions = self._recent_navigation_directions[
                            -int(self._navigation_direction_history_window) :
                        ]
                region = navigation_state_estimate.get("agent_pos_region", {})
                if isinstance(region, dict):
                    rx = int(region.get("x", -1))
                    ry = int(region.get("y", -1))
                    if rx >= 0 and ry >= 0:
                        self._last_known_agent_pos_region = (rx, ry)
                        target_region_key = f"{rx}:{ry}"
                        self._region_visit_counts[target_region_key] = int(
                            self._region_visit_counts.get(target_region_key, 0) + 1
                        )
                        if source_region_key != "NA":
                            region_action_key = f"{source_region_key}|a{action_id}"
                            target_histogram = self._region_action_transition_counts.setdefault(
                                region_action_key,
                                {},
                            )
                            target_histogram[target_region_key] = int(
                                target_histogram.get(target_region_key, 0) + 1
                            )
            else:
                self._navigation_blocked_count += 1
                action_stats["blocked"] = int(action_stats.get("blocked", 0) + 1)
                if edge_key is not None:
                    self._blocked_edge_counts[edge_key] = (
                        int(self._blocked_edge_counts.get(edge_key, 0)) + 1
                    )

        if action_id == 6:
            bucket = self._click_context_bucket_from_candidate(executed_candidate)
            stats = self._click_bucket_stats.setdefault(
                bucket,
                {"attempts": 0, "non_no_change": 0, "progress": 0},
            )
            stats["attempts"] = int(stats.get("attempts", 0) + 1)
            if obs_change_type != "NO_CHANGE":
                stats["non_no_change"] = int(stats.get("non_no_change", 0) + 1)
            if level_delta > 0:
                stats["progress"] = int(stats.get("progress", 0) + 1)

            subcluster = self._click_context_subcluster_from_candidate(executed_candidate)
            sub_stats = self._click_subcluster_stats.setdefault(
                subcluster,
                {"attempts": 0, "non_no_change": 0, "progress": 0},
            )
            sub_stats["attempts"] = int(sub_stats.get("attempts", 0) + 1)
            if obs_change_type != "NO_CHANGE":
                sub_stats["non_no_change"] = int(sub_stats.get("non_no_change", 0) + 1)
            if level_delta > 0:
                sub_stats["progress"] = int(sub_stats.get("progress", 0) + 1)

    def _operability_diagnostics_v1(self) -> dict[str, Any]:
        nav_attempts = int(self._navigation_attempt_count)
        nav_blocked = int(self._navigation_blocked_count)
        nav_moved = int(self._navigation_moved_count)
        click_summary: dict[str, Any] = {}
        for bucket, stats in sorted(self._click_bucket_stats.items()):
            attempts = int(stats.get("attempts", 0))
            non_no_change = int(stats.get("non_no_change", 0))
            progress = int(stats.get("progress", 0))
            click_summary[str(bucket)] = {
                "attempts": attempts,
                "non_no_change": non_no_change,
                "progress": progress,
                "non_no_change_rate": float(non_no_change / float(max(1, attempts))),
                "progress_rate": float(progress / float(max(1, attempts))),
            }
        click_subcluster_summary: dict[str, Any] = {}
        for subcluster, stats in sorted(self._click_subcluster_stats.items()):
            attempts = int(stats.get("attempts", 0))
            non_no_change = int(stats.get("non_no_change", 0))
            progress = int(stats.get("progress", 0))
            click_subcluster_summary[str(subcluster)] = {
                "attempts": attempts,
                "non_no_change": non_no_change,
                "progress": progress,
                "non_no_change_rate": float(non_no_change / float(max(1, attempts))),
                "progress_rate": float(progress / float(max(1, attempts))),
            }
        navigation_action_summary: dict[str, Any] = {}
        for action_key, stats in sorted(self._navigation_action_stats.items()):
            attempts = int(stats.get("attempts", 0))
            blocked = int(stats.get("blocked", 0))
            moved = int(stats.get("moved", 0))
            navigation_action_summary[str(action_key)] = {
                "attempts": int(attempts),
                "blocked": int(blocked),
                "moved": int(moved),
                "blocked_rate": float(blocked / float(max(1, attempts))),
                "moved_rate": float(moved / float(max(1, attempts))),
            }
        region_action_transition_flat: list[tuple[str, int]] = []
        for region_action_key, target_histogram in self._region_action_transition_counts.items():
            if not isinstance(target_histogram, dict):
                continue
            for target_region_key, count in target_histogram.items():
                transition_count = int(max(0, count))
                if transition_count <= 0:
                    continue
                region_action_transition_flat.append(
                    (f"{region_action_key}|to={target_region_key}", transition_count)
                )
        return {
            "schema_name": "active_inference_operability_diagnostics_v1",
            "schema_version": 1,
            "navigation_attempt_count": nav_attempts,
            "navigation_moved_count": nav_moved,
            "navigation_blocked_count": nav_blocked,
            "navigation_blocked_rate": float(nav_blocked / float(max(1, nav_attempts))),
            "blocked_edge_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._blocked_edge_counts.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )
            },
            "navigation_action_stats": navigation_action_summary,
            "region_visit_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._region_visit_counts.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )
            },
            "region_action_transition_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    region_action_transition_flat,
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
            "click_bucket_effectiveness": click_summary,
            "click_subcluster_effectiveness": click_subcluster_summary,
        }

    def _navigation_sequence_diagnostics_v1(self) -> dict[str, Any]:
        return {
            "schema_name": "active_inference_navigation_sequence_diagnostics_v1",
            "schema_version": 1,
            "history_window": int(self._navigation_direction_history_window),
            "history_length": int(len(self._recent_navigation_directions)),
            "recent_navigation_directions": [
                str(v) for v in self._recent_navigation_directions[-32:]
            ],
            "direction_visit_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._navigation_direction_visit_count.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )
            },
            "direction_sequence_visit_histogram": {
                str(key): int(value)
                for (key, value) in sorted(
                    self._navigation_direction_sequence_visit_count.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
        }

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
        action_id = int(executed_candidate.action_id)
        expected_dir = {
            1: (0, -1),  # up
            2: (0, 1),   # down
            3: (-1, 0),  # left
            4: (1, 0),   # right
        }.get(action_id)
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

                # Navigation moves are one-tile translations; prefer displacements
                # that are close to one-step and aligned with the commanded direction.
                shift_error = abs(int(shift) - 5)
                if shift < 2:
                    shift_error += 6
                direction_penalty = 0
                if expected_dir is not None:
                    expected_dx_sign, expected_dy_sign = expected_dir
                    if expected_dx_sign < 0 and delta_x >= 0:
                        direction_penalty += 30
                    elif expected_dx_sign > 0 and delta_x <= 0:
                        direction_penalty += 30
                    if expected_dy_sign < 0 and delta_y >= 0:
                        direction_penalty += 30
                    elif expected_dy_sign > 0 and delta_y <= 0:
                        direction_penalty += 30
                    if expected_dx_sign != 0 and abs(delta_y) > abs(delta_x):
                        direction_penalty += 12
                    if expected_dy_sign != 0 and abs(delta_x) > abs(delta_y):
                        direction_penalty += 12

                score = (area_gap * 10) + (shift_error * 6) + direction_penalty
                if self._tracked_agent_token_digest is not None:
                    if str(previous.digest) != self._tracked_agent_token_digest:
                        score += 8
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

    def _is_progress_proxy_event(self, signature: Any) -> bool:
        # Keep the proxy conservative: only reset stagnation on strong progress signals.
        try:
            level_delta = int(getattr(signature, "level_delta", 0))
        except Exception:
            level_delta = 0
        if level_delta > 0:
            return True
        state_transition = str(getattr(signature, "state_transition", ""))
        if state_transition.endswith("->WIN"):
            return True
        return False

    def _reasoning_for_forced_reset(self, packet: ObservationPacketV1) -> dict[str, Any]:
        memory_policy_payload, exploration_policy_payload = (
            self._reasoning_policy_payloads_v1(packet)
        )
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
            "transition_record_previous_step_v1": dict(self._latest_transition_record),
            "transition_graph_summary_v1": self._transition_graph_summary_v1(),
            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
            "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
            "no_change_streak": int(self._no_change_streak),
            "stagnation_streak": int(self._stagnation_streak),
            "memory_policy_v1": memory_policy_payload,
            "exploration_policy_v1": exploration_policy_payload,
            "operability_diagnostics_v1": self._operability_diagnostics_v1(),
        }

    def _reasoning_for_failure(
        self,
        packet: ObservationPacketV1 | None,
        diagnostics: StageDiagnosticsCollectorV1,
        failure_code: str,
        failure_message: str,
    ) -> dict[str, Any]:
        memory_policy_payload, exploration_policy_payload = (
            self._reasoning_policy_payloads_v1(packet)
        )
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
            "transition_record_previous_step_v1": dict(self._latest_transition_record),
            "transition_graph_summary_v1": self._transition_graph_summary_v1(),
            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
            "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
            "no_change_streak": int(self._no_change_streak),
            "stagnation_streak": int(self._stagnation_streak),
            "memory_policy_v1": memory_policy_payload,
            "exploration_policy_v1": exploration_policy_payload,
            "operability_diagnostics_v1": self._operability_diagnostics_v1(),
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
        transition_record: TransitionRecordV1 | None = None
        selection_diagnostics: dict[str, Any] = {}
        action_space_constraint_report: dict[str, Any] = {}
        navigation_state_estimate: dict[str, Any] = {}
        exploration_policy_payload: dict[str, Any] = {}
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
                tracked_token_before = self._tracked_agent_token_digest
                navigation_state_estimate = self._estimate_navigation_state(
                    self._previous_representation,
                    representation,
                    self._previous_action_candidate,
                )
                transition_record = self._build_transition_record_v1(
                    previous_packet=self._previous_packet,
                    current_packet=packet,
                    previous_representation=self._previous_representation,
                    current_representation=representation,
                    executed_candidate=self._previous_action_candidate,
                    observed_signature=causal_signature,
                    navigation_state_estimate=navigation_state_estimate,
                    tracked_token_before=tracked_token_before,
                )
                self._update_transition_graph_v1(transition_record)
                self._update_operability_stats_v1(
                    executed_candidate=self._previous_action_candidate,
                    causal_signature=causal_signature,
                    navigation_state_estimate=navigation_state_estimate,
                )
                self._latest_navigation_state_estimate = dict(navigation_state_estimate)
                progress_proxy_event = self._is_progress_proxy_event(causal_signature)
                if str(causal_signature.obs_change_type) == "NO_CHANGE":
                    self._no_change_streak += 1
                else:
                    self._no_change_streak = 0
                if progress_proxy_event:
                    self._stagnation_streak = 0
                else:
                    self._stagnation_streak += 1
                posterior_report = dict(self.hypothesis_bank.last_posterior_delta_report)
                diagnostics.finish_ok(
                    "causal_update",
                    {
                        "updated": True,
                        "signature_digest": causal_signature.signature_digest,
                        "event_tags": list(causal_signature.event_tags),
                        "obs_change_type": str(causal_signature.obs_change_type),
                        "no_change_streak": int(self._no_change_streak),
                        "stagnation_streak": int(self._stagnation_streak),
                        "progress_proxy_event": bool(progress_proxy_event),
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
                        "transition_record_previous_step_v1": (
                            transition_record.to_dict()
                            if transition_record is not None
                            else None
                        ),
                        "transition_graph_summary_v1": self._transition_graph_summary_v1(),
                        "operability_diagnostics_v1": self._operability_diagnostics_v1(),
                    },
                )
            else:
                self._no_change_streak = 0
                self._stagnation_streak = 0
                self._latest_navigation_state_estimate = {}
                self._latest_transition_record = {}
                diagnostics.finish_ok(
                    "causal_update",
                    {
                        "updated": False,
                        "reason": "insufficient_history",
                        "stagnation_streak": int(self._stagnation_streak),
                    },
                )
        except Exception as exc:
            diagnostics.finish_rejected(
                "causal_update",
                f"causal_update_error::{type(exc).__name__}",
            )
            self._latest_navigation_state_estimate = {}
            self._latest_transition_record = {}

        remaining_budget = max(0, int(self.MAX_ACTIONS - int(self.action_counter)))
        early_probe_budget_remaining = max(
            0,
            int(self.early_probe_budget - int(self.action_counter)),
        )
        effective_explore_steps = self._effective_explore_steps(packet)
        exploration_policy_payload = self._exploration_policy_v1(
            packet=packet,
            effective_explore_steps=effective_explore_steps,
            remaining_budget=remaining_budget,
            early_probe_budget_remaining=early_probe_budget_remaining,
        )
        diagnostics.start("phase_determination")
        try:
            phase = self.policy.determine_phase(
                action_counter=self.action_counter,
                remaining_budget=remaining_budget,
                hypothesis_bank=self.hypothesis_bank,
                explore_steps_override=effective_explore_steps,
            )
            diagnostics.finish_ok(
                "phase_determination",
                {
                    "phase": phase,
                    "posterior_entropy_bits": float(self.hypothesis_bank.posterior_entropy()),
                    "remaining_budget": int(remaining_budget),
                    "effective_explore_steps": int(effective_explore_steps),
                    "exploration_budget_remaining": int(
                        exploration_policy_payload.get("exploration_budget_remaining", 0)
                    ),
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
            self._no_change_streak = 0
            self._stagnation_streak = 0
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
            stop_loss_reason = "none"
            if self._no_change_streak >= self.no_change_stop_loss_steps:
                stop_loss_reason = "no_change_streak_threshold"
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
            elif self._stagnation_streak >= self.stagnation_stop_loss_steps:
                stop_loss_reason = "stagnation_streak_threshold"
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
                        "stagnation_streak": int(self._stagnation_streak),
                        "threshold": int(self.no_change_stop_loss_steps),
                        "stagnation_threshold": int(self.stagnation_stop_loss_steps),
                        "reason": str(stop_loss_reason),
                        "selected_action_id": int(selected_candidate.action_id),
                    },
                )
                self._no_change_streak = 0
                self._stagnation_streak = 0
                action.reasoning = {
                    "schema_name": "active_inference_reasoning_v2",
                    "schema_version": 2,
                    "phase": phase,
                    "selected_candidate": selected_candidate.to_dict(),
                    "reason": "stop_loss_guard_triggered",
                    "stop_loss_reason": str(stop_loss_reason),
                    "hypothesis_summary": self.hypothesis_bank.summary(),
                    "posterior_delta_report_previous_step": dict(
                        self.hypothesis_bank.last_posterior_delta_report
                    ),
                    "action_space_constraint_report_v1": dict(action_space_constraint_report),
                    "transition_record_previous_step_v1": (
                        transition_record.to_dict() if transition_record is not None else None
                    ),
                    "transition_graph_summary_v1": self._transition_graph_summary_v1(),
                    "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
                    "representation_summary": representation.summary,
                    "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                    "remaining_budget": int(remaining_budget),
                    "early_probe_budget_remaining": int(early_probe_budget_remaining),
                    "no_change_streak": int(self._no_change_streak),
                    "stagnation_streak": int(self._stagnation_streak),
                }
            else:
                diagnostics.finish_ok(
                    "stop_loss_guard",
                    {
                        "triggered": False,
                        "no_change_streak": int(self._no_change_streak),
                        "stagnation_streak": int(self._stagnation_streak),
                        "threshold": int(self.no_change_stop_loss_steps),
                        "stagnation_threshold": int(self.stagnation_stop_loss_steps),
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
                            "transition_record_previous_step_v1": (
                                transition_record.to_dict()
                                if transition_record is not None
                                else None
                            ),
                            "transition_graph_summary_v1": self._transition_graph_summary_v1(),
                            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
                            "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                            "remaining_budget": int(remaining_budget),
                            "early_probe_budget_remaining": int(early_probe_budget_remaining),
                            "no_change_streak": int(self._no_change_streak),
                            "stagnation_streak": int(self._stagnation_streak),
                            "stage_diagnostics_v1": diagnostics.to_dicts(),
                            "bottleneck_stage_v1": diagnostics.bottleneck_stage(),
                        }
                    else:
                        state_digest_current = self._state_digest_v1(packet, representation)
                        control_schema = self._control_schema_posterior()
                        navigation_target_features = self._navigation_target_features_v1(
                            representation
                        )
                        region_graph_snapshot = self._region_graph_snapshot_v1()
                        for candidate in candidates:
                            action_key = str(int(candidate.action_id))
                            action_posterior = dict(control_schema.get(action_key, {}))
                            candidate.metadata["control_schema_observed_posterior"] = dict(
                                action_posterior
                            )
                            candidate.metadata["candidate_cluster_id"] = self._candidate_cluster_id(
                                candidate
                            )
                            candidate.metadata["candidate_subcluster_id"] = (
                                self._candidate_subcluster_id(candidate)
                            )
                            candidate.metadata["transition_exploration_stats"] = (
                                self._transition_exploration_stats_v1(
                                    state_digest_current=str(state_digest_current),
                                    candidate=candidate,
                                )
                            )
                            if int(candidate.action_id) in (1, 2, 3, 4):
                                candidate.metadata["blocked_edge_observed_stats"] = (
                                    self._navigation_candidate_stats(int(candidate.action_id))
                                )
                                candidate.metadata["predicted_region_features_v1"] = (
                                    self._predicted_region_features_v1(
                                        action_id=int(candidate.action_id),
                                        action_posterior=action_posterior,
                                    )
                                )
                                target_payload = dict(navigation_target_features)
                                target_payload["candidate_action_id"] = int(candidate.action_id)
                                candidate.metadata["navigation_target_features_v1"] = target_payload
                                candidate.metadata["region_graph_snapshot_v1"] = dict(
                                    region_graph_snapshot
                                )
                            if int(candidate.action_id) == 6:
                                click_bucket = self._click_context_bucket_from_candidate(candidate)
                                click_subcluster = self._click_context_subcluster_from_candidate(
                                    candidate
                                )
                                candidate.metadata["click_bucket_observed_stats"] = dict(
                                    self._click_bucket_stats.get(click_bucket, {})
                                )
                                candidate.metadata["click_subcluster_observed_stats"] = dict(
                                    self._click_subcluster_stats.get(click_subcluster, {})
                                )

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
                            cluster_select_count=self._cluster_select_count,
                            subcluster_select_count=self._subcluster_select_count,
                            early_probe_budget_remaining=early_probe_budget_remaining,
                            no_change_streak=self._no_change_streak,
                            stagnation_streak=self._stagnation_streak,
                            direction_sequence_visit_count=self._navigation_direction_sequence_visit_count,
                            direction_visit_count=self._navigation_direction_visit_count,
                            recent_navigation_directions=self._recent_navigation_directions,
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
                            "transition_record_previous_step_v1": (
                                transition_record.to_dict()
                                if transition_record is not None
                                else None
                            ),
                            "transition_graph_summary_v1": self._transition_graph_summary_v1(),
                            "selection_diagnostics_v1": dict(selection_diagnostics),
                            "navigation_state_estimate_v1": dict(self._latest_navigation_state_estimate),
                            "navigation_sequence_diagnostics_v1": self._navigation_sequence_diagnostics_v1(),
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
            action.reasoning.setdefault("memory_policy_v1", self._memory_policy_v1())
            action.reasoning.setdefault(
                "exploration_policy_v1",
                dict(exploration_policy_payload),
            )
            action.reasoning.setdefault(
                "transition_record_previous_step_v1",
                transition_record.to_dict() if transition_record is not None else None,
            )
            action.reasoning.setdefault(
                "transition_graph_summary_v1",
                self._transition_graph_summary_v1(),
            )
            action.reasoning.setdefault(
                "operability_diagnostics_v1",
                self._operability_diagnostics_v1(),
            )
            action.reasoning.setdefault(
                "navigation_sequence_diagnostics_v1",
                self._navigation_sequence_diagnostics_v1(),
            )
            action.reasoning.setdefault("no_change_streak", int(self._no_change_streak))
            action.reasoning.setdefault("stagnation_streak", int(self._stagnation_streak))
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
                    "stagnation_streak": int(self._stagnation_streak),
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
                    "transition_record_previous_step_v1": (
                        transition_record.to_dict() if transition_record is not None else None
                    ),
                    "transition_graph_summary_v1": self._transition_graph_summary_v1(),
                    "selection_diagnostics_v1": dict(selection_diagnostics),
                    "memory_policy_v1": self._memory_policy_v1(),
                    "exploration_policy_v1": dict(exploration_policy_payload),
                    "operability_diagnostics_v1": self._operability_diagnostics_v1(),
                    "navigation_sequence_diagnostics_v1": self._navigation_sequence_diagnostics_v1(),
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
        selected_cluster_id = self._candidate_cluster_id(selected_candidate)
        selected_subcluster_id = self._candidate_subcluster_id(selected_candidate)
        self._action_select_count[selected_action_id] = (
            int(self._action_select_count.get(selected_action_id, 0)) + 1
        )
        self._candidate_select_count[selected_candidate_id] = (
            int(self._candidate_select_count.get(selected_candidate_id, 0)) + 1
        )
        self._cluster_select_count[selected_cluster_id] = (
            int(self._cluster_select_count.get(selected_cluster_id, 0)) + 1
        )
        self._subcluster_select_count[selected_subcluster_id] = (
            int(self._subcluster_select_count.get(selected_subcluster_id, 0)) + 1
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
                    "memory_policy_v1": self._memory_policy_v1(),
                    "exploration_policy_config_v1": {
                        "base_explore_steps": int(self.exploration_base_steps),
                        "exploration_min_steps": int(self.exploration_min_steps),
                        "exploration_max_steps": int(self.exploration_max_steps),
                        "exploration_fraction": float(self.exploration_fraction),
                        "action6_bucket_probe_min_attempts": int(
                            self.action6_bucket_probe_min_attempts
                        ),
                        "action6_subcluster_probe_min_attempts": int(
                            self.action6_subcluster_probe_min_attempts
                        ),
                        "action6_probe_score_margin": float(self.action6_probe_score_margin),
                        "action_cost_in_objective": "off_hard",
                        "action_cost_enable_requested": bool(
                            self.action_cost_objective_enable_requested
                        ),
                        "action_cost_override_blocked": bool(
                            self.action_cost_objective_override_blocked
                        ),
                    },
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
                    "final_transition_record_previous_step_v1": dict(
                        self._latest_transition_record
                    ),
                    "final_transition_graph_summary_v1": self._transition_graph_summary_v1(),
                    "final_operability_diagnostics_v1": self._operability_diagnostics_v1(),
                    "final_navigation_sequence_diagnostics_v1": self._navigation_sequence_diagnostics_v1(),
                    "final_control_schema_posterior": self._control_schema_posterior(),
                    "final_action_select_count": {
                        str(key): int(value)
                        for (key, value) in sorted(self._action_select_count.items())
                    },
                    "final_cluster_select_count": {
                        str(key): int(value)
                        for (key, value) in sorted(self._cluster_select_count.items())
                    },
                    "final_subcluster_select_count": {
                        str(key): int(value)
                        for (key, value) in sorted(self._subcluster_select_count.items())
                    },
                    "available_actions_trajectory_v1": self._available_actions_trajectory_summary(),
                    "final_no_change_streak": int(self._no_change_streak),
                    "final_stagnation_streak": int(self._stagnation_streak),
                }
            )
            self.trace_recorder.close()
            self._trace_closed = True
        super().cleanup(scorecard)
