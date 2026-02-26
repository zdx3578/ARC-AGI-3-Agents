from __future__ import annotations

import hashlib
import json
import math
import os
import traceback
import uuid
from pathlib import Path
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ...agent import Agent
from ...runtime_settings import (
    get_runtime_bool,
    get_runtime_float,
    get_runtime_int,
    get_runtime_setting,
    get_runtime_str,
)
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
from .navigation_audit_v1 import run_navigation_map_audit_v1
from .navigation_map_v1 import build_navigation_map_snapshot_v1
from .trace import ActiveInferenceTraceRecorderV1


def _cfg_value(name: str, default: Any) -> Any:
    return get_runtime_setting(name, default, section="active_inference")


def _cfg_int(name: str, default: int) -> int:
    return get_runtime_int(name, int(default), section="active_inference")


def _cfg_float(name: str, default: float) -> float:
    return get_runtime_float(name, float(default), section="active_inference")


def _cfg_bool(name: str, default: bool) -> bool:
    return get_runtime_bool(name, bool(default), section="active_inference")


def _cfg_weight_overrides() -> dict[str, dict[str, float]]:
    raw_any = _cfg_value("ACTIVE_INFERENCE_PHASE_WEIGHT_OVERRIDES_JSON", {})
    parsed: Any = raw_any
    if isinstance(raw_any, str):
        raw = raw_any.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
    if not isinstance(parsed, dict):
        return {}

    try:
        parsed_dict: dict[str, Any] = dict(parsed)
    except Exception:
        return {}

    out: dict[str, dict[str, float]] = {}
    for phase_any, values_any in parsed_dict.items():
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
        self.cross_episode_memory_enable_requested = _cfg_bool(
            "ACTIVE_INFERENCE_ENABLE_CROSS_EPISODE_MEMORY",
            False,
        )
        self.cross_episode_memory_override_blocked = bool(
            self.cross_episode_memory_enable_requested
        )
        self.action_cost_objective_hard_off = True
        self.action_cost_objective_enable_requested = _cfg_bool(
            "ACTIVE_INFERENCE_ENABLE_ACTION_COST_OBJECTIVE",
            False,
        )
        self.action_cost_objective_override_blocked = bool(
            self.action_cost_objective_enable_requested
        )
        self.MAX_ACTIONS = max(1, _cfg_int("ACTIVE_INFERENCE_MAX_ACTIONS", 80))
        self.component_connectivity = (
            4 if _cfg_int("ACTIVE_INFERENCE_COMPONENT_CONNECTIVITY", 8) == 4 else 8
        )
        self.max_action6_points = max(1, _cfg_int("ACTIVE_INFERENCE_MAX_ACTION6_POINTS", 16))
        self.top_k_reasoning = max(1, _cfg_int("ACTIVE_INFERENCE_TOP_K_REASONING", 5))
        self.trace_candidate_limit = max(1, _cfg_int("ACTIVE_INFERENCE_TRACE_CANDIDATE_LIMIT", 30))
        self.trace_include_full_representation = _cfg_bool(
            "ACTIVE_INFERENCE_TRACE_INCLUDE_FULL_REPRESENTATION",
            False,
        )
        self.frame_chain_window = max(1, _cfg_int("ACTIVE_INFERENCE_FRAME_CHAIN_WINDOW", 8))
        self.available_actions_history_window = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_ACTION_SPACE_HISTORY_WINDOW", 24),
        )
        self.rollout_horizon = max(1, _cfg_int("ACTIVE_INFERENCE_ROLLOUT_HORIZON", 2))
        self.rollout_discount = max(
            0.0,
            min(1.0, _cfg_float("ACTIVE_INFERENCE_ROLLOUT_DISCOUNT", 0.55)),
        )
        self.rollout_max_candidates = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_ROLLOUT_MAX_CANDIDATES", 8),
        )
        self.rollout_only_in_exploit = _cfg_bool(
            "ACTIVE_INFERENCE_ROLLOUT_ONLY_IN_EXPLOIT",
            True,
        )
        self.region_revisit_hard_threshold = max(
            4,
            _cfg_int("ACTIVE_INFERENCE_REGION_REVISIT_HARD_THRESHOLD", 24),
        )
        self.sequence_rollout_frontier_weight = max(
            0.0,
            _cfg_float("ACTIVE_INFERENCE_SEQUENCE_ROLLOUT_FRONTIER_WEIGHT", 0.35),
        )
        self.sequence_rollout_direction_weight = max(
            0.0,
            _cfg_float("ACTIVE_INFERENCE_SEQUENCE_ROLLOUT_DIRECTION_WEIGHT", 0.25),
        )
        self.sequence_probe_score_margin = max(
            0.0,
            _cfg_float("ACTIVE_INFERENCE_SEQUENCE_PROBE_SCORE_MARGIN", 0.28),
        )
        self.sequence_probe_trigger_steps = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_SEQUENCE_PROBE_TRIGGER_STEPS", 20),
        )
        self.coverage_sweep_target_regions = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_COVERAGE_SWEEP_TARGET_REGIONS", 24),
        )
        self.coverage_sweep_score_margin = max(
            0.0,
            _cfg_float("ACTIVE_INFERENCE_COVERAGE_SWEEP_SCORE_MARGIN", 0.42),
        )
        self.coverage_resweep_interval = max(
            0,
            _cfg_int("ACTIVE_INFERENCE_COVERAGE_RESWEEP_INTERVAL", 96),
        )
        self.coverage_resweep_span = max(
            0,
            _cfg_int("ACTIVE_INFERENCE_COVERAGE_RESWEEP_SPAN", 24),
        )
        self.coverage_sweep_direction_retry_limit = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_COVERAGE_SWEEP_DIRECTION_RETRY_LIMIT", 8),
        )
        self.coverage_sweep_min_region_visits = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_COVERAGE_SWEEP_MIN_REGION_VISITS", 2),
        )
        self.coverage_prepass_steps = max(
            0,
            _cfg_int(
                "ACTIVE_INFERENCE_COVERAGE_PREPASS_STEPS",
                min(300, int(self.MAX_ACTIONS)),
            ),
        )
        self.coverage_prepass_passes = max(
            1,
            min(2, _cfg_int("ACTIVE_INFERENCE_COVERAGE_PREPASS_PASSES", 1)),
        )
        self.coverage_matrix_sweep_enabled = _cfg_bool(
            "ACTIVE_INFERENCE_COVERAGE_MATRIX_SWEEP_ENABLED",
            True,
        )
        self.coverage_sweep_force_in_exploit = _cfg_bool(
            "ACTIVE_INFERENCE_COVERAGE_SWEEP_FORCE_IN_EXPLOIT",
            True,
        )
        self.high_info_focus_release_after_first_pass = _cfg_bool(
            "ACTIVE_INFERENCE_HIGH_INFO_RELEASE_AFTER_FIRST_PASS",
            True,
        )
        self.high_info_focus_release_action_counter = _cfg_int(
            "ACTIVE_INFERENCE_HIGH_INFO_RELEASE_ACTION_COUNTER",
            -1,
        )
        self.enable_navigation_confidence_gating = _cfg_bool(
            "ACTIVE_INFERENCE_NAV_CONFIDENCE_GATING_ENABLED",
            True,
        )
        self.enable_sequence_causal_term = _cfg_bool(
            "ACTIVE_INFERENCE_SEQUENCE_CAUSAL_TERM_ENABLED",
            True,
        )
        self.sequence_causal_window_steps = max(
            4,
            _cfg_int("ACTIVE_INFERENCE_SEQUENCE_CAUSAL_WINDOW_STEPS", 24),
        )
        self.sequence_causal_verify_window_steps = max(
            2,
            _cfg_int(
                "ACTIVE_INFERENCE_SEQUENCE_CAUSAL_VERIFY_WINDOW_STEPS",
                max(4, int(self.sequence_causal_window_steps // 3)),
            ),
        )
        self.sequence_causal_trigger_region_key = str(
            get_runtime_str(
                "ACTIVE_INFERENCE_SEQUENCE_CAUSAL_TRIGGER_REGION",
                "NA",
                section="active_inference",
            ).strip()
            or "NA"
        )
        self.sequence_causal_target_region_key = str(
            get_runtime_str(
                "ACTIVE_INFERENCE_SEQUENCE_CAUSAL_TARGET_REGION",
                "NA",
                section="active_inference",
            ).strip()
            or "NA"
        )
        self.high_info_focus_window_steps = max(
            4,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_FOCUS_WINDOW_STEPS", 16),
        )
        self.high_info_chain_lock_window_steps = max(
            1,
            _cfg_int(
                "ACTIVE_INFERENCE_HIGH_INFO_CHAIN_LOCK_WINDOW_STEPS",
                max(3, int(self.high_info_focus_window_steps // 3)),
            ),
        )
        self.high_info_target_commit_window_steps = max(
            2,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_TARGET_COMMIT_WINDOW_STEPS", 10),
        )
        self.high_info_chain_lock_miss_limit = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_CHAIN_LOCK_MISS_LIMIT", 3),
        )
        self.high_info_target_commit_miss_limit = max(
            1,
            _cfg_int(
                "ACTIVE_INFERENCE_HIGH_INFO_TARGET_COMMIT_MISS_LIMIT",
                max(4, int(self.high_info_target_commit_window_steps)),
            ),
        )
        self.high_info_focus_max_targets = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_FOCUS_MAX_TARGETS", 3),
        )
        self.high_info_focus_min_trigger_score = max(
            0.0,
            min(1.0, _cfg_float("ACTIVE_INFERENCE_HIGH_INFO_MIN_TRIGGER_SCORE", 0.50)),
        )
        self.high_info_strong_change_pixels = max(
            64,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_STRONG_CHANGE_PIXELS", 512),
        )
        self.high_info_min_samples_per_target = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_MIN_SAMPLES_PER_TARGET", 2),
        )
        self.high_info_coupled_min_samples = max(
            int(self.high_info_min_samples_per_target),
            _cfg_int(
                "ACTIVE_INFERENCE_HIGH_INFO_COUPLED_MIN_SAMPLES",
                max(3, int(self.high_info_min_samples_per_target) + 1),
            ),
        )
        self.high_info_coupled_score_floor = max(
            0.0,
            min(
                1.0,
                _cfg_float("ACTIVE_INFERENCE_HIGH_INFO_COUPLED_SCORE_FLOOR", 0.76),
            ),
        )
        self.high_info_retrigger_cooldown_steps = max(
            0,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_RETRIGGER_COOLDOWN_STEPS", 10),
        )
        self.high_info_simultaneous_focus_enabled = _cfg_bool(
            "ACTIVE_INFERENCE_HIGH_INFO_SIMULTANEOUS_FOCUS_ENABLED",
            True,
        )
        self.high_info_simultaneous_min_region_pixels = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_SIMULTANEOUS_MIN_REGION_PIXELS", 12),
        )
        self.high_info_simultaneous_top_region_ratio = max(
            0.0,
            min(
                1.0,
                _cfg_float("ACTIVE_INFERENCE_HIGH_INFO_SIMULTANEOUS_TOP_REGION_RATIO", 0.30),
            ),
        )
        self.high_info_simultaneous_max_targets = max(
            1,
            _cfg_int(
                "ACTIVE_INFERENCE_HIGH_INFO_SIMULTANEOUS_MAX_TARGETS",
                max(3, int(self.high_info_focus_max_targets) + 1),
            ),
        )
        self.high_info_simultaneous_min_total_pixels = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_SIMULTANEOUS_MIN_TOTAL_PIXELS", 24),
        )
        self.high_info_reachability_graph_min_edges = max(
            0,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_REACHABILITY_GRAPH_MIN_EDGES", 14),
        )
        self.high_info_reachability_graph_min_regions = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_REACHABILITY_GRAPH_MIN_REGIONS", 6),
        )
        self.high_info_novelty_protocol_enabled = _cfg_bool(
            "ACTIVE_INFERENCE_HIGH_INFO_NOVELTY_PROTOCOL_ENABLED",
            True,
        )
        self.high_info_novelty_retrigger_extra_samples = max(
            0,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_NOVELTY_RETRIGGER_EXTRA_SAMPLES", 2),
        )
        self.high_info_novelty_related_extra_samples = max(
            0,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_NOVELTY_RELATED_EXTRA_SAMPLES", 1),
        )
        self.high_info_novelty_max_related_targets = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_NOVELTY_MAX_RELATED_TARGETS", 3),
        )
        self.high_info_novelty_stats_max_entries = max(
            4,
            _cfg_int("ACTIVE_INFERENCE_HIGH_INFO_NOVELTY_STATS_MAX_ENTRIES", 24),
        )
        # When disabled, high-info focus becomes purely evidence-driven (triggered by actual
        # high-diff/novel events), rather than rearming on heuristically "coupled" regions.
        self.high_info_idle_rearm_enabled = _cfg_bool(
            "ACTIVE_INFERENCE_HIGH_INFO_IDLE_REARM_ENABLED",
            False,
        )
        self.orientation_alignment_min_similarity = max(
            0.35,
            min(0.95, _cfg_float("ACTIVE_INFERENCE_ORIENTATION_MIN_SIMILARITY", 0.68)),
        )
        self.orientation_alignment_improve_delta = max(
            0.01,
            min(0.30, _cfg_float("ACTIVE_INFERENCE_ORIENTATION_IMPROVE_DELTA", 0.04)),
        )
        self.enable_empirical_region_override = _cfg_bool(
            "ACTIVE_INFERENCE_ENABLE_EMPIRICAL_REGION_OVERRIDE",
            True,
        )
        self.early_probe_budget = max(
            0,
            _cfg_int(
                "ACTIVE_INFERENCE_EARLY_PROBE_BUDGET",
                max(8, min(512, int(round(float(self.MAX_ACTIONS) * 0.08)))),
            ),
        )
        self.action6_bucket_probe_min_attempts = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_ACTION6_BUCKET_PROBE_MIN_ATTEMPTS", 3),
        )
        self.action6_subcluster_probe_min_attempts = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_ACTION6_SUBCLUSTER_PROBE_MIN_ATTEMPTS", 2),
        )
        self.action6_probe_score_margin = max(
            0.0,
            _cfg_float("ACTIVE_INFERENCE_ACTION6_PROBE_SCORE_MARGIN", 0.06),
        )
        self.action6_explore_probe_score_margin = max(
            0.0,
            _cfg_float("ACTIVE_INFERENCE_ACTION6_EXPLORE_PROBE_SCORE_MARGIN", 0.12),
        )
        self.action6_stagnation_step_threshold = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_ACTION6_STAGNATION_STEP_THRESHOLD", 12),
        )
        self.stagnation_probe_trigger_steps = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_STAGNATION_PROBE_TRIGGER_STEPS", 24),
        )
        self.stagnation_probe_score_margin = max(
            0.0,
            _cfg_float("ACTIVE_INFERENCE_STAGNATION_PROBE_SCORE_MARGIN", 0.22),
        )
        self.stagnation_probe_min_action_usage_gap = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_STAGNATION_PROBE_MIN_ACTION_USAGE_GAP", 8),
        )
        self.stagnation_stop_loss_steps = max(
            1,
            _cfg_int(
                "ACTIVE_INFERENCE_STAGNATION_STOP_LOSS_STEPS",
                max(80, int(round(float(self.MAX_ACTIONS) * 0.45))),
            ),
        )
        self.no_change_stop_loss_steps = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_NO_CHANGE_STOP_LOSS_STEPS", 3),
        )
        self.stop_on_game_over = _cfg_bool(
            "ACTIVE_INFERENCE_STOP_ON_GAME_OVER",
            True,
        )

        explore_steps = max(1, _cfg_int("ACTIVE_INFERENCE_EXPLORE_STEPS", 20))
        self.exploration_base_steps = int(explore_steps)
        self.exploration_min_steps = max(
            1,
            _cfg_int("ACTIVE_INFERENCE_EXPLORATION_MIN_STEPS", 20),
        )
        self.exploration_max_steps = max(
            self.exploration_min_steps,
            _cfg_int(
                "ACTIVE_INFERENCE_EXPLORATION_MAX_STEPS",
                max(120, int(round(float(self.MAX_ACTIONS) * 0.70))),
            ),
        )
        self.exploration_fraction = max(
            0.0,
            min(1.0, _cfg_float("ACTIVE_INFERENCE_EXPLORATION_FRACTION", 0.35)),
        )
        exploit_entropy_threshold = max(
            0.0, _cfg_float("ACTIVE_INFERENCE_EXPLOIT_ENTROPY_THRESHOLD", 0.9)
        )
        weight_overrides = _cfg_weight_overrides()
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
            coverage_prepass_passes=self.coverage_prepass_passes,
            coverage_sweep_score_margin=self.coverage_sweep_score_margin,
            coverage_resweep_interval=self.coverage_resweep_interval,
            coverage_resweep_span=self.coverage_resweep_span,
            coverage_sweep_direction_retry_limit=self.coverage_sweep_direction_retry_limit,
            coverage_matrix_sweep_enabled=self.coverage_matrix_sweep_enabled,
            coverage_sweep_force_in_exploit=self.coverage_sweep_force_in_exploit,
            high_info_focus_release_after_first_pass=self.high_info_focus_release_after_first_pass,
            high_info_focus_release_action_counter=self.high_info_focus_release_action_counter,
            navigation_confidence_gating_enabled=self.enable_navigation_confidence_gating,
            sequence_causal_term_enabled=self.enable_sequence_causal_term,
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
        self._tracked_agent_anchor_xy: tuple[int, int] | None = None
        self._tracked_agent_color: int | None = None
        self._tracked_agent_area_ema: float | None = None
        # Cached arena bbox used by high-info diff masking as a fallback when segmentation drifts.
        # Stored as (min_x, min_y, max_x, max_y) in pixel coordinates.
        self._high_info_cached_arena_bbox: tuple[int, int, int, int] | None = None
        self._navigation_anchor_jump_reject_count = 0
        self._navigation_anchor_jump_streak = 0
        self._last_known_agent_pos_region: tuple[int, int] | None = None
        self._latest_observed_agent_pos_region: tuple[int, int] | None = None
        self._latest_navigation_state_estimate: dict[str, Any] = {}
        self._navigation_step_displacement_history_window = max(
            32,
            _cfg_int("ACTIVE_INFERENCE_NAVIGATION_STEP_HISTORY_WINDOW", 128),
        )
        self._navigation_step_displacement_history: list[int] = []
        self._action_select_count: dict[int, int] = {}
        self._candidate_select_count: dict[str, int] = {}
        self._cluster_select_count: dict[str, int] = {}
        self._subcluster_select_count: dict[str, int] = {}
        self._navigation_direction_history_window = max(
            8,
            _cfg_int("ACTIVE_INFERENCE_DIRECTION_HISTORY_WINDOW", 256),
        )
        self._recent_navigation_directions: list[str] = []
        self._navigation_direction_visit_count: dict[str, int] = {}
        self._navigation_direction_sequence_visit_count: dict[str, int] = {}
        self._navigation_attempt_count = 0
        self._navigation_blocked_count = 0
        self._navigation_moved_count = 0
        self._navigation_implausible_transition_count = 0
        self._navigation_match_count = 0
        self._navigation_semantic_compare_count = 0
        self._navigation_semantic_mismatch_count = 0
        self._navigation_action_stats: dict[str, dict[str, int]] = {}
        self._orientation_action_stats: dict[str, dict[str, int]] = {}
        self._latest_orientation_alignment_state_v1: dict[str, Any] = {
            "schema_name": "active_inference_orientation_alignment_state_v1",
            "schema_version": 1,
            "enabled": False,
            "detected": False,
            "aligned": False,
            "similarity": 0.0,
            "best_rotation_deg": -1,
            "rotation_bucket": "rot_unknown",
            "source_digest": "NA",
            "target_digest": "NA",
            "source_area": 0,
            "target_area": 0,
            "reason": "uninitialized",
        }
        self._latest_navigation_target_features_v1: dict[str, Any] = {
            "schema_name": "active_inference_navigation_target_features_v1",
            "schema_version": 1,
            "enabled": False,
            "cross_like_enabled": False,
            "cross_like_target_region": {"x": -1, "y": -1},
            "gate_like_enabled": False,
            "gate_like_target_region": {"x": -1, "y": -1},
            "orientation_alignment_v1": dict(self._latest_orientation_alignment_state_v1),
            "targets": [],
        }
        self._blocked_edge_counts: dict[str, int] = {}
        self._edge_attempt_counts: dict[str, int] = {}
        self._region_visit_counts: dict[str, int] = {}
        self._region_action_transition_counts: dict[str, dict[str, int]] = {}
        self._region_action_event_counts: dict[str, dict[str, int]] = {}
        self._region_action_non_no_change_counts: dict[str, int] = {}
        self._region_action_strong_change_counts: dict[str, int] = {}
        self._region_action_progress_counts: dict[str, int] = {}
        self._region_action_palette_change_counts: dict[str, int] = {}
        self._region_action_palette_delta_total_sum: dict[str, int] = {}
        self._region_action_ui_side_effect_counts: dict[str, int] = {}
        self._region_action_terminal_failure_counts: dict[str, int] = {}
        self._navigation_ui_reject_count = 0
        self._click_bucket_stats: dict[str, dict[str, int]] = {}
        self._click_subcluster_stats: dict[str, dict[str, int]] = {}
        self._state_visit_count: dict[str, int] = {}
        self._state_action_visit_count: dict[str, int] = {}
        self._transition_edge_visit_count: dict[str, int] = {}
        self._state_outgoing_edges: dict[str, set[str]] = {}
        self._latest_transition_record: dict[str, Any] = {}
        self._sequence_causal_state_v1: dict[str, Any] = {
            "enabled": bool(self.enable_sequence_causal_term),
            "trigger_region_key": str(self.sequence_causal_trigger_region_key),
            "target_region_key": str(self.sequence_causal_target_region_key),
            "trigger_region_key_effective": str(self.sequence_causal_trigger_region_key),
            "target_region_key_effective": str(self.sequence_causal_target_region_key),
            "window_steps": int(self.sequence_causal_window_steps),
            "verify_window_steps": int(self.sequence_causal_verify_window_steps),
            "active": False,
            "stage": "idle",
            "steps_remaining": 0,
            "trigger_count": 0,
            "target_reach_count": 0,
            "success_count": 0,
            "timeout_count": 0,
            "trigger_action_counter": -1,
            "deadline_action_counter": -1,
            "verify_deadline_action_counter": -1,
            "last_reached_action_counter": -1,
            "verify_action_ids": [],
            "last_status": "idle",
        }
        self._high_info_focus_state_v1: dict[str, Any] = {
            "enabled": True,
            "active": False,
            "stage": "idle",
            "window_steps": int(self.high_info_focus_window_steps),
            "steps_remaining": 0,
            "trigger_count": 0,
            "completion_count": 0,
            "timeout_count": 0,
            "source_region_key": "NA",
            "trigger_event_type": "NA",
            "trigger_action_counter": -1,
            "deadline_action_counter": -1,
            "current_target_region_key": "NA",
            "target_miss_streak": 0,
            "target_region_queue": [],
            "pending_region_queue": [],
            "pending_region_scores": {},
            "target_region_scores": {},
            "target_required_samples": {},
            "target_sample_counts": {},
            "completed_target_regions": [],
            "coupled_region_keys": [],
            "primary_coupled_region_key": "NA",
            "secondary_coupled_region_key": "NA",
            "simultaneous_changed_region_keys": [],
            "simultaneous_reachable_region_keys": [],
            "simultaneous_unknown_region_keys": [],
            "simultaneous_unreachable_region_keys": [],
            "simultaneous_anchor_region_key": "NA",
            "simultaneous_changed_total_pixels": 0,
            "region_recent_change_pixels": {},
            "region_change_magnitude": {},
            "region_change_magnitude_ema": {},
            "region_change_delta": {},
            "region_sudden_spike_keys": [],
            "last_trigger_changed_pixels": 0,
            "last_trigger_changed_region_diff_map": {},
            "cross_region_key": "NA",
            "gate_region_key": "NA",
            "verify_action_ids": [],
            "chain_lock_active": False,
            "chain_lock_window_steps": int(self.high_info_chain_lock_window_steps),
            "chain_lock_steps_remaining": 0,
            "chain_lock_target_region_key": "NA",
            "chain_lock_miss_limit": int(self.high_info_chain_lock_miss_limit),
            "target_commit_active": False,
            "target_commit_window_steps": int(self.high_info_target_commit_window_steps),
            "target_commit_miss_limit": int(self.high_info_target_commit_miss_limit),
            "chain_lock_last_status": "idle",
            "interaction_chain_active": False,
            "interaction_chain_generation": 0,
            "interaction_target_chain": [],
            "interaction_target_index": 0,
            "interaction_last_status": "idle",
            "priority_subqueue_active": False,
            "priority_subqueue_keys": [],
            "inner_loop_active": False,
            "inner_loop_reason": "NA",
            "inner_loop_queue": [],
            "inner_loop_current_target_region_key": "NA",
            "novelty_protocol_active": False,
            "novelty_source_region_key": "NA",
            "novelty_related_region_keys": [],
            "last_novelty_signature": "NA",
            "novelty_trigger_count": 0,
            "novelty_last_action_counter": -1,
            "novelty_signature_stats": {},
            "novelty_baseline_sample_counts": {},
            "last_status": "idle",
        }

        self.trace_enabled = _cfg_bool("ACTIVE_INFERENCE_TRACE_ENABLED", True)
        self.trace_recorder: ActiveInferenceTraceRecorderV1 | None = None
        self._trace_closed = False
        self.navigation_map_audit_enabled = _cfg_bool(
            "ACTIVE_INFERENCE_NAVIGATION_MAP_AUDIT_ENABLED",
            True,
        )
        self.navigation_map_audit_subdir = str(
            _cfg_value("ACTIVE_INFERENCE_NAVIGATION_MAP_AUDIT_SUBDIR", "navigation_checks")
        )
        self._latest_navigation_map_snapshot_v1: dict[str, Any] = {}
        self._latest_navigation_map_audit_v1: dict[str, Any] = {
            "schema_name": "active_inference_navigation_map_audit_v1",
            "schema_version": 1,
            "enabled": False,
            "reason": "uninitialized",
        }
        if self.trace_enabled:
            trace_root = get_runtime_str("RECORDINGS_DIR", "recordings", section="runtime")
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
        if latest_frame.state is GameState.WIN:
            return True
        if self.stop_on_game_over and latest_frame.state is GameState.GAME_OVER:
            return True
        return False

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
        values = self._patch_values_v1(frame, x=x, y=y, radius=radius)
        return self._stable_digest_payload_v1(values, prefix="patch:")

    def _patch_values_v1(
        self,
        frame: list[list[int]],
        *,
        x: int,
        y: int,
        radius: int = 2,
    ) -> list[int]:
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
        return values

    def _patch_context_v1(
        self,
        frame: list[list[int]],
        *,
        x: int,
        y: int,
        radius: int = 2,
    ) -> dict[str, Any]:
        values = self._patch_values_v1(frame, x=int(x), y=int(y), radius=int(radius))
        histogram: dict[int, int] = {}
        for value in values:
            key = int(value)
            histogram[key] = int(histogram.get(key, 0) + 1)
        width = int((2 * int(radius)) + 1)
        return {
            "schema_name": "active_inference_patch_context_v1",
            "schema_version": 1,
            "center": {"x": int(x), "y": int(y)},
            "radius": int(radius),
            "width": int(width),
            "height": int(width),
            "digest": self._stable_digest_payload_v1(values, prefix="patch:"),
            "color_histogram": {str(k): int(v) for (k, v) in sorted(histogram.items())},
            "values": [int(v) for v in values],
        }

    @staticmethod
    def _action_direction_vector_v1(action_id: int) -> tuple[int, int] | None:
        mapping: dict[int, tuple[int, int]] = {
            1: (0, -1),  # up
            2: (0, 1),   # down
            3: (-1, 0),  # left
            4: (1, 0),   # right
        }
        return mapping.get(int(action_id))

    def _trace_object_snapshot_v1(
        self,
        representation: RepresentationStateV1,
        *,
        max_objects: int = 12,
    ) -> dict[str, Any]:
        agent_pos_xy = self._current_agent_position_xy_v1(representation)
        targets = self._navigation_key_targets_v1(
            representation,
            agent_pos_xy=agent_pos_xy,
        )
        target_by_digest = {
            str(row.get("digest", "NA")): {
                "kind": str(row.get("kind", "salient")),
                "salience": float(row.get("salience", 0.0)),
                "target_priority": float(row.get("target_priority", 0.0)),
            }
            for row in targets
            if isinstance(row, dict)
        }
        tracked_digest = str(self._tracked_agent_token_digest or "NA")
        tracked_obj = (
            self._find_object_by_digest_v1(representation, tracked_digest)
            if tracked_digest != "NA"
            else None
        )
        rows: list[dict[str, Any]] = []
        for obj in representation.object_nodes:
            key_target = target_by_digest.get(str(obj.digest), {})
            distance_from_agent = -1
            if agent_pos_xy is not None:
                ax, ay = agent_pos_xy
                distance_from_agent = int(
                    abs(int(obj.centroid_x) - int(ax))
                    + abs(int(obj.centroid_y) - int(ay))
                )
            rows.append(
                {
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
                    "touches_boundary": bool(obj.touches_boundary),
                    "distance_from_agent": int(distance_from_agent),
                    "is_key_target": bool(str(obj.digest) in target_by_digest),
                    "key_target_kind": str(key_target.get("kind", "na")),
                    "key_target_salience": float(key_target.get("salience", 0.0)),
                    "key_target_priority": float(key_target.get("target_priority", 0.0)),
                    "is_tracked_agent_candidate": bool(str(obj.digest) == tracked_digest),
                }
            )

        rows.sort(
            key=lambda row: (
                not bool(row.get("is_tracked_agent_candidate", False)),
                not bool(row.get("is_key_target", False)),
                -int(row.get("area", 0)),
                int(row.get("distance_from_agent", 10**9)),
                str(row.get("digest", "")),
            )
        )
        trimmed = rows[: int(max(1, max_objects))]
        return {
            "schema_name": "active_inference_trace_object_snapshot_v1",
            "schema_version": 1,
            "object_count": int(len(representation.object_nodes)),
            "agent_pos_xy": (
                {"x": int(agent_pos_xy[0]), "y": int(agent_pos_xy[1])}
                if agent_pos_xy is not None
                else {"x": -1, "y": -1}
            ),
            "tracked_agent_digest": str(tracked_digest),
            "tracked_agent_present": bool(tracked_obj is not None),
            "tracked_agent_object": dict(tracked_obj) if tracked_obj is not None else None,
            "key_targets": [
                {
                    "digest": str(row.get("digest", "NA")),
                    "kind": str(row.get("kind", "salient")),
                    "salience": float(row.get("salience", 0.0)),
                    "target_priority": float(row.get("target_priority", 0.0)),
                    "x": int(row.get("centroid_x", -1)),
                    "y": int(row.get("centroid_y", -1)),
                    "distance_from_agent": int(row.get("distance_from_agent", -1)),
                }
                for row in targets[:4]
            ],
            "top_objects": trimmed,
        }

    def _trace_neighborhood_context_v1(
        self,
        *,
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
        patch_radius: int = 2,
    ) -> dict[str, Any]:
        anchors: list[dict[str, Any]] = []
        added_keys: set[str] = set()
        agent_pos_xy = self._current_agent_position_xy_v1(representation)
        targets = self._navigation_key_targets_v1(
            representation,
            agent_pos_xy=agent_pos_xy,
        )

        def add_anchor(
            *,
            label: str,
            x: int,
            y: int,
            kind: str,
            digest: str = "NA",
        ) -> None:
            if int(x) < 0 or int(y) < 0:
                return
            key = f"{int(x)}:{int(y)}:{str(label)}"
            if key in added_keys:
                return
            added_keys.add(key)
            anchors.append(
                {
                    "label": str(label),
                    "kind": str(kind),
                    "digest": str(digest),
                    "region_key": self._region_key_from_xy_v1(
                        int(max(0, min(7, int(x) // 8))),
                        int(max(0, min(7, int(y) // 8))),
                    ),
                    "patch_context_v1": self._patch_context_v1(
                        packet.frame,
                        x=int(x),
                        y=int(y),
                        radius=int(patch_radius),
                    ),
                }
            )

        if agent_pos_xy is not None:
            add_anchor(
                label="agent",
                x=int(agent_pos_xy[0]),
                y=int(agent_pos_xy[1]),
                kind="agent",
                digest=str(self._tracked_agent_token_digest or "NA"),
            )
        if targets:
            primary = targets[0]
            if isinstance(primary, dict):
                add_anchor(
                    label="primary_target",
                    x=int(primary.get("centroid_x", -1)),
                    y=int(primary.get("centroid_y", -1)),
                    kind=str(primary.get("kind", "salient")),
                    digest=str(primary.get("digest", "NA")),
                )
            cross_like = next(
                (
                    row
                    for row in targets
                    if isinstance(row, dict)
                    and str(row.get("kind", "")) == "cross_like"
                ),
                None,
            )
            if isinstance(cross_like, dict):
                add_anchor(
                    label="cross_like_target",
                    x=int(cross_like.get("centroid_x", -1)),
                    y=int(cross_like.get("centroid_y", -1)),
                    kind="cross_like",
                    digest=str(cross_like.get("digest", "NA")),
                )
        return {
            "schema_name": "active_inference_trace_neighborhood_context_v1",
            "schema_version": 1,
            "anchor_count": int(len(anchors)),
            "anchors": anchors,
        }

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
    def _orientation_rotation_bucket_v1(rotation_deg: int) -> str:
        mapping = {
            0: "rot_0",
            90: "rot_90",
            180: "rot_180",
            270: "rot_270",
        }
        return str(mapping.get(int(rotation_deg), "rot_unknown"))

    def _empty_orientation_alignment_state_v1(self, *, reason: str) -> dict[str, Any]:
        return {
            "schema_name": "active_inference_orientation_alignment_state_v1",
            "schema_version": 1,
            "enabled": True,
            "detected": False,
            "aligned": False,
            "similarity": 0.0,
            "best_rotation_deg": -1,
            "rotation_bucket": "rot_unknown",
            "source_digest": "NA",
            "target_digest": "NA",
            "source_area": 0,
            "target_area": 0,
            "source_color": -1,
            "target_color": -1,
            "source_bbox": [-1, -1, -1, -1],
            "target_bbox": [-1, -1, -1, -1],
            "reason": str(reason),
        }

    def _extract_object_cells_from_frame_v1(
        self,
        *,
        frame: list[list[int]],
        object_row: dict[str, Any],
    ) -> set[tuple[int, int]]:
        if not frame or not frame[0]:
            return set()
        frame_height = int(len(frame))
        frame_width = int(len(frame[0]))
        color = int(object_row.get("color", -1))
        bbox = object_row.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) < 4:
            return set()
        min_x = int(max(0, min(frame_width - 1, int(bbox[0]))))
        min_y = int(max(0, min(frame_height - 1, int(bbox[1]))))
        max_x = int(max(0, min(frame_width - 1, int(bbox[2]))))
        max_y = int(max(0, min(frame_height - 1, int(bbox[3]))))
        if min_x > max_x or min_y > max_y:
            return set()

        centroid_x = int(object_row.get("centroid_x", -1))
        centroid_y = int(object_row.get("centroid_y", -1))
        seeds: list[tuple[int, int]] = []
        if (
            min_x <= centroid_x <= max_x
            and min_y <= centroid_y <= max_y
            and int(frame[centroid_y][centroid_x]) == color
        ):
            seeds.append((int(centroid_x), int(centroid_y)))
        if not seeds:
            best_seed: tuple[int, int] | None = None
            best_distance = 10**9
            for y in range(min_y, max_y + 1):
                row = frame[y]
                for x in range(min_x, max_x + 1):
                    if int(row[x]) != color:
                        continue
                    if centroid_x >= 0 and centroid_y >= 0:
                        distance = int(abs(int(x) - centroid_x) + abs(int(y) - centroid_y))
                    else:
                        distance = 0
                    if distance < best_distance:
                        best_distance = int(distance)
                        best_seed = (int(x), int(y))
            if best_seed is None:
                return set()
            seeds.append(best_seed)

        cells: set[tuple[int, int]] = set()
        stack: list[tuple[int, int]] = list(seeds)
        visited: set[tuple[int, int]] = set(seeds)
        while stack:
            x, y = stack.pop()
            if x < min_x or x > max_x or y < min_y or y > max_y:
                continue
            if int(frame[y][x]) != color:
                continue
            cells.add((int(x), int(y)))
            neighbors = (
                (int(x - 1), int(y)),
                (int(x + 1), int(y)),
                (int(x), int(y - 1)),
                (int(x), int(y + 1)),
            )
            for nx, ny in neighbors:
                if nx < min_x or nx > max_x or ny < min_y or ny > max_y:
                    continue
                key = (int(nx), int(ny))
                if key in visited:
                    continue
                visited.add(key)
                stack.append(key)
        return cells

    @staticmethod
    def _rasterize_cells_to_grid_v1(
        *,
        cells: set[tuple[int, int]],
        bbox: list[int],
        grid_size: int = 8,
    ) -> set[tuple[int, int]]:
        if not cells or not isinstance(bbox, list) or len(bbox) < 4:
            return set()
        min_x = int(bbox[0])
        min_y = int(bbox[1])
        max_x = int(bbox[2])
        max_y = int(bbox[3])
        width = int(max(1, max_x - min_x + 1))
        height = int(max(1, max_y - min_y + 1))
        g = int(max(2, grid_size))
        points: set[tuple[int, int]] = set()
        for x, y in cells:
            local_x = int(max(0, min(width - 1, int(x) - min_x)))
            local_y = int(max(0, min(height - 1, int(y) - min_y)))
            gx = int(min(g - 1, max(0, int((local_x * g) / float(width)))))
            gy = int(min(g - 1, max(0, int((local_y * g) / float(height)))))
            points.add((int(gx), int(gy)))
        return points

    @staticmethod
    def _rotate_grid_points_v1(
        *,
        points: set[tuple[int, int]],
        steps: int,
        grid_size: int = 8,
    ) -> set[tuple[int, int]]:
        g = int(max(2, grid_size))
        out = set((int(x), int(y)) for (x, y) in points)
        for _ in range(int(steps) % 4):
            out = {(int(g - 1 - y), int(x)) for (x, y) in out}
        return out

    @staticmethod
    def _iou_points_v1(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = int(len(a & b))
        union = int(len(a | b))
        return float(inter / float(max(1, union)))

    def _orientation_alignment_state_from_targets_v1(
        self,
        *,
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
        targets: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not packet.frame or not packet.frame[0]:
            return self._empty_orientation_alignment_state_v1(reason="empty_frame")
        if not targets:
            return self._empty_orientation_alignment_state_v1(reason="no_targets")

        cross_target = next(
            (
                row
                for row in targets
                if isinstance(row, dict) and str(row.get("kind", "")) == "cross_like"
            ),
            None,
        )
        if not isinstance(cross_target, dict):
            return self._empty_orientation_alignment_state_v1(reason="no_cross_target")

        target_digest = str(cross_target.get("digest", "NA"))
        target_obj = self._find_object_by_digest_v1(representation, target_digest)
        if not isinstance(target_obj, dict):
            return self._empty_orientation_alignment_state_v1(reason="target_not_found")

        target_color = int(target_obj.get("color", -1))
        target_area = int(max(0, target_obj.get("area", 0)))
        source_candidates: list[dict[str, Any]] = []
        for obj in representation.object_nodes:
            if str(obj.digest) == str(target_digest):
                continue
            if int(obj.color) != int(target_color):
                continue
            source_candidates.append(
                {
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
                    "touches_boundary": bool(obj.touches_boundary),
                }
            )
        if not source_candidates:
            return self._empty_orientation_alignment_state_v1(reason="no_same_color_source")

        source_candidates.sort(
            key=lambda row: (
                1 if bool(row.get("touches_boundary", False)) else 0,
                -int(max(0, row.get("area", 0))),
                str(row.get("digest", "")),
            )
        )
        source_obj = source_candidates[0]
        if int(source_obj.get("area", 0)) < int(target_area):
            source_candidates.sort(
                key=lambda row: (
                    -int(max(0, row.get("area", 0))),
                    1 if bool(row.get("touches_boundary", False)) else 0,
                    str(row.get("digest", "")),
                )
            )
            source_obj = source_candidates[0]

        source_bbox = source_obj.get("bbox", [-1, -1, -1, -1])
        target_bbox = target_obj.get("bbox", [-1, -1, -1, -1])
        source_cells = self._extract_object_cells_from_frame_v1(
            frame=packet.frame,
            object_row=source_obj,
        )
        target_cells = self._extract_object_cells_from_frame_v1(
            frame=packet.frame,
            object_row=target_obj,
        )
        if len(source_cells) < 3 or len(target_cells) < 3:
            return self._empty_orientation_alignment_state_v1(
                reason="insufficient_component_cells"
            )

        source_points = self._rasterize_cells_to_grid_v1(
            cells=source_cells,
            bbox=source_bbox if isinstance(source_bbox, list) else [-1, -1, -1, -1],
            grid_size=8,
        )
        target_points = self._rasterize_cells_to_grid_v1(
            cells=target_cells,
            bbox=target_bbox if isinstance(target_bbox, list) else [-1, -1, -1, -1],
            grid_size=8,
        )
        if not source_points or not target_points:
            return self._empty_orientation_alignment_state_v1(
                reason="empty_raster_points"
            )

        best_rotation = -1
        best_similarity = -1.0
        for rotation_index, rotation_deg in enumerate((0, 90, 180, 270)):
            rotated_points = self._rotate_grid_points_v1(
                points=source_points,
                steps=int(rotation_index),
                grid_size=8,
            )
            similarity = self._iou_points_v1(rotated_points, target_points)
            if float(similarity) > float(best_similarity):
                best_similarity = float(similarity)
                best_rotation = int(rotation_deg)

        aligned = bool(
            int(best_rotation) == 0
            and float(best_similarity) >= float(self.orientation_alignment_min_similarity)
        )
        return {
            "schema_name": "active_inference_orientation_alignment_state_v1",
            "schema_version": 1,
            "enabled": True,
            "detected": True,
            "aligned": bool(aligned),
            "similarity": float(max(0.0, min(1.0, best_similarity))),
            "best_rotation_deg": int(best_rotation),
            "rotation_bucket": self._orientation_rotation_bucket_v1(int(best_rotation)),
            "source_digest": str(source_obj.get("digest", "NA")),
            "target_digest": str(target_obj.get("digest", "NA")),
            "source_area": int(max(0, source_obj.get("area", 0))),
            "target_area": int(max(0, target_obj.get("area", 0))),
            "source_color": int(source_obj.get("color", -1)),
            "target_color": int(target_obj.get("color", -1)),
            "source_bbox": [
                int(v) for v in (source_bbox if isinstance(source_bbox, list) else [-1, -1, -1, -1])[:4]
            ],
            "target_bbox": [
                int(v) for v in (target_bbox if isinstance(target_bbox, list) else [-1, -1, -1, -1])[:4]
            ],
            "reason": "ok",
        }

    def _update_orientation_action_stats_v1(
        self,
        *,
        executed_action_id: int,
        previous_alignment: dict[str, Any] | None,
        current_alignment: dict[str, Any] | None,
    ) -> None:
        action_id = int(executed_action_id)
        if action_id not in (1, 2, 3, 4):
            return
        prev = previous_alignment if isinstance(previous_alignment, dict) else {}
        cur = current_alignment if isinstance(current_alignment, dict) else {}
        if not (
            bool(prev.get("enabled", False))
            and bool(prev.get("detected", False))
            and bool(cur.get("enabled", False))
            and bool(cur.get("detected", False))
        ):
            return
        action_key = str(action_id)
        stats = self._orientation_action_stats.setdefault(
            action_key,
            {"attempts": 0, "improved": 0, "regressed": 0, "aligned_hit": 0},
        )
        stats["attempts"] = int(stats.get("attempts", 0) + 1)
        prev_similarity = float(max(0.0, min(1.0, prev.get("similarity", 0.0))))
        cur_similarity = float(max(0.0, min(1.0, cur.get("similarity", 0.0))))
        delta_similarity = float(cur_similarity - prev_similarity)
        if delta_similarity >= float(self.orientation_alignment_improve_delta):
            stats["improved"] = int(stats.get("improved", 0) + 1)
        elif delta_similarity <= -float(self.orientation_alignment_improve_delta):
            stats["regressed"] = int(stats.get("regressed", 0) + 1)
        if bool(cur.get("aligned", False)):
            stats["aligned_hit"] = int(stats.get("aligned_hit", 0) + 1)

    def _orientation_alignment_candidate_features_v1(
        self,
        *,
        candidate: ActionCandidateV1,
        orientation_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        state = orientation_state if isinstance(orientation_state, dict) else {}
        action_id = int(candidate.action_id)
        action_stats = self._orientation_action_stats.get(str(action_id), {})
        attempts = int(max(0, action_stats.get("attempts", 0)))
        improved = int(max(0, action_stats.get("improved", 0)))
        regressed = int(max(0, action_stats.get("regressed", 0)))
        aligned_hit = int(max(0, action_stats.get("aligned_hit", 0)))
        improve_rate = float(improved / float(max(1, attempts)))
        regress_rate = float(regressed / float(max(1, attempts)))
        aligned_hit_rate = float(aligned_hit / float(max(1, attempts)))
        enabled = bool(state.get("enabled", False)) and bool(state.get("detected", False))
        similarity = float(max(0.0, min(1.0, state.get("similarity", 0.0))))
        aligned = bool(state.get("aligned", False))
        bonus_hint = 0.0
        penalty_hint = 0.0
        if enabled and action_id in (1, 2, 3, 4):
            mismatch = float(max(0.0, 1.0 - similarity))
            bonus_hint = float(
                max(0.0, (0.62 * improve_rate) + (0.38 * aligned_hit_rate))
                * (0.40 + (0.60 * mismatch))
            )
            penalty_hint = float(
                max(0.0, regress_rate) * (0.25 + (0.75 * mismatch))
            )
            if aligned:
                bonus_hint = float(max(bonus_hint, 0.18 * aligned_hit_rate))
        return {
            "schema_name": "active_inference_orientation_alignment_features_v1",
            "schema_version": 1,
            "enabled": bool(enabled),
            "detected": bool(state.get("detected", False)),
            "aligned": bool(aligned),
            "similarity": float(similarity),
            "best_rotation_deg": int(state.get("best_rotation_deg", -1)),
            "rotation_bucket": str(state.get("rotation_bucket", "rot_unknown")),
            "action_attempts": int(attempts),
            "action_improve_rate": float(max(0.0, min(1.0, improve_rate))),
            "action_regress_rate": float(max(0.0, min(1.0, regress_rate))),
            "action_aligned_hit_rate": float(max(0.0, min(1.0, aligned_hit_rate))),
            "bonus_hint": float(max(0.0, bonus_hint)),
            "penalty_hint": float(max(0.0, penalty_hint)),
        }

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

    def _navigation_step_projection_features_v1(
        self,
        *,
        action_id: int,
        action_posterior: dict[str, Any] | None,
        navigation_target_features: dict[str, Any] | None,
        predicted_region_features: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": "active_inference_navigation_step_projection_features_v1",
            "schema_version": 1,
            "enabled": False,
            "action_id": int(action_id),
            "target_kind": "unknown",
            "target_salience": 0.0,
            "target_digest": "NA",
            "projection_source": "none",
            "agent_pos_xy": {"x": -1, "y": -1},
            "target_pos_xy": {"x": -1, "y": -1},
            "expected_step_pos_xy": {"dx": 0.0, "dy": 0.0},
            "distance_before": -1.0,
            "distance_after": -1.0,
            "distance_delta": 0.0,
            "distance_delta_normalized": 0.0,
            "step_manhattan": 0.0,
            "alignment": 0.0,
            "confidence": 0.0,
            "bonus_hint": 0.0,
            "penalty_hint": 0.0,
            "moves_toward_target": False,
            "moves_away_from_target": False,
            "command_direction_bucket": "dir_unknown",
        }
        if int(action_id) not in (1, 2, 3, 4):
            payload["reason"] = "non_navigation_action"
            return payload

        command_vec = self._action_direction_vector_v1(int(action_id))
        if command_vec is not None:
            payload["command_direction_bucket"] = self._direction_bucket_from_delta_v1(
                int(command_vec[0]),
                int(command_vec[1]),
            )

        target = (
            navigation_target_features
            if isinstance(navigation_target_features, dict)
            else {}
        )
        if not bool(target.get("enabled", False)):
            payload["reason"] = "navigation_target_disabled"
            return payload

        agent_pos = target.get("agent_pos_xy", {})
        target_pos = target.get("target_pos_xy", {})
        if not isinstance(agent_pos, dict) or not isinstance(target_pos, dict):
            payload["reason"] = "invalid_target_payload"
            return payload
        agent_x = int(agent_pos.get("x", -1))
        agent_y = int(agent_pos.get("y", -1))
        target_x = int(target_pos.get("x", -1))
        target_y = int(target_pos.get("y", -1))
        if min(agent_x, agent_y, target_x, target_y) < 0:
            payload["reason"] = "missing_agent_or_target_position"
            return payload
        payload["agent_pos_xy"] = {"x": int(agent_x), "y": int(agent_y)}
        payload["target_pos_xy"] = {"x": int(target_x), "y": int(target_y)}
        payload["target_kind"] = str(target.get("target_kind", "unknown"))
        payload["target_salience"] = float(
            max(0.0, min(1.0, float(target.get("target_salience", 0.0))))
        )
        payload["target_digest"] = str(target.get("target_digest", "NA"))

        expected_dx = 0.0
        expected_dy = 0.0
        projection_source = "command_default"
        projection_confidence = 0.25

        predicted = (
            predicted_region_features
            if isinstance(predicted_region_features, dict)
            else {}
        )
        predicted_enabled = bool(predicted.get("enabled", False))
        if predicted_enabled:
            current_region = predicted.get("current_region", {})
            next_region = predicted.get("predicted_region", {})
            if isinstance(current_region, dict) and isinstance(next_region, dict):
                crx = int(current_region.get("x", -1))
                cry = int(current_region.get("y", -1))
                nrx = int(next_region.get("x", -1))
                nry = int(next_region.get("y", -1))
                if min(crx, cry, nrx, nry) >= 0:
                    expected_dx = float((int(nrx) - int(crx)) * 8)
                    expected_dy = float((int(nry) - int(cry)) * 8)
                    projection_source = "predicted_region"
                    projection_confidence = float(
                        max(
                            0.35,
                            min(1.0, float(predicted.get("confidence", 0.0))),
                        )
                    )

        if projection_source != "predicted_region":
            posterior = action_posterior if isinstance(action_posterior, dict) else {}
            mass = 0.0
            weighted_dx = 0.0
            weighted_dy = 0.0
            dominant_prob = 0.0
            for delta_key, value in posterior.items():
                try:
                    p_raw = float(value)
                except Exception:
                    p_raw = 0.0
                if p_raw <= 0.0:
                    continue
                dx, dy = self._parse_delta_key_v1(str(delta_key))
                weighted_dx += float(p_raw) * float(dx)
                weighted_dy += float(p_raw) * float(dy)
                mass += float(p_raw)
            if mass > 1.0e-9:
                for _, value in posterior.items():
                    try:
                        p_raw = float(value)
                    except Exception:
                        p_raw = 0.0
                    if p_raw <= 0.0:
                        continue
                    dominant_prob = max(dominant_prob, float(p_raw / mass))
                expected_dx = float(weighted_dx / mass)
                expected_dy = float(weighted_dy / mass)
                projection_source = "posterior_expected_delta"
                projection_confidence = float(max(0.30, min(1.0, dominant_prob)))

        if (
            projection_source == "command_default"
            and command_vec is not None
        ):
            command_dx, command_dy = command_vec
            expected_dx = float(int(command_dx) * 5)
            expected_dy = float(int(command_dy) * 5)

        target_dx = float(target_x - agent_x)
        target_dy = float(target_y - agent_y)
        distance_before = float(abs(target_dx) + abs(target_dy))
        projected_agent_x = float(agent_x) + float(expected_dx)
        projected_agent_y = float(agent_y) + float(expected_dy)
        distance_after = float(
            abs(float(target_x) - projected_agent_x)
            + abs(float(target_y) - projected_agent_y)
        )
        distance_delta = float(distance_after - distance_before)
        distance_delta_normalized = float(
            distance_delta / float(max(1.0, distance_before))
        )
        step_manhattan = float(abs(float(expected_dx)) + abs(float(expected_dy)))
        toward_gain = float(
            max(0.0, distance_before - distance_after)
            / float(max(1.0, step_manhattan))
        )
        away_gain = float(
            max(0.0, distance_after - distance_before)
            / float(max(1.0, step_manhattan))
        )

        alignment = 0.0
        step_norm = float(math.sqrt((expected_dx * expected_dx) + (expected_dy * expected_dy)))
        target_norm = float(math.sqrt((target_dx * target_dx) + (target_dy * target_dy)))
        if step_norm > 1.0e-9 and target_norm > 1.0e-9:
            alignment = float(
                max(
                    -1.0,
                    min(
                        1.0,
                        ((expected_dx * target_dx) + (expected_dy * target_dy))
                        / float(step_norm * target_norm),
                    ),
                )
            )

        target_salience = float(payload.get("target_salience", 0.0))
        confidence = float(max(0.0, min(1.0, projection_confidence)))
        bonus_hint = float(max(0.0, toward_gain) * target_salience * confidence)
        penalty_hint = float(max(0.0, away_gain) * target_salience * confidence)
        payload.update(
            {
                "enabled": True,
                "projection_source": str(projection_source),
                "expected_step_pos_xy": {"dx": float(expected_dx), "dy": float(expected_dy)},
                "distance_before": float(distance_before),
                "distance_after": float(distance_after),
                "distance_delta": float(distance_delta),
                "distance_delta_normalized": float(distance_delta_normalized),
                "step_manhattan": float(step_manhattan),
                "alignment": float(alignment),
                "confidence": float(confidence),
                "bonus_hint": float(bonus_hint),
                "penalty_hint": float(penalty_hint),
                "moves_toward_target": bool(distance_after + 1.0e-6 < distance_before),
                "moves_away_from_target": bool(distance_after > distance_before + 1.0e-6),
            }
        )
        return payload

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
    def _parse_region_key_v1(region_key: str) -> tuple[int, int] | None:
        try:
            srow, scol = str(region_key).split(":", 1)
            row = int(srow)
            col = int(scol)
        except Exception:
            return None
        # Region grid is fixed 8x8 (0..7). Reject drifted/UI-sourced keys early so they
        # cannot poison reachability, masks, or queues.
        if row < 0 or col < 0 or row > 7 or col > 7:
            return None
        # Region keys are stored as row:col. Internal geometry keeps (x, y)=(col, row).
        return (int(col), int(row))

    @staticmethod
    def _region_key_from_xy_v1(region_x: int, region_y: int) -> str:
        col = int(region_x)
        row = int(region_y)
        if col < 0 or row < 0 or col > 7 or row > 7:
            return "NA"
        return f"{int(row)}:{int(col)}"

    @classmethod
    def _region_key_from_region_payload_v1(
        cls,
        payload: dict[str, Any] | None,
    ) -> str:
        if not isinstance(payload, dict):
            return "NA"
        rx = int(payload.get("x", -1))
        ry = int(payload.get("y", -1))
        if rx < 0 or ry < 0:
            return "NA"
        region_key = cls._region_key_from_xy_v1(int(rx), int(ry))
        return str(region_key) if cls._parse_region_key_v1(region_key) is not None else "NA"

    def _high_info_coupled_regions_v1(
        self,
        *,
        fallback_source_region_key: str = "NA",
        reachable_region_set: set[str] | None = None,
        reachability_graph_ready: bool = False,
    ) -> dict[str, Any]:
        nav = (
            self._latest_navigation_target_features_v1
            if isinstance(self._latest_navigation_target_features_v1, dict)
            else {}
        )
        nav_region_hints: dict[str, float] = {}
        nav_region_kind_rank: dict[str, int] = {}
        targets = nav.get("targets", [])
        if isinstance(targets, list):
            for row in targets:
                if not isinstance(row, dict):
                    continue
                x = int(row.get("x", -1))
                y = int(row.get("y", -1))
                if x < 0 or y < 0:
                    continue
                region_key = self._region_key_from_xy_v1(
                    int(max(0, min(7, x // 8))),
                    int(max(0, min(7, y // 8))),
                )
                if self._parse_region_key_v1(region_key) is None:
                    continue
                hint = float(max(0.0, row.get("target_priority", row.get("salience", 0.0))))
                target_kind = str(row.get("kind", ""))
                kind_bonus = 0.0
                kind_rank = 0
                if target_kind == "cross_like":
                    kind_bonus = 0.22
                    kind_rank = 2
                elif target_kind == "salient":
                    kind_bonus = 0.08
                    kind_rank = 1
                hint = float(min(1.0, hint + kind_bonus))
                nav_region_hints[str(region_key)] = max(
                    float(nav_region_hints.get(str(region_key), 0.0)),
                    float(min(1.0, hint)),
                )
                nav_region_kind_rank[str(region_key)] = int(
                    max(int(nav_region_kind_rank.get(str(region_key), 0)), int(kind_rank))
                )

        orientation_state = nav.get("orientation_alignment_v1", {})
        if not isinstance(orientation_state, dict):
            orientation_state = (
                self._latest_orientation_alignment_state_v1
                if isinstance(self._latest_orientation_alignment_state_v1, dict)
                else {}
            )
        orientation_enabled = bool(orientation_state.get("enabled", False))
        orientation_detected = bool(orientation_state.get("detected", False))
        orientation_aligned = bool(orientation_state.get("aligned", False))
        orientation_misaligned = bool(
            orientation_enabled and orientation_detected and (not orientation_aligned)
        )

        scoreboard = self._high_info_region_scoreboard_v1(max_regions=64)
        rows_raw = scoreboard.get("rows", [])
        if not isinstance(rows_raw, list):
            rows_raw = []
        rows: list[dict[str, Any]] = []
        for row in rows_raw:
            if not isinstance(row, dict):
                continue
            region_key = str(row.get("region_key", "NA"))
            if self._parse_region_key_v1(region_key) is None:
                continue
            rows.append(dict(row))
        row_by_region = {
            str(row.get("region_key", "NA")): dict(row)
            for row in rows
            if self._parse_region_key_v1(str(row.get("region_key", "NA"))) is not None
        }

        transition_pair_counts: dict[tuple[str, str], int] = {}
        transition_source_totals: dict[str, int] = {}
        for region_action_key, target_histogram in self._region_action_transition_counts.items():
            if not isinstance(target_histogram, dict):
                continue
            try:
                source_region_key, _ = str(region_action_key).split("|a", 1)
            except Exception:
                continue
            if self._parse_region_key_v1(source_region_key) is None:
                continue
            total = 0
            for target_region_key, count_raw in target_histogram.items():
                target_key = str(target_region_key)
                if self._parse_region_key_v1(target_key) is None:
                    continue
                if not self._region_step_plausible_v1(
                    str(source_region_key),
                    str(target_key),
                    max_axis_step=1,
                ):
                    continue
                count = int(max(0, count_raw))
                if count <= 0:
                    continue
                total += int(count)
                pair_key = (str(source_region_key), str(target_key))
                transition_pair_counts[pair_key] = int(
                    transition_pair_counts.get(pair_key, 0) + int(count)
                )
            if total > 0:
                transition_source_totals[str(source_region_key)] = int(
                    transition_source_totals.get(str(source_region_key), 0) + int(total)
                )

        fallback_key = str(fallback_source_region_key)
        if self._parse_region_key_v1(fallback_key) is None:
            fallback_key = "NA"
        candidate_region_keys: set[str] = set(row_by_region.keys())
        candidate_region_keys.update(str(k) for k in nav_region_hints.keys())
        if fallback_key != "NA":
            candidate_region_keys.add(str(fallback_key))

        reachable_keys: set[str] = set()
        if isinstance(reachable_region_set, set):
            for region_key in reachable_region_set:
                key = str(region_key)
                if self._parse_region_key_v1(key) is None:
                    continue
                reachable_keys.add(str(key))
        frontier_keys: set[str] = set(reachable_keys)
        for region_key in list(reachable_keys):
            parsed = self._parse_region_key_v1(str(region_key))
            if parsed is None:
                continue
            rx, ry = parsed
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = int(rx + dx)
                ny = int(ry + dy)
                if nx < 0 or ny < 0 or nx > 7 or ny > 7:
                    continue
                frontier_keys.add(self._region_key_from_xy_v1(int(nx), int(ny)))
        if self._parse_region_key_v1(str(fallback_key)) is not None:
            frontier_keys.add(str(fallback_key))

        def _region_reachability_state(region_key: str) -> str:
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                return "invalid"
            if not bool(reachability_graph_ready):
                return "unknown"
            if key in reachable_keys:
                return "reachable"
            if key in frontier_keys:
                return "frontier"
            return "out_of_graph"

        def _region_base_score(region_key: str) -> float:
            row = row_by_region.get(str(region_key), {})
            info_score = float(max(0.0, row.get("info_score", 0.0)))
            coupling_score = float(max(0.0, row.get("coupling_signal_score", 0.0)))
            cc_rate = float(max(0.0, row.get("cc_count_change_rate", 0.0)))
            strong_rate = float(max(0.0, row.get("strong_change_rate", 0.0)))
            progress_rate = float(max(0.0, row.get("progress_rate", 0.0)))
            ui_suppression = float(max(0.0, min(1.0, row.get("ui_suppression", 0.0))))
            monotony_penalty = float(max(0.0, row.get("monotony_penalty", 0.0)))
            attempts = int(max(0, row.get("attempts", 0)))
            visit_count = int(max(0, self._region_visit_counts.get(str(region_key), 0)))
            novelty = float(max(0.0, 1.0 - min(1.0, float(visit_count) / 12.0)))
            nav_hint = float(max(0.0, min(1.0, nav_region_hints.get(str(region_key), 0.0))))
            nav_kind_rank = int(max(0, min(2, nav_region_kind_rank.get(str(region_key), 0))))
            nav_kind_bonus = float(0.05 * float(nav_kind_rank))
            support = float(1.0 - math.exp(-float(max(0, attempts)) / 4.0))
            support_prior = float(min(0.90, 0.35 + (0.45 * nav_hint)))
            base = float(
                (0.46 * info_score)
                + (0.16 * coupling_score)
                + (0.10 * cc_rate)
                + (0.07 * strong_rate)
                + (0.07 * progress_rate)
                + (0.14 * nav_hint)
            )
            base = float(base * (1.0 - (0.85 * ui_suppression)))
            base = float(base + (0.08 * novelty))
            base = float(base + nav_kind_bonus)
            stagnation_penalty = 0.0
            if attempts >= 4 and progress_rate <= 0.0:
                stagnation_penalty = float(
                    min(
                        0.35,
                        float(monotony_penalty) + (0.006 * float(max(0, visit_count - 10))),
                    )
                )
                base = float(max(0.0, base - stagnation_penalty))
            exploration_boost = 0.0
            if nav_kind_rank >= 2 and attempts < 2:
                exploration_boost = float(0.28 + (0.08 * float(max(0, 1 - attempts))))
            elif nav_hint >= 0.75 and attempts <= 0:
                exploration_boost = 0.12
            base = float(base + exploration_boost)
            if attempts <= 0 and nav_hint >= 0.70:
                unseen_hint_bonus = float(0.05 + (0.04 * float(nav_kind_rank)))
                base = float(base + unseen_hint_bonus)
            base = float(max(base, (0.18 * nav_hint * novelty)))
            base = float(max(support_prior, (0.30 + (0.70 * support))) * base)
            if str(region_key) == str(fallback_key):
                base = float(base + 0.05)
            reachability_state = _region_reachability_state(str(region_key))
            if reachability_state == "out_of_graph":
                if nav_kind_rank >= 2 and nav_hint >= 0.80:
                    base = float(base * 0.82)
                elif str(region_key) == str(fallback_key):
                    base = float(base * 0.88)
                else:
                    base = float(base * 0.58)
            elif reachability_state == "frontier":
                base = float(base + 0.04)
            parsed_region = self._parse_region_key_v1(str(region_key))
            if parsed_region is not None:
                rx, ry = parsed_region
                peripheral_corner = bool(
                    (int(ry) <= 1 or int(ry) >= 6)
                    and (int(rx) <= 1 or int(rx) >= 6)
                )
                if (
                    peripheral_corner
                    and attempts <= 0
                    and visit_count <= 0
                    and reachability_state != "reachable"
                ):
                    base = float(base * 0.72)
            return float(max(0.0, min(1.0, base)))

        primary_candidates: list[tuple[float, int, int, int, int, str]] = []
        for region_key in candidate_region_keys:
            if self._parse_region_key_v1(str(region_key)) is None:
                continue
            score = float(_region_base_score(str(region_key)))
            kind_rank = int(max(0, min(2, nav_region_kind_rank.get(str(region_key), 0))))
            reachability_state = _region_reachability_state(str(region_key))
            reachability_priority = 1
            if reachability_state == "reachable":
                reachability_priority = 0
            elif reachability_state == "out_of_graph":
                reachability_priority = 2
            visit_count = int(max(0, self._region_visit_counts.get(str(region_key), 0)))
            attempts = int(max(0, row_by_region.get(str(region_key), {}).get("attempts", 0)))
            primary_candidates.append(
                (
                    float(score),
                    int(reachability_priority),
                    -int(kind_rank),
                    int(attempts),
                    int(visit_count),
                    str(region_key),
                )
            )
        primary_candidates.sort(
            key=lambda item: (
                -float(item[0]),
                int(item[1]),
                int(item[2]),
                int(item[3]),
                int(item[4]),
                str(item[5]),
            )
        )

        primary_region_key = "NA"
        primary_region_score = 0.0
        if primary_candidates and float(primary_candidates[0][0]) >= 0.12:
            primary_region_key = str(primary_candidates[0][5])
            primary_region_score = float(primary_candidates[0][0])
        elif fallback_key != "NA":
            primary_region_key = str(fallback_key)
            primary_region_score = float(_region_base_score(primary_region_key))

        secondary_region_key = "NA"
        secondary_region_score = 0.0
        pair_affinity_score = 0.0
        if self._parse_region_key_v1(primary_region_key) is not None:
            secondary_candidates: list[tuple[float, int, int, int, str, float, int]] = []
            for region_key in candidate_region_keys:
                key = str(region_key)
                if self._parse_region_key_v1(key) is None or key == str(primary_region_key):
                    continue
                row = row_by_region.get(key, {})
                cc_rate = float(max(0.0, row.get("cc_count_change_rate", 0.0)))
                strong_rate = float(max(0.0, row.get("strong_change_rate", 0.0)))
                progress_rate = float(max(0.0, row.get("progress_rate", 0.0)))
                structural_signal = float(max(cc_rate, strong_rate, progress_rate))
                forward = float(
                    transition_pair_counts.get((str(primary_region_key), key), 0)
                ) / float(max(1, transition_source_totals.get(str(primary_region_key), 0)))
                backward = float(
                    transition_pair_counts.get((key, str(primary_region_key)), 0)
                ) / float(max(1, transition_source_totals.get(key, 0)))
                pair_affinity = float(max(0.0, min(1.0, max(forward, backward))))
                distance = int(self._region_distance_v1(str(primary_region_key), key))
                distance_term = float(
                    max(0.0, 1.0 - min(1.0, float(max(0, distance - 1)) / 6.0))
                )
                base = float(_region_base_score(key))
                attempts = int(max(0, row.get("attempts", 0)))
                visit_count = int(max(0, self._region_visit_counts.get(key, 0)))
                nav_hint = float(max(0.0, min(1.0, nav_region_hints.get(key, 0.0))))
                nav_kind_rank = int(max(0, min(2, nav_region_kind_rank.get(key, 0))))
                monotony_penalty = float(max(0.0, row.get("monotony_penalty", 0.0)))
                locality_penalty = 0.0
                if distance <= 1 and structural_signal < 0.35:
                    locality_penalty = float(
                        min(
                            0.30,
                            (1.0 - structural_signal)
                            * (0.10 + (0.03 * float(min(10, attempts)))),
                        )
                    )
                stale_penalty = 0.0
                if attempts >= 6 and progress_rate <= 0.0 and structural_signal < 0.20:
                    stale_penalty = float(min(0.22, 0.04 * float(attempts - 5)))
                loop_penalty = 0.0
                if progress_rate <= 0.0:
                    if visit_count >= 8:
                        loop_penalty += float(min(0.30, 0.02 * float(visit_count - 7)))
                    if attempts >= 4 and nav_hint < 0.30:
                        loop_penalty += float(min(0.24, 0.04 * float(attempts - 3)))
                    if monotony_penalty > 0.0:
                        loop_penalty += float(min(0.30, 0.60 * monotony_penalty))
                far_structural_bonus = 0.0
                if distance >= 2 and structural_signal >= 0.60:
                    far_structural_bonus = 0.08
                anchor_bonus = 0.0
                if nav_hint >= 0.70 and attempts <= 3:
                    anchor_bonus += 0.14
                elif nav_kind_rank >= 2 and attempts <= 4:
                    anchor_bonus += 0.10
                if orientation_misaligned and nav_kind_rank >= 2:
                    anchor_bonus += 0.08
                weak_evidence = bool(
                    progress_rate <= 0.0
                    and structural_signal < 0.12
                    and pair_affinity < 0.14
                )
                weak_evidence_penalty = 0.0
                if weak_evidence:
                    weak_evidence_penalty += 0.22
                    if nav_kind_rank <= 0 and nav_hint < 0.70:
                        weak_evidence_penalty += 0.12
                reachability_state = _region_reachability_state(str(key))
                reachability_penalty = 0.0
                reachability_priority = 1
                if reachability_state == "reachable":
                    reachability_priority = 0
                elif reachability_state == "out_of_graph":
                    reachability_priority = 3
                    if nav_kind_rank >= 2 and nav_hint >= 0.82:
                        reachability_penalty = 0.08
                        reachability_priority = 2
                    else:
                        reachability_penalty = 0.26
                score = float(
                    (0.50 * base)
                    + (0.20 * pair_affinity)
                    + (0.08 * distance_term)
                    + (0.16 * structural_signal)
                    + (0.16 * nav_hint)
                    + float(far_structural_bonus)
                    + float(anchor_bonus)
                    - float(locality_penalty)
                    - float(stale_penalty)
                    - float(loop_penalty)
                    - float(weak_evidence_penalty)
                    - float(reachability_penalty)
                )
                secondary_candidates.append(
                    (
                        float(score),
                        int(reachability_priority),
                        int(attempts),
                        int(visit_count),
                        str(key),
                        float(pair_affinity),
                        int(distance),
                    )
                )
            secondary_candidates.sort(
                key=lambda item: (
                    -float(item[0]),
                    int(item[1]),
                    int(item[2]),
                    int(item[3]),
                    str(item[4]),
                )
            )
            if secondary_candidates and float(secondary_candidates[0][0]) >= 0.16:
                chosen_secondary = secondary_candidates[0]
                chosen_distance = int(chosen_secondary[6])
                if chosen_distance <= 1:
                    near_score = float(chosen_secondary[0])
                    far_alternative = next(
                        (
                            row
                            for row in secondary_candidates
                            if int(row[6]) >= 2
                            and float(row[0]) >= float(near_score - 0.08)
                        ),
                        None,
                    )
                    if far_alternative is not None:
                        chosen_secondary = far_alternative
                secondary_region_key = str(chosen_secondary[4])
                secondary_region_score = float(chosen_secondary[0])
                pair_affinity_score = float(chosen_secondary[5])
            elif fallback_key != "NA" and str(fallback_key) != str(primary_region_key):
                secondary_region_key = str(fallback_key)
                secondary_region_score = float(_region_base_score(secondary_region_key))

        coupled_region_keys: list[str] = []
        if (
            self._parse_region_key_v1(primary_region_key) is not None
            and str(primary_region_key) not in coupled_region_keys
        ):
            coupled_region_keys.append(str(primary_region_key))
        if (
            self._parse_region_key_v1(secondary_region_key) is not None
            and str(secondary_region_key) not in coupled_region_keys
        ):
            coupled_region_keys.append(str(secondary_region_key))
        if (
            fallback_key != "NA"
            and self._parse_region_key_v1(fallback_key) is not None
            and fallback_key not in coupled_region_keys
            and len(coupled_region_keys) < 2
        ):
            coupled_region_keys.append(str(fallback_key))

        return {
            "selection_method": "dynamic_online_pair_v1",
            "primary_region_key": str(primary_region_key),
            "secondary_region_key": str(secondary_region_key),
            "primary_region_score": float(primary_region_score),
            "secondary_region_score": float(secondary_region_score),
            "pair_affinity_score": float(pair_affinity_score),
            "coupled_region_keys": list(coupled_region_keys),
            "fallback_source_region_key": str(fallback_key),
            "cross_region_key": str(primary_region_key),
            "gate_region_key": str(secondary_region_key),
            "orientation_aligned": bool(orientation_aligned),
            "orientation_misaligned": bool(orientation_misaligned),
        }

    @classmethod
    def _region_distance_v1(
        cls,
        source_region_key: str,
        target_region_key: str,
    ) -> int:
        source = cls._parse_region_key_v1(source_region_key)
        target = cls._parse_region_key_v1(target_region_key)
        if source is None or target is None:
            return 10**6
        sx, sy = source
        tx, ty = target
        return int(abs(int(sx) - int(tx)) + abs(int(sy) - int(ty)))

    @classmethod
    def _region_step_plausible_v1(
        cls,
        source_region_key: str,
        target_region_key: str,
        *,
        max_axis_step: int = 1,
    ) -> bool:
        source = cls._parse_region_key_v1(source_region_key)
        target = cls._parse_region_key_v1(target_region_key)
        if source is None or target is None:
            return False
        limit = int(max(0, max_axis_step))
        sx, sy = source
        tx, ty = target
        dx = abs(int(sx) - int(tx))
        dy = abs(int(sy) - int(ty))
        return bool(dx <= limit and dy <= limit)

    def _region_graph_adjacency_v1(
        self,
        *,
        min_edge_count: int = 1,
    ) -> dict[str, dict[str, int]]:
        threshold = int(max(1, min_edge_count))
        adjacency: dict[str, dict[str, int]] = {}
        for region_action_key, target_histogram in self._region_action_transition_counts.items():
            if not isinstance(target_histogram, dict):
                continue
            try:
                source_region_key, _ = str(region_action_key).split("|a", 1)
            except Exception:
                continue
            if self._parse_region_key_v1(str(source_region_key)) is None:
                continue
            for target_region_key, count_raw in target_histogram.items():
                target_key = str(target_region_key)
                if self._parse_region_key_v1(target_key) is None:
                    continue
                count = int(max(0, count_raw))
                if count < threshold:
                    continue
                if not self._region_step_plausible_v1(
                    str(source_region_key),
                    str(target_key),
                    max_axis_step=1,
                ):
                    continue
                if str(source_region_key) == str(target_key):
                    continue
                source_hist = adjacency.setdefault(str(source_region_key), {})
                source_hist[str(target_key)] = max(
                    int(source_hist.get(str(target_key), 0)),
                    int(count),
                )
                adjacency.setdefault(str(target_key), {})
        return adjacency

    def _region_reachable_set_v1(
        self,
        adjacency: dict[str, dict[str, int]],
        *,
        start_region_key: str,
    ) -> set[str]:
        start_key = str(start_region_key)
        if self._parse_region_key_v1(start_key) is None:
            return set()
        if not adjacency:
            return {str(start_key)}
        queue: list[str] = [str(start_key)]
        visited: set[str] = {str(start_key)}
        cursor = 0
        while cursor < len(queue):
            node = str(queue[cursor])
            cursor += 1
            neighbors = adjacency.get(str(node), {})
            if not isinstance(neighbors, dict):
                continue
            for neighbor_key in neighbors.keys():
                neighbor = str(neighbor_key)
                if self._parse_region_key_v1(neighbor) is None:
                    continue
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        return visited

    def _region_route_distance_v1(
        self,
        adjacency: dict[str, dict[str, int]],
        *,
        start_region_key: str,
        goal_region_key: str,
    ) -> int:
        start_key = str(start_region_key)
        goal_key = str(goal_region_key)
        if self._parse_region_key_v1(start_key) is None:
            return 10**6
        if self._parse_region_key_v1(goal_key) is None:
            return 10**6
        if str(start_key) == str(goal_key):
            return 0
        if not adjacency:
            return 10**6
        if str(start_key) not in adjacency:
            return 10**6
        queue: list[tuple[str, int]] = [(str(start_key), 0)]
        visited: set[str] = {str(start_key)}
        cursor = 0
        while cursor < len(queue):
            node, distance = queue[cursor]
            cursor += 1
            neighbors = adjacency.get(str(node), {})
            if not isinstance(neighbors, dict):
                continue
            for neighbor_key in neighbors.keys():
                neighbor = str(neighbor_key)
                if neighbor in visited:
                    continue
                if self._parse_region_key_v1(neighbor) is None:
                    continue
                next_distance = int(distance + 1)
                if str(neighbor) == str(goal_key):
                    return int(next_distance)
                visited.add(neighbor)
                queue.append((str(neighbor), int(next_distance)))
        return 10**6

    def _changed_region_diff_map_v1(
        self,
        *,
        frame_before: list[list[int]],
        frame_after: list[list[int]],
        max_regions: int = 64,
        allowed_pixel_mask: list[list[bool]] | None = None,
    ) -> dict[str, int]:
        if not frame_before or not frame_after:
            return {}
        usable_height = int(min(len(frame_before), len(frame_after)))
        if usable_height <= 0:
            return {}
        diff_counts: dict[str, int] = {}
        for y in range(usable_height):
            row_before = frame_before[y]
            row_after = frame_after[y]
            if not isinstance(row_before, list) or not isinstance(row_after, list):
                continue
            mask_row = (
                allowed_pixel_mask[y]
                if isinstance(allowed_pixel_mask, list)
                and y < len(allowed_pixel_mask)
                and isinstance(allowed_pixel_mask[y], list)
                else None
            )
            usable_width = int(min(len(row_before), len(row_after)))
            for x in range(usable_width):
                if (
                    isinstance(mask_row, list)
                    and x < len(mask_row)
                    and not bool(mask_row[x])
                ):
                    continue
                if int(row_before[x]) == int(row_after[x]):
                    continue
                rx = int(max(0, min(7, int(x) // 8)))
                ry = int(max(0, min(7, int(y) // 8)))
                region_key = self._region_key_from_xy_v1(int(rx), int(ry))
                diff_counts[str(region_key)] = int(diff_counts.get(str(region_key), 0) + 1)
        rows = sorted(diff_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))
        if int(max_regions) > 0:
            rows = rows[: int(max_regions)]
        return {str(k): int(v) for (k, v) in rows if int(v) > 0}

    @staticmethod
    def _walkable_component_mask_from_anchor_v1(
        frame: list[list[int]],
        *,
        anchor_x: int,
        anchor_y: int,
        search_radius: int = 6,
        min_component_pixels: int = 80,
    ) -> tuple[list[list[bool]] | None, dict[str, Any]]:
        meta: dict[str, Any] = {
            "enabled": False,
            "anchor_xy": {"x": int(anchor_x), "y": int(anchor_y)},
            "selected_color": -1,
            "component_pixels": 0,
            "dilated_pixels": 0,
            "candidate_count": 0,
        }
        if not frame:
            meta["reason"] = "empty_frame"
            return None, meta
        height = int(len(frame))
        width = int(min(len(row) for row in frame if isinstance(row, list)) or 0)
        if height <= 0 or width <= 0:
            meta["reason"] = "invalid_dimensions"
            return None, meta
        if anchor_x < 0 or anchor_y < 0 or anchor_x >= width or anchor_y >= height:
            meta["reason"] = "invalid_anchor"
            return None, meta

        def _component_mask_for_seed(seed_x: int, seed_y: int, color: int) -> list[list[bool]]:
            mask = [[False for _ in range(width)] for _ in range(height)]
            queue: list[tuple[int, int]] = [(int(seed_x), int(seed_y))]
            head = 0
            mask[int(seed_y)][int(seed_x)] = True
            while head < len(queue):
                cx, cy = queue[head]
                head += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = int(cx + dx)
                    ny = int(cy + dy)
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if mask[ny][nx]:
                        continue
                    if int(frame[ny][nx]) != int(color):
                        continue
                    mask[ny][nx] = True
                    queue.append((int(nx), int(ny)))
            return mask

        def _mask_pixels(mask: list[list[bool]]) -> int:
            return int(sum(1 for row in mask for v in row if bool(v)))

        center_color = int(frame[int(anchor_y)][int(anchor_x)])
        color_hist: dict[int, int] = {}
        nearest_xy_by_color: dict[int, tuple[int, int, int]] = {}
        for y in range(max(0, int(anchor_y) - int(search_radius)), min(height, int(anchor_y) + int(search_radius) + 1)):
            row = frame[y]
            if not isinstance(row, list):
                continue
            for x in range(max(0, int(anchor_x) - int(search_radius)), min(width, int(anchor_x) + int(search_radius) + 1)):
                dist = int(abs(int(x) - int(anchor_x)) + abs(int(y) - int(anchor_y)))
                if dist < 2 or dist > int(search_radius):
                    continue
                color = int(row[x])
                color_hist[color] = int(color_hist.get(color, 0) + 1)
                prev = nearest_xy_by_color.get(color)
                if prev is None or dist < int(prev[0]):
                    nearest_xy_by_color[color] = (int(dist), int(x), int(y))

        ranked_colors = sorted(
            color_hist.items(),
            key=lambda item: (-int(item[1]), 0 if int(item[0]) != int(center_color) else 1, int(item[0])),
        )
        candidate_colors = [int(color) for (color, count) in ranked_colors if int(count) >= 4][:6]
        if int(center_color) not in candidate_colors:
            candidate_colors.append(int(center_color))
        meta["candidate_count"] = int(len(candidate_colors))

        best_mask: list[list[bool]] | None = None
        best_pixels = 0
        best_color = -1
        for color in candidate_colors:
            if int(color) in nearest_xy_by_color:
                _, sx, sy = nearest_xy_by_color[int(color)]
            else:
                sx = int(anchor_x)
                sy = int(anchor_y)
            if int(frame[int(sy)][int(sx)]) != int(color):
                found = False
                for y in range(max(0, int(anchor_y) - int(search_radius)), min(height, int(anchor_y) + int(search_radius) + 1)):
                    row = frame[y]
                    if not isinstance(row, list):
                        continue
                    for x in range(max(0, int(anchor_x) - int(search_radius)), min(width, int(anchor_x) + int(search_radius) + 1)):
                        if int(row[x]) == int(color):
                            sx = int(x)
                            sy = int(y)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    continue
            mask = _component_mask_for_seed(int(sx), int(sy), int(color))
            pixels = _mask_pixels(mask)
            if pixels > int(best_pixels):
                best_pixels = int(pixels)
                best_mask = mask
                best_color = int(color)

        if best_mask is None or int(best_pixels) < int(min_component_pixels):
            meta["reason"] = "component_too_small"
            meta["component_pixels"] = int(best_pixels)
            return None, meta

        dilated = [[bool(v) for v in row] for row in best_mask]
        for _ in range(3):
            expanded = [list(row) for row in dilated]
            for y in range(height):
                for x in range(width):
                    if not dilated[y][x]:
                        continue
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx = int(x + dx)
                        ny = int(y + dy)
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        expanded[ny][nx] = True
            dilated = expanded

        meta["enabled"] = True
        meta["selected_color"] = int(best_color)
        meta["component_pixels"] = int(best_pixels)
        meta["dilated_pixels"] = int(_mask_pixels(dilated))
        return dilated, meta

    def _high_info_allowed_pixel_mask_v1(
        self,
        *,
        frame_before: list[list[int]],
        frame_after: list[list[int]],
        previous_representation: RepresentationStateV1,
        current_representation: RepresentationStateV1,
        navigation_state_estimate: dict[str, Any],
        tracked_token_before: str | None,
    ) -> tuple[list[list[bool]] | None, dict[str, Any]]:
        meta: dict[str, Any] = {
            "enabled": False,
            "mask_pixels": 0,
            "mask_ratio": 0.0,
            "method": "region_allowlist_v2",
            "visited_region_count": 0,
            "map_region_count": 0,
            "frontier_region_count": 0,
            "allowed_region_count": 0,
            "seed_region_keys": [],
            "trusted_seed_region_keys": [],
            "dropped_seed_region_keys": [],
            "map_region_keys": [],
            "frontier_region_keys": [],
        }
        if not frame_before or not frame_after:
            meta["reason"] = "empty_frame"
            return None, meta
        height = int(min(len(frame_before), len(frame_after)))
        width = int(
            min(
                min((len(row) for row in frame_before if isinstance(row, list)), default=0),
                min((len(row) for row in frame_after if isinstance(row, list)), default=0),
            )
        )
        if height <= 0 or width <= 0:
            meta["reason"] = "invalid_dimensions"
            return None, meta

        before_seed: tuple[int, int] | None = None
        after_seed: tuple[int, int] | None = None
        if tracked_token_before:
            tracked_before = self._find_object_by_digest_v1(
                previous_representation,
                str(tracked_token_before),
            )
            if isinstance(tracked_before, dict):
                bx = int(tracked_before.get("centroid_x", -1))
                by = int(tracked_before.get("centroid_y", -1))
                if bx >= 0 and by >= 0:
                    before_seed = (int(bx), int(by))
        if before_seed is None and self._tracked_agent_anchor_xy is not None:
            before_seed = (
                int(self._tracked_agent_anchor_xy[0]),
                int(self._tracked_agent_anchor_xy[1]),
            )
        if isinstance(navigation_state_estimate, dict):
            pos = navigation_state_estimate.get("agent_pos_xy", {})
            if isinstance(pos, dict):
                ax = int(pos.get("x", -1))
                ay = int(pos.get("y", -1))
                if ax >= 0 and ay >= 0:
                    after_seed = (int(ax), int(ay))
        if after_seed is None and self._tracked_agent_token_digest:
            tracked_after = self._find_object_by_digest_v1(
                current_representation,
                str(self._tracked_agent_token_digest),
            )
            if isinstance(tracked_after, dict):
                ax = int(tracked_after.get("centroid_x", -1))
                ay = int(tracked_after.get("centroid_y", -1))
                if ax >= 0 and ay >= 0:
                    after_seed = (int(ax), int(ay))
        if after_seed is None and before_seed is not None:
            after_seed = (int(before_seed[0]), int(before_seed[1]))

        def _arena_bbox_candidate(rep: RepresentationStateV1) -> tuple[float, dict[str, Any]] | None:
            nodes = list(getattr(rep, "object_nodes", []) or [])
            if not nodes:
                return None
            seed_xy = after_seed if after_seed is not None else before_seed
            frame_pixels = int(max(1, height * width))
            min_area = int(max(64, frame_pixels * 0.05))
            best: tuple[float, dict[str, Any]] | None = None
            for node in nodes:
                try:
                    if bool(getattr(node, "touches_boundary", False)):
                        continue
                    area = int(getattr(node, "area", 0))
                    if area < min_area:
                        continue
                    min_x = int(getattr(node, "bbox_min_x", -1))
                    min_y = int(getattr(node, "bbox_min_y", -1))
                    max_x = int(getattr(node, "bbox_max_x", -1))
                    max_y = int(getattr(node, "bbox_max_y", -1))
                    if min_x < 0 or min_y < 0 or max_x < 0 or max_y < 0:
                        continue
                    if min_x > max_x or min_y > max_y:
                        continue
                    if max_x >= width or max_y >= height:
                        continue
                    bbox_w = int(max_x - min_x + 1)
                    bbox_h = int(max_y - min_y + 1)
                    if bbox_w < 8 or bbox_h < 8:
                        continue
                    bbox_area = int(bbox_w * bbox_h)
                    if bbox_area <= 0:
                        continue
                    fill_ratio = float(area) / float(max(1, bbox_area))
                    if fill_ratio < 0.18:
                        continue
                    if seed_xy is not None:
                        sx, sy = int(seed_xy[0]), int(seed_xy[1])
                        if not (min_x <= sx <= max_x and min_y <= sy <= max_y):
                            continue
                    cx = int(getattr(node, "centroid_x", (min_x + max_x) // 2))
                    cy = int(getattr(node, "centroid_y", (min_y + max_y) // 2))
                except Exception:
                    continue
                center_dist = int(abs(cx - (width // 2)) + abs(cy - (height // 2)))
                score = float(area) + (200.0 * float(fill_ratio)) - (0.35 * float(center_dist))
                payload = {
                    "digest": str(getattr(node, "digest", "")),
                    "color": int(getattr(node, "color", -1)),
                    "area": int(area),
                    "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
                    "bbox_area": int(bbox_area),
                    "fill_ratio": float(round(fill_ratio, 4)),
                    "seed_required": bool(seed_xy is not None),
                }
                if best is None or score > float(best[0]):
                    best = (float(score), payload)
            return best

        # Prefer an arena bounding box mask inferred from the representation itself.
        # This keeps the full map visible to high-info detection (including remote effects),
        # while excluding HUD/hub pixels outside the arena.
        arena = _arena_bbox_candidate(current_representation) or _arena_bbox_candidate(
            previous_representation
        )
        if arena is not None:
            _, payload = arena
            bbox = payload.get("bbox", [-1, -1, -1, -1])
            if isinstance(bbox, list) and len(bbox) >= 4:
                min_x, min_y, max_x, max_y = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                margin = 2
                min_x = int(max(0, min_x - margin))
                min_y = int(max(0, min_y - margin))
                max_x = int(min(width - 1, max_x + margin))
                max_y = int(min(height - 1, max_y + margin))
                mask = [[False for _ in range(width)] for _ in range(height)]
                for y in range(min_y, max_y + 1):
                    row = mask[y]
                    for x in range(min_x, max_x + 1):
                        row[x] = True
                mask_pixels = int((max_x - min_x + 1) * (max_y - min_y + 1))
                frame_pixels = int(max(1, height * width))
                mask_ratio = float(mask_pixels) / float(frame_pixels)
                if mask_ratio < 0.06:
                    meta["reason"] = "arena_mask_too_small"
                    meta["arena_object"] = dict(payload)
                    meta["arena_bbox"] = [int(min_x), int(min_y), int(max_x), int(max_y)]
                    if self._high_info_cached_arena_bbox is not None:
                        # Fall back to the last known good bbox rather than disabling masking.
                        cached = tuple(int(v) for v in self._high_info_cached_arena_bbox)
                        cmin_x, cmin_y, cmax_x, cmax_y = cached
                        mask = [[False for _ in range(width)] for _ in range(height)]
                        for y in range(max(0, cmin_y), min(height - 1, cmax_y) + 1):
                            row = mask[y]
                            for x in range(max(0, cmin_x), min(width - 1, cmax_x) + 1):
                                row[x] = True
                        mask_pixels = int((cmax_x - cmin_x + 1) * (cmax_y - cmin_y + 1))
                        frame_pixels = int(max(1, height * width))
                        meta["enabled"] = True
                        meta["method"] = "arena_bbox_cached_v1"
                        meta["mask_pixels"] = int(mask_pixels)
                        meta["mask_ratio"] = float(mask_pixels) / float(frame_pixels)
                        meta["arena_bbox"] = [int(cmin_x), int(cmin_y), int(cmax_x), int(cmax_y)]
                        meta["arena_margin"] = 0
                        meta["reason"] = "fallback_cached_small"
                        return mask, meta
                    return None, meta
                if mask_ratio > 0.80:
                    meta["reason"] = "arena_mask_too_large"
                    meta["arena_object"] = dict(payload)
                    meta["arena_bbox"] = [int(min_x), int(min_y), int(max_x), int(max_y)]
                    if self._high_info_cached_arena_bbox is not None:
                        cached = tuple(int(v) for v in self._high_info_cached_arena_bbox)
                        cmin_x, cmin_y, cmax_x, cmax_y = cached
                        mask = [[False for _ in range(width)] for _ in range(height)]
                        for y in range(max(0, cmin_y), min(height - 1, cmax_y) + 1):
                            row = mask[y]
                            for x in range(max(0, cmin_x), min(width - 1, cmax_x) + 1):
                                row[x] = True
                        mask_pixels = int((cmax_x - cmin_x + 1) * (cmax_y - cmin_y + 1))
                        frame_pixels = int(max(1, height * width))
                        meta["enabled"] = True
                        meta["method"] = "arena_bbox_cached_v1"
                        meta["mask_pixels"] = int(mask_pixels)
                        meta["mask_ratio"] = float(mask_pixels) / float(frame_pixels)
                        meta["arena_bbox"] = [int(cmin_x), int(cmin_y), int(cmax_x), int(cmax_y)]
                        meta["arena_margin"] = 0
                        meta["reason"] = "fallback_cached_large"
                        return mask, meta
                    return None, meta
                meta["enabled"] = True
                meta["method"] = "arena_bbox_v1"
                meta["mask_pixels"] = int(mask_pixels)
                meta["mask_ratio"] = float(mask_ratio)
                meta["arena_object"] = dict(payload)
                meta["arena_bbox"] = [int(min_x), int(min_y), int(max_x), int(max_y)]
                meta["arena_margin"] = int(margin)
                # Cache for future frames where segmentation may temporarily drift.
                self._high_info_cached_arena_bbox = (
                    int(min_x),
                    int(min_y),
                    int(max_x),
                    int(max_y),
                )
                return mask, meta

        # Fallback to the last known good arena bbox. This prevents HUD-driven global diffs
        # from being treated as high-info when tracking temporarily drifts.
        if self._high_info_cached_arena_bbox is not None:
            cmin_x, cmin_y, cmax_x, cmax_y = (
                int(self._high_info_cached_arena_bbox[0]),
                int(self._high_info_cached_arena_bbox[1]),
                int(self._high_info_cached_arena_bbox[2]),
                int(self._high_info_cached_arena_bbox[3]),
            )
            cmin_x = int(max(0, min(width - 1, cmin_x)))
            cmin_y = int(max(0, min(height - 1, cmin_y)))
            cmax_x = int(max(0, min(width - 1, cmax_x)))
            cmax_y = int(max(0, min(height - 1, cmax_y)))
            if cmin_x <= cmax_x and cmin_y <= cmax_y:
                mask = [[False for _ in range(width)] for _ in range(height)]
                for y in range(cmin_y, cmax_y + 1):
                    row = mask[y]
                    for x in range(cmin_x, cmax_x + 1):
                        row[x] = True
                mask_pixels = int((cmax_x - cmin_x + 1) * (cmax_y - cmin_y + 1))
                frame_pixels = int(max(1, height * width))
                meta["enabled"] = True
                meta["method"] = "arena_bbox_cached_v1"
                meta["mask_pixels"] = int(mask_pixels)
                meta["mask_ratio"] = float(mask_pixels) / float(frame_pixels)
                meta["arena_bbox"] = [int(cmin_x), int(cmin_y), int(cmax_x), int(cmax_y)]
                meta["arena_margin"] = 0
                meta["reason"] = "fallback_cached_missing"
                return mask, meta

        def _region_key_from_pixel(px: int, py: int) -> str:
            rx = int(max(0, min(7, int(px) // 8)))
            ry = int(max(0, min(7, int(py) // 8)))
            return self._region_key_from_xy_v1(int(rx), int(ry))

        def _neighbor_region_keys(region_key: str) -> list[str]:
            parsed = self._parse_region_key_v1(str(region_key))
            if parsed is None:
                return []
            rx, ry = int(parsed[0]), int(parsed[1])
            neighbors: list[str] = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = int(rx + dx)
                ny = int(ry + dy)
                if nx < 0 or ny < 0 or nx > 7 or ny > 7:
                    continue
                neighbors.append(self._region_key_from_xy_v1(int(nx), int(ny)))
            return neighbors

        seed_region_keys: set[str] = set()
        if before_seed is not None:
            seed_region_keys.add(_region_key_from_pixel(int(before_seed[0]), int(before_seed[1])))
        if after_seed is not None:
            seed_region_keys.add(_region_key_from_pixel(int(after_seed[0]), int(after_seed[1])))
        nav_region_payload = navigation_state_estimate.get("agent_pos_region", {})
        if isinstance(nav_region_payload, dict):
            nav_key = self._region_key_from_region_payload_v1(nav_region_payload)
            if self._parse_region_key_v1(str(nav_key)) is not None:
                seed_region_keys.add(str(nav_key))
        if self._last_known_agent_pos_region is not None:
            seed_region_keys.add(
                self._region_key_from_xy_v1(
                    int(self._last_known_agent_pos_region[0]),
                    int(self._last_known_agent_pos_region[1]),
                )
            )
        if self._latest_observed_agent_pos_region is not None:
            seed_region_keys.add(
                self._region_key_from_xy_v1(
                    int(self._latest_observed_agent_pos_region[0]),
                    int(self._latest_observed_agent_pos_region[1]),
                )
            )
        seed_region_keys = {
            str(key) for key in seed_region_keys if self._parse_region_key_v1(str(key)) is not None
        }

        visited_region_keys: set[str] = {
            str(region_key)
            for (region_key, count) in self._region_visit_counts.items()
            if int(count) > 0 and self._parse_region_key_v1(str(region_key)) is not None
        }
        if not visited_region_keys:
            visited_region_keys = set(seed_region_keys)

        map_region_keys: set[str] = set()
        if visited_region_keys:
            nav_anchor_key = "NA"
            if isinstance(nav_region_payload, dict):
                nav_anchor_key = self._region_key_from_region_payload_v1(nav_region_payload)
            if self._parse_region_key_v1(str(nav_anchor_key)) is None:
                nav_anchor_key = "NA"
            primary_anchor_candidates: list[str] = []
            if str(nav_anchor_key) in visited_region_keys:
                primary_anchor_candidates.append(str(nav_anchor_key))
            if self._last_known_agent_pos_region is not None:
                last_key = self._region_key_from_xy_v1(
                    int(self._last_known_agent_pos_region[0]),
                    int(self._last_known_agent_pos_region[1]),
                )
                if str(last_key) in visited_region_keys:
                    primary_anchor_candidates.append(str(last_key))
            if self._latest_observed_agent_pos_region is not None:
                obs_key = self._region_key_from_xy_v1(
                    int(self._latest_observed_agent_pos_region[0]),
                    int(self._latest_observed_agent_pos_region[1]),
                )
                if str(obs_key) in visited_region_keys:
                    primary_anchor_candidates.append(str(obs_key))
            anchor_candidates = [
                str(key)
                for key in primary_anchor_candidates
                if self._parse_region_key_v1(str(key)) is not None
            ]
            if not anchor_candidates:
                anchor_candidates = [str(key) for key in seed_region_keys if str(key) in visited_region_keys]
            if not anchor_candidates and seed_region_keys and visited_region_keys:
                nearest_anchor = min(
                    visited_region_keys,
                    key=lambda key: (
                        min(
                            self._region_distance_v1(str(key), str(seed_key))
                            for seed_key in seed_region_keys
                        ),
                        -int(self._region_visit_counts.get(str(key), 0)),
                        str(key),
                    ),
                )
                anchor_candidates = [str(nearest_anchor)]
            if not anchor_candidates and visited_region_keys:
                anchor_candidates = [
                    str(
                        max(
                            visited_region_keys,
                            key=lambda key: (
                                int(self._region_visit_counts.get(str(key), 0)),
                                -int(self._parse_region_key_v1(str(key))[1]) if self._parse_region_key_v1(str(key)) is not None else 0,
                                -int(self._parse_region_key_v1(str(key))[0]) if self._parse_region_key_v1(str(key)) is not None else 0,
                            ),
                        )
                    )
                ]
            if anchor_candidates:
                queue = [str(anchor_candidates[0])]
                head = 0
                map_region_keys.add(str(anchor_candidates[0]))
                while head < len(queue):
                    current_key = str(queue[head])
                    head += 1
                    for neighbor_key in _neighbor_region_keys(str(current_key)):
                        key = str(neighbor_key)
                        if key in map_region_keys or key not in visited_region_keys:
                            continue
                        map_region_keys.add(str(key))
                        queue.append(str(key))
        if not map_region_keys:
            map_region_keys = set(visited_region_keys)

        frontier_region_keys: set[str] = set()
        for region_key in list(map_region_keys):
            for neighbor_key in _neighbor_region_keys(str(region_key)):
                if str(neighbor_key) in map_region_keys:
                    continue
                frontier_region_keys.add(str(neighbor_key))

        trusted_seed_region_keys: set[str] = set()
        dropped_seed_region_keys: set[str] = set()
        for seed_key in seed_region_keys:
            key = str(seed_key)
            if key in map_region_keys or key in frontier_region_keys:
                trusted_seed_region_keys.add(key)
            else:
                dropped_seed_region_keys.add(key)

        allowed_region_keys: set[str] = set(map_region_keys)
        allowed_region_keys.update(str(v) for v in frontier_region_keys)
        allowed_region_keys.update(str(v) for v in trusted_seed_region_keys)
        allowed_region_keys = {
            str(key)
            for key in allowed_region_keys
            if self._parse_region_key_v1(str(key)) is not None
        }
        if not allowed_region_keys:
            meta["reason"] = "no_allowed_regions"
            return None, meta

        mask = [[False for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                region_key = _region_key_from_pixel(int(x), int(y))
                if str(region_key) in allowed_region_keys:
                    mask[y][x] = True

        mask_pixels = int(sum(1 for row in mask for v in row if bool(v)))
        frame_pixels = int(max(1, height * width))
        mask_ratio = float(mask_pixels / float(frame_pixels))
        if mask_pixels <= 0:
            meta["reason"] = "mask_empty"
            meta["mask_pixels"] = int(mask_pixels)
            meta["mask_ratio"] = float(mask_ratio)
            return None, meta

        meta["enabled"] = True
        meta["mask_pixels"] = int(mask_pixels)
        meta["mask_ratio"] = float(mask_ratio)
        meta["visited_region_count"] = int(len(visited_region_keys))
        meta["map_region_count"] = int(len(map_region_keys))
        meta["frontier_region_count"] = int(len(frontier_region_keys))
        meta["allowed_region_count"] = int(len(allowed_region_keys))
        meta["seed_region_keys"] = [str(v) for v in sorted(seed_region_keys)[:16]]
        meta["trusted_seed_region_keys"] = [str(v) for v in sorted(trusted_seed_region_keys)[:16]]
        meta["dropped_seed_region_keys"] = [str(v) for v in sorted(dropped_seed_region_keys)[:16]]
        meta["map_region_keys"] = [str(v) for v in sorted(map_region_keys)[:32]]
        meta["frontier_region_keys"] = [str(v) for v in sorted(frontier_region_keys)[:32]]
        return mask, meta

    def _simultaneous_changed_region_info_v1(
        self,
        *,
        transition_record: TransitionRecordV1 | None,
        source_region_key: str,
        anchor_region_key: str,
        region_adjacency: dict[str, dict[str, int]],
        reachable_region_set: set[str],
        reachability_graph_ready: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "changed_region_keys": [],
            "reachable_region_keys": [],
            "unknown_region_keys": [],
            "unreachable_region_keys": [],
            "anchor_region_key": str(anchor_region_key),
            "changed_total_pixels": 0,
            "changed_region_diff_map": {},
        }
        if not bool(self.high_info_simultaneous_focus_enabled):
            return result
        if transition_record is None:
            return result
        effect_summary = transition_record.effect_summary
        if not isinstance(effect_summary, dict):
            return result
        diff_map_raw = effect_summary.get("changed_region_diff_map_v1", {})
        if not isinstance(diff_map_raw, dict):
            return result
        diff_map: dict[str, int] = {}
        for region_key_raw, count_raw in diff_map_raw.items():
            region_key = str(region_key_raw)
            if self._parse_region_key_v1(region_key) is None:
                continue
            count = int(max(0, count_raw))
            if count <= 0:
                continue
            diff_map[str(region_key)] = int(count)
        if not diff_map:
            return result
        changed_total_pixels = int(sum(int(v) for v in diff_map.values()))
        if int(changed_total_pixels) < int(self.high_info_simultaneous_min_total_pixels):
            return result
        result["changed_total_pixels"] = int(changed_total_pixels)
        result["changed_region_diff_map"] = {
            str(k): int(v)
            for (k, v) in sorted(diff_map.items(), key=lambda item: (-int(item[1]), str(item[0])))
        }
        top_count = int(max(diff_map.values()))
        min_pixels = int(max(1, int(self.high_info_simultaneous_min_region_pixels)))
        top_ratio_floor = float(
            max(0.0, min(1.0, float(self.high_info_simultaneous_top_region_ratio)))
        )
        sorted_regions = sorted(
            diff_map.items(),
            key=lambda item: (
                -int(item[1]),
                int(
                    self._region_distance_v1(
                        str(anchor_region_key),
                        str(item[0]),
                    )
                ),
                str(item[0]),
            ),
        )
        selected: list[str] = []
        for region_key, count in sorted_regions:
            if int(count) < int(min_pixels):
                continue
            ratio_to_top = float(count) / float(max(1, top_count))
            if ratio_to_top < float(top_ratio_floor):
                continue
            selected.append(str(region_key))
        if len(selected) < 2:
            for region_key, count in sorted_regions:
                if int(count) < int(min_pixels):
                    continue
                key = str(region_key)
                if key in selected:
                    continue
                selected.append(key)
                if len(selected) >= 2:
                    break
        source_key = str(source_region_key)
        if self._parse_region_key_v1(source_key) is not None and source_key not in selected:
            selected.insert(0, str(source_key))
        deduped_selected: list[str] = []
        for region_key in selected:
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                continue
            if key in deduped_selected:
                continue
            deduped_selected.append(key)
        if len(deduped_selected) < 2:
            return result
        max_targets = int(max(1, self.high_info_simultaneous_max_targets))
        if max_targets > 0:
            deduped_selected = deduped_selected[:max_targets]
        reachable_sorted: list[tuple[int, int, str]] = []
        unknown_sorted: list[tuple[int, int, str]] = []
        unreachable_sorted: list[tuple[int, int, str]] = []
        reachable_set = {str(v) for v in reachable_region_set if self._parse_region_key_v1(str(v))}
        for region_key in deduped_selected:
            key = str(region_key)
            count = int(max(0, diff_map.get(key, 0)))
            route_distance = int(
                self._region_route_distance_v1(
                    region_adjacency,
                    start_region_key=str(anchor_region_key),
                    goal_region_key=str(key),
                )
            )
            if route_distance >= 10**6:
                route_distance = int(
                    self._region_distance_v1(
                        str(anchor_region_key),
                        str(key),
                    )
                )
            row = (int(route_distance), -int(count), str(key))
            if key in reachable_set or str(key) == str(anchor_region_key):
                reachable_sorted.append(row)
            else:
                # Treat non-reachable keys as unknown, even if the reachability graph is \"ready\".
                # Empirical region graphs can be incomplete early in exploration; over-pruning here
                # prevents the agent from following remote high-diff effects (e.g., trigger->exit).
                unknown_sorted.append(row)
        reachable_sorted.sort()
        unknown_sorted.sort()
        unreachable_sorted.sort()
        changed_region_keys = [
            str(row[2]) for row in (reachable_sorted + unknown_sorted + unreachable_sorted)
        ]
        result["changed_region_keys"] = list(changed_region_keys)
        result["reachable_region_keys"] = [str(row[2]) for row in reachable_sorted]
        result["unknown_region_keys"] = [str(row[2]) for row in unknown_sorted]
        result["unreachable_region_keys"] = [str(row[2]) for row in unreachable_sorted]
        return result

    def _current_region_key_v1(self) -> str:
        """Return the current coarse region key used for routing/coverage.

        **Hard rule (stability):** when navigation tracking is matched and not flagged as a
        peripheral UI candidate, prefer the navigation-derived region (agent_pos_region)
        over the representation-derived observed region. The representation-derived region
        is only used as a fallback when navigation tracking is unavailable or implausible.

        This prevents region-key jitter near cell boundaries (e.g., centroid rounding) from
        corrupting the region graph / route planning and causing long action loops.
        """
        latest = (
            self._latest_navigation_state_estimate
            if isinstance(self._latest_navigation_state_estimate, dict)
            else {}
        )

        # Representation-derived region (fallback only).
        observed_key = "NA"
        if self._latest_observed_agent_pos_region is not None:
            orx, ory = self._latest_observed_agent_pos_region
            observed_key = self._region_key_from_xy_v1(int(orx), int(ory))

        # Navigation-derived region (preferred when matched).
        latest_key = "NA"
        region = latest.get("agent_pos_region", {})
        nav_matched = bool(latest.get("matched", False)) and isinstance(region, dict)
        nav_ui_candidate = bool(latest.get("peripheral_ui_candidate", False))
        if nav_matched:
            rx = int(region.get("x", -1))
            ry = int(region.get("y", -1))
            if rx >= 0 and ry >= 0:
                latest_key = self._region_key_from_xy_v1(int(rx), int(ry))

        last_key = "NA"
        if self._last_known_agent_pos_region is not None:
            last_rx, last_ry = self._last_known_agent_pos_region
            last_key = self._region_key_from_xy_v1(int(last_rx), int(last_ry))

        def _plausible_from_last(candidate_key: str) -> bool:
            if self._parse_region_key_v1(str(candidate_key)) is None:
                return False
            if self._parse_region_key_v1(str(last_key)) is None:
                return True
            return bool(
                self._region_step_plausible_v1(
                    str(last_key),
                    str(candidate_key),
                    max_axis_step=1,
                )
            )

        # Prefer navigation region when available (and not UI).
        if (
            nav_matched
            and (not nav_ui_candidate)
            and self._parse_region_key_v1(str(latest_key)) is not None
        ):
            if self._parse_region_key_v1(str(last_key)) is not None and not _plausible_from_last(
                str(latest_key)
            ):
                # Navigation jumped implausibly; try observed if plausible, else stick to last.
                if _plausible_from_last(str(observed_key)):
                    return str(observed_key)
                return str(last_key)
            return str(latest_key)

        # If navigation is missing/unreliable, fall back to representation-derived observed region.
        if _plausible_from_last(str(observed_key)):
            return str(observed_key)

        # Final fallback: keep last-known region if available.
        if self._parse_region_key_v1(str(last_key)) is not None:
            return str(last_key)

        if self._parse_region_key_v1(str(latest_key)) is not None:
            return str(latest_key)

        return "NA"

    def _update_observed_agent_region_from_representation_v1(
        self,
        representation: RepresentationStateV1,
    ) -> None:
        object_nodes = list(getattr(representation, "object_nodes", []))
        if not object_nodes:
            return
        frame_width = int(max(1, getattr(representation, "frame_width", 1)))
        frame_height = int(max(1, getattr(representation, "frame_height", 1)))
        peripheral_margin_x = max(1, int(round(float(frame_width) * 0.08)))
        peripheral_margin_y = max(1, int(round(float(frame_height) * 0.08)))
        ui_max_area = max(4, int(round(float(frame_width * frame_height) * 0.004)))
        ui_max_side = max(2, int(round(float(min(frame_width, frame_height)) * 0.08)))

        def peripheral_ui_likelihood(node: Any) -> float:
            try:
                area = int(getattr(node, "area", 0))
                cx = int(getattr(node, "centroid_x", -1))
                cy = int(getattr(node, "centroid_y", -1))
                min_x = int(getattr(node, "bbox_min_x", cx))
                max_x = int(getattr(node, "bbox_max_x", cx))
                min_y = int(getattr(node, "bbox_min_y", cy))
                max_y = int(getattr(node, "bbox_max_y", cy))
            except Exception:
                return 0.0
            if cx < 0 or cy < 0:
                return 0.0
            width = max(1, int(max_x - min_x + 1))
            height = max(1, int(max_y - min_y + 1))
            near_periphery = bool(
                cx < peripheral_margin_x
                or cx >= max(0, frame_width - peripheral_margin_x)
                or cy < peripheral_margin_y
                or cy >= max(0, frame_height - peripheral_margin_y)
            )
            if not near_periphery:
                return 0.0
            area_score = (
                1.0
                if area <= ui_max_area
                else max(0.0, 1.0 - (float(area - ui_max_area) / float(max(1, ui_max_area * 2))))
            )
            side = max(width, height)
            side_score = 1.0 if side <= ui_max_side else 0.0
            boundary_score = 0.25 if bool(getattr(node, "touches_boundary", False)) else 0.0
            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        (0.65 * area_score) + (0.25 * side_score) + float(boundary_score),
                    ),
                )
            )

        last_region = self._last_known_agent_pos_region
        last_cx = None
        last_cy = None
        if last_region is not None:
            last_cx = (int(last_region[0]) * 8) + 4
            last_cy = (int(last_region[1]) * 8) + 4
        anchor_x = None
        anchor_y = None
        if self._tracked_agent_anchor_xy is not None:
            anchor_x = int(self._tracked_agent_anchor_xy[0])
            anchor_y = int(self._tracked_agent_anchor_xy[1])
        expected_action_id = int(self._previous_action_candidate.action_id) if self._previous_action_candidate is not None else -1
        expected_dir = self._action_direction_vector_v1(expected_action_id)
        expected_step = 5
        expected_x = anchor_x
        expected_y = anchor_y
        if anchor_x is not None and anchor_y is not None and expected_dir is not None:
            ex, ey = expected_dir
            expected_x = int(anchor_x + (int(ex) * int(expected_step)))
            expected_y = int(anchor_y + (int(ey) * int(expected_step)))

        best_node = None
        best_score = float("inf")
        for node in object_nodes:
            area = int(getattr(node, "area", 0))
            cx = int(getattr(node, "centroid_x", -1))
            cy = int(getattr(node, "centroid_y", -1))
            if area <= 0 or cx < 0 or cy < 0:
                continue
            ui_likelihood = float(peripheral_ui_likelihood(node))
            if ui_likelihood >= 0.95:
                continue
            score = 0.0
            if self._tracked_agent_area_ema is not None:
                score += 0.55 * float(abs(float(area) - float(self._tracked_agent_area_ema)))
            else:
                score += 0.10 * float(abs(area - 14))
            if self._tracked_agent_color is not None and int(getattr(node, "color", -1)) != int(self._tracked_agent_color):
                score += 8.0
            if self._tracked_agent_token_digest and str(getattr(node, "digest", "")) == str(self._tracked_agent_token_digest):
                score -= 10.0
            if bool(getattr(node, "touches_boundary", False)):
                score += 12.0
            score += float(16.0 * ui_likelihood)
            if last_cx is not None and last_cy is not None:
                score += 0.06 * float(abs(cx - int(last_cx)) + abs(cy - int(last_cy)))
            if expected_x is not None and expected_y is not None:
                expected_dist = int(abs(cx - int(expected_x)) + abs(cy - int(expected_y)))
                score += 0.28 * float(expected_dist)
                if expected_dist > 20:
                    score += 80.0
            elif anchor_x is not None and anchor_y is not None:
                anchor_dist = int(abs(cx - int(anchor_x)) + abs(cy - int(anchor_y)))
                score += 0.22 * float(anchor_dist)
                if anchor_dist > 16:
                    score += 64.0
            if last_region is not None:
                last_key = self._region_key_from_xy_v1(int(last_region[0]), int(last_region[1]))
                candidate_key = self._region_key_from_xy_v1(
                    int(max(0, min(7, cx // 8))),
                    int(max(0, min(7, cy // 8))),
                )
                if not self._region_step_plausible_v1(
                    str(last_key),
                    str(candidate_key),
                    max_axis_step=1,
                ):
                    score += 14.0
            if score < best_score:
                best_score = float(score)
                best_node = node
        if best_node is None:
            return
        rx = int(max(0, min(7, int(best_node.centroid_x) // 8)))
        ry = int(max(0, min(7, int(best_node.centroid_y) // 8)))
        self._latest_observed_agent_pos_region = (int(rx), int(ry))

    def _navigation_semantic_features_v1(self) -> dict[str, Any]:
        compare_count = int(max(0, self._navigation_semantic_compare_count))
        mismatch_count = int(max(0, self._navigation_semantic_mismatch_count))
        nav_attempts = int(max(0, self._navigation_attempt_count))
        nav_match_count = int(max(0, self._navigation_match_count))
        mismatch_rate = float(mismatch_count / float(max(1, compare_count)))
        consistency = float(max(0.0, min(1.0, 1.0 - mismatch_rate)))
        match_rate = float(nav_match_count / float(max(1, nav_attempts)))
        sample_confidence = float(1.0 - math.exp(-float(compare_count) / 24.0))
        confidence_prior = 0.75
        confidence_raw = float((0.55 * consistency) + (0.45 * match_rate))
        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    (sample_confidence * confidence_raw)
                    + ((1.0 - sample_confidence) * confidence_prior),
                ),
            )
        )
        return {
            "schema_name": "active_inference_navigation_semantic_features_v1",
            "schema_version": 1,
            "enabled": bool(self.enable_navigation_confidence_gating),
            "compare_count": int(compare_count),
            "mismatch_count": int(mismatch_count),
            "mismatch_rate": float(mismatch_rate),
            "consistency": float(consistency),
            "match_rate": float(match_rate),
            "sample_confidence": float(sample_confidence),
            "confidence": float(confidence),
            "low_confidence": bool(confidence < 0.55),
            "high_mismatch": bool(compare_count >= 12 and mismatch_rate >= 0.40),
        }

    def _sequence_causal_state_snapshot_v1(self) -> dict[str, Any]:
        state = (
            self._sequence_causal_state_v1
            if isinstance(self._sequence_causal_state_v1, dict)
            else {}
        )
        return {
            "schema_name": "active_inference_sequence_causal_state_v1",
            "schema_version": 1,
            "enabled": bool(state.get("enabled", False)),
            "trigger_region_key": str(state.get("trigger_region_key", "NA")),
            "target_region_key": str(state.get("target_region_key", "NA")),
            "trigger_region_key_effective": str(
                state.get(
                    "trigger_region_key_effective",
                    state.get("trigger_region_key", "NA"),
                )
            ),
            "target_region_key_effective": str(
                state.get(
                    "target_region_key_effective",
                    state.get("target_region_key", "NA"),
                )
            ),
            "window_steps": int(state.get("window_steps", 0)),
            "verify_window_steps": int(state.get("verify_window_steps", 0)),
            "active": bool(state.get("active", False)),
            "stage": str(state.get("stage", "idle")),
            "steps_remaining": int(max(0, state.get("steps_remaining", 0))),
            "trigger_count": int(max(0, state.get("trigger_count", 0))),
            "target_reach_count": int(max(0, state.get("target_reach_count", 0))),
            "success_count": int(max(0, state.get("success_count", 0))),
            "timeout_count": int(max(0, state.get("timeout_count", 0))),
            "trigger_action_counter": int(state.get("trigger_action_counter", -1)),
            "deadline_action_counter": int(state.get("deadline_action_counter", -1)),
            "verify_deadline_action_counter": int(
                state.get("verify_deadline_action_counter", -1)
            ),
            "last_reached_action_counter": int(state.get("last_reached_action_counter", -1)),
            "verify_action_ids": [
                int(v)
                for v in state.get("verify_action_ids", [])
                if isinstance(v, int) or str(v).isdigit()
            ],
            "last_status": str(state.get("last_status", "idle")),
        }

    def _high_info_focus_state_snapshot_v1(self) -> dict[str, Any]:
        state = (
            self._high_info_focus_state_v1
            if isinstance(self._high_info_focus_state_v1, dict)
            else {}
        )
        verify_action_ids = state.get("verify_action_ids", [])
        if not isinstance(verify_action_ids, list):
            verify_action_ids = []
        queue = state.get("target_region_queue", [])
        if not isinstance(queue, list):
            queue = []
        pending_queue = state.get("pending_region_queue", [])
        if not isinstance(pending_queue, list):
            pending_queue = []
        pending_scores = state.get("pending_region_scores", {})
        if not isinstance(pending_scores, dict):
            pending_scores = {}
        completed = state.get("completed_target_regions", [])
        if not isinstance(completed, list):
            completed = []
        scores = state.get("target_region_scores", {})
        if not isinstance(scores, dict):
            scores = {}
        sample_counts = state.get("target_sample_counts", {})
        if not isinstance(sample_counts, dict):
            sample_counts = {}
        required_samples = state.get("target_required_samples", {})
        if not isinstance(required_samples, dict):
            required_samples = {}
        coupled_regions = state.get("coupled_region_keys", [])
        if not isinstance(coupled_regions, list):
            coupled_regions = []
        simultaneous_regions = state.get("simultaneous_changed_region_keys", [])
        if not isinstance(simultaneous_regions, list):
            simultaneous_regions = []
        simultaneous_reachable_regions = state.get("simultaneous_reachable_region_keys", [])
        if not isinstance(simultaneous_reachable_regions, list):
            simultaneous_reachable_regions = []
        simultaneous_unknown_regions = state.get("simultaneous_unknown_region_keys", [])
        if not isinstance(simultaneous_unknown_regions, list):
            simultaneous_unknown_regions = []
        simultaneous_unreachable_regions = state.get("simultaneous_unreachable_region_keys", [])
        if not isinstance(simultaneous_unreachable_regions, list):
            simultaneous_unreachable_regions = []
        interaction_chain = state.get("interaction_target_chain", [])
        if not isinstance(interaction_chain, list):
            interaction_chain = []
        priority_subqueue = state.get("priority_subqueue_keys", [])
        if not isinstance(priority_subqueue, list):
            priority_subqueue = []
        inner_loop_queue = state.get("inner_loop_queue", [])
        if not isinstance(inner_loop_queue, list):
            inner_loop_queue = []
        inner_loop_queue = [
            str(v)
            for v in inner_loop_queue
            if self._parse_region_key_v1(str(v)) is not None
        ][:16]
        novelty_related_region_keys = state.get("novelty_related_region_keys", [])
        if not isinstance(novelty_related_region_keys, list):
            novelty_related_region_keys = []
        novelty_signature_stats = state.get("novelty_signature_stats", {})
        if not isinstance(novelty_signature_stats, dict):
            novelty_signature_stats = {}
        novelty_signature_stats_sanitized: dict[str, dict[str, Any]] = {}
        for signature_raw, entry_raw in novelty_signature_stats.items():
            if not isinstance(entry_raw, dict):
                continue
            signature = str(signature_raw)
            source_region_key = str(entry_raw.get("source_region_key", "NA"))
            related_region_keys = entry_raw.get("related_region_keys", [])
            if not isinstance(related_region_keys, list):
                related_region_keys = []
            novelty_signature_stats_sanitized[signature] = {
                "count": int(max(0, entry_raw.get("count", 0))),
                "progress_hits": int(max(0, entry_raw.get("progress_hits", 0))),
                "avg_changed_pixels": float(max(0.0, entry_raw.get("avg_changed_pixels", 0.0))),
                "avg_simultaneous_pixels": float(
                    max(0.0, entry_raw.get("avg_simultaneous_pixels", 0.0))
                ),
                "source_region_key": str(source_region_key),
                "related_region_keys": [str(v) for v in related_region_keys[:8]],
                "last_action_counter": int(entry_raw.get("last_action_counter", -1)),
            }
        region_recent_change_pixels = state.get("region_recent_change_pixels", {})
        if not isinstance(region_recent_change_pixels, dict):
            region_recent_change_pixels = {}
        region_change_magnitude = state.get("region_change_magnitude", {})
        if not isinstance(region_change_magnitude, dict):
            region_change_magnitude = {}
        region_change_magnitude_ema = state.get("region_change_magnitude_ema", {})
        if not isinstance(region_change_magnitude_ema, dict):
            region_change_magnitude_ema = {}
        region_change_delta = state.get("region_change_delta", {})
        if not isinstance(region_change_delta, dict):
            region_change_delta = {}
        region_sudden_spike_keys = state.get("region_sudden_spike_keys", [])
        if not isinstance(region_sudden_spike_keys, list):
            region_sudden_spike_keys = []
        last_trigger_changed_region_diff_map = state.get(
            "last_trigger_changed_region_diff_map",
            {},
        )
        if not isinstance(last_trigger_changed_region_diff_map, dict):
            last_trigger_changed_region_diff_map = {}
        return {
            "schema_name": "active_inference_high_info_focus_state_v1",
            "schema_version": 1,
            "enabled": bool(state.get("enabled", False)),
            "active": bool(state.get("active", False)),
            "stage": str(state.get("stage", "idle")),
            "window_steps": int(max(0, state.get("window_steps", 0))),
            "steps_remaining": int(max(0, state.get("steps_remaining", 0))),
            "trigger_count": int(max(0, state.get("trigger_count", 0))),
            "completion_count": int(max(0, state.get("completion_count", 0))),
            "timeout_count": int(max(0, state.get("timeout_count", 0))),
            "source_region_key": str(state.get("source_region_key", "NA")),
            "trigger_event_type": str(state.get("trigger_event_type", "NA")),
            "trigger_action_counter": int(state.get("trigger_action_counter", -1)),
            "deadline_action_counter": int(state.get("deadline_action_counter", -1)),
            "current_target_region_key": str(state.get("current_target_region_key", "NA")),
            "target_miss_streak": int(max(0, state.get("target_miss_streak", 0))),
            "target_region_queue": [str(v) for v in queue[:8]],
            "pending_region_queue": [str(v) for v in pending_queue[:16]],
            "pending_region_scores": {
                str(k): float(v)
                for (k, v) in sorted(
                    pending_scores.items(),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )[:16]
            },
            "target_region_scores": {
                str(k): float(v)
                for (k, v) in sorted(scores.items(), key=lambda item: str(item[0]))[:16]
            },
            "target_required_samples": {
                str(k): int(max(1, v))
                for (k, v) in sorted(required_samples.items(), key=lambda item: str(item[0]))[
                    :16
                ]
            },
            "target_sample_counts": {
                str(k): int(max(0, v))
                for (k, v) in sorted(sample_counts.items(), key=lambda item: str(item[0]))[:32]
            },
            "completed_target_regions": [str(v) for v in completed[-8:]],
            "coupled_region_keys": [str(v) for v in coupled_regions[:8]],
            "primary_coupled_region_key": str(state.get("primary_coupled_region_key", "NA")),
            "secondary_coupled_region_key": str(
                state.get("secondary_coupled_region_key", "NA")
            ),
            "simultaneous_changed_region_keys": [str(v) for v in simultaneous_regions[:8]],
            "simultaneous_reachable_region_keys": [
                str(v) for v in simultaneous_reachable_regions[:8]
            ],
            "simultaneous_unknown_region_keys": [str(v) for v in simultaneous_unknown_regions[:8]],
            "simultaneous_unreachable_region_keys": [
                str(v) for v in simultaneous_unreachable_regions[:8]
            ],
            "simultaneous_anchor_region_key": str(
                state.get("simultaneous_anchor_region_key", "NA")
            ),
            "simultaneous_changed_total_pixels": int(
                max(0, state.get("simultaneous_changed_total_pixels", 0))
            ),
            "region_recent_change_pixels": {
                str(k): int(max(0, v))
                for (k, v) in sorted(
                    region_recent_change_pixels.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:16]
            },
            "region_change_magnitude": {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in sorted(
                    region_change_magnitude.items(),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )[:16]
            },
            "region_change_magnitude_ema": {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in sorted(
                    region_change_magnitude_ema.items(),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )[:16]
            },
            "region_change_delta": {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in sorted(
                    region_change_delta.items(),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )[:16]
            },
            "region_sudden_spike_keys": [str(v) for v in region_sudden_spike_keys[:16]],
            "last_trigger_changed_pixels": int(
                max(0, state.get("last_trigger_changed_pixels", 0))
            ),
            "last_trigger_changed_region_diff_map": {
                str(k): int(max(0, v))
                for (k, v) in sorted(
                    last_trigger_changed_region_diff_map.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:16]
            },
            "cross_region_key": str(state.get("cross_region_key", "NA")),
            "gate_region_key": str(state.get("gate_region_key", "NA")),
            "verify_action_ids": [int(v) for v in verify_action_ids],
            "chain_lock_active": bool(state.get("chain_lock_active", False)),
            "chain_lock_window_steps": int(
                max(1, state.get("chain_lock_window_steps", self.high_info_chain_lock_window_steps))
            ),
            "chain_lock_steps_remaining": int(
                max(0, state.get("chain_lock_steps_remaining", 0))
            ),
            "chain_lock_target_region_key": str(
                state.get("chain_lock_target_region_key", "NA")
            ),
            "chain_lock_miss_limit": int(max(1, state.get("chain_lock_miss_limit", 1))),
            "target_commit_active": bool(state.get("target_commit_active", False)),
            "target_commit_window_steps": int(
                max(
                    1,
                    state.get(
                        "target_commit_window_steps",
                        self.high_info_target_commit_window_steps,
                    ),
                )
            ),
            "target_commit_miss_limit": int(
                max(
                    1,
                    state.get(
                        "target_commit_miss_limit",
                        self.high_info_target_commit_miss_limit,
                    ),
                )
            ),
            "chain_lock_last_status": str(state.get("chain_lock_last_status", "idle")),
            "interaction_chain_active": bool(state.get("interaction_chain_active", False)),
            "interaction_chain_generation": int(
                max(0, state.get("interaction_chain_generation", 0))
            ),
            "interaction_target_chain": [str(v) for v in interaction_chain[:16]],
            "interaction_target_index": int(max(0, state.get("interaction_target_index", 0))),
            "interaction_last_status": str(state.get("interaction_last_status", "idle")),
            "priority_subqueue_active": bool(state.get("priority_subqueue_active", False)),
            "priority_subqueue_keys": [str(v) for v in priority_subqueue[:16]],
            "inner_loop_active": bool(
                state.get("inner_loop_active", False) and bool(inner_loop_queue)
            ),
            "inner_loop_reason": str(state.get("inner_loop_reason", "NA")),
            "inner_loop_queue": [str(v) for v in inner_loop_queue[:16]],
            "inner_loop_current_target_region_key": str(
                state.get("inner_loop_current_target_region_key", "NA")
            ),
            "novelty_protocol_active": bool(state.get("novelty_protocol_active", False)),
            "novelty_source_region_key": str(state.get("novelty_source_region_key", "NA")),
            "novelty_related_region_keys": [str(v) for v in novelty_related_region_keys[:16]],
            "last_novelty_signature": str(state.get("last_novelty_signature", "NA")),
            "novelty_trigger_count": int(max(0, state.get("novelty_trigger_count", 0))),
            "novelty_last_action_counter": int(state.get("novelty_last_action_counter", -1)),
            "novelty_signature_stats": {
                str(k): dict(v)
                for (k, v) in sorted(
                    novelty_signature_stats_sanitized.items(),
                    key=lambda item: (
                        -int(item[1].get("count", 0)),
                        -int(item[1].get("last_action_counter", -1)),
                        str(item[0]),
                    ),
                )[:16]
            },
            "last_status": str(state.get("last_status", "idle")),
        }

    @staticmethod
    def _coupling_signal_profile_v1(
        *,
        cc_rate: float,
        strong_change_rate: float,
        non_no_change_rate: float,
        entropy_norm: float,
        palette_change_rate: float,
        palette_delta_mean_norm: float,
        progress_rate: float,
    ) -> dict[str, Any]:
        components = {
            "structural_cc": float(max(0.0, min(1.0, cc_rate))),
            "visual_palette": float(
                max(
                    0.0,
                    min(1.0, max(float(palette_change_rate), float(palette_delta_mean_norm))),
                )
            ),
            "dynamic_change": float(max(0.0, min(1.0, strong_change_rate))),
            "behavioral_response": float(max(0.0, min(1.0, non_no_change_rate))),
            "surprise_entropy": float(max(0.0, min(1.0, entropy_norm))),
            "progress_delta": float(max(0.0, min(1.0, progress_rate))),
        }
        weights = {
            "structural_cc": 0.24,
            "visual_palette": 0.22,
            "dynamic_change": 0.18,
            "behavioral_response": 0.14,
            "surprise_entropy": 0.12,
            "progress_delta": 0.10,
        }
        if components["progress_delta"] > 0.0:
            weights["progress_delta"] += 0.10
            weights["behavioral_response"] = max(
                0.04,
                float(weights["behavioral_response"] - 0.06),
            )
            weights["surprise_entropy"] = max(
                0.04,
                float(weights["surprise_entropy"] - 0.04),
            )
        weight_sum = float(sum(weights.values()))
        if weight_sum > 0.0:
            weights = {
                str(k): float(v / weight_sum)
                for (k, v) in weights.items()
            }
        score = float(
            sum(
                float(weights.get(key, 0.0)) * float(components.get(key, 0.0))
                for key in components
            )
        )
        dominant_key, dominant_value = max(
            components.items(),
            key=lambda item: (float(item[1]), str(item[0])),
            default=("unknown", 0.0),
        )
        return {
            "score": float(max(0.0, min(1.0, score))),
            "kind": str(dominant_key),
            "kind_value": float(max(0.0, min(1.0, dominant_value))),
            "components": {
                str(k): float(v)
                for (k, v) in components.items()
            },
            "weights": {
                str(k): float(v)
                for (k, v) in weights.items()
            },
        }

    def _high_info_region_scoreboard_v1(
        self,
        *,
        max_regions: int = 12,
    ) -> dict[str, Any]:
        region_rows: dict[str, dict[str, Any]] = {}
        for region_action_key, histogram_raw in self._region_action_event_counts.items():
            if not isinstance(histogram_raw, dict):
                continue
            try:
                region_key, action_token = str(region_action_key).split("|a", 1)
                action_id = int(action_token)
            except Exception:
                continue
            if self._parse_region_key_v1(str(region_key)) is None:
                continue
            histogram = {
                str(k): int(max(0, v))
                for (k, v) in histogram_raw.items()
                if int(max(0, v)) > 0
            }
            attempts = int(sum(histogram.values()))
            if attempts <= 0:
                continue
            non_no_change = int(self._region_action_non_no_change_counts.get(region_action_key, 0))
            strong_change = int(self._region_action_strong_change_counts.get(region_action_key, 0))
            progress_count = int(self._region_action_progress_counts.get(region_action_key, 0))
            ui_side_effect_count = int(
                self._region_action_ui_side_effect_counts.get(region_action_key, 0)
            )
            terminal_failure_count = int(
                self._region_action_terminal_failure_counts.get(region_action_key, 0)
            )
            cc_count_change = int(histogram.get("CC_COUNT_CHANGE", 0))
            palette_change_count = int(
                self._region_action_palette_change_counts.get(region_action_key, 0)
            )
            palette_delta_total_sum = int(
                self._region_action_palette_delta_total_sum.get(region_action_key, 0)
            )
            entropy = 0.0
            for count in histogram.values():
                p = float(count / float(max(1, attempts)))
                if p <= 0.0:
                    continue
                entropy -= p * math.log2(p)
            entropy_norm = float(entropy / max(1.0, math.log2(float(max(2, len(histogram))))))
            non_no_change_rate = float(non_no_change / float(max(1, attempts)))
            strong_change_rate = float(strong_change / float(max(1, attempts)))
            progress_rate = float(progress_count / float(max(1, attempts)))
            ui_side_effect_rate = float(ui_side_effect_count / float(max(1, attempts)))
            terminal_failure_rate = float(terminal_failure_count / float(max(1, attempts)))
            cc_rate = float(cc_count_change / float(max(1, attempts)))
            palette_change_rate = float(palette_change_count / float(max(1, attempts)))
            palette_delta_mean = float(palette_delta_total_sum / float(max(1, attempts)))
            palette_delta_mean_norm = float(
                palette_delta_mean / (palette_delta_mean + 16.0)
            )
            coupling_profile = self._coupling_signal_profile_v1(
                cc_rate=cc_rate,
                strong_change_rate=strong_change_rate,
                non_no_change_rate=non_no_change_rate,
                entropy_norm=entropy_norm,
                palette_change_rate=palette_change_rate,
                palette_delta_mean_norm=palette_delta_mean_norm,
                progress_rate=progress_rate,
            )
            visit_count = int(self._region_visit_counts.get(str(region_key), 0))
            visit_novelty = float(max(0.0, 1.0 - min(1.0, float(visit_count) / 10.0)))
            ui_suppression = float(
                max(
                    0.0,
                    min(
                        1.0,
                        (0.85 * ui_side_effect_rate) + (0.65 * terminal_failure_rate),
                    ),
                )
            )
            monotony_penalty = 0.0
            if attempts >= 4 and progress_rate <= 0.0:
                deterministic = float(max(0.0, min(1.0, 1.0 - float(entropy_norm))))
                stagnant_visits = int(max(0, visit_count - 8))
                monotony_penalty = float(
                    min(
                        0.42,
                        (0.18 * deterministic) + (0.012 * float(stagnant_visits)),
                    )
                )
                if (
                    non_no_change_rate >= 0.95
                    and entropy_norm <= 0.05
                    and attempts >= 6
                ):
                    monotony_penalty = float(min(0.55, monotony_penalty + 0.12))
            info_score = float(
                max(
                    0.0,
                    min(
                        1.0,
                        (
                            (0.92 * float(coupling_profile.get("score", 0.0)))
                            + (0.06 * visit_novelty)
                        )
                        * (1.0 - ui_suppression)
                        - float(monotony_penalty),
                    ),
                )
            )
            current = region_rows.get(str(region_key))
            row = {
                "region_key": str(region_key),
                "best_action_id": int(action_id),
                "info_score": float(info_score),
                "attempts": int(attempts),
                "event_entropy_norm": float(entropy_norm),
                "non_no_change_rate": float(non_no_change_rate),
                "strong_change_rate": float(strong_change_rate),
                "progress_rate": float(progress_rate),
                "ui_side_effect_rate": float(ui_side_effect_rate),
                "terminal_failure_rate": float(terminal_failure_rate),
                "ui_suppression": float(ui_suppression),
                "cc_count_change_rate": float(cc_rate),
                "palette_change_rate": float(palette_change_rate),
                "palette_delta_mean": float(palette_delta_mean),
                "palette_delta_mean_norm": float(palette_delta_mean_norm),
                "coupling_signal_score": float(coupling_profile.get("score", 0.0)),
                "coupling_signal_kind": str(coupling_profile.get("kind", "unknown")),
                "coupling_components": dict(coupling_profile.get("components", {})),
                "coupling_weights": dict(coupling_profile.get("weights", {})),
                "monotony_penalty": float(monotony_penalty),
                "visit_count": int(visit_count),
                "event_histogram": {str(k): int(v) for (k, v) in sorted(histogram.items())},
            }
            if current is None or float(row["info_score"]) > float(current.get("info_score", 0.0)):
                region_rows[str(region_key)] = row

        rows = sorted(
            region_rows.values(),
            key=lambda row: (
                -float(row.get("info_score", 0.0)),
                int(row.get("visit_count", 10**9)),
                -int(row.get("attempts", 0)),
                str(row.get("region_key", "")),
            ),
        )
        if len(rows) > int(max_regions):
            rows = rows[: int(max_regions)]
        return {
            "schema_name": "active_inference_high_info_region_scoreboard_v1",
            "schema_version": 1,
            "region_count": int(len(region_rows)),
            "rows": rows,
        }

    def _region_action_semantics_v1(
        self,
        *,
        action_id: int,
        current_region_key: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_name": "active_inference_region_action_semantics_v1",
            "schema_version": 1,
            "enabled": False,
            "action_id": int(action_id),
            "current_region_key": str(current_region_key),
            "attempts": 0,
            "moved_count": 0,
            "blocked_count": 0,
            "moved_rate": 0.0,
            "blocked_rate": 0.0,
            "non_no_change_rate": 0.0,
            "strong_change_rate": 0.0,
            "progress_rate": 0.0,
            "ui_side_effect_rate": 0.0,
            "terminal_failure_rate": 0.0,
            "ui_suppression": 0.0,
            "cc_count_change_rate": 0.0,
            "palette_change_rate": 0.0,
            "palette_delta_mean": 0.0,
            "palette_delta_mean_norm": 0.0,
            "event_entropy_norm": 0.0,
            "info_trigger_score": 0.0,
            "coupling_signal_score": 0.0,
            "coupling_signal_score_raw": 0.0,
            "coupling_signal_kind": "unknown",
            "coupling_components": {},
            "coupling_weights": {},
            "edge_status": "unknown",
            "event_histogram": {},
        }
        if int(action_id) not in (1, 2, 3, 4):
            payload["reason"] = "non_navigation_action"
            return payload
        if self._parse_region_key_v1(str(current_region_key)) is None:
            payload["reason"] = "unknown_region"
            return payload
        region_action_key = f"{str(current_region_key)}|a{int(action_id)}"
        histogram_raw = self._region_action_event_counts.get(region_action_key, {})
        if not isinstance(histogram_raw, dict):
            histogram_raw = {}
        histogram = {
            str(k): int(max(0, v))
            for (k, v) in histogram_raw.items()
            if int(max(0, v)) > 0
        }
        attempts = int(sum(histogram.values()))
        edge_key = f"region={str(current_region_key)}|action={int(action_id)}"
        blocked_count = int(self._blocked_edge_counts.get(edge_key, 0))
        moved_count = int(
            sum(
                int(max(0, count))
                for count in self._region_action_transition_counts.get(region_action_key, {}).values()
            )
        )
        if attempts <= 0:
            payload["attempts"] = int(max(blocked_count, moved_count))
            payload["reason"] = "insufficient_evidence"
            return payload
        non_no_change_count = int(self._region_action_non_no_change_counts.get(region_action_key, 0))
        strong_change_count = int(self._region_action_strong_change_counts.get(region_action_key, 0))
        progress_count = int(self._region_action_progress_counts.get(region_action_key, 0))
        ui_side_effect_count = int(
            self._region_action_ui_side_effect_counts.get(region_action_key, 0)
        )
        terminal_failure_count = int(
            self._region_action_terminal_failure_counts.get(region_action_key, 0)
        )
        cc_count_change_count = int(histogram.get("CC_COUNT_CHANGE", 0))
        palette_change_count = int(
            self._region_action_palette_change_counts.get(region_action_key, 0)
        )
        palette_delta_total_sum = int(
            self._region_action_palette_delta_total_sum.get(region_action_key, 0)
        )
        entropy = 0.0
        for count in histogram.values():
            p = float(count / float(max(1, attempts)))
            if p <= 0.0:
                continue
            entropy -= p * math.log2(p)
        entropy_norm = float(entropy / max(1.0, math.log2(float(max(2, len(histogram))))))
        moved_rate = float(moved_count / float(max(1, attempts)))
        blocked_rate = float(blocked_count / float(max(1, attempts)))
        non_no_change_rate = float(non_no_change_count / float(max(1, attempts)))
        strong_change_rate = float(strong_change_count / float(max(1, attempts)))
        progress_rate = float(progress_count / float(max(1, attempts)))
        ui_side_effect_rate = float(ui_side_effect_count / float(max(1, attempts)))
        terminal_failure_rate = float(terminal_failure_count / float(max(1, attempts)))
        ui_suppression = float(
            max(
                0.0,
                min(
                    1.0,
                    (0.85 * ui_side_effect_rate) + (0.65 * terminal_failure_rate),
                ),
            )
        )
        cc_rate = float(cc_count_change_count / float(max(1, attempts)))
        palette_change_rate = float(palette_change_count / float(max(1, attempts)))
        palette_delta_mean = float(palette_delta_total_sum / float(max(1, attempts)))
        palette_delta_mean_norm = float(
            palette_delta_mean / (palette_delta_mean + 16.0)
        )
        coupling_profile = self._coupling_signal_profile_v1(
            cc_rate=cc_rate,
            strong_change_rate=strong_change_rate,
            non_no_change_rate=non_no_change_rate,
            entropy_norm=entropy_norm,
            palette_change_rate=palette_change_rate,
            palette_delta_mean_norm=palette_delta_mean_norm,
            progress_rate=progress_rate,
        )
        coupling_score_raw = float(coupling_profile.get("score", 0.0))
        coupling_score_effective = float(coupling_score_raw * (1.0 - ui_suppression))
        info_trigger_score = float(
            max(
                0.0,
                min(
                    1.0,
                    (0.96 * float(coupling_score_effective)),
                ),
            )
        )
        if attempts < 2:
            edge_status = "unknown"
        elif blocked_rate >= 0.75 and moved_rate <= 0.25:
            edge_status = "blocked"
        elif moved_rate >= 0.55 and blocked_rate <= 0.45:
            edge_status = "passable"
        else:
            edge_status = "mixed"
        payload.update(
            {
                "enabled": True,
                "attempts": int(attempts),
                "moved_count": int(moved_count),
                "blocked_count": int(blocked_count),
                "moved_rate": float(moved_rate),
                "blocked_rate": float(blocked_rate),
                "non_no_change_rate": float(non_no_change_rate),
                "strong_change_rate": float(strong_change_rate),
                "progress_rate": float(progress_rate),
                "ui_side_effect_rate": float(ui_side_effect_rate),
                "terminal_failure_rate": float(terminal_failure_rate),
                "ui_suppression": float(ui_suppression),
                "cc_count_change_rate": float(cc_rate),
                "palette_change_rate": float(palette_change_rate),
                "palette_delta_mean": float(palette_delta_mean),
                "palette_delta_mean_norm": float(palette_delta_mean_norm),
                "event_entropy_norm": float(entropy_norm),
                "info_trigger_score": float(info_trigger_score),
                "coupling_signal_score": float(coupling_score_effective),
                "coupling_signal_score_raw": float(coupling_score_raw),
                "coupling_signal_kind": str(coupling_profile.get("kind", "unknown")),
                "coupling_components": dict(coupling_profile.get("components", {})),
                "coupling_weights": dict(coupling_profile.get("weights", {})),
                "edge_status": str(edge_status),
                "event_histogram": {str(k): int(v) for (k, v) in sorted(histogram.items())},
            }
        )
        return payload

    def _high_info_focus_candidate_features_v1(
        self,
        *,
        candidate: ActionCandidateV1,
        predicted_region_features: dict[str, Any] | None,
    ) -> dict[str, Any]:
        state = self._high_info_focus_state_snapshot_v1()
        raw_current_region_key = self._current_region_key_v1()
        current_region_key = str(raw_current_region_key)
        predicted_current_region_key = "NA"
        target_region_key = str(state.get("current_target_region_key", "NA"))
        inner_loop_active = bool(state.get("inner_loop_active", False))
        inner_loop_target_region_key = str(
            state.get("inner_loop_current_target_region_key", "NA")
        )
        if (
            inner_loop_active
            and self._parse_region_key_v1(str(inner_loop_target_region_key)) is not None
        ):
            # Inner-loop target is authoritative while loop is active.
            target_region_key = str(inner_loop_target_region_key)
        simultaneous_region_keys = state.get("simultaneous_changed_region_keys", [])
        if not isinstance(simultaneous_region_keys, list):
            simultaneous_region_keys = []
        simultaneous_reachable_keys = state.get("simultaneous_reachable_region_keys", [])
        if not isinstance(simultaneous_reachable_keys, list):
            simultaneous_reachable_keys = []
        simultaneous_unreachable_keys = state.get("simultaneous_unreachable_region_keys", [])
        if not isinstance(simultaneous_unreachable_keys, list):
            simultaneous_unreachable_keys = []
        simultaneous_region_set = {
            str(v)
            for v in simultaneous_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        }
        simultaneous_reachable_set = {
            str(v)
            for v in simultaneous_reachable_keys
            if self._parse_region_key_v1(str(v)) is not None
        }
        simultaneous_unreachable_set = {
            str(v)
            for v in simultaneous_unreachable_keys
            if self._parse_region_key_v1(str(v)) is not None
        }
        predicted_region_key = str(current_region_key)
        predicted_region_visit_count = 0
        predicted_edge_attempts = 0
        predicted_edge_blocked_rate = 0.0
        if isinstance(predicted_region_features, dict):
            current_key = str(predicted_region_features.get("current_region_key", "NA"))
            if self._parse_region_key_v1(current_key) is not None:
                predicted_current_region_key = str(current_key)
            key = str(predicted_region_features.get("predicted_region_key", "NA"))
            if self._parse_region_key_v1(key) is not None:
                predicted_region_key = str(key)
            predicted_region_visit_count = int(
                max(0, predicted_region_features.get("predicted_region_visit_count", 0))
            )
            predicted_edge_attempts = int(
                max(0, predicted_region_features.get("edge_attempts", 0))
            )
            predicted_edge_blocked_rate = float(
                max(
                    0.0,
                    min(1.0, predicted_region_features.get("edge_blocked_rate", 0.0)),
                )
            )
        raw_current_region_tuple = self._parse_region_key_v1(raw_current_region_key)
        predicted_current_region_tuple = self._parse_region_key_v1(predicted_current_region_key)
        if predicted_current_region_tuple is not None:
            if raw_current_region_tuple is None:
                current_region_key = str(predicted_current_region_key)
            else:
                dx = int(abs(int(predicted_current_region_tuple[0]) - int(raw_current_region_tuple[0])))
                dy = int(abs(int(predicted_current_region_tuple[1]) - int(raw_current_region_tuple[1])))
                if max(dx, dy) <= 1:
                    current_region_key = str(predicted_current_region_key)
        current_region_tuple = self._parse_region_key_v1(current_region_key)
        current_region_mismatch = bool(
            raw_current_region_tuple is not None
            and predicted_current_region_tuple is not None
            and str(raw_current_region_key) != str(predicted_current_region_key)
        )
        region_adjacency = self._region_graph_adjacency_v1(min_edge_count=1)
        route_distance_before = int(
            self._region_route_distance_v1(
                region_adjacency,
                start_region_key=str(current_region_key),
                goal_region_key=str(target_region_key),
            )
        )
        route_distance_after = int(
            self._region_route_distance_v1(
                region_adjacency,
                start_region_key=str(predicted_region_key),
                goal_region_key=str(target_region_key),
            )
        )
        distance_before = int(route_distance_before)
        if distance_before >= 10**6:
            distance_before = int(
                self._region_distance_v1(
                    str(current_region_key),
                    str(target_region_key),
                )
            )
        distance_after = int(route_distance_after)
        if distance_after >= 10**6:
            distance_after = int(
                self._region_distance_v1(
                    str(predicted_region_key),
                    str(target_region_key),
                )
            )
        distance_delta = int(distance_after - distance_before)
        verify_action_ids = state.get("verify_action_ids", [])
        if not isinstance(verify_action_ids, list):
            verify_action_ids = []
        verify_action_set = {int(v) for v in verify_action_ids}
        action_id = int(candidate.action_id)
        reaches_target = bool(
            self._parse_region_key_v1(predicted_region_key) is not None
            and str(predicted_region_key) == str(target_region_key)
        )
        stays_in_current_region = bool(
            current_region_tuple is not None
            and self._parse_region_key_v1(predicted_region_key) is not None
            and str(predicted_region_key) == str(current_region_key)
        )
        hard_block_loop_risk = bool(
            stays_in_current_region
            and int(predicted_edge_attempts) >= 12
            and float(predicted_edge_blocked_rate) >= 0.65
        )
        moves_toward_target = bool(distance_after < distance_before)
        moves_away_target = bool(distance_after > distance_before)
        in_target_now = bool(
            self._parse_region_key_v1(current_region_key) is not None
            and str(current_region_key) == str(target_region_key)
        )
        verify_action_candidate = bool(
            str(state.get("stage", "idle")) == "verify"
            and in_target_now
            and action_id in verify_action_set
        )
        target_score = float(
            state.get("target_region_scores", {}).get(target_region_key, 0.0)
            if isinstance(state.get("target_region_scores", {}), dict)
            else 0.0
        )
        sample_counts = state.get("target_sample_counts", {})
        if not isinstance(sample_counts, dict):
            sample_counts = {}
        required_samples_map = state.get("target_required_samples", {})
        if not isinstance(required_samples_map, dict):
            required_samples_map = {}
        region_recent_change_pixels = state.get("region_recent_change_pixels", {})
        if not isinstance(region_recent_change_pixels, dict):
            region_recent_change_pixels = {}
        region_change_magnitude_map = state.get("region_change_magnitude", {})
        if not isinstance(region_change_magnitude_map, dict):
            region_change_magnitude_map = {}
        region_change_magnitude_ema_map = state.get("region_change_magnitude_ema", {})
        if not isinstance(region_change_magnitude_ema_map, dict):
            region_change_magnitude_ema_map = {}
        region_change_delta_map = state.get("region_change_delta", {})
        if not isinstance(region_change_delta_map, dict):
            region_change_delta_map = {}
        region_sudden_spike_keys = state.get("region_sudden_spike_keys", [])
        if not isinstance(region_sudden_spike_keys, list):
            region_sudden_spike_keys = []
        region_sudden_spike_set = {
            str(v)
            for v in region_sudden_spike_keys
            if self._parse_region_key_v1(str(v)) is not None
        }
        coupled_regions = state.get("coupled_region_keys", [])
        if not isinstance(coupled_regions, list):
            coupled_regions = []
        coupled_region_set = {str(v) for v in coupled_regions}
        alternate_coupled_distance = 10**6
        if self._parse_region_key_v1(predicted_region_key) is not None:
            for region_key in coupled_region_set:
                if str(region_key) == str(target_region_key):
                    continue
                candidate_distance = int(
                    self._region_distance_v1(
                        str(predicted_region_key),
                        str(region_key),
                    )
                )
                if candidate_distance < alternate_coupled_distance:
                    alternate_coupled_distance = int(candidate_distance)
        target_sample_count = int(max(0, sample_counts.get(target_region_key, 0)))
        required_samples = int(
            max(
                1,
                required_samples_map.get(
                    target_region_key,
                    int(self.high_info_min_samples_per_target),
                ),
            )
        )
        remaining_samples = int(
            max(0, int(required_samples) - int(target_sample_count))
        )
        target_recent_change_pixels = int(
            max(0, region_recent_change_pixels.get(str(target_region_key), 0))
        )
        target_change_magnitude = float(
            max(
                0.0,
                min(
                    1.0,
                    region_change_magnitude_map.get(
                        str(target_region_key),
                        region_change_magnitude_ema_map.get(str(target_region_key), 0.0),
                    ),
                ),
            )
        )
        target_change_magnitude_ema = float(
            max(0.0, min(1.0, region_change_magnitude_ema_map.get(str(target_region_key), 0.0)))
        )
        target_change_delta = float(
            max(0.0, min(1.0, region_change_delta_map.get(str(target_region_key), 0.0)))
        )
        target_sudden_spike = bool(str(target_region_key) in region_sudden_spike_set)
        persistent_self_loop_risk = bool(
            stays_in_current_region
            and int(predicted_edge_attempts) >= 36
            and int(predicted_region_visit_count) >= 18
            and int(remaining_samples) > 0
        )
        high_block_loop_risk = bool(
            hard_block_loop_risk or persistent_self_loop_risk
        )
        target_is_simultaneous = bool(str(target_region_key) in simultaneous_region_set)
        target_is_reachable_simultaneous = bool(
            str(target_region_key) in simultaneous_reachable_set
        )
        target_is_unreachable_simultaneous = bool(
            str(target_region_key) in simultaneous_unreachable_set
        )
        bonus_hint = 0.0
        penalty_hint = 0.0
        interaction_chain_active = bool(state.get("interaction_chain_active", False))
        chain_lock_target_region_key = str(state.get("chain_lock_target_region_key", "NA"))
        chain_lock_window_steps = int(
            max(1, state.get("chain_lock_window_steps", self.high_info_chain_lock_window_steps))
        )
        chain_lock_steps_remaining = int(max(0, state.get("chain_lock_steps_remaining", 0)))
        target_commit_active = bool(state.get("target_commit_active", False))
        novelty_protocol_active = bool(state.get("novelty_protocol_active", False))
        novelty_source_region_key = str(state.get("novelty_source_region_key", "NA"))
        novelty_related_region_keys = state.get("novelty_related_region_keys", [])
        if not isinstance(novelty_related_region_keys, list):
            novelty_related_region_keys = []
        novelty_related_region_set = {
            str(v)
            for v in novelty_related_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        }
        target_is_novelty_source = bool(
            self._parse_region_key_v1(str(target_region_key)) is not None
            and str(target_region_key) == str(novelty_source_region_key)
        )
        target_is_novelty_related = bool(
            self._parse_region_key_v1(str(target_region_key)) is not None
            and str(target_region_key) in novelty_related_region_set
        )
        chain_lock_active = bool(
            state.get("chain_lock_active", False)
            and bool(state.get("active", False))
            and self._parse_region_key_v1(str(chain_lock_target_region_key)) is not None
            and chain_lock_steps_remaining > 0
        )
        chain_lock_target_match = bool(
            chain_lock_active and str(target_region_key) == str(chain_lock_target_region_key)
        )
        if bool(state.get("active", False)):
            if verify_action_candidate:
                bonus_hint += 1.0
            elif action_id in (1, 2, 3, 4):
                if reaches_target:
                    bonus_hint += 0.85
                elif moves_toward_target:
                    bonus_hint += 0.45
                if moves_away_target:
                    penalty_hint += 0.55
            elif str(state.get("stage", "idle")) == "verify":
                penalty_hint += 0.20
            sample_bonus = 0.0
            coupled_sample_bonus = 0.0
            if remaining_samples > 0:
                sample_bonus = float(min(0.45, 0.15 * float(remaining_samples)))
                if str(target_region_key) in coupled_region_set:
                    coupled_sample_bonus = float(min(0.48, 0.18 * float(remaining_samples)))
            if action_id in (1, 2, 3, 4):
                bonus_hint += float(0.12 * target_change_magnitude)
                if reaches_target or moves_toward_target:
                    bonus_hint += float(sample_bonus + coupled_sample_bonus)
                    if target_sudden_spike:
                        bonus_hint += float(0.24 + (0.16 * target_change_delta))
                    if interaction_chain_active:
                        bonus_hint += 0.12
                    if target_is_reachable_simultaneous:
                        bonus_hint += 0.28
                elif moves_away_target:
                    penalty_hint += float(
                        min(0.42, 0.14 * float(max(0, remaining_samples)))
                    )
                    if str(target_region_key) in coupled_region_set:
                        penalty_hint += float(
                            min(0.16, 0.06 * float(max(0, remaining_samples)))
                        )
                    if target_sudden_spike:
                        penalty_hint += float(0.58 + (0.18 * target_change_delta))
                    if interaction_chain_active:
                        penalty_hint += 0.36
                    if target_is_reachable_simultaneous:
                        penalty_hint += 0.54
                else:
                    bonus_hint += float(0.10 * sample_bonus)
            elif verify_action_candidate:
                bonus_hint += float(0.18 * sample_bonus)
            if target_is_unreachable_simultaneous and action_id in (1, 2, 3, 4):
                penalty_hint += 1.40
        urgency = float(
            min(
                1.0,
                max(
                    0.0,
                    1.0
                    - (
                        float(state.get("steps_remaining", 0))
                        / float(max(1, state.get("window_steps", 1)))
                    ),
                ),
            )
        )
        if interaction_chain_active and bool(state.get("active", False)):
            if action_id in (1, 2, 3, 4):
                if reaches_target or moves_toward_target:
                    bonus_hint += 0.18
                if moves_away_target:
                    penalty_hint += 0.38
                if chain_lock_active and chain_lock_target_match:
                    if reaches_target:
                        bonus_hint += 0.62
                    elif moves_toward_target:
                        bonus_hint += 0.40
                    elif moves_away_target:
                        penalty_hint += 0.84
                if high_block_loop_risk:
                    penalty_hint += 1.20
                    bonus_hint = float(0.45 * bonus_hint)
                elif (
                    stays_in_current_region
                    and int(predicted_edge_attempts) >= 24
                ):
                    penalty_hint += float(
                        min(
                            0.85,
                            0.20 + (0.02 * float(int(predicted_edge_attempts) - 23)),
                        )
                    )
        if current_region_mismatch and action_id in (1, 2, 3, 4):
            penalty_hint += 0.28
        bonus_hint = float((0.75 + (0.25 * urgency)) * bonus_hint)
        return {
            "schema_name": "active_inference_high_info_focus_features_v1",
            "schema_version": 1,
            "enabled": bool(state.get("enabled", False)),
            "active": bool(state.get("active", False)),
            "stage": str(state.get("stage", "idle")),
            "raw_current_region_key": str(raw_current_region_key),
            "predicted_current_region_key": str(predicted_current_region_key),
            "current_region_key": str(current_region_key),
            "current_region_mismatch": bool(current_region_mismatch),
            "target_region_key": str(target_region_key),
            "predicted_region_key": str(predicted_region_key),
            "predicted_region_visit_count": int(predicted_region_visit_count),
            "predicted_edge_attempts": int(predicted_edge_attempts),
            "predicted_edge_blocked_rate": float(predicted_edge_blocked_rate),
            "queue_length": int(len(state.get("target_region_queue", []))),
            "steps_remaining": int(max(0, state.get("steps_remaining", 0))),
            "target_score": float(target_score),
            "target_sample_count": int(target_sample_count),
            "required_samples": int(required_samples),
            "remaining_samples": int(remaining_samples),
            "target_recent_change_pixels": int(target_recent_change_pixels),
            "target_change_magnitude": float(target_change_magnitude),
            "target_change_magnitude_ema": float(target_change_magnitude_ema),
            "target_change_delta": float(target_change_delta),
            "target_sudden_spike": bool(target_sudden_spike),
            "distance_before": int(distance_before),
            "distance_after": int(distance_after),
            "distance_delta": int(distance_delta),
            "target_route_distance_before": int(route_distance_before),
            "target_route_distance_after": int(route_distance_after),
            "alternate_coupled_distance": int(alternate_coupled_distance),
            "moves_toward_target_region": bool(moves_toward_target),
            "moves_away_target_region": bool(moves_away_target),
            "reaches_target_region": bool(reaches_target),
            "stays_in_current_region": bool(stays_in_current_region),
            "target_is_simultaneous_region": bool(target_is_simultaneous),
            "target_is_reachable_simultaneous": bool(target_is_reachable_simultaneous),
            "target_is_unreachable_simultaneous": bool(target_is_unreachable_simultaneous),
            "high_block_loop_risk": bool(high_block_loop_risk),
            "verify_action_ids": [int(v) for v in sorted(verify_action_set)],
            "verify_action_candidate": bool(verify_action_candidate),
            "interaction_chain_active": bool(interaction_chain_active),
            "chain_lock_active": bool(chain_lock_active),
            "chain_lock_window_steps": int(chain_lock_window_steps),
            "chain_lock_steps_remaining": int(chain_lock_steps_remaining),
            "chain_lock_target_region_key": str(chain_lock_target_region_key),
            "chain_lock_target_match": bool(chain_lock_target_match),
            "target_commit_active": bool(target_commit_active),
            "novelty_protocol_active": bool(novelty_protocol_active),
            "novelty_source_region_key": str(novelty_source_region_key),
            "target_is_novelty_source": bool(target_is_novelty_source),
            "target_is_novelty_related": bool(target_is_novelty_related),
            "novelty_related_region_count": int(len(novelty_related_region_set)),
            "bonus_hint": float(max(0.0, bonus_hint)),
            "penalty_hint": float(max(0.0, penalty_hint)),
        }

    def _sequence_causal_candidate_features_v1(
        self,
        *,
        candidate: ActionCandidateV1,
        predicted_region_features: dict[str, Any] | None,
    ) -> dict[str, Any]:
        state = self._sequence_causal_state_snapshot_v1()
        trigger_region_key = str(
            state.get(
                "trigger_region_key_effective",
                state.get("trigger_region_key", "NA"),
            )
        )
        target_region_key = str(
            state.get(
                "target_region_key_effective",
                state.get("target_region_key", "NA"),
            )
        )
        current_region_key = self._current_region_key_v1()
        steps_remaining = int(max(0, state.get("steps_remaining", 0)))
        action_id = int(candidate.action_id)
        predicted_key = current_region_key
        if isinstance(predicted_region_features, dict):
            candidate_key = str(predicted_region_features.get("predicted_region_key", "NA"))
            if self._parse_region_key_v1(candidate_key) is not None:
                predicted_key = str(candidate_key)
        current_distance = int(
            self._region_distance_v1(current_region_key, target_region_key)
        )
        predicted_distance = int(
            self._region_distance_v1(predicted_key, target_region_key)
        )
        distance_delta = int(predicted_distance - current_distance)
        reaches_target = bool(
            self._parse_region_key_v1(predicted_key) is not None
            and str(predicted_key) == str(target_region_key)
        )
        advances_to_target = bool(
            predicted_distance < current_distance
        )
        moves_away_from_target = bool(
            predicted_distance > current_distance
        )
        verify_action_ids_raw = state.get("verify_action_ids", [])
        if not isinstance(verify_action_ids_raw, list):
            verify_action_ids_raw = []
        verify_action_ids = {
            int(v) for v in verify_action_ids_raw if isinstance(v, int) or str(v).isdigit()
        }
        verify_actions_available = bool(verify_action_ids)
        verify_action_candidate = bool(
            str(state.get("stage", "idle")) == "verify"
            and (
                action_id in verify_action_ids
                if verify_actions_available
                else action_id in (1, 2, 3, 4)
            )
        )
        bonus_hint = 0.0
        penalty_hint = 0.0
        stage = str(state.get("stage", "idle"))
        if bool(state.get("active", False)):
            if stage == "seek_target":
                if reaches_target:
                    bonus_hint += 1.0
                elif advances_to_target:
                    bonus_hint += 0.45
                if moves_away_from_target and action_id in (1, 2, 3, 4):
                    penalty_hint += 0.35
            elif stage == "verify":
                if verify_action_candidate:
                    bonus_hint += 0.9
                elif action_id in (1, 2, 3, 4) and moves_away_from_target:
                    penalty_hint += 0.60
            urgency = float(
                min(
                    1.0,
                    max(
                        0.0,
                        1.0
                        - (
                            float(steps_remaining)
                            / float(max(1, int(state.get("window_steps", 1))))
                        ),
                    ),
                )
            )
            bonus_hint = float((0.70 + (0.30 * urgency)) * bonus_hint)
        return {
            "schema_name": "active_inference_sequence_causal_features_v1",
            "schema_version": 1,
            "enabled": bool(state.get("enabled", False)),
            "active": bool(state.get("active", False)),
            "stage": str(stage),
            "trigger_region_key": str(trigger_region_key),
            "target_region_key": str(target_region_key),
            "current_region_key": str(current_region_key),
            "predicted_region_key": str(predicted_key),
            "steps_remaining": int(steps_remaining),
            "current_distance_to_target": int(current_distance),
            "predicted_distance_to_target": int(predicted_distance),
            "distance_delta_to_target": int(distance_delta),
            "advances_to_target": bool(advances_to_target),
            "moves_away_from_target": bool(moves_away_from_target),
            "reaches_target": bool(reaches_target),
            "verify_action_candidate": bool(verify_action_candidate),
            "verify_action_ids": [int(v) for v in sorted(verify_action_ids)],
            "bonus_hint": float(max(0.0, bonus_hint)),
            "penalty_hint": float(max(0.0, penalty_hint)),
        }

    def _update_sequence_causal_state_v1(
        self,
        *,
        current_packet: ObservationPacketV1,
        transition_record: TransitionRecordV1 | None,
        causal_signature: Any,
        navigation_state_estimate: dict[str, Any],
        executed_candidate: ActionCandidateV1 | None,
    ) -> None:
        state = self._sequence_causal_state_v1
        if not isinstance(state, dict):
            return
        if not bool(state.get("enabled", False)):
            state["active"] = False
            state["stage"] = "idle"
            state["steps_remaining"] = 0
            state["target_required_samples"] = {}
            state["last_status"] = "disabled"
            return
        verify_action_ids = sorted(
            {
                int(v)
                for v in current_packet.available_actions
                if int(v) > 4 and int(v) != 0
            }
        )
        if not verify_action_ids:
            verify_action_ids = sorted(
                {
                    int(v)
                    for v in current_packet.available_actions
                    if int(v) in (1, 2, 3, 4)
                }
            )
        state["verify_action_ids"] = [int(v) for v in verify_action_ids]
        current_counter = int(getattr(current_packet, "action_counter", 0))
        if bool(state.get("active", False)):
            deadline_counter = int(state.get("deadline_action_counter", -1))
            if deadline_counter >= 0 and current_counter > deadline_counter:
                state["active"] = False
                state["stage"] = "idle"
                state["steps_remaining"] = 0
                state["timeout_count"] = int(state.get("timeout_count", 0) + 1)
                state["last_status"] = "seek_timeout"
        trigger_region_key = str(state.get("trigger_region_key", "NA"))
        target_region_key = str(state.get("target_region_key", "NA"))
        source_region_key = "NA"
        if transition_record is not None:
            action_context = transition_record.action_context
            if isinstance(action_context, dict):
                source_region_key = str(action_context.get("action_region_before", "NA"))
        nav_region = navigation_state_estimate.get("agent_pos_region", {})
        nav_region_key = "NA"
        if isinstance(nav_region, dict):
            nav_rx = int(nav_region.get("x", -1))
            nav_ry = int(nav_region.get("y", -1))
            if nav_rx >= 0 and nav_ry >= 0:
                nav_region_key = self._region_key_from_xy_v1(int(nav_rx), int(nav_ry))

        coupled_info = self._high_info_coupled_regions_v1(
            fallback_source_region_key=str(source_region_key),
        )
        effective_trigger_region_key = str(trigger_region_key)
        effective_target_region_key = str(target_region_key)
        primary_region_key = str(
            coupled_info.get("primary_region_key", coupled_info.get("cross_region_key", "NA"))
        )
        secondary_region_key = str(
            coupled_info.get("secondary_region_key", coupled_info.get("gate_region_key", "NA"))
        )
        if self._parse_region_key_v1(primary_region_key) is not None:
            effective_trigger_region_key = str(primary_region_key)
        if self._parse_region_key_v1(secondary_region_key) is not None:
            effective_target_region_key = str(secondary_region_key)
        if str(effective_trigger_region_key) == str(effective_target_region_key):
            effective_trigger_region_key = str(trigger_region_key)
            effective_target_region_key = str(target_region_key)
        state["trigger_region_key_effective"] = str(effective_trigger_region_key)
        state["target_region_key_effective"] = str(effective_target_region_key)

        obs_change_type = str(getattr(causal_signature, "obs_change_type", ""))
        changed_pixels = int(max(0, getattr(causal_signature, "changed_pixel_count", 0)))
        level_delta_now = int(max(0, getattr(causal_signature, "level_delta", 0)))
        verify_meaningful_event = bool(
            level_delta_now > 0
            or obs_change_type
            in (
                "CC_COUNT_CHANGE",
                "GLOBAL_PATTERN_CHANGE",
                "METADATA_PROGRESS_CHANGE",
            )
            or changed_pixels >= int(self.high_info_strong_change_pixels)
        )
        in_trigger_region = bool(
            str(source_region_key) == str(effective_trigger_region_key)
            or str(nav_region_key) == str(effective_trigger_region_key)
        )
        orientation_aligned = bool(coupled_info.get("orientation_aligned", False))
        trigger_match = bool(
            in_trigger_region
            and (
                obs_change_type == "CC_COUNT_CHANGE"
                or (
                    orientation_aligned
                    and str(obs_change_type) not in ("", "NO_CHANGE")
                )
            )
        )
        if trigger_match:
            state["active"] = True
            state["stage"] = "seek_target"
            state["trigger_count"] = int(state.get("trigger_count", 0) + 1)
            state["trigger_action_counter"] = int(current_counter)
            state["deadline_action_counter"] = int(
                current_counter + int(state.get("window_steps", 24))
            )
            state["verify_deadline_action_counter"] = -1
            state["steps_remaining"] = int(state.get("window_steps", 24))
            state["last_status"] = "triggered"

        if not bool(state.get("active", False)):
            return

        remaining = int(state.get("deadline_action_counter", current_counter) - current_counter)
        state["steps_remaining"] = int(max(0, remaining))
        stage = str(state.get("stage", "idle"))
        in_target_region = bool(
            str(nav_region_key) == str(effective_target_region_key)
            or str(source_region_key) == str(effective_target_region_key)
        )
        if stage == "seek_target" and in_target_region:
            state["stage"] = "verify"
            state["target_reach_count"] = int(state.get("target_reach_count", 0) + 1)
            state["last_reached_action_counter"] = int(current_counter)
            state["verify_deadline_action_counter"] = int(
                current_counter + int(state.get("verify_window_steps", 6))
            )
            state["last_status"] = "target_reached"
            stage = "verify"

        executed_action_id = int(executed_candidate.action_id) if executed_candidate else -1
        if stage == "verify":
            verify_deadline = int(state.get("verify_deadline_action_counter", -1))
            verified = bool(
                in_target_region
                and executed_action_id in set(int(v) for v in verify_action_ids)
                and verify_meaningful_event
            )
            if verified:
                state["active"] = False
                state["stage"] = "idle"
                state["steps_remaining"] = 0
                state["success_count"] = int(state.get("success_count", 0) + 1)
                state["last_status"] = "verified_success"
            elif verify_deadline >= 0 and current_counter > verify_deadline:
                state["active"] = False
                state["stage"] = "idle"
                state["steps_remaining"] = 0
                state["timeout_count"] = int(state.get("timeout_count", 0) + 1)
                state["last_status"] = "verify_timeout"

    def _update_high_info_focus_state_v1(
        self,
        *,
        current_packet: ObservationPacketV1,
        transition_record: TransitionRecordV1 | None,
        causal_signature: Any,
        navigation_state_estimate: dict[str, Any],
        executed_candidate: ActionCandidateV1 | None,
    ) -> None:
        state = self._high_info_focus_state_v1
        if not isinstance(state, dict):
            return
        if not bool(state.get("enabled", False)):
            state["active"] = False
            state["stage"] = "idle"
            state["steps_remaining"] = 0
            state["target_miss_streak"] = 0
            state["simultaneous_changed_region_keys"] = []
            state["simultaneous_reachable_region_keys"] = []
            state["simultaneous_unknown_region_keys"] = []
            state["simultaneous_unreachable_region_keys"] = []
            state["simultaneous_anchor_region_key"] = "NA"
            state["simultaneous_changed_total_pixels"] = 0
            state["region_recent_change_pixels"] = {}
            state["region_change_magnitude"] = {}
            state["region_change_magnitude_ema"] = {}
            state["region_change_delta"] = {}
            state["region_sudden_spike_keys"] = []
            state["last_trigger_changed_pixels"] = 0
            state["last_trigger_changed_region_diff_map"] = {}
            state["interaction_chain_active"] = False
            state["interaction_target_chain"] = []
            state["interaction_target_index"] = 0
            state["interaction_last_status"] = "disabled"
            state["pending_region_queue"] = []
            state["pending_region_scores"] = {}
            state["chain_lock_active"] = False
            state["chain_lock_window_steps"] = int(self.high_info_chain_lock_window_steps)
            state["chain_lock_steps_remaining"] = 0
            state["chain_lock_target_region_key"] = "NA"
            state["chain_lock_miss_limit"] = int(self.high_info_chain_lock_miss_limit)
            state["target_commit_active"] = False
            state["target_commit_window_steps"] = int(self.high_info_target_commit_window_steps)
            state["target_commit_miss_limit"] = int(self.high_info_target_commit_miss_limit)
            state["chain_lock_last_status"] = "disabled"
            state["novelty_protocol_active"] = False
            state["novelty_source_region_key"] = "NA"
            state["novelty_related_region_keys"] = []
            state["last_novelty_signature"] = "NA"
            state["novelty_trigger_count"] = 0
            state["novelty_last_action_counter"] = -1
            state["novelty_signature_stats"] = {}
            state["inner_loop_active"] = False
            state["inner_loop_reason"] = "NA"
            state["inner_loop_queue"] = []
            state["inner_loop_current_target_region_key"] = "NA"
            state["trigger_region_key_effective"] = str(
                state.get("trigger_region_key", self.sequence_causal_trigger_region_key)
            )
            state["target_region_key_effective"] = str(
                state.get("target_region_key", self.sequence_causal_target_region_key)
            )
            state["last_status"] = "disabled"
            return

        verify_action_ids = sorted(
            {
                int(v)
                for v in current_packet.available_actions
                if int(v) > 4 and int(v) != 0
            }
        )
        state["verify_action_ids"] = [int(v) for v in verify_action_ids]
        current_counter = int(getattr(current_packet, "action_counter", 0))
        deadline_counter = int(state.get("deadline_action_counter", -1))
        if bool(state.get("active", False)) and deadline_counter >= 0 and current_counter > deadline_counter:
            state["active"] = False
            state["stage"] = "idle"
            state["steps_remaining"] = 0
            state["target_miss_streak"] = 0
            state["target_region_queue"] = []
            state["pending_region_queue"] = []
            state["pending_region_scores"] = {}
            state["current_target_region_key"] = "NA"
            state["target_region_scores"] = {}
            state["target_required_samples"] = {}
            state["simultaneous_changed_region_keys"] = []
            state["simultaneous_reachable_region_keys"] = []
            state["simultaneous_unknown_region_keys"] = []
            state["simultaneous_unreachable_region_keys"] = []
            state["simultaneous_anchor_region_key"] = "NA"
            state["simultaneous_changed_total_pixels"] = 0
            state["region_recent_change_pixels"] = {}
            state["region_change_magnitude"] = {}
            state["region_change_magnitude_ema"] = {}
            state["region_change_delta"] = {}
            state["region_sudden_spike_keys"] = []
            state["last_trigger_changed_pixels"] = 0
            state["last_trigger_changed_region_diff_map"] = {}
            state["interaction_chain_active"] = False
            state["interaction_target_chain"] = []
            state["interaction_target_index"] = 0
            state["interaction_last_status"] = "timeout"
            state["chain_lock_active"] = False
            state["chain_lock_window_steps"] = int(self.high_info_chain_lock_window_steps)
            state["chain_lock_steps_remaining"] = 0
            state["chain_lock_target_region_key"] = "NA"
            state["chain_lock_miss_limit"] = int(self.high_info_chain_lock_miss_limit)
            state["target_commit_active"] = False
            state["target_commit_window_steps"] = int(self.high_info_target_commit_window_steps)
            state["target_commit_miss_limit"] = int(self.high_info_target_commit_miss_limit)
            state["chain_lock_last_status"] = "timeout"
            state["priority_subqueue_active"] = False
            state["priority_subqueue_keys"] = []
            state["novelty_protocol_active"] = False
            state["novelty_source_region_key"] = "NA"
            state["novelty_related_region_keys"] = []
            state["inner_loop_active"] = False
            state["inner_loop_reason"] = "NA"
            state["inner_loop_queue"] = []
            state["inner_loop_current_target_region_key"] = "NA"
            state["timeout_count"] = int(state.get("timeout_count", 0) + 1)
            state["last_status"] = "timeout"

        score_memory = state.get("target_region_scores", {})
        if not isinstance(score_memory, dict):
            score_memory = {}
        score_memory = {
            str(k): float(v)
            for (k, v) in score_memory.items()
            if self._parse_region_key_v1(str(k)) is not None and float(v) > 0.0
        }
        completed_regions = state.get("completed_target_regions", [])
        if not isinstance(completed_regions, list):
            completed_regions = []
        completed_regions = [
            str(v) for v in completed_regions if self._parse_region_key_v1(str(v)) is not None
        ][-16:]
        target_sample_counts = state.get("target_sample_counts", {})
        if not isinstance(target_sample_counts, dict):
            target_sample_counts = {}
        target_sample_counts = {
            str(k): int(max(0, v))
            for (k, v) in target_sample_counts.items()
            if self._parse_region_key_v1(str(k)) is not None
        }
        target_required_samples = state.get("target_required_samples", {})
        if not isinstance(target_required_samples, dict):
            target_required_samples = {}
        target_required_samples = {
            str(k): int(max(1, v))
            for (k, v) in target_required_samples.items()
            if self._parse_region_key_v1(str(k)) is not None
        }
        simultaneous_changed_region_keys = state.get("simultaneous_changed_region_keys", [])
        if not isinstance(simultaneous_changed_region_keys, list):
            simultaneous_changed_region_keys = []
        simultaneous_changed_region_keys = [
            str(v)
            for v in simultaneous_changed_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        ]
        simultaneous_reachable_region_keys = state.get("simultaneous_reachable_region_keys", [])
        if not isinstance(simultaneous_reachable_region_keys, list):
            simultaneous_reachable_region_keys = []
        simultaneous_reachable_region_keys = [
            str(v)
            for v in simultaneous_reachable_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        ]
        simultaneous_unknown_region_keys = state.get("simultaneous_unknown_region_keys", [])
        if not isinstance(simultaneous_unknown_region_keys, list):
            simultaneous_unknown_region_keys = []
        simultaneous_unknown_region_keys = [
            str(v)
            for v in simultaneous_unknown_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        ]
        simultaneous_unreachable_region_keys = state.get("simultaneous_unreachable_region_keys", [])
        if not isinstance(simultaneous_unreachable_region_keys, list):
            simultaneous_unreachable_region_keys = []
        simultaneous_unreachable_region_keys = [
            str(v)
            for v in simultaneous_unreachable_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        ]
        simultaneous_changed_total_pixels = int(
            max(0, state.get("simultaneous_changed_total_pixels", 0))
        )
        simultaneous_anchor_region_key = str(state.get("simultaneous_anchor_region_key", "NA"))
        interaction_target_chain = state.get("interaction_target_chain", [])
        if not isinstance(interaction_target_chain, list):
            interaction_target_chain = []
        interaction_target_chain = [
            str(v)
            for v in interaction_target_chain
            if self._parse_region_key_v1(str(v)) is not None
        ]
        interaction_target_index = int(max(0, state.get("interaction_target_index", 0)))
        interaction_chain_active = bool(state.get("interaction_chain_active", False))
        priority_subqueue_keys = state.get("priority_subqueue_keys", [])
        if not isinstance(priority_subqueue_keys, list):
            priority_subqueue_keys = []
        deduped_priority_subqueue: list[str] = []
        for region_key in priority_subqueue_keys:
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                continue
            if key in deduped_priority_subqueue:
                continue
            deduped_priority_subqueue.append(key)
        priority_subqueue_keys = list(deduped_priority_subqueue)
        inner_loop_queue_raw = state.get("inner_loop_queue", [])
        if not isinstance(inner_loop_queue_raw, list):
            inner_loop_queue_raw = []
        inner_loop_queue = [
            str(v)
            for v in inner_loop_queue_raw
            if self._parse_region_key_v1(str(v)) is not None
        ][:16]
        inner_loop_active = bool(
            state.get("inner_loop_active", False) and bool(inner_loop_queue)
        )
        inner_loop_reason = str(state.get("inner_loop_reason", "NA"))
        if not inner_loop_active:
            inner_loop_reason = "NA"
        pending_region_queue = state.get("pending_region_queue", [])
        if not isinstance(pending_region_queue, list):
            pending_region_queue = []
        pending_region_queue = [
            str(v)
            for v in pending_region_queue
            if self._parse_region_key_v1(str(v)) is not None
        ][:64]
        pending_region_scores = state.get("pending_region_scores", {})
        if not isinstance(pending_region_scores, dict):
            pending_region_scores = {}
        pending_region_scores = {
            str(k): float(max(0.0, v))
            for (k, v) in pending_region_scores.items()
            if self._parse_region_key_v1(str(k)) is not None and float(v) > 0.0
        }
        novelty_protocol_active = bool(state.get("novelty_protocol_active", False))
        novelty_source_region_key = str(state.get("novelty_source_region_key", "NA"))
        if self._parse_region_key_v1(novelty_source_region_key) is None:
            novelty_source_region_key = "NA"
        novelty_related_region_keys = state.get("novelty_related_region_keys", [])
        if not isinstance(novelty_related_region_keys, list):
            novelty_related_region_keys = []
        novelty_related_region_keys = [
            str(v)
            for v in novelty_related_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        ][:16]
        last_novelty_signature = str(state.get("last_novelty_signature", "NA"))
        novelty_trigger_count = int(max(0, state.get("novelty_trigger_count", 0)))
        novelty_last_action_counter = int(state.get("novelty_last_action_counter", -1))
        novelty_signature_stats_raw = state.get("novelty_signature_stats", {})
        if not isinstance(novelty_signature_stats_raw, dict):
            novelty_signature_stats_raw = {}
        novelty_signature_stats: dict[str, dict[str, Any]] = {}
        for signature_raw, entry_raw in novelty_signature_stats_raw.items():
            if not isinstance(entry_raw, dict):
                continue
            signature = str(signature_raw)
            source_key = str(entry_raw.get("source_region_key", "NA"))
            if self._parse_region_key_v1(source_key) is None:
                source_key = "NA"
            related_keys_raw = entry_raw.get("related_region_keys", [])
            if not isinstance(related_keys_raw, list):
                related_keys_raw = []
            related_keys = [
                str(v)
                for v in related_keys_raw
                if self._parse_region_key_v1(str(v)) is not None
            ][:8]
            novelty_signature_stats[signature] = {
                "count": int(max(0, entry_raw.get("count", 0))),
                "progress_hits": int(max(0, entry_raw.get("progress_hits", 0))),
                "avg_changed_pixels": float(max(0.0, entry_raw.get("avg_changed_pixels", 0.0))),
                "avg_simultaneous_pixels": float(
                    max(0.0, entry_raw.get("avg_simultaneous_pixels", 0.0))
                ),
                "source_region_key": str(source_key),
                "related_region_keys": list(related_keys),
                "last_action_counter": int(entry_raw.get("last_action_counter", -1)),
            }
        priority_subqueue_active = bool(
            state.get("priority_subqueue_active", False)
            and bool(priority_subqueue_keys)
        )
        chain_lock_window_steps = int(
            max(1, state.get("chain_lock_window_steps", self.high_info_chain_lock_window_steps))
        )
        chain_lock_miss_limit = int(
            max(1, state.get("chain_lock_miss_limit", self.high_info_chain_lock_miss_limit))
        )
        target_commit_window_steps = int(
            max(
                1,
                state.get(
                    "target_commit_window_steps",
                    self.high_info_target_commit_window_steps,
                ),
            )
        )
        target_commit_miss_limit = int(
            max(
                1,
                state.get(
                    "target_commit_miss_limit",
                    self.high_info_target_commit_miss_limit,
                ),
            )
        )
        chain_lock_active = bool(state.get("chain_lock_active", False))
        chain_lock_steps_remaining = int(max(0, state.get("chain_lock_steps_remaining", 0)))
        chain_lock_target_region_key = str(state.get("chain_lock_target_region_key", "NA"))
        chain_lock_last_status = str(state.get("chain_lock_last_status", "idle"))
        target_commit_active = bool(state.get("target_commit_active", False))

        def _disable_chain_lock(status: str) -> None:
            nonlocal chain_lock_active
            nonlocal chain_lock_steps_remaining
            nonlocal chain_lock_target_region_key
            nonlocal chain_lock_last_status
            nonlocal chain_lock_miss_limit
            nonlocal target_commit_active
            chain_lock_active = False
            chain_lock_steps_remaining = 0
            chain_lock_target_region_key = "NA"
            chain_lock_miss_limit = int(max(1, self.high_info_chain_lock_miss_limit))
            target_commit_active = False
            chain_lock_last_status = str(status)

        def _arm_chain_lock(
            target_region_key: str,
            status: str,
            *,
            focus_commit: bool = False,
        ) -> None:
            nonlocal chain_lock_active
            nonlocal chain_lock_steps_remaining
            nonlocal chain_lock_target_region_key
            nonlocal chain_lock_last_status
            nonlocal chain_lock_miss_limit
            nonlocal target_commit_active
            key = str(target_region_key)
            if self._parse_region_key_v1(key) is not None and bool(state.get("active", False)):
                chain_lock_active = True
                if focus_commit:
                    chain_lock_steps_remaining = int(
                        max(chain_lock_window_steps, target_commit_window_steps)
                    )
                    chain_lock_miss_limit = int(
                        max(chain_lock_miss_limit, target_commit_miss_limit)
                    )
                    target_commit_active = True
                else:
                    chain_lock_steps_remaining = int(chain_lock_window_steps)
                    chain_lock_miss_limit = int(max(1, chain_lock_miss_limit))
                    target_commit_active = False
                chain_lock_target_region_key = str(key)
                chain_lock_last_status = str(status)
            else:
                _disable_chain_lock(status)

        def _enqueue_pending_regions(
            region_keys: list[str],
            *,
            score_hint: dict[str, float] | None = None,
        ) -> None:
            nonlocal pending_region_queue
            nonlocal pending_region_scores
            lock_key = str(chain_lock_target_region_key)
            hint = score_hint if isinstance(score_hint, dict) else {}
            for region_key in region_keys:
                key = str(region_key)
                if self._parse_region_key_v1(key) is None:
                    continue
                if key == lock_key:
                    continue
                if key not in pending_region_queue:
                    pending_region_queue.append(str(key))
                score_value = float(max(0.0, hint.get(str(key), 0.0)))
                if score_value <= 0.0:
                    score_value = float(max(0.0, pending_region_scores.get(str(key), 0.0)))
                if score_value > 0.0:
                    pending_region_scores[str(key)] = float(
                        max(float(pending_region_scores.get(str(key), 0.0)), score_value)
                    )

        def _flush_pending_regions_to_state() -> None:
            state["pending_region_queue"] = [str(v) for v in pending_region_queue[:64]]
            state["pending_region_scores"] = {
                str(k): float(v)
                for (k, v) in sorted(
                    pending_region_scores.items(),
                    key=lambda item: (-float(item[1]), str(item[0])),
                )[:64]
            }

        def _flush_novelty_regions_to_state() -> None:
            state["novelty_protocol_active"] = bool(novelty_protocol_active)
            state["novelty_source_region_key"] = str(novelty_source_region_key)
            state["novelty_related_region_keys"] = [
                str(v) for v in novelty_related_region_keys[:16]
            ]
            state["last_novelty_signature"] = str(last_novelty_signature)
            state["novelty_trigger_count"] = int(max(0, novelty_trigger_count))
            state["novelty_last_action_counter"] = int(novelty_last_action_counter)
            state["novelty_signature_stats"] = {
                str(k): dict(v)
                for (k, v) in sorted(
                    novelty_signature_stats.items(),
                    key=lambda item: (
                        -int(item[1].get("count", 0)),
                        -int(item[1].get("last_action_counter", -1)),
                        str(item[0]),
                    ),
                )[: int(max(4, self.high_info_novelty_stats_max_entries))]
            }

        def _flush_inner_loop_state_to_state() -> None:
            state["inner_loop_active"] = bool(inner_loop_active and bool(inner_loop_queue))
            state["inner_loop_reason"] = str(
                inner_loop_reason if (inner_loop_active and inner_loop_queue) else "NA"
            )
            state["inner_loop_queue"] = [str(v) for v in inner_loop_queue[:16]]
            state["inner_loop_current_target_region_key"] = (
                str(inner_loop_queue[0])
                if (inner_loop_active and inner_loop_queue)
                else "NA"
            )

        def _activate_inner_loop_v1(region_keys: list[str], reason: str) -> bool:
            nonlocal inner_loop_active
            nonlocal inner_loop_reason
            nonlocal inner_loop_queue
            nonlocal priority_subqueue_active
            nonlocal priority_subqueue_keys
            nonlocal interaction_chain_active
            nonlocal interaction_target_chain
            nonlocal interaction_target_index
            ordered: list[str] = []
            for region_key in region_keys:
                key = str(region_key)
                if self._parse_region_key_v1(key) is None:
                    continue
                if key in ordered:
                    continue
                ordered.append(str(key))
            if not ordered:
                return False
            inner_loop_active = True
            inner_loop_reason = str(reason)
            inner_loop_queue = list(ordered[:16])
            priority_subqueue_active = True
            priority_subqueue_keys = [str(inner_loop_queue[0])]
            interaction_chain_active = False
            interaction_target_chain = []
            interaction_target_index = 0
            _arm_chain_lock(str(inner_loop_queue[0]), f"inner_loop_{str(reason)}", focus_commit=True)
            # Ensure inner-loop has enough budget to finish at least one source<->target cycle.
            base_window = int(max(1, state.get("window_steps", self.high_info_focus_window_steps)))
            desired_window = int(max(base_window, 12 + (6 * len(inner_loop_queue))))
            state["window_steps"] = int(desired_window)
            state["deadline_action_counter"] = int(
                max(
                    int(state.get("deadline_action_counter", -1)),
                    int(current_counter + desired_window),
                )
            )
            state["steps_remaining"] = int(
                max(0, int(state.get("deadline_action_counter", current_counter)) - int(current_counter))
            )
            return True

        def _compact_novelty_stats_v1() -> None:
            nonlocal novelty_signature_stats
            novelty_signature_stats = {
                str(k): dict(v)
                for (k, v) in sorted(
                    novelty_signature_stats.items(),
                    key=lambda item: (
                        -int(item[1].get("count", 0)),
                        -int(item[1].get("last_action_counter", -1)),
                        str(item[0]),
                    ),
                )[: int(max(4, self.high_info_novelty_stats_max_entries))]
            }

        def _record_novelty_pattern_v1(
            signature: str,
            *,
            source_region_key_for_stats: str,
            related_region_keys_for_stats: list[str],
            changed_pixels_for_stats: int,
            simultaneous_pixels_for_stats: int,
            progress_hit: bool,
        ) -> None:
            nonlocal novelty_signature_stats
            key = str(signature)
            previous = novelty_signature_stats.get(str(key), {})
            prev_count = int(max(0, previous.get("count", 0)))
            count_now = int(prev_count + 1)
            prev_avg_changed = float(max(0.0, previous.get("avg_changed_pixels", 0.0)))
            prev_avg_sim = float(max(0.0, previous.get("avg_simultaneous_pixels", 0.0)))
            avg_changed_now = float(
                ((prev_avg_changed * float(prev_count)) + float(max(0, changed_pixels_for_stats)))
                / float(max(1, count_now))
            )
            avg_sim_now = float(
                ((prev_avg_sim * float(prev_count)) + float(max(0, simultaneous_pixels_for_stats)))
                / float(max(1, count_now))
            )
            progress_hits_now = int(max(0, previous.get("progress_hits", 0))) + (
                1 if bool(progress_hit) else 0
            )
            novelty_signature_stats[str(key)] = {
                "count": int(count_now),
                "progress_hits": int(progress_hits_now),
                "avg_changed_pixels": float(avg_changed_now),
                "avg_simultaneous_pixels": float(avg_sim_now),
                "source_region_key": str(source_region_key_for_stats),
                "related_region_keys": [
                    str(v)
                    for v in related_region_keys_for_stats
                    if self._parse_region_key_v1(str(v)) is not None
                ][:8],
                "last_action_counter": int(current_counter),
            }
            _compact_novelty_stats_v1()

        if (
            self._parse_region_key_v1(str(chain_lock_target_region_key)) is None
            or chain_lock_steps_remaining <= 0
        ):
            _disable_chain_lock("inactive")
        min_samples_per_target = int(max(1, self.high_info_min_samples_per_target))
        coupled_min_samples = int(max(min_samples_per_target, self.high_info_coupled_min_samples))
        coupled_progress_locked = bool(int(current_packet.levels_completed) <= 0)
        high_value_threshold = float(
            max(0.0, min(1.0, float(self.high_info_focus_min_trigger_score)))
        )

        source_region_key = "NA"
        if transition_record is not None:
            context = transition_record.action_context
            if isinstance(context, dict):
                source_region_key = str(context.get("action_region_before", "NA"))
        nav_region_key = self._current_region_key_v1()
        if self._parse_region_key_v1(source_region_key) is None:
            source_region_key = str(nav_region_key)
        region_adjacency = self._region_graph_adjacency_v1(min_edge_count=1)
        graph_edge_count = int(
            sum(len(hist) for hist in region_adjacency.values() if isinstance(hist, dict))
        )
        known_region_count = int(len(self._region_visit_counts))
        reachability_graph_ready = bool(
            graph_edge_count >= int(self.high_info_reachability_graph_min_edges)
            and known_region_count >= int(self.high_info_reachability_graph_min_regions)
        )
        reachability_anchor_key = str(nav_region_key)
        if self._parse_region_key_v1(reachability_anchor_key) is None:
            reachability_anchor_key = str(source_region_key)
        reachable_region_set = self._region_reachable_set_v1(
            region_adjacency,
            start_region_key=str(reachability_anchor_key),
        )
        if self._parse_region_key_v1(str(reachability_anchor_key)) is not None:
            reachable_region_set.add(str(reachability_anchor_key))
        simultaneous_changed_set = set(simultaneous_changed_region_keys)
        simultaneous_reachable_set = set(simultaneous_reachable_region_keys)
        simultaneous_unknown_set = set(simultaneous_unknown_region_keys)
        simultaneous_unreachable_set = set(simultaneous_unreachable_region_keys)
        reachable_or_frontier_set: set[str] = {
            str(v)
            for v in reachable_region_set
            if self._parse_region_key_v1(str(v)) is not None
        }
        for region_key in list(reachable_or_frontier_set):
            parsed = self._parse_region_key_v1(str(region_key))
            if parsed is None:
                continue
            rx, ry = parsed
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = int(rx + dx)
                ny = int(ry + dy)
                if nx < 0 or ny < 0 or nx > 7 or ny > 7:
                    continue
                reachable_or_frontier_set.add(self._region_key_from_xy_v1(int(nx), int(ny)))
        if self._parse_region_key_v1(str(source_region_key)) is not None:
            reachable_or_frontier_set.add(str(source_region_key))
        if self._parse_region_key_v1(str(nav_region_key)) is not None:
            reachable_or_frontier_set.add(str(nav_region_key))

        def _reachable_or_frontier_region_v1(region_key: str) -> bool:
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                return False
            if not bool(reachability_graph_ready):
                return True
            return bool(key in reachable_or_frontier_set)

        coupled_info = self._high_info_coupled_regions_v1(
            fallback_source_region_key=str(source_region_key),
            reachable_region_set=set(reachable_region_set),
            reachability_graph_ready=bool(reachability_graph_ready),
        )
        coupled_region_keys = {
            str(v)
            for v in coupled_info.get("coupled_region_keys", [])
            if self._parse_region_key_v1(str(v)) is not None
        }
        primary_coupled_region_key = str(
            coupled_info.get("primary_region_key", coupled_info.get("cross_region_key", "NA"))
        )
        secondary_coupled_region_key = str(
            coupled_info.get("secondary_region_key", coupled_info.get("gate_region_key", "NA"))
        )
        if self._parse_region_key_v1(primary_coupled_region_key) is not None:
            coupled_region_keys.add(str(primary_coupled_region_key))
        if self._parse_region_key_v1(secondary_coupled_region_key) is not None:
            coupled_region_keys.add(str(secondary_coupled_region_key))
        orientation_misaligned = bool(coupled_info.get("orientation_misaligned", False))
        coupled_floor = float(
            max(
                self.high_info_coupled_score_floor,
                self.high_info_focus_min_trigger_score,
            )
        )

        def _required_samples(region_key: str) -> int:
            key = str(region_key)
            if key in coupled_region_keys:
                return int(coupled_min_samples)
            if key in simultaneous_reachable_set:
                return int(max(min_samples_per_target + 1, coupled_min_samples))
            return int(max(1, target_required_samples.get(key, min_samples_per_target)))

        obs_change_type = str(getattr(causal_signature, "obs_change_type", ""))
        changed_pixels = int(max(0, getattr(causal_signature, "changed_pixel_count", 0)))
        effect_summary = transition_record.effect_summary if transition_record is not None else {}
        if not isinstance(effect_summary, dict):
            effect_summary = {}
        changed_region_diff_map_raw = effect_summary.get("changed_region_diff_map_v1", {})
        if not isinstance(changed_region_diff_map_raw, dict):
            changed_region_diff_map_raw = {}
        region_recent_change_pixels: dict[str, int] = {}
        for region_key_raw, pixels_raw in changed_region_diff_map_raw.items():
            region_key = str(region_key_raw)
            if self._parse_region_key_v1(region_key) is None:
                continue
            pixels = int(max(0, pixels_raw))
            if pixels <= 0:
                continue
            region_recent_change_pixels[str(region_key)] = int(pixels)
        total_changed_pixels_for_regions = int(
            max(1, changed_pixels, sum(int(v) for v in region_recent_change_pixels.values()))
        )
        region_change_magnitude_prev = state.get("region_change_magnitude_ema", {})
        if not isinstance(region_change_magnitude_prev, dict):
            region_change_magnitude_prev = {}
        region_change_magnitude_ema: dict[str, float] = {}
        for region_key_raw, magnitude_raw in region_change_magnitude_prev.items():
            region_key = str(region_key_raw)
            if self._parse_region_key_v1(region_key) is None:
                continue
            magnitude = float(max(0.0, min(1.0, magnitude_raw)))
            if magnitude <= 0.0:
                continue
            decayed = float(0.92 * magnitude)
            if decayed >= 0.01:
                region_change_magnitude_ema[str(region_key)] = float(decayed)
        region_change_magnitude_now: dict[str, float] = {}
        region_change_delta_now: dict[str, float] = {}
        for region_key, pixels in region_recent_change_pixels.items():
            ratio = float(pixels) / float(max(1, total_changed_pixels_for_regions))
            density = float(min(1.0, float(pixels) / 64.0))
            magnitude_now = float(max(0.0, min(1.0, (0.65 * ratio) + (0.35 * density))))
            prev_ema = float(max(0.0, min(1.0, region_change_magnitude_ema.get(region_key, 0.0))))
            ema_now = float((0.72 * prev_ema) + (0.28 * magnitude_now))
            region_change_magnitude_now[str(region_key)] = float(magnitude_now)
            region_change_delta_now[str(region_key)] = float(max(0.0, magnitude_now - prev_ema))
            region_change_magnitude_ema[str(region_key)] = float(max(0.0, min(1.0, ema_now)))
        region_sudden_spike_keys_now: set[str] = set()
        for region_key, magnitude_now in region_change_magnitude_now.items():
            delta_now = float(max(0.0, region_change_delta_now.get(region_key, 0.0)))
            pixels_now = int(max(0, region_recent_change_pixels.get(region_key, 0)))
            non_source_large_change = bool(
                str(region_key) != str(source_region_key)
                and pixels_now >= int(max(8, self.high_info_simultaneous_min_region_pixels))
                and int(changed_pixels) >= int(self.high_info_simultaneous_min_total_pixels)
                and float(magnitude_now) >= 0.22
            )
            spike_detected = bool(
                delta_now >= 0.22
                or (
                    float(magnitude_now) >= 0.68
                    and pixels_now >= int(max(8, self.high_info_simultaneous_min_region_pixels))
                )
                or non_source_large_change
            )
            if not spike_detected:
                continue
            if _reachable_or_frontier_region_v1(str(region_key)):
                region_sudden_spike_keys_now.add(str(region_key))
        strong_event = bool(
            obs_change_type in ("CC_COUNT_CHANGE", "GLOBAL_PATTERN_CHANGE", "METADATA_PROGRESS_CHANGE")
            or changed_pixels >= int(self.high_info_strong_change_pixels)
        )
        action_id = int(executed_candidate.action_id) if executed_candidate is not None else -1
        trigger_semantics = self._region_action_semantics_v1(
            action_id=int(action_id),
            current_region_key=str(source_region_key),
        )
        trigger_score = float(trigger_semantics.get("info_trigger_score", 0.0))
        trigger_attempts = int(max(0, trigger_semantics.get("attempts", 0)))
        source_visit_count = int(max(0, self._region_visit_counts.get(str(source_region_key), 0)))
        source_in_coupled_region = bool(str(source_region_key) in coupled_region_keys)
        coupled_probe_trigger = bool(
            source_in_coupled_region
            and source_visit_count >= int(max(2, coupled_min_samples - 1))
            and (
                str(obs_change_type) != "NO_CHANGE"
                or int(changed_pixels) >= 2
            )
        )
        level_delta_now = int(max(0, getattr(causal_signature, "level_delta", 0)))
        trigger_reliable = bool(
            trigger_attempts >= int(max(2, min_samples_per_target))
            or (source_in_coupled_region and trigger_attempts >= 1)
            or coupled_probe_trigger
            or level_delta_now > 0
            or strong_event
        )
        non_coupled_high_trigger_threshold = float(
            max(self.high_info_focus_min_trigger_score, 0.62)
        )
        source_trigger_gate = bool(
            source_in_coupled_region
            or trigger_score >= non_coupled_high_trigger_threshold
            or level_delta_now > 0
            or obs_change_type == "METADATA_PROGRESS_CHANGE"
            or obs_change_type == "GLOBAL_PATTERN_CHANGE"
            or int(changed_pixels) >= int(self.high_info_strong_change_pixels)
        )
        should_trigger = bool(
            (
                strong_event
                or coupled_probe_trigger
                or (
                    source_in_coupled_region
                    and trigger_score >= 0.34
                )
            )
            and trigger_reliable
            and source_trigger_gate
            and self._parse_region_key_v1(str(source_region_key)) is not None
            and (
                trigger_score >= float(self.high_info_focus_min_trigger_score)
                or source_in_coupled_region
                or coupled_probe_trigger
                or int(changed_pixels) >= int(self.high_info_strong_change_pixels)
            )
        )
        if should_trigger and bool(state.get("active", False)):
            if (
                (not source_in_coupled_region)
                and trigger_score < float(max(0.72, non_coupled_high_trigger_threshold))
                and level_delta_now <= 0
                and str(obs_change_type) != "METADATA_PROGRESS_CHANGE"
                and int(changed_pixels) < int(self.high_info_strong_change_pixels)
            ):
                should_trigger = False
            previous_trigger_counter = int(state.get("trigger_action_counter", -1))
            previous_source_region_key = str(state.get("source_region_key", "NA"))
            retrigger_cooldown = int(max(0, self.high_info_retrigger_cooldown_steps))
            steps_since_trigger = (
                int(current_counter - previous_trigger_counter)
                if previous_trigger_counter >= 0
                else 10**6
            )
            high_conf_retrigger = bool(
                level_delta_now > 0
                or obs_change_type == "METADATA_PROGRESS_CHANGE"
                or trigger_score >= max(
                    0.75,
                    float(self.high_info_focus_min_trigger_score) + 0.20,
                )
            )
            if (
                retrigger_cooldown > 0
                and steps_since_trigger < retrigger_cooldown
                and (not high_conf_retrigger)
            ):
                should_trigger = False
            elif (
                retrigger_cooldown > 0
                and str(previous_source_region_key) == str(source_region_key)
                and steps_since_trigger < retrigger_cooldown
            ):
                should_trigger = False
            elif (
                retrigger_cooldown > 0
                and steps_since_trigger < max(3, int(retrigger_cooldown // 2))
                and (not strong_event)
                and level_delta_now <= 0
            ):
                should_trigger = False
            elif (
                (not strong_event)
                and level_delta_now <= 0
                and int(changed_pixels) < int(self.high_info_simultaneous_min_total_pixels)
            ):
                should_trigger = False
        if should_trigger and bool(state.get("active", False)) and bool(target_commit_active):
            # Commit window: keep the current high-value target fixed.
            # New evidence is still collected into the queue via region-change bookkeeping.
            should_trigger = False

        simultaneous_info = self._simultaneous_changed_region_info_v1(
            transition_record=transition_record,
            source_region_key=str(source_region_key),
            anchor_region_key=str(reachability_anchor_key),
            region_adjacency=region_adjacency,
            reachable_region_set=reachable_region_set,
            reachability_graph_ready=reachability_graph_ready,
        )
        simultaneous_changed_now = [
            str(v) for v in simultaneous_info.get("changed_region_keys", [])
        ]
        simultaneous_reachable_now = [
            str(v) for v in simultaneous_info.get("reachable_region_keys", [])
        ]
        simultaneous_unknown_now = [
            str(v) for v in simultaneous_info.get("unknown_region_keys", [])
        ]
        simultaneous_unreachable_now = [
            str(v) for v in simultaneous_info.get("unreachable_region_keys", [])
        ]
        simultaneous_anchor_now = str(
            simultaneous_info.get("anchor_region_key", reachability_anchor_key)
        )
        simultaneous_pixels_now = int(
            max(0, simultaneous_info.get("changed_total_pixels", 0))
        )
        remote_effect_detected = False
        if simultaneous_changed_now:
            # Detect "remote effects": an action in the source region causes a non-adjacent
            # region to change in the same step. This is a strong causal signal even if the
            # primary event type is a translation.
            for region_key in simultaneous_changed_now:
                key = str(region_key)
                if self._parse_region_key_v1(key) is None:
                    continue
                if key == str(source_region_key):
                    continue
                src_parsed = self._parse_region_key_v1(str(source_region_key))
                key_parsed = self._parse_region_key_v1(str(key))
                if src_parsed is None or key_parsed is None:
                    continue
                # Treat near-neighbors (including diagonals and 2-step spillover) as "local";
                # require Chebyshev distance >= 3 to qualify as a remote effect.
                dx = int(abs(int(key_parsed[0]) - int(src_parsed[0])))
                dy = int(abs(int(key_parsed[1]) - int(src_parsed[1])))
                if max(dx, dy) >= 3:
                    remote_effect_detected = True
                    break
        simultaneous_refresh = bool(
            simultaneous_changed_now
            and int(simultaneous_pixels_now) >= int(self.high_info_simultaneous_min_total_pixels)
            and (len(simultaneous_changed_now) >= 2 or level_delta_now > 0)
            # Even under a commit lock, we still want to capture simultaneous/remote effects
            # (e.g., trigger region causing the exit region to change). These are the exact
            # "high-info" events we want to chase after prepass.
            and (
                should_trigger
                or strong_event
                or remote_effect_detected
                or level_delta_now > 0
            )
        )
        if simultaneous_refresh:
            simultaneous_changed_region_keys = list(simultaneous_changed_now)
            simultaneous_reachable_region_keys = list(simultaneous_reachable_now)
            simultaneous_unknown_region_keys = list(simultaneous_unknown_now)
            simultaneous_unreachable_region_keys = list(simultaneous_unreachable_now)
            simultaneous_anchor_region_key = str(simultaneous_anchor_now)
            simultaneous_changed_total_pixels = int(simultaneous_pixels_now)
            simultaneous_changed_set = set(simultaneous_changed_region_keys)
            simultaneous_reachable_set = set(simultaneous_reachable_region_keys)
            simultaneous_unknown_set = set(simultaneous_unknown_region_keys)
            simultaneous_unreachable_set = set(simultaneous_unreachable_region_keys)
        novelty_related_region_candidates: list[str] = []
        if self.high_info_novelty_protocol_enabled:
            novelty_related_region_candidates = [
                str(region_key)
                for region_key in simultaneous_reachable_region_keys
                if self._parse_region_key_v1(str(region_key)) is not None
                and str(region_key) != str(source_region_key)
                and not (
                    str(region_key) in simultaneous_unreachable_set
                    and bool(reachability_graph_ready)
                )
            ]
            if not novelty_related_region_candidates:
                novelty_related_region_candidates = [
                    str(region_key)
                    for region_key in simultaneous_changed_region_keys
                    if self._parse_region_key_v1(str(region_key)) is not None
                    and str(region_key) != str(source_region_key)
                    and not (
                        str(region_key) in simultaneous_unreachable_set
                        and bool(reachability_graph_ready)
                    )
                ]
            novelty_related_region_candidates = sorted(
                list(dict.fromkeys(novelty_related_region_candidates)),
                key=lambda key: (
                    -int(max(0, region_recent_change_pixels.get(str(key), 0))),
                    -float(max(0.0, region_change_magnitude_now.get(str(key), 0.0))),
                    -float(max(0.0, region_change_delta_now.get(str(key), 0.0))),
                    int(
                        self._region_route_distance_v1(
                            region_adjacency,
                            start_region_key=str(source_region_key),
                            goal_region_key=str(key),
                        )
                    ),
                    str(key),
                ),
            )[: int(max(1, self.high_info_novelty_max_related_targets))]
        active_novelty_related_targets: list[str] = [
            str(v)
            for v in novelty_related_region_candidates
            if self._parse_region_key_v1(str(v)) is not None
            and str(v) != str(source_region_key)
            and not (
                str(v) in simultaneous_unreachable_set
                and bool(reachability_graph_ready)
            )
        ][: int(max(1, self.high_info_novelty_max_related_targets))]
        # Keep only genuinely remote effects (Chebyshev distance >= 3). This prevents
        # routine movement/shape spillover across adjacent regions from arming novelty.
        if active_novelty_related_targets:
            src_parsed = self._parse_region_key_v1(str(source_region_key))
            if src_parsed is not None:
                remote_only: list[str] = []
                for region_key in active_novelty_related_targets:
                    key = str(region_key)
                    parsed = self._parse_region_key_v1(key)
                    if parsed is None:
                        continue
                    dx = int(abs(int(parsed[0]) - int(src_parsed[0])))
                    dy = int(abs(int(parsed[1]) - int(src_parsed[1])))
                    if max(dx, dy) >= 3:
                        remote_only.append(str(key))
                if remote_only:
                    active_novelty_related_targets = list(remote_only)[: int(max(1, self.high_info_novelty_max_related_targets))]
        novelty_event_detected = bool(
            self.high_info_novelty_protocol_enabled
            and (
                should_trigger
                or strong_event
                or remote_effect_detected
                or int(level_delta_now) > 0
            )
            and self._parse_region_key_v1(str(source_region_key)) is not None
            and bool(active_novelty_related_targets)
            and (
                len(simultaneous_changed_region_keys) >= 2
                or len(active_novelty_related_targets) >= 2
                or int(level_delta_now) > 0
                or bool(str(obs_change_type) in ("GLOBAL_PATTERN_CHANGE", "CC_COUNT_CHANGE"))
            )
            # Novelty protocol is only meaningful when there is evidence of a remote effect.
            and (remote_effect_detected or int(level_delta_now) > 0)
            and int(max(changed_pixels, simultaneous_changed_total_pixels, simultaneous_pixels_now))
            >= int(self.high_info_simultaneous_min_total_pixels)
        )
        if novelty_event_detected and inner_loop_active and inner_loop_queue:
            # Inner-loop isolation: do not let newly observed events preempt an active
            # high-info verification loop. New evidence is still recorded and can be
            # consumed after the current loop drains.
            if str(source_region_key) not in set(str(v) for v in inner_loop_queue):
                novelty_event_detected = False
        novelty_signature = "NA"
        if novelty_event_detected:
            novelty_signature = str(
                "src="
                + str(source_region_key)
                + "|evt="
                + str(obs_change_type)
                + "|rel="
                + ",".join(str(v) for v in active_novelty_related_targets[:3])
            )
            # If we're already in a high-info lock window, arming the novelty protocol must not
            # depend on re-triggering. Otherwise we'd miss the common pattern:
            # "trigger region -> remote change" while locked on the trigger.
            if bool(state.get("active", False)) and (not should_trigger):
                novelty_protocol_active = True
                novelty_source_region_key = str(source_region_key)
                novelty_related_region_keys = list(active_novelty_related_targets)
                last_novelty_signature = str(novelty_signature)
                novelty_trigger_count = int(novelty_trigger_count + 1)
                novelty_last_action_counter = int(current_counter)
                _record_novelty_pattern_v1(
                    str(novelty_signature),
                    source_region_key_for_stats=str(source_region_key),
                    related_region_keys_for_stats=list(active_novelty_related_targets),
                    changed_pixels_for_stats=int(changed_pixels),
                    simultaneous_pixels_for_stats=int(
                        max(simultaneous_pixels_now, simultaneous_changed_total_pixels)
                    ),
                    progress_hit=bool(level_delta_now > 0),
                )
                # Verify-first: when a source action causes a remote effect, do not immediately
                # force extra retriggers of the source. Retriggering without verification can
                # destroy the causal state we're trying to exploit (e.g., cyclic toggles).
                baseline_counts: dict[str, int] = {}
                source_key = str(source_region_key)
                source_sample_count = int(max(0, target_sample_counts.get(str(source_key), 0)))
                # Count the triggering visit itself as a source sample so the protocol can
                # immediately chase remote effects next.
                source_sample_count = int(max(1, source_sample_count))
                target_sample_counts[str(source_key)] = int(
                    max(int(target_sample_counts.get(str(source_key), 0)), int(source_sample_count))
                )
                baseline_counts[str(source_key)] = int(source_sample_count)
                target_required_samples[str(source_key)] = 1
                score_memory[str(source_region_key)] = max(
                    float(score_memory.get(str(source_region_key), 0.0)),
                    0.90,
                )
                for region_key in active_novelty_related_targets:
                    related_key = str(region_key)
                    related_sample_count = int(
                        max(0, target_sample_counts.get(str(related_key), 0))
                    )
                    baseline_counts[str(related_key)] = int(related_sample_count)
                    related_required = int(
                        max(
                            _required_samples(str(related_key)),
                            related_sample_count
                            + int(1 + self.high_info_novelty_related_extra_samples),
                        )
                    )
                    target_required_samples[str(related_key)] = int(
                        max(
                            target_required_samples.get(str(related_key), 0),
                            related_required,
                        )
                    )
                    score_memory[str(related_key)] = max(
                        float(score_memory.get(str(related_key), 0.0)),
                        0.88,
                    )
                state["novelty_baseline_sample_counts"] = dict(baseline_counts)
                # Extend the focus window so the agent has enough budget to travel
                # source <-> remote and satisfy the extra sampling requirements.
                max_dist = 0
                for key in active_novelty_related_targets[:6]:
                    dist = int(self._region_distance_v1(str(source_region_key), str(key)))
                    if dist > max_dist:
                        max_dist = int(dist)
                desired_window = int(max(int(state.get("window_steps", 0)), 6 * (max_dist + 1)))
                desired_window = int(min(96, max(16, desired_window)))
                state["window_steps"] = int(max(int(state.get("window_steps", 0)), desired_window))
                state["deadline_action_counter"] = int(
                    max(
                        int(state.get("deadline_action_counter", -1)),
                        int(current_counter + desired_window),
                    )
                )
                # Immediate "remote effect" chase: if the current lock is on the source, retarget
                # the lock to the most affected remote region.
                if chain_lock_active and bool(active_novelty_related_targets):
                    lock_key = str(chain_lock_target_region_key)
                    remote_rows: list[tuple[int, int, int, str]] = []
                    for region_key in active_novelty_related_targets:
                        key = str(region_key)
                        if self._parse_region_key_v1(key) is None:
                            continue
                        route_distance = int(
                            self._region_route_distance_v1(
                                region_adjacency,
                                start_region_key=str(source_region_key),
                                goal_region_key=str(key),
                            )
                        )
                        if route_distance >= 10**6:
                            route_distance = int(
                                self._region_distance_v1(
                                    str(source_region_key),
                                    str(key),
                                )
                            )
                        diff_pixels = int(max(0, region_recent_change_pixels.get(str(key), 0)))
                        # Prefer genuinely remote effects (>=2 region steps away).
                        remote_priority = 0 if route_distance >= 2 else 1
                        remote_rows.append(
                            (int(remote_priority), -int(diff_pixels), int(route_distance), str(key))
                        )
                    remote_rows.sort()
                    best_remote = str(remote_rows[0][3]) if remote_rows else str("NA")
                    if (
                        self._parse_region_key_v1(best_remote) is not None
                        and best_remote != lock_key
                        and lock_key == str(source_region_key)
                    ):
                        _arm_chain_lock(best_remote, "novelty_remote_override", focus_commit=True)
        simultaneous_focus_candidates = [
            str(v)
            for v in simultaneous_changed_region_keys
            if self._parse_region_key_v1(str(v)) is not None
            and not (
                str(v) in simultaneous_unreachable_set
                and bool(reachability_graph_ready)
            )
        ]
        simultaneous_focus_candidate_set = set(simultaneous_focus_candidates)
        simultaneous_priority_set = set(simultaneous_reachable_region_keys) | set(
            simultaneous_unknown_region_keys
        )
        if not simultaneous_priority_set:
            simultaneous_priority_set = set(simultaneous_focus_candidate_set)
        simultaneous_touches_coupled = bool(
            bool(simultaneous_focus_candidate_set & set(coupled_region_keys))
            or str(source_region_key) in set(coupled_region_keys)
            or level_delta_now > 0
        )
        simultaneous_focus_reset = bool(
            simultaneous_refresh
            and len(simultaneous_focus_candidates) >= 2
            and simultaneous_touches_coupled
        )
        if simultaneous_focus_reset:
            ordered_priority_rows: list[tuple[int, float, int, int, str]] = []
            for region_key in simultaneous_focus_candidates:
                route_distance = int(
                    self._region_route_distance_v1(
                        region_adjacency,
                        start_region_key=str(source_region_key),
                        goal_region_key=str(region_key),
                    )
                )
                if route_distance >= 10**6:
                    route_distance = int(
                        self._region_distance_v1(
                            str(source_region_key),
                            str(region_key),
                        )
                    )
                region_diff = int(
                    max(
                        0,
                        (simultaneous_info.get("changed_region_diff_map", {}) or {}).get(
                            str(region_key),
                            0,
                        ),
                    )
                )
                change_magnitude = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            max(
                                region_change_magnitude_now.get(str(region_key), 0.0),
                                region_change_magnitude_ema.get(str(region_key), 0.0),
                            ),
                        ),
                    )
                )
                sudden_priority = (
                    0 if str(region_key) in region_sudden_spike_keys_now else 1
                )
                ordered_priority_rows.append(
                    (
                        int(sudden_priority),
                        float(-change_magnitude),
                        -int(region_diff),
                        int(route_distance),
                        str(region_key),
                    )
                )
            ordered_priority_rows.sort()
            priority_subqueue_keys = [str(row[4]) for row in ordered_priority_rows]
            priority_subqueue_active = bool(len(priority_subqueue_keys) >= 2)
        for region_key in simultaneous_reachable_set:
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                continue
            if float(region_change_delta_now.get(key, 0.0)) >= 0.14:
                region_sudden_spike_keys_now.add(str(key))
        region_event_priority_keys: set[str] = set()
        for region_key, pixels in region_recent_change_pixels.items():
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                continue
            if key == str(source_region_key):
                continue
            magnitude = float(
                max(
                    0.0,
                    min(
                        1.0,
                        max(
                            region_change_magnitude_now.get(key, 0.0),
                            region_change_magnitude_ema.get(key, 0.0),
                        ),
                    ),
                )
            )
            if (
                int(pixels) >= int(max(8, self.high_info_simultaneous_min_region_pixels))
                and magnitude >= 0.20
            ):
                region_event_priority_keys.add(str(key))
        region_sudden_spike_priority_keys = {
            str(region_key)
            for region_key in (set(region_sudden_spike_keys_now) | set(region_event_priority_keys))
            if self._parse_region_key_v1(str(region_key)) is not None
            and (
                _reachable_or_frontier_region_v1(str(region_key))
                or str(region_key) in simultaneous_priority_set
                or str(region_key) in region_event_priority_keys
            )
        }
        region_change_magnitude_effective: dict[str, float] = {}
        for region_key in set(region_change_magnitude_ema.keys()) | set(region_change_magnitude_now.keys()):
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                continue
            effective = float(
                max(
                    0.0,
                    min(
                        1.0,
                        max(
                            region_change_magnitude_ema.get(key, 0.0),
                            region_change_magnitude_now.get(key, 0.0),
                        ),
                    ),
                )
            )
            if effective > 0.0:
                region_change_magnitude_effective[str(key)] = float(effective)

        def _is_reachable_diff_override_candidate_v1(
            region_key: str,
            *,
            anchor_region_key: str,
            source_region_key_hint: str,
        ) -> bool:
            key = str(region_key)
            if self._parse_region_key_v1(key) is None:
                return False
            if str(key) in simultaneous_unreachable_set and bool(reachability_graph_ready):
                return False
            if not _reachable_or_frontier_region_v1(str(key)):
                return False
            if str(key) == str(source_region_key_hint):
                return False
            if (
                self._parse_region_key_v1(str(anchor_region_key)) is not None
                and str(key) == str(anchor_region_key)
            ):
                return False
            if (
                self._parse_region_key_v1(str(nav_region_key)) is not None
                and str(key) == str(nav_region_key)
            ):
                return False
            changed_px = int(max(0, region_recent_change_pixels.get(str(key), 0)))
            min_pixels = int(max(6, int(self.high_info_simultaneous_min_region_pixels // 2)))
            if changed_px < min_pixels:
                return False
            change_magnitude = float(max(0.0, region_change_magnitude_effective.get(str(key), 0.0)))
            change_delta = float(max(0.0, region_change_delta_now.get(str(key), 0.0)))
            visit_count = int(max(0, self._region_visit_counts.get(str(key), 0)))
            novelty_gate = bool(
                change_delta >= 0.08
                or (change_magnitude >= 0.32 and visit_count <= 4)
                or (change_magnitude >= 0.45 and visit_count <= 12)
            )
            if not novelty_gate:
                return False
            route_distance = int(
                self._region_route_distance_v1(
                    region_adjacency,
                    start_region_key=str(anchor_region_key),
                    goal_region_key=str(key),
                )
            )
            if route_distance >= 10**6:
                route_distance = int(
                    self._region_distance_v1(
                        str(anchor_region_key),
                        str(key),
                    )
                )
            if route_distance > 6:
                return False
            return True

        def _collect_hot_targets(anchor_region_key: str) -> dict[str, float]:
            target_scores_local: dict[str, float] = {}
            scoreboard = self._high_info_region_scoreboard_v1(max_regions=64)
            rows = scoreboard.get("rows", [])
            if not isinstance(rows, list):
                rows = []
            rows_by_region_key: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                region_key = str(row.get("region_key", "NA"))
                if self._parse_region_key_v1(region_key) is None:
                    continue
                rows_by_region_key[str(region_key)] = dict(row)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                region_key = str(row.get("region_key", "NA"))
                if self._parse_region_key_v1(region_key) is None:
                    continue
                ui_suppression_hint = float(
                    max(0.0, min(1.0, row.get("ui_suppression", 0.0)))
                )
                event_override = bool(
                    str(region_key) in region_event_priority_keys
                    and ui_suppression_hint < 0.70
                )
                if (
                    not _reachable_or_frontier_region_v1(str(region_key))
                    and str(region_key) not in coupled_region_keys
                    and str(region_key) not in simultaneous_priority_set
                    and not event_override
                ):
                    continue
                info_score = float(max(0.0, row.get("info_score", 0.0)))
                if info_score <= 0.0:
                    continue
                visit_count = int(self._region_visit_counts.get(region_key, 0))
                recent_change_pixels = int(max(0, region_recent_change_pixels.get(region_key, 0)))
                recent_change_magnitude = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            region_change_magnitude_now.get(
                                region_key,
                                region_change_magnitude_ema.get(region_key, 0.0),
                            ),
                        ),
                    )
                )
                recent_change_delta = float(
                    max(0.0, min(1.0, region_change_delta_now.get(region_key, 0.0)))
                )
                sudden_spike = bool(region_key in region_sudden_spike_keys_now)
                non_source_event_change = bool(
                    str(region_key) != str(source_region_key)
                    and recent_change_pixels
                    >= int(max(8, self.high_info_simultaneous_min_region_pixels))
                    and int(changed_pixels) >= int(self.high_info_simultaneous_min_total_pixels)
                )
                novelty_bonus = float(max(0.0, 1.0 - min(1.0, float(visit_count) / 10.0))) * 0.20
                revisit_penalty = float(min(0.18, float(visit_count) / 20.0))
                stale_loop_penalty = 0.0
                if coupled_progress_locked and region_key not in coupled_region_keys:
                    progress_rate = float(max(0.0, row.get("progress_rate", 0.0)))
                    coupling_score = float(max(0.0, row.get("coupling_signal_score", 0.0)))
                    monotony_penalty = float(max(0.0, row.get("monotony_penalty", 0.0)))
                    if (
                        visit_count >= 8
                        and info_score <= 0.58
                        and coupling_score <= 0.52
                        and progress_rate <= 0.0
                    ):
                        stale_loop_penalty = float(min(0.34, 0.04 * float(visit_count - 7)))
                    if monotony_penalty > 0.0 and progress_rate <= 0.0:
                        stale_loop_penalty = float(
                            stale_loop_penalty + min(0.28, monotony_penalty)
                        )
                    if visit_count >= 24 and progress_rate <= 0.0 and coupling_score <= 0.82:
                        stale_loop_penalty = float(
                            stale_loop_penalty + min(0.30, 0.01 * float(visit_count - 23))
                        )
                score = float(
                    max(
                        0.0,
                        info_score + novelty_bonus - revisit_penalty - stale_loop_penalty,
                    )
                )
                score = float(
                    score
                    + (0.22 * recent_change_magnitude)
                    + (
                        (0.28 + (0.22 * recent_change_delta))
                        if sudden_spike
                        else 0.0
                    )
                    + (
                        0.20 + (0.14 * recent_change_magnitude)
                        if non_source_event_change
                        else 0.0
                    )
                )
                if sudden_spike and _reachable_or_frontier_region_v1(str(region_key)):
                    spike_floor = float(
                        min(
                            1.0,
                            0.78
                            + (0.14 * recent_change_magnitude)
                            + (0.08 if recent_change_pixels >= 12 else 0.0),
                        )
                    )
                    score = float(max(score, spike_floor))
                if region_key == str(anchor_region_key) and strong_event:
                    anchor_sample_count = int(
                        max(0, target_sample_counts.get(str(region_key), 0))
                    )
                    anchor_required_samples = int(_required_samples(str(region_key)))
                    if anchor_sample_count < anchor_required_samples:
                        anchor_floor = 0.62
                        if (
                            int(source_visit_count) >= int(max(4, coupled_min_samples + 1))
                            and coupled_progress_locked
                        ):
                            anchor_floor = 0.48
                        score = float(max(score, anchor_floor))
                target_scores_local[region_key] = max(
                    float(target_scores_local.get(region_key, 0.0)),
                    float(score),
                )
            if simultaneous_changed_set:
                max_targets = int(max(1, self.high_info_simultaneous_max_targets))
                ordered_changed = [
                    str(v)
                    for v in simultaneous_changed_region_keys
                    if self._parse_region_key_v1(str(v)) is not None
                ][:max_targets]
                for idx, region_key in enumerate(ordered_changed):
                    if (
                        str(region_key) in simultaneous_unreachable_set
                        and bool(reachability_graph_ready)
                    ):
                        continue
                    route_distance = int(
                        self._region_route_distance_v1(
                            region_adjacency,
                            start_region_key=str(anchor_region_key),
                            goal_region_key=str(region_key),
                        )
                    )
                    if route_distance >= 10**6:
                        route_distance = int(
                            self._region_distance_v1(
                                str(anchor_region_key),
                                str(region_key),
                            )
                        )
                    route_bonus = float(max(0.0, 0.18 - (0.04 * float(route_distance))))
                    rank_penalty = float(0.05 * float(idx))
                    base_score = float(0.74 + route_bonus - rank_penalty)
                    change_magnitude = float(
                        max(
                            0.0,
                            min(
                                1.0,
                                region_change_magnitude_now.get(
                                    str(region_key),
                                    region_change_magnitude_ema.get(str(region_key), 0.0),
                                ),
                            ),
                        )
                    )
                    change_delta = float(
                        max(0.0, min(1.0, region_change_delta_now.get(str(region_key), 0.0)))
                    )
                    base_score = float(base_score + (0.14 * change_magnitude))
                    if str(region_key) in region_sudden_spike_keys_now:
                        base_score = float(base_score + 0.16 + (0.10 * change_delta))
                    if str(region_key) in simultaneous_reachable_set:
                        base_score = float(base_score + 0.16)
                    elif str(region_key) in simultaneous_unknown_set:
                        base_score = float(base_score + 0.08)
                    if str(region_key) == str(anchor_region_key):
                        base_score = float(base_score + 0.06)
                    target_scores_local[str(region_key)] = max(
                        float(target_scores_local.get(str(region_key), 0.0)),
                        float(max(0.0, min(1.0, base_score))),
                    )

            changed_bbox = getattr(causal_signature, "changed_bbox", None)
            frame_height = int(len(current_packet.frame))
            frame_width = int(len(current_packet.frame[0])) if frame_height > 0 else 0
            frame_area = int(max(1, frame_height * frame_width))
            bbox_values: tuple[int, int, int, int] | None = None
            if isinstance(changed_bbox, dict):
                min_x = int(changed_bbox.get("min_x", -1))
                max_x = int(changed_bbox.get("max_x", -1))
                min_y = int(changed_bbox.get("min_y", -1))
                max_y = int(changed_bbox.get("max_y", -1))
                bbox_values = (int(min_x), int(min_y), int(max_x), int(max_y))
            elif isinstance(changed_bbox, (tuple, list)) and len(changed_bbox) >= 4:
                bbox_values = (
                    int(changed_bbox[0]),
                    int(changed_bbox[1]),
                    int(changed_bbox[2]),
                    int(changed_bbox[3]),
                )
            if (
                bbox_values is not None
                and frame_height > 0
                and frame_width > 0
                and int(changed_pixels) < int(0.65 * float(frame_area))
            ):
                min_x, min_y, max_x, max_y = bbox_values
                if min_x >= 0 and max_x >= min_x and min_y >= 0 and max_y >= min_y:
                    min_rx = int(max(0, min(7, min_x // 8)))
                    max_rx = int(max(0, min(7, max_x // 8)))
                    min_ry = int(max(0, min(7, min_y // 8)))
                    max_ry = int(max(0, min(7, max_y // 8)))
                    bbox_bonus = float(min(0.22, float(changed_pixels) / 1024.0))
                    for ry in range(min_ry, max_ry + 1):
                        for rx in range(min_rx, max_rx + 1):
                            key = self._region_key_from_xy_v1(int(rx), int(ry))
                            if not _reachable_or_frontier_region_v1(str(key)):
                                continue
                            row_bias = 0.06 if int(ry) <= 2 else 0.0
                            target_scores_local[key] = max(
                                float(target_scores_local.get(key, 0.0)),
                                float(0.52 + bbox_bonus + row_bias),
                            )
            for region_key in coupled_region_keys:
                base_floor = float(coupled_floor)
                region_row = rows_by_region_key.get(str(region_key), {})
                region_visit_count = int(max(0, self._region_visit_counts.get(str(region_key), 0)))
                region_progress_rate = float(max(0.0, region_row.get("progress_rate", 0.0)))
                region_monotony_penalty = float(max(0.0, region_row.get("monotony_penalty", 0.0)))
                if not _reachable_or_frontier_region_v1(str(region_key)):
                    base_floor = float(min(base_floor, 0.58))
                if region_key == str(primary_coupled_region_key):
                    if orientation_misaligned:
                        base_floor = float(min(1.0, base_floor + 0.14))
                    if strong_event and str(source_region_key) == str(primary_coupled_region_key):
                        base_floor = float(min(1.0, base_floor + 0.10))
                elif region_key == str(secondary_coupled_region_key):
                    if strong_event and str(source_region_key) == str(primary_coupled_region_key):
                        base_floor = float(min(1.0, base_floor + 0.16))
                    else:
                        base_floor = float(min(1.0, base_floor + 0.06))
                elif strong_event and str(source_region_key) == str(primary_coupled_region_key):
                    base_floor = float(min(1.0, base_floor + 0.05))
                required_samples = int(_required_samples(str(region_key)))
                region_sample_count = int(max(0, target_sample_counts.get(str(region_key), 0)))
                if (
                    coupled_progress_locked
                    and region_progress_rate <= 0.0
                    and region_sample_count >= required_samples
                    and region_visit_count >= int(max(12, required_samples + 6))
                    and region_monotony_penalty >= 0.12
                ):
                    base_floor = float(min(base_floor, 0.52))
                target_scores_local[str(region_key)] = max(
                    float(target_scores_local.get(str(region_key), 0.0)),
                    float(base_floor),
                )
            return target_scores_local

        def _rank_queue(
            target_scores_raw: dict[str, float],
            *,
            anchor_region_key: str,
            completed_recent: set[str],
            sample_counts: dict[str, int],
        ) -> list[str]:
            current_target = str(state.get("current_target_region_key", "NA"))
            rows: list[
                tuple[int, int, int, int, int, int, int, int, float, float, int, int, str]
            ] = []
            for region_key, score in target_scores_raw.items():
                region_key = str(region_key)
                if self._parse_region_key_v1(region_key) is None:
                    continue
                if (
                    region_key in simultaneous_unreachable_set
                    and bool(reachability_graph_ready)
                ):
                    continue
                reachable_or_frontier = bool(_reachable_or_frontier_region_v1(str(region_key)))
                # Hard gate: unreachable regions do not enter high-info planning queue.
                if not reachable_or_frontier:
                    continue
                score_value = float(score)
                if score_value <= 0.0:
                    continue
                sample_count = int(max(0, sample_counts.get(region_key, 0)))
                required_samples = int(_required_samples(region_key))
                remaining_samples = int(max(0, int(required_samples) - sample_count))
                if remaining_samples <= 0 and region_key in completed_recent:
                    continue
                coupled_priority = 0 if region_key in coupled_region_keys else 1
                carry_priority = 0 if region_key == current_target else 1
                completed_priority = 1 if region_key in completed_recent else 0
                remaining_priority = 0 if remaining_samples > 0 else 1
                simultaneous_priority = 2
                if region_key in simultaneous_reachable_set:
                    simultaneous_priority = 0
                elif region_key in simultaneous_unknown_set:
                    simultaneous_priority = 1
                reachability_priority = 0 if reachable_or_frontier else 1
                high_value_priority = (
                    0 if score_value >= float(high_value_threshold) else 1
                )
                unsampled_bonus = 0.12 if sample_count <= 0 else 0.0
                adjusted_score = float(score_value + unsampled_bonus)
                dynamic_change_score = float(
                    max(
                        0.0,
                        min(
                            1.0,
                            max(
                                region_change_magnitude_now.get(region_key, 0.0),
                                region_change_magnitude_ema.get(region_key, 0.0),
                            ),
                        ),
                    )
                )
                sudden_priority_active = bool(
                    (
                        region_key in region_sudden_spike_priority_keys
                        or region_key in region_event_priority_keys
                    )
                    and (
                        dynamic_change_score >= 0.20
                        or score_value >= float(max(high_value_threshold, 0.62))
                    )
                )
                adjusted_score = float(adjusted_score + (0.18 * dynamic_change_score))
                if sudden_priority_active and reachable_or_frontier:
                    adjusted_score = float(adjusted_score + 0.36)
                if (not reachable_or_frontier) and region_key in coupled_region_keys:
                    adjusted_score = float(adjusted_score - 0.18)
                route_distance_graph = int(
                    self._region_route_distance_v1(
                        region_adjacency,
                        start_region_key=str(anchor_region_key),
                        goal_region_key=str(region_key),
                    )
                )
                if route_distance_graph >= 10**6:
                    continue
                rows.append(
                    (
                        0 if sudden_priority_active else 1,
                        int(simultaneous_priority),
                        int(reachability_priority),
                        int(coupled_priority),
                        int(remaining_priority),
                        int(high_value_priority),
                        int(carry_priority),
                        int(completed_priority),
                        float(-dynamic_change_score),
                        float(-adjusted_score),
                        int(route_distance_graph),
                        int(self._region_visit_counts.get(region_key, 0)),
                        region_key,
                    )
                )
            rows.sort()
            queue = [str(row[-1]) for row in rows]
            if not queue:
                return []
            sudden_head = [key for key in queue if key in region_sudden_spike_priority_keys]
            simultaneous_head = [
                key
                for key in queue
                if (
                    key not in set(sudden_head)
                    and (key in simultaneous_reachable_set or key in simultaneous_unknown_set)
                )
            ]
            coupled_head = [
                key
                for key in queue
                if key in coupled_region_keys
                and key not in set(sudden_head)
                and key not in simultaneous_head
            ]
            non_coupled = [
                key
                for key in queue
                if key not in set(sudden_head)
                and key not in coupled_region_keys
                and key not in simultaneous_head
            ]
            if sudden_head:
                queue = sudden_head + simultaneous_head + coupled_head + non_coupled
            elif simultaneous_head:
                queue = simultaneous_head + coupled_head + non_coupled
            elif coupled_progress_locked and coupled_head and non_coupled:
                high_non_coupled = [
                    str(key)
                    for key in non_coupled
                    if float(target_scores_raw.get(str(key), 0.0))
                    >= float(max(0.58, high_value_threshold))
                ]
                reordered: list[str] = []
                reordered.append(str(coupled_head[0]))
                if high_non_coupled:
                    first_high = str(high_non_coupled[0])
                    if first_high not in reordered:
                        reordered.append(first_high)
                for key in list(coupled_head[1:]) + list(high_non_coupled[1:]) + list(non_coupled):
                    skey = str(key)
                    if skey not in reordered:
                        reordered.append(skey)
                queue = list(reordered)
            else:
                queue = coupled_head + non_coupled
            max_targets = int(max(self.high_info_focus_max_targets, len(coupled_head)))
            if max_targets > 0:
                queue = queue[:max_targets]
            return list(queue)

        def _build_interaction_chain(
            queue_keys: list[str],
            *,
            source_key: str,
        ) -> list[str]:
            normalized: list[str] = []
            for region_key in queue_keys:
                key = str(region_key)
                if self._parse_region_key_v1(key) is None:
                    continue
                if key in normalized:
                    continue
                normalized.append(str(key))
            preferred = [key for key in normalized if str(key) != str(source_key)]
            if preferred:
                normalized = list(preferred)
            sudden_head = [key for key in normalized if key in region_sudden_spike_priority_keys]
            if sudden_head:
                sudden_set = set(str(v) for v in sudden_head)
                normalized = list(sudden_head) + [
                    str(v) for v in normalized if str(v) not in sudden_set
                ]
            if coupled_progress_locked and not sudden_head:
                coupled_only = [key for key in normalized if key in coupled_region_keys]
                if len(coupled_only) >= 2:
                    normalized = [str(coupled_only[0]), str(coupled_only[1])]
                elif len(coupled_only) == 1:
                    spillover = [key for key in normalized if key not in coupled_only]
                    normalized = [str(coupled_only[0])]
                    if spillover:
                        normalized.append(str(spillover[0]))
            max_region_count = int(max(2, min(5, max(2, self.high_info_focus_max_targets))))
            normalized = normalized[:max_region_count]
            if not normalized:
                return []
            if len(normalized) == 1:
                return [str(normalized[0]), str(normalized[0])]
            chain: list[str] = []
            # Pairwise mutual coverage among top high-info regions.
            for idx, left in enumerate(normalized):
                for right in normalized[idx + 1 :]:
                    chain.extend([str(left), str(right), str(right), str(left)])
            deduped: list[str] = []
            for key in chain:
                if not deduped or str(deduped[-1]) != str(key):
                    deduped.append(str(key))
            max_chain_len = int(max(4, min(24, 4 * max_region_count)))
            return deduped[:max_chain_len]

        if should_trigger:
            was_active_before_trigger = bool(state.get("active", False))
            new_targets = _collect_hot_targets(str(source_region_key))
            merged_scores: dict[str, float] = {}
            for region_key, old_score in score_memory.items():
                sample_count = int(max(0, target_sample_counts.get(str(region_key), 0)))
                required_samples = int(_required_samples(str(region_key)))
                retention_floor = (
                    0.22 if sample_count < int(required_samples) else 0.0
                )
                merged_scores[str(region_key)] = float(max(0.0, 0.86 * float(old_score)))
                if retention_floor > 0.0:
                    merged_scores[str(region_key)] = float(
                        max(float(merged_scores[str(region_key)]), float(retention_floor))
                    )
            for region_key, new_score in new_targets.items():
                merged_scores[str(region_key)] = max(
                    float(merged_scores.get(str(region_key), 0.0)),
                    float(new_score),
                )
            if novelty_event_detected:
                novelty_protocol_active = True
                novelty_source_region_key = str(source_region_key)
                novelty_related_region_keys = list(active_novelty_related_targets)
                last_novelty_signature = str(novelty_signature)
                novelty_trigger_count = int(novelty_trigger_count + 1)
                novelty_last_action_counter = int(current_counter)
                _record_novelty_pattern_v1(
                    str(novelty_signature),
                    source_region_key_for_stats=str(source_region_key),
                    related_region_keys_for_stats=list(active_novelty_related_targets),
                    changed_pixels_for_stats=int(changed_pixels),
                    simultaneous_pixels_for_stats=int(
                        max(simultaneous_pixels_now, simultaneous_changed_total_pixels)
                    ),
                    progress_hit=bool(level_delta_now > 0),
                )
                baseline_counts: dict[str, int] = {}
                source_key = str(source_region_key)
                source_sample_count = int(max(0, target_sample_counts.get(str(source_key), 0)))
                source_sample_count = int(max(1, source_sample_count))
                target_sample_counts[str(source_key)] = int(
                    max(int(target_sample_counts.get(str(source_key), 0)), int(source_sample_count))
                )
                baseline_counts[str(source_key)] = int(source_sample_count)
                # Verify-first: keep the source satisfied initially so the next pending target
                # becomes the remote region. Retriggers are scheduled later if verification fails.
                target_required_samples[str(source_key)] = 1
                merged_scores[str(source_region_key)] = max(
                    float(merged_scores.get(str(source_region_key), 0.0)),
                    0.90,
                )
                for idx, region_key in enumerate(active_novelty_related_targets):
                    related_key = str(region_key)
                    related_sample_count = int(max(0, target_sample_counts.get(str(related_key), 0)))
                    baseline_counts[str(related_key)] = int(related_sample_count)
                    related_required = int(
                        max(
                            _required_samples(str(related_key)),
                            related_sample_count + int(1 + self.high_info_novelty_related_extra_samples),
                        )
                    )
                    target_required_samples[str(related_key)] = int(
                        max(
                            target_required_samples.get(str(related_key), 0),
                            related_required,
                        )
                    )
                    novelty_related_floor = float(max(0.82, 0.94 - (0.06 * float(idx))))
                    merged_scores[str(related_key)] = max(
                        float(merged_scores.get(str(related_key), 0.0)),
                        float(novelty_related_floor),
                    )
                state["novelty_baseline_sample_counts"] = dict(baseline_counts)
                # Prioritize chasing the remote effect region(s) immediately.
                novelty_chain_keys = [str(v) for v in active_novelty_related_targets]
                if len(novelty_chain_keys) >= 1:
                    priority_subqueue_active = True
                    priority_subqueue_keys = list(novelty_chain_keys[:16])
                    interaction_chain_active = True
                    interaction_target_chain = list(novelty_chain_keys[:16])
                    interaction_target_index = 0
                    state["interaction_last_status"] = "novelty_armed"
            focus_lock_active = bool(
                simultaneous_focus_reset
                or (
                    was_active_before_trigger
                    and len(coupled_region_keys) >= 2
                    and int(changed_pixels) >= int(self.high_info_simultaneous_min_total_pixels)
                )
            )
            if novelty_event_detected:
                focus_lock_active = True
            focus_lock_keys: set[str] = set()
            if simultaneous_changed_set:
                ordered_simultaneous = [
                    str(v)
                    for v in simultaneous_changed_region_keys
                    if self._parse_region_key_v1(str(v)) is not None
                ]
                for idx, region_key in enumerate(ordered_simultaneous):
                    if (
                        str(region_key) in simultaneous_unreachable_set
                        and bool(reachability_graph_ready)
                    ):
                        continue
                    base_score = float(max(0.64, 0.94 - (0.08 * float(idx))))
                    if str(region_key) in simultaneous_reachable_set:
                        base_score = float(min(1.0, base_score + 0.08))
                    merged_scores[str(region_key)] = max(
                        float(merged_scores.get(str(region_key), 0.0)),
                        float(base_score),
                    )
            if focus_lock_active:
                if simultaneous_focus_reset and simultaneous_focus_candidate_set:
                    focus_lock_keys = set(simultaneous_focus_candidate_set)
                    if self._parse_region_key_v1(str(source_region_key)) is not None:
                        focus_lock_keys.add(str(source_region_key))
                    for idx, region_key in enumerate(simultaneous_focus_candidates):
                        focus_boost = float(max(0.74, 0.98 - (0.08 * float(idx))))
                        merged_scores[str(region_key)] = max(
                            float(merged_scores.get(str(region_key), 0.0)),
                            float(focus_boost),
                        )
                else:
                    focus_lock_keys = {
                        str(v)
                        for v in simultaneous_changed_region_keys
                        if self._parse_region_key_v1(str(v)) is not None
                    }
                    for idx, region_key in enumerate(sorted(coupled_region_keys)):
                        if self._parse_region_key_v1(str(region_key)) is None:
                            continue
                        focus_lock_keys.add(str(region_key))
                        coupled_floor_boost = float(max(0.72, 0.94 - (0.10 * float(idx))))
                        merged_scores[str(region_key)] = max(
                            float(merged_scores.get(str(region_key), 0.0)),
                            float(coupled_floor_boost),
                        )
                    if self._parse_region_key_v1(str(source_region_key)) is not None:
                        focus_lock_keys.add(str(source_region_key))
                if focus_lock_keys and (not simultaneous_focus_reset):
                    merged_scores = {
                        str(region_key): float(score)
                        for (region_key, score) in merged_scores.items()
                        if str(region_key) in focus_lock_keys
                    }
            min_keep = float(max(0.08, 0.35 * float(self.high_info_focus_min_trigger_score)))
            mandatory_keep = {
                str(region_key)
                for (region_key, sample_count) in target_sample_counts.items()
                if self._parse_region_key_v1(str(region_key)) is not None
                and int(sample_count) < int(_required_samples(str(region_key)))
                and (
                    (not focus_lock_active)
                    or str(region_key) in focus_lock_keys
                )
            }
            merged_scores = {
                str(region_key): float(score)
                for (region_key, score) in merged_scores.items()
                if self._parse_region_key_v1(str(region_key)) is not None
                and (
                    float(score) >= min_keep
                    or str(region_key) in mandatory_keep
                )
                and not (
                    str(region_key) in simultaneous_unreachable_set
                    and bool(reachability_graph_ready)
                )
            }
            for region_key in merged_scores.keys():
                target_sample_counts.setdefault(str(region_key), 0)
                target_required_samples[str(region_key)] = int(
                    max(
                        _required_samples(str(region_key)),
                        target_required_samples.get(str(region_key), 0),
                    )
                )
            if simultaneous_focus_reset and priority_subqueue_keys:
                for region_key in priority_subqueue_keys:
                    target_sample_counts.setdefault(str(region_key), 0)
                    target_required_samples[str(region_key)] = int(
                        max(
                            target_required_samples.get(str(region_key), 0),
                            _required_samples(str(region_key)),
                            2,
                        )
                    )
            score_memory = dict(merged_scores)
            state["active"] = bool(score_memory)
            state["stage"] = "seek"
            state["window_steps"] = int(self.high_info_focus_window_steps)
            state["steps_remaining"] = int(self.high_info_focus_window_steps)
            state["trigger_count"] = int(state.get("trigger_count", 0) + 1)
            state["source_region_key"] = str(source_region_key)
            state["trigger_event_type"] = str(obs_change_type)
            state["trigger_action_counter"] = int(current_counter)
            state["deadline_action_counter"] = int(
                max(
                    int(state.get("deadline_action_counter", -1)),
                    int(current_counter + int(self.high_info_focus_window_steps)),
                )
            )
            # If a novelty/remote-effect event was detected, the default window (often small)
            # is not enough to execute the required "source<->remote" travel + resampling.
            # Extend the deadline dynamically based on region distance.
            if novelty_event_detected or remote_effect_detected:
                related = (
                    list(active_novelty_related_targets)
                    if active_novelty_related_targets
                    else [
                        str(k)
                        for k in simultaneous_changed_now
                        if self._parse_region_key_v1(str(k)) is not None
                        and str(k) != str(source_region_key)
                    ]
                )
                max_dist = 0
                for key in related[:6]:
                    dist = int(self._region_distance_v1(str(source_region_key), str(key)))
                    if dist > max_dist:
                        max_dist = int(dist)
                desired_window = int(max(int(self.high_info_focus_window_steps), 6 * (max_dist + 1)))
                desired_window = int(min(96, max(16, desired_window)))
                state["window_steps"] = int(max(int(state.get("window_steps", 0)), desired_window))
                state["deadline_action_counter"] = int(
                    max(
                        int(state.get("deadline_action_counter", -1)),
                        int(current_counter + desired_window),
                    )
                )
            state["primary_coupled_region_key"] = str(primary_coupled_region_key)
            state["secondary_coupled_region_key"] = str(secondary_coupled_region_key)
            state["simultaneous_changed_region_keys"] = [
                str(v) for v in simultaneous_changed_region_keys[:16]
            ]
            state["simultaneous_reachable_region_keys"] = [
                str(v) for v in simultaneous_reachable_region_keys[:16]
            ]
            state["simultaneous_unknown_region_keys"] = [
                str(v) for v in simultaneous_unknown_region_keys[:16]
            ]
            state["simultaneous_unreachable_region_keys"] = [
                str(v) for v in simultaneous_unreachable_region_keys[:16]
            ]
            state["simultaneous_anchor_region_key"] = str(simultaneous_anchor_region_key)
            state["simultaneous_changed_total_pixels"] = int(
                max(0, simultaneous_changed_total_pixels)
            )
            state["region_recent_change_pixels"] = {
                str(k): int(max(0, v))
                for (k, v) in region_recent_change_pixels.items()
                if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
            }
            state["region_change_magnitude"] = {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in region_change_magnitude_effective.items()
                if self._parse_region_key_v1(str(k)) is not None
            }
            state["region_change_magnitude_ema"] = {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in region_change_magnitude_ema.items()
                if self._parse_region_key_v1(str(k)) is not None
            }
            state["region_change_delta"] = {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in region_change_delta_now.items()
                if self._parse_region_key_v1(str(k)) is not None
            }
            state["region_sudden_spike_keys"] = [
                str(v) for v in sorted(region_sudden_spike_priority_keys)[:16]
            ]
            state["last_trigger_changed_pixels"] = int(max(0, changed_pixels))
            state["last_trigger_changed_region_diff_map"] = {
                str(k): int(max(0, v))
                for (k, v) in region_recent_change_pixels.items()
                if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
            }
            state["cross_region_key"] = str(primary_coupled_region_key)
            state["gate_region_key"] = str(secondary_coupled_region_key)
            state["coupled_region_keys"] = [
                str(v) for v in sorted(coupled_region_keys)
            ]
            seed_queue = _rank_queue(
                score_memory,
                anchor_region_key=str(source_region_key),
                completed_recent=set(),
                sample_counts=target_sample_counts,
            )
            if priority_subqueue_active and priority_subqueue_keys:
                priority_seed_queue = [
                    str(v) for v in seed_queue if str(v) in set(priority_subqueue_keys)
                ]
                if len(priority_seed_queue) >= 2:
                    interaction_chain = _build_interaction_chain(
                        list(priority_seed_queue),
                        source_key=str(source_region_key),
                    )
                    priority_subqueue_keys = list(priority_seed_queue)
                else:
                    priority_subqueue_active = False
                    priority_subqueue_keys = []
                    interaction_chain = _build_interaction_chain(
                        list(seed_queue),
                        source_key=str(source_region_key),
                    )
            else:
                interaction_chain = _build_interaction_chain(
                    list(seed_queue),
                    source_key=str(source_region_key),
                )
            primary_seed_target = str(seed_queue[0]) if seed_queue else "NA"
            if self._parse_region_key_v1(str(primary_seed_target)) is not None:
                _arm_chain_lock(
                    str(primary_seed_target),
                    "focus_commit_armed",
                    focus_commit=True,
                )
            interaction_chain_active = bool(len(interaction_chain) >= 2)
            interaction_target_chain = list(interaction_chain)
            interaction_target_index = 0
            state["interaction_chain_active"] = bool(interaction_chain_active)
            if interaction_chain_active:
                state["interaction_chain_generation"] = int(
                    max(0, state.get("interaction_chain_generation", 0)) + 1
                )
                state["interaction_target_chain"] = list(interaction_chain)
                state["interaction_target_index"] = 0
                state["interaction_last_status"] = "armed"
            else:
                state["interaction_target_chain"] = []
                state["interaction_target_index"] = 0
                state["interaction_last_status"] = "insufficient_chain"
            state["chain_lock_active"] = bool(chain_lock_active)
            state["chain_lock_window_steps"] = int(chain_lock_window_steps)
            state["chain_lock_steps_remaining"] = int(max(0, chain_lock_steps_remaining))
            state["chain_lock_target_region_key"] = str(chain_lock_target_region_key)
            state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
            state["target_commit_active"] = bool(target_commit_active)
            state["target_commit_window_steps"] = int(target_commit_window_steps)
            state["target_commit_miss_limit"] = int(target_commit_miss_limit)
            state["chain_lock_last_status"] = str(chain_lock_last_status)
            state["priority_subqueue_active"] = bool(priority_subqueue_active)
            state["priority_subqueue_keys"] = [str(v) for v in priority_subqueue_keys[:16]]
            state["pending_region_queue"] = []
            state["pending_region_scores"] = {}
            state["last_status"] = (
                "retriggered_chain_focus_reset"
                if simultaneous_focus_reset
                else ("retriggered_chain" if was_active_before_trigger else "triggered")
            )

        if not bool(state.get("active", False)):
            # Evidence-driven rearm: if we have a pending high-info queue (e.g., collected during
            # prepass or deferred under hard locks), start a new focus window from that queue.
            if coupled_progress_locked and pending_region_queue and pending_region_scores:
                rearm_scores: dict[str, float] = {}
                for region_key in pending_region_queue[:16]:
                    key = str(region_key)
                    if self._parse_region_key_v1(key) is None:
                        continue
                    score = float(max(0.0, pending_region_scores.get(key, 0.0)))
                    if score <= 0.0:
                        continue
                    rearm_scores[key] = float(score)
                    target_sample_counts.setdefault(str(key), 0)
                    target_required_samples[str(key)] = int(_required_samples(str(key)))
                if rearm_scores:
                    rearm_anchor = str(nav_region_key)
                    if self._parse_region_key_v1(rearm_anchor) is None:
                        rearm_anchor = str(source_region_key)
                    rearm_queue = _rank_queue(
                        rearm_scores,
                        anchor_region_key=str(rearm_anchor),
                        completed_recent=set(),
                        sample_counts=target_sample_counts,
                    )
                    if rearm_queue:
                        rearm_chain = _build_interaction_chain(
                            list(rearm_queue),
                            source_key=str(rearm_anchor),
                        )
                        interaction_chain_active = bool(len(rearm_chain) >= 2)
                        state["active"] = True
                        state["stage"] = "seek"
                        state["window_steps"] = int(self.high_info_focus_window_steps)
                        state["trigger_action_counter"] = int(current_counter)
                        state["deadline_action_counter"] = int(
                            max(
                                int(state.get("deadline_action_counter", -1)),
                                int(current_counter + int(self.high_info_focus_window_steps)),
                            )
                        )
                        state["steps_remaining"] = int(
                            max(0, int(state["deadline_action_counter"]) - int(current_counter))
                        )
                        state["source_region_key"] = str(rearm_anchor)
                        state["trigger_event_type"] = "PENDING_REARM"
                        state["target_miss_streak"] = 0
                        state["current_target_region_key"] = str(rearm_queue[0])
                        state["target_region_queue"] = list(rearm_queue)
                        state["target_region_scores"] = dict(rearm_scores)
                        state["target_sample_counts"] = dict(target_sample_counts)
                        state["target_required_samples"] = dict(target_required_samples)
                        state["completed_target_regions"] = list(completed_regions[-16:])
                        # Arm a commit lock on the queue head to avoid immediate retarget churn.
                        if self._parse_region_key_v1(str(rearm_queue[0])) is not None:
                            _arm_chain_lock(str(rearm_queue[0]), "pending_rearm", focus_commit=True)
                        state["interaction_chain_active"] = bool(interaction_chain_active)
                        state["interaction_target_chain"] = list(rearm_chain[:16]) if interaction_chain_active else []
                        state["interaction_target_index"] = 0
                        state["interaction_last_status"] = (
                            "pending_rearm_armed" if interaction_chain_active else "pending_rearm_singleton"
                        )
                        state["chain_lock_active"] = bool(chain_lock_active)
                        state["chain_lock_window_steps"] = int(chain_lock_window_steps)
                        state["chain_lock_steps_remaining"] = int(
                            max(0, chain_lock_steps_remaining)
                        )
                        state["chain_lock_target_region_key"] = str(chain_lock_target_region_key)
                        state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
                        state["target_commit_active"] = bool(target_commit_active)
                        state["target_commit_window_steps"] = int(target_commit_window_steps)
                        state["target_commit_miss_limit"] = int(target_commit_miss_limit)
                        state["chain_lock_last_status"] = str(chain_lock_last_status)
                        state["priority_subqueue_active"] = False
                        state["priority_subqueue_keys"] = []
                        state["pending_region_queue"] = []
                        state["pending_region_scores"] = {}
                        _flush_inner_loop_state_to_state()
                        _flush_novelty_regions_to_state()
                        state["last_status"] = "pending_rearm"
                        return

            if self.high_info_idle_rearm_enabled and coupled_progress_locked:
                rearm_scores: dict[str, float] = {}
                if self._parse_region_key_v1(str(primary_coupled_region_key)) is not None:
                    rearm_scores[str(primary_coupled_region_key)] = float(
                        max(float(coupled_floor), 0.62)
                    )
                if (
                    self._parse_region_key_v1(str(secondary_coupled_region_key)) is not None
                    and str(secondary_coupled_region_key) != str(primary_coupled_region_key)
                ):
                    rearm_scores[str(secondary_coupled_region_key)] = float(
                        max(float(coupled_floor - 0.08), 0.54)
                    )
                if rearm_scores:
                    rearm_anchor = str(nav_region_key)
                    if self._parse_region_key_v1(rearm_anchor) is None:
                        rearm_anchor = str(source_region_key)
                    for region_key in rearm_scores.keys():
                        target_sample_counts.setdefault(str(region_key), 0)
                        target_required_samples[str(region_key)] = int(
                            _required_samples(str(region_key))
                        )
                    rearm_queue = _rank_queue(
                        rearm_scores,
                        anchor_region_key=str(rearm_anchor),
                        completed_recent=set(),
                        sample_counts=target_sample_counts,
                    )
                    if rearm_queue:
                        rearm_chain = _build_interaction_chain(
                            list(rearm_queue),
                            source_key=str(rearm_anchor),
                        )
                        interaction_chain_active = bool(len(rearm_chain) >= 2)
                        state["active"] = True
                        state["stage"] = "seek"
                        state["window_steps"] = int(self.high_info_focus_window_steps)
                        state["steps_remaining"] = int(
                            max(2, int(self.high_info_focus_window_steps // 2))
                        )
                        state["trigger_action_counter"] = int(current_counter)
                        state["deadline_action_counter"] = int(
                            max(
                                int(state.get("deadline_action_counter", -1)),
                                int(current_counter + int(self.high_info_focus_window_steps)),
                            )
                        )
                        state["source_region_key"] = str(rearm_anchor)
                        state["trigger_event_type"] = "IDLE_REARM"
                        state["target_miss_streak"] = 0
                        state["current_target_region_key"] = str(rearm_queue[0])
                        state["target_region_queue"] = list(rearm_queue)
                        state["target_region_scores"] = dict(rearm_scores)
                        state["target_sample_counts"] = dict(target_sample_counts)
                        state["target_required_samples"] = dict(target_required_samples)
                        state["completed_target_regions"] = list(completed_regions[-16:])
                        state["primary_coupled_region_key"] = str(primary_coupled_region_key)
                        state["secondary_coupled_region_key"] = str(secondary_coupled_region_key)
                        state["simultaneous_changed_region_keys"] = [
                            str(v) for v in simultaneous_changed_region_keys[:16]
                        ]
                        state["simultaneous_reachable_region_keys"] = [
                            str(v) for v in simultaneous_reachable_region_keys[:16]
                        ]
                        state["simultaneous_unknown_region_keys"] = [
                            str(v) for v in simultaneous_unknown_region_keys[:16]
                        ]
                        state["simultaneous_unreachable_region_keys"] = [
                            str(v) for v in simultaneous_unreachable_region_keys[:16]
                        ]
                        state["simultaneous_anchor_region_key"] = str(
                            simultaneous_anchor_region_key
                        )
                        state["simultaneous_changed_total_pixels"] = int(
                            max(0, simultaneous_changed_total_pixels)
                        )
                        state["region_recent_change_pixels"] = {
                            str(k): int(max(0, v))
                            for (k, v) in region_recent_change_pixels.items()
                            if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
                        }
                        state["region_change_magnitude"] = {
                            str(k): float(max(0.0, min(1.0, v)))
                            for (k, v) in region_change_magnitude_effective.items()
                            if self._parse_region_key_v1(str(k)) is not None
                        }
                        state["region_change_magnitude_ema"] = {
                            str(k): float(max(0.0, min(1.0, v)))
                            for (k, v) in region_change_magnitude_ema.items()
                            if self._parse_region_key_v1(str(k)) is not None
                        }
                        state["region_change_delta"] = {
                            str(k): float(max(0.0, min(1.0, v)))
                            for (k, v) in region_change_delta_now.items()
                            if self._parse_region_key_v1(str(k)) is not None
                        }
                        state["region_sudden_spike_keys"] = [
                            str(v) for v in sorted(region_sudden_spike_priority_keys)[:16]
                        ]
                        state["last_trigger_changed_pixels"] = int(max(0, changed_pixels))
                        state["last_trigger_changed_region_diff_map"] = {
                            str(k): int(max(0, v))
                            for (k, v) in region_recent_change_pixels.items()
                            if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
                        }
                        state["cross_region_key"] = str(primary_coupled_region_key)
                        state["gate_region_key"] = str(secondary_coupled_region_key)
                        state["coupled_region_keys"] = [
                            str(v) for v in sorted(coupled_region_keys)
                        ]
                        state["interaction_chain_active"] = bool(interaction_chain_active)
                        state["interaction_target_chain"] = list(rearm_chain[:16])
                        state["interaction_target_index"] = 0
                        state["interaction_last_status"] = (
                            "rearm_tracking"
                            if interaction_chain_active
                            else "rearm_single"
                        )
                        _arm_chain_lock(
                            str(rearm_queue[0]),
                            "rearm_focus_commit",
                            focus_commit=True,
                        )
                        state["chain_lock_active"] = bool(chain_lock_active)
                        state["chain_lock_window_steps"] = int(chain_lock_window_steps)
                        state["chain_lock_steps_remaining"] = int(
                            max(0, chain_lock_steps_remaining)
                        )
                        state["chain_lock_target_region_key"] = str(
                            chain_lock_target_region_key
                        )
                        state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
                        state["target_commit_active"] = bool(target_commit_active)
                        state["target_commit_window_steps"] = int(target_commit_window_steps)
                        state["target_commit_miss_limit"] = int(target_commit_miss_limit)
                        state["chain_lock_last_status"] = str(chain_lock_last_status)
                        state["priority_subqueue_active"] = False
                        state["priority_subqueue_keys"] = []
                        state["pending_region_queue"] = []
                        state["pending_region_scores"] = {}
                        _flush_inner_loop_state_to_state()
                        _flush_novelty_regions_to_state()
                        state["last_status"] = "idle_rearm"
                        return
            state["target_region_scores"] = dict(score_memory)
            state["target_miss_streak"] = 0
            state["completed_target_regions"] = list(completed_regions[-16:])
            state["target_sample_counts"] = dict(target_sample_counts)
            state["target_required_samples"] = dict(target_required_samples)
            state["primary_coupled_region_key"] = str(primary_coupled_region_key)
            state["secondary_coupled_region_key"] = str(secondary_coupled_region_key)
            state["simultaneous_changed_region_keys"] = [
                str(v) for v in simultaneous_changed_region_keys[:16]
            ]
            state["simultaneous_reachable_region_keys"] = [
                str(v) for v in simultaneous_reachable_region_keys[:16]
            ]
            state["simultaneous_unknown_region_keys"] = [
                str(v) for v in simultaneous_unknown_region_keys[:16]
            ]
            state["simultaneous_unreachable_region_keys"] = [
                str(v) for v in simultaneous_unreachable_region_keys[:16]
            ]
            state["simultaneous_anchor_region_key"] = str(simultaneous_anchor_region_key)
            state["simultaneous_changed_total_pixels"] = int(
                max(0, simultaneous_changed_total_pixels)
            )
            state["region_recent_change_pixels"] = {
                str(k): int(max(0, v))
                for (k, v) in region_recent_change_pixels.items()
                if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
            }
            state["region_change_magnitude"] = {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in region_change_magnitude_effective.items()
                if self._parse_region_key_v1(str(k)) is not None
            }
            state["region_change_magnitude_ema"] = {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in region_change_magnitude_ema.items()
                if self._parse_region_key_v1(str(k)) is not None
            }
            state["region_change_delta"] = {
                str(k): float(max(0.0, min(1.0, v)))
                for (k, v) in region_change_delta_now.items()
                if self._parse_region_key_v1(str(k)) is not None
            }
            state["region_sudden_spike_keys"] = [
                str(v) for v in sorted(region_sudden_spike_priority_keys)[:16]
            ]
            state["last_trigger_changed_pixels"] = int(max(0, changed_pixels))
            state["last_trigger_changed_region_diff_map"] = {
                str(k): int(max(0, v))
                for (k, v) in region_recent_change_pixels.items()
                if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
            }
            state["cross_region_key"] = str(primary_coupled_region_key)
            state["gate_region_key"] = str(secondary_coupled_region_key)
            state["coupled_region_keys"] = [str(v) for v in sorted(coupled_region_keys)]
            state["interaction_chain_active"] = bool(interaction_chain_active)
            state["interaction_target_chain"] = list(interaction_target_chain[:16])
            state["interaction_target_index"] = int(max(0, interaction_target_index))
            _disable_chain_lock("inactive")
            state["chain_lock_active"] = bool(chain_lock_active)
            state["chain_lock_window_steps"] = int(chain_lock_window_steps)
            state["chain_lock_steps_remaining"] = int(max(0, chain_lock_steps_remaining))
            state["chain_lock_target_region_key"] = str(chain_lock_target_region_key)
            state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
            state["target_commit_active"] = bool(target_commit_active)
            state["target_commit_window_steps"] = int(target_commit_window_steps)
            state["target_commit_miss_limit"] = int(target_commit_miss_limit)
            state["chain_lock_last_status"] = str(chain_lock_last_status)
            state["priority_subqueue_active"] = bool(priority_subqueue_active)
            state["priority_subqueue_keys"] = [str(v) for v in priority_subqueue_keys[:16]]
            state["pending_region_queue"] = []
            state["pending_region_scores"] = {}
            _flush_inner_loop_state_to_state()
            _flush_novelty_regions_to_state()
            return

        if not score_memory:
            score_memory = _collect_hot_targets(str(source_region_key))
        if novelty_protocol_active:
            protocol_order = [
                str(v)
                for v in [str(novelty_source_region_key)] + list(novelty_related_region_keys)
                if self._parse_region_key_v1(str(v)) is not None
            ]
            protocol_pending: list[str] = []
            for idx, region_key in enumerate(protocol_order):
                sample_count = int(max(0, target_sample_counts.get(str(region_key), 0)))
                required_samples = int(
                    max(
                        1,
                        target_required_samples.get(str(region_key), _required_samples(str(region_key))),
                    )
                )
                if sample_count >= required_samples:
                    continue
                protocol_pending.append(str(region_key))
                protocol_floor = float(max(0.74, 0.95 - (0.08 * float(idx))))
                score_memory[str(region_key)] = max(
                    float(score_memory.get(str(region_key), 0.0)),
                    float(protocol_floor),
                )
            if protocol_pending:
                _activate_inner_loop_v1(
                    [str(v) for v in protocol_pending[:16]],
                    "novelty_protocol",
                )
                state["last_status"] = "inner_loop_armed"
            else:
                # If we didn't make progress, schedule a retrigger+verify cycle rather than
                # immediately disabling the protocol.
                progress_hit = bool(level_delta_now > 0 or str(obs_change_type) == "METADATA_PROGRESS_CHANGE")
                if progress_hit:
                    novelty_protocol_active = False
                    novelty_source_region_key = "NA"
                    novelty_related_region_keys = []
                    inner_loop_active = False
                    inner_loop_reason = "NA"
                    inner_loop_queue = []
                else:
                    src_key = str(novelty_source_region_key)
                    if self._parse_region_key_v1(src_key) is None:
                        novelty_protocol_active = False
                        novelty_source_region_key = "NA"
                        novelty_related_region_keys = []
                        inner_loop_active = False
                        inner_loop_reason = "NA"
                        inner_loop_queue = []
                    else:
                        src_count = int(max(0, target_sample_counts.get(src_key, 0)))
                        target_required_samples[src_key] = int(
                            max(
                                int(target_required_samples.get(src_key, 0)),
                                int(src_count + 1),
                            )
                        )
                        score_memory[src_key] = max(float(score_memory.get(src_key, 0.0)), 0.92)
                        for region_key in list(novelty_related_region_keys):
                            key = str(region_key)
                            if self._parse_region_key_v1(key) is None:
                                continue
                            related_count = int(max(0, target_sample_counts.get(key, 0)))
                            target_required_samples[key] = int(
                                max(
                                    int(target_required_samples.get(key, 0)),
                                    int(related_count + 1 + self.high_info_novelty_related_extra_samples),
                                )
                            )
                            score_memory[key] = max(float(score_memory.get(key, 0.0)), 0.88)
                        _activate_inner_loop_v1(
                            [str(src_key)] + [str(v) for v in novelty_related_region_keys],
                            "novelty_cycle",
                        )
                        _enqueue_pending_regions(
                            [str(src_key)] + [str(v) for v in novelty_related_region_keys],
                            score_hint={str(k): float(v) for (k, v) in score_memory.items()},
                        )
                        state["last_status"] = "novelty_cycle_extend"
        completed_recent = set(str(v) for v in completed_regions[-8:])
        anchor_key = str(nav_region_key)
        if self._parse_region_key_v1(anchor_key) is None:
            anchor_key = str(source_region_key)
        queue = _rank_queue(
            score_memory,
            anchor_region_key=str(anchor_key),
            completed_recent=completed_recent,
            sample_counts=target_sample_counts,
        )

        def _inner_loop_hard_active_v1() -> bool:
            return bool(
                inner_loop_active
                and inner_loop_queue
                and self._parse_region_key_v1(str(inner_loop_queue[0])) is not None
            )

        if _inner_loop_hard_active_v1():
            inner_lock_key = str(inner_loop_queue[0])
            # Keep an isolated target stream for the active inner loop.
            queue = [str(inner_lock_key)] + [
                str(v) for v in queue if str(v) != str(inner_lock_key)
            ]
            state["last_status"] = "inner_loop_queue_isolated"

        def _commit_lock_hard_active_v1() -> bool:
            if _inner_loop_hard_active_v1():
                return False
            return bool(
                target_commit_active
                and chain_lock_active
                and chain_lock_steps_remaining > 0
                and self._parse_region_key_v1(str(chain_lock_target_region_key)) is not None
            )

        def _force_hard_lock_queue_v1(status: str) -> bool:
            nonlocal queue
            nonlocal target_miss_streak
            nonlocal priority_subqueue_active
            nonlocal priority_subqueue_keys
            nonlocal interaction_chain_active
            nonlocal interaction_target_chain
            nonlocal interaction_target_index
            if inner_loop_active and inner_loop_queue:
                lock_key = str(inner_loop_queue[0])
                if self._parse_region_key_v1(lock_key) is None:
                    return False
                # NOTE: inner-loop hard lock is armed on activation; do not re-arm here (prevents miss/timeout from working).
                deferred = [str(v) for v in queue if str(v) != str(lock_key)]
                if deferred:
                    _enqueue_pending_regions(
                        deferred,
                        score_hint={str(k): float(v) for (k, v) in score_memory.items()},
                    )
                queue = [str(lock_key)]
                priority_subqueue_active = True
                priority_subqueue_keys = [str(lock_key)]
                interaction_chain_active = False
                interaction_target_chain = []
                interaction_target_index = 0
                state["last_status"] = f"{str(status)}_inner_loop"
                return True
            if not _commit_lock_hard_active_v1():
                return False
            lock_key = str(chain_lock_target_region_key)
            deferred = [str(v) for v in queue if str(v) != str(lock_key)]
            if deferred:
                _enqueue_pending_regions(
                    deferred,
                    score_hint={str(k): float(v) for (k, v) in score_memory.items()},
                )
            queue = [str(lock_key)]
            target_miss_streak = 0
            state["last_status"] = str(status)
            return True

        if (
            (not _inner_loop_hard_active_v1())
            and chain_lock_active
            and self._parse_region_key_v1(str(chain_lock_target_region_key)) is not None
            and str(chain_lock_target_region_key) not in set(str(v) for v in queue)
        ):
            queue = [str(chain_lock_target_region_key)] + [str(v) for v in queue]
            state["last_status"] = "chain_lock_reinjected_target"
        _force_hard_lock_queue_v1("target_commit_hard_lock")
        if queue and (strong_event or int(changed_pixels) >= int(self.high_info_simultaneous_min_total_pixels)):
            source_change_magnitude = float(
                max(0.0, min(1.0, region_change_magnitude_effective.get(str(source_region_key), 0.0)))
            )
            event_priority_candidates = sorted(
                [
                    str(region_key)
                    for region_key in region_sudden_spike_priority_keys
                    if self._parse_region_key_v1(str(region_key)) is not None
                    and str(region_key) != str(source_region_key)
                    and float(region_change_magnitude_effective.get(str(region_key), 0.0)) >= 0.20
                ],
                key=lambda key: (
                    -float(region_change_magnitude_effective.get(str(key), 0.0)),
                    int(
                        self._region_route_distance_v1(
                            region_adjacency,
                            start_region_key=str(anchor_key),
                            goal_region_key=str(key),
                        )
                    ),
                    str(key),
                ),
            )
            if event_priority_candidates:
                top_event_key = str(event_priority_candidates[0])
                top_event_mag = float(
                    max(0.0, min(1.0, region_change_magnitude_effective.get(str(top_event_key), 0.0)))
                )
                if (
                    top_event_mag >= float(max(0.20, source_change_magnitude - 0.28))
                    and str(top_event_key) != str(queue[0])
                ):
                    event_priority_window = [
                        str(v)
                        for v in event_priority_candidates[
                            : int(max(1, min(3, self.high_info_focus_max_targets)))
                        ]
                    ]
                    if _commit_lock_hard_active_v1():
                        _enqueue_pending_regions(
                            [str(v) for v in event_priority_window],
                            score_hint={
                                str(k): float(max(0.0, region_change_magnitude_effective.get(str(k), 0.0)))
                                for k in event_priority_window
                            },
                        )
                        state["last_status"] = "target_commit_pending_event_priority"
                    else:
                        priority_subqueue_active = bool(event_priority_window)
                        priority_subqueue_keys = list(event_priority_window)
                        if (
                            chain_lock_active
                            and self._parse_region_key_v1(str(chain_lock_target_region_key)) is not None
                        ):
                            locked_key = str(chain_lock_target_region_key)
                            queue = [str(locked_key)] + [
                                str(v)
                                for v in event_priority_window
                                if str(v) != str(locked_key)
                            ] + [
                                str(v)
                                for v in queue
                                if str(v) not in set(event_priority_window)
                                and str(v) != str(locked_key)
                            ]
                            state["last_status"] = "event_change_queued_under_lock"
                        else:
                            queue = list(event_priority_window) + [
                                str(v) for v in queue if str(v) not in set(event_priority_window)
                            ]
                            state["last_status"] = "event_change_priority_lock"
        _force_hard_lock_queue_v1("target_commit_hard_lock")
        if not queue:
            state["active"] = False
            state["stage"] = "idle"
            state["steps_remaining"] = 0
            state["target_miss_streak"] = 0
            state["current_target_region_key"] = "NA"
            state["target_region_queue"] = []
            state["pending_region_queue"] = []
            state["pending_region_scores"] = {}
            state["target_region_scores"] = {}
            state["simultaneous_changed_region_keys"] = []
            state["simultaneous_reachable_region_keys"] = []
            state["simultaneous_unknown_region_keys"] = []
            state["simultaneous_unreachable_region_keys"] = []
            state["simultaneous_anchor_region_key"] = "NA"
            state["simultaneous_changed_total_pixels"] = 0
            state["region_recent_change_pixels"] = {}
            state["region_change_magnitude"] = {}
            state["region_change_magnitude_ema"] = {}
            state["region_change_delta"] = {}
            state["region_sudden_spike_keys"] = []
            state["last_trigger_changed_pixels"] = 0
            state["last_trigger_changed_region_diff_map"] = {}
            state["interaction_chain_active"] = False
            state["interaction_target_chain"] = []
            state["interaction_target_index"] = 0
            state["interaction_last_status"] = "completed"
            _disable_chain_lock("completed")
            state["chain_lock_active"] = bool(chain_lock_active)
            state["chain_lock_window_steps"] = int(chain_lock_window_steps)
            state["chain_lock_steps_remaining"] = int(max(0, chain_lock_steps_remaining))
            state["chain_lock_target_region_key"] = str(chain_lock_target_region_key)
            state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
            state["target_commit_active"] = bool(target_commit_active)
            state["target_commit_window_steps"] = int(target_commit_window_steps)
            state["target_commit_miss_limit"] = int(target_commit_miss_limit)
            state["chain_lock_last_status"] = str(chain_lock_last_status)
            state["priority_subqueue_active"] = False
            state["priority_subqueue_keys"] = []
            state["completion_count"] = int(state.get("completion_count", 0) + 1)
            state["last_status"] = "completed"
            state["completed_target_regions"] = list(completed_regions[-16:])
            state["target_sample_counts"] = dict(target_sample_counts)
            state["target_required_samples"] = dict(target_required_samples)
            state["primary_coupled_region_key"] = str(primary_coupled_region_key)
            state["secondary_coupled_region_key"] = str(secondary_coupled_region_key)
            state["cross_region_key"] = str(primary_coupled_region_key)
            state["gate_region_key"] = str(secondary_coupled_region_key)
            state["coupled_region_keys"] = [str(v) for v in sorted(coupled_region_keys)]
            novelty_protocol_active = False
            novelty_source_region_key = "NA"
            novelty_related_region_keys = []
            inner_loop_active = False
            inner_loop_reason = "NA"
            inner_loop_queue = []
            _flush_inner_loop_state_to_state()
            _flush_novelty_regions_to_state()
            return

        if (not _commit_lock_hard_active_v1()) and interaction_chain_active and interaction_target_chain:
            interaction_target_index = int(
                max(0, min(len(interaction_target_chain) - 1, interaction_target_index))
            )
            interaction_target_key = str(interaction_target_chain[interaction_target_index])
            if self._parse_region_key_v1(interaction_target_key) is not None:
                if (
                    (not chain_lock_active)
                    and str(chain_lock_target_region_key) != str(interaction_target_key)
                ):
                    _arm_chain_lock(str(interaction_target_key), "retarget")
                effective_target_key = str(interaction_target_key)
                if chain_lock_active and (
                    self._parse_region_key_v1(str(chain_lock_target_region_key)) is not None
                ):
                    effective_target_key = str(chain_lock_target_region_key)
                queue = [str(effective_target_key)] + [
                    str(v) for v in queue if str(v) != str(effective_target_key)
                ]
                state["interaction_chain_active"] = True
                state["interaction_target_chain"] = list(interaction_target_chain[:16])
                state["interaction_target_index"] = int(interaction_target_index)
                state["interaction_last_status"] = "tracking"

        if (not _commit_lock_hard_active_v1()) and priority_subqueue_active and priority_subqueue_keys:
            refreshed_priority_subqueue: list[str] = []
            for region_key in priority_subqueue_keys:
                key = str(region_key)
                if self._parse_region_key_v1(key) is None:
                    continue
                if key in refreshed_priority_subqueue:
                    continue
                required_samples = int(
                    max(
                        target_required_samples.get(str(key), 0),
                        _required_samples(str(key)),
                        2,
                    )
                )
                target_required_samples[str(key)] = int(required_samples)
                sample_count = int(max(0, target_sample_counts.get(str(key), 0)))
                if sample_count < required_samples:
                    refreshed_priority_subqueue.append(str(key))
            priority_subqueue_keys = list(refreshed_priority_subqueue)
            priority_subqueue_active = bool(priority_subqueue_keys)
            if priority_subqueue_active:
                queue = [str(v) for v in priority_subqueue_keys] + [
                    str(v) for v in queue if str(v) not in set(priority_subqueue_keys)
                ]
                state["last_status"] = "priority_subqueue_tracking"
            else:
                state["last_status"] = "priority_subqueue_completed"
        _force_hard_lock_queue_v1("target_commit_hard_lock")

        reachable_diff_priority_active = False
        reachable_diff_priority_key = "NA"
        reachable_diff_override_enabled = bool(
            (not chain_lock_active)
            and (not target_commit_active)
        )
        override_anchor_region_key = str(nav_region_key)
        if self._parse_region_key_v1(override_anchor_region_key) is None:
            override_anchor_region_key = str(source_region_key)
        final_reachable_diff_candidates = [
            str(region_key)
            for region_key in simultaneous_reachable_region_keys
            if _is_reachable_diff_override_candidate_v1(
                str(region_key),
                anchor_region_key=str(override_anchor_region_key),
                source_region_key_hint=str(source_region_key),
            )
        ]
        if final_reachable_diff_candidates and reachable_diff_override_enabled:
            final_reachable_diff_candidates.sort(
                key=lambda key: (
                    -int(max(0, region_recent_change_pixels.get(str(key), 0))),
                    -float(max(0.0, region_change_magnitude_effective.get(str(key), 0.0))),
                    -float(max(0.0, region_change_delta_now.get(str(key), 0.0))),
                    int(
                        self._region_route_distance_v1(
                            region_adjacency,
                            start_region_key=str(override_anchor_region_key),
                            goal_region_key=str(key),
                        )
                    ),
                    str(key),
                )
            )
            forced_final_key = str(final_reachable_diff_candidates[0])
            reachable_diff_priority_active = True
            reachable_diff_priority_key = str(forced_final_key)
            if queue and str(queue[0]) != str(forced_final_key):
                queue = [str(forced_final_key)] + [
                    str(v) for v in queue if str(v) != str(forced_final_key)
                ]
                if interaction_chain_active and interaction_target_chain:
                    for idx, chain_key in enumerate(interaction_target_chain):
                        if str(chain_key) == str(forced_final_key):
                            interaction_target_index = int(idx)
                            break
                    state["interaction_target_index"] = int(max(0, interaction_target_index))
                    state["last_status"] = "reachable_diff_priority_override"
        _force_hard_lock_queue_v1("target_commit_hard_lock")

        previous_target_region_key = str(state.get("current_target_region_key", "NA"))
        target_region_key = str(queue[0])
        target_miss_streak = int(max(0, state.get("target_miss_streak", 0)))
        if (
            self._parse_region_key_v1(str(previous_target_region_key)) is not None
            and str(target_region_key) == str(previous_target_region_key)
            and str(nav_region_key) != str(previous_target_region_key)
        ):
            target_miss_streak = int(target_miss_streak + 1)
        elif str(nav_region_key) == str(target_region_key):
            target_miss_streak = 0
        elif str(target_region_key) != str(previous_target_region_key):
            target_miss_streak = 0
        novelty_related_set = {
            str(v)
            for v in novelty_related_region_keys
            if self._parse_region_key_v1(str(v)) is not None
        }
        novelty_drop_miss_limit = int(max(4, min(10, self.high_info_chain_lock_miss_limit)))
        if (
            novelty_protocol_active
            and str(target_region_key) in novelty_related_set
            and str(nav_region_key) != str(target_region_key)
            and int(target_miss_streak) >= int(novelty_drop_miss_limit)
        ):
            dropped_key = str(target_region_key)
            novelty_related_region_keys = [
                str(v) for v in novelty_related_region_keys if str(v) != str(dropped_key)
            ]
            priority_subqueue_keys = [
                str(v) for v in priority_subqueue_keys if str(v) != str(dropped_key)
            ]
            pending_region_queue = [
                str(v) for v in pending_region_queue if str(v) != str(dropped_key)
            ]
            pending_region_scores.pop(str(dropped_key), None)
            if str(dropped_key) in score_memory:
                score_memory[str(dropped_key)] = float(
                    min(float(score_memory.get(str(dropped_key), 0.0)), 0.02)
                )
            queue = [str(v) for v in queue if str(v) != str(dropped_key)]
            target_miss_streak = 0
            if not novelty_related_region_keys:
                novelty_protocol_active = False
                novelty_source_region_key = "NA"
            state["last_status"] = "novelty_drop_unreachable_related"
            if queue:
                target_region_key = str(queue[0])
            else:
                target_region_key = "NA"
        if chain_lock_active:
            if self._parse_region_key_v1(str(chain_lock_target_region_key)) is None:
                _disable_chain_lock("invalid_target")
            else:
                locked_target_key = str(chain_lock_target_region_key)
                if str(target_region_key) != str(locked_target_key):
                    if str(locked_target_key) in set(str(v) for v in queue):
                        queue = [str(locked_target_key)] + [
                            str(v) for v in queue if str(v) != str(locked_target_key)
                        ]
                    else:
                        queue = [str(locked_target_key)] + [str(v) for v in queue]
                    target_region_key = str(locked_target_key)
                    target_miss_streak = 0
                    state["last_status"] = "chain_lock_retarget"
                nav_in_lock_target = bool(str(nav_region_key) == str(locked_target_key))
                chain_lock_steps_remaining = int(max(0, chain_lock_steps_remaining - 1))
                if chain_lock_steps_remaining <= 0:
                    _disable_chain_lock("window_expired")
                elif not nav_in_lock_target:
                    if int(target_miss_streak) >= int(chain_lock_miss_limit):
                        if inner_loop_active:
                            # Inner-loop is a *hard focus*; if we cannot re-enter the target region for too long,
                            # we must drop the lock to avoid infinite attractors.
                            missed_key = str(chain_lock_target)
                            _disable_chain_lock("inner_loop_miss_limit")
                            if missed_key and missed_key not in pending_region_queue:
                                pending_region_queue.append(missed_key)
                            inner_loop_active = False
                            inner_loop_reason = "miss_limit"
                            inner_loop_queue = []
                            inner_loop_current_target_region_key = "NA"
                            chain_lock_last_status = "tracking_inner_loop_miss_limit_disabled"
                        elif target_commit_active:
                            chain_lock_last_status = "tracking_commit_hold"
                        else:
                            _disable_chain_lock("miss_limit")
                    else:
                        chain_lock_last_status = "tracking"
                else:
                    chain_lock_last_status = "in_target"
        hard_commit_mode = _commit_lock_hard_active_v1()
        if (not hard_commit_mode) and pending_region_queue:
            pending_order = sorted(
                [str(v) for v in pending_region_queue if self._parse_region_key_v1(str(v)) is not None],
                key=lambda key: (
                    -float(max(0.0, pending_region_scores.get(str(key), 0.0))),
                    int(
                        self._region_route_distance_v1(
                            region_adjacency,
                            start_region_key=str(anchor_key),
                            goal_region_key=str(key),
                        )
                    ),
                    str(key),
                ),
            )
            if pending_order:
                queue = list(pending_order) + [str(v) for v in queue if str(v) not in set(pending_order)]
                state["last_status"] = "pending_queue_merged"
            pending_region_queue = []
            pending_region_scores = {}
        miss_streak_limit = int(max(12, 2 * int(self.high_info_focus_window_steps)))
        if (
            (not chain_lock_active)
            and (not hard_commit_mode)
            and (not reachable_diff_priority_active)
            and int(target_miss_streak) >= int(miss_streak_limit)
            and len(queue) > 1
        ):
            alternate_queue = [
                str(v)
                for v in queue[1:]
                if self._parse_region_key_v1(str(v)) is not None
            ]
            simultaneous_alternate_queue = [
                str(v) for v in alternate_queue if str(v) in simultaneous_priority_set
            ]
            coupled_alternate_queue = [
                str(v) for v in alternate_queue if str(v) in coupled_region_keys
            ]
            if simultaneous_alternate_queue:
                alternate_queue = list(simultaneous_alternate_queue) + [
                    str(v)
                    for v in alternate_queue
                    if str(v) not in simultaneous_alternate_queue
                ]
            elif coupled_alternate_queue:
                alternate_queue = list(coupled_alternate_queue) + [
                    str(v) for v in alternate_queue if str(v) not in coupled_alternate_queue
                ]
            if alternate_queue:
                forced_target = str(alternate_queue[0])
                queue = [str(forced_target)] + [
                    str(v) for v in queue if str(v) != str(forced_target)
                ]
                target_region_key = str(forced_target)
                target_miss_streak = 0
                state["last_status"] = "retarget_on_miss_streak"
                if interaction_chain_active and interaction_target_chain:
                    for idx, chain_key in enumerate(interaction_target_chain):
                        if str(chain_key) == str(forced_target):
                            interaction_target_index = int(idx)
                            break
                state["interaction_target_index"] = int(max(0, interaction_target_index))
        _force_hard_lock_queue_v1("target_commit_hard_lock")
        if queue:
            target_region_key = str(queue[0])
        if _inner_loop_hard_active_v1():
            target_region_key = str(inner_loop_queue[0])
            queue = [str(target_region_key)] + [
                str(v) for v in queue if str(v) != str(target_region_key)
            ]
            state["last_status"] = "inner_loop_target_authoritative"

        deadline = int(state.get("deadline_action_counter", current_counter))
        state["steps_remaining"] = int(max(0, deadline - current_counter))
        state["target_miss_streak"] = int(max(0, target_miss_streak))
        state["current_target_region_key"] = str(target_region_key)
        state["target_region_queue"] = list(queue)
        state["target_region_scores"] = dict(score_memory)
        _flush_pending_regions_to_state()
        _flush_inner_loop_state_to_state()
        _flush_novelty_regions_to_state()
        state["completed_target_regions"] = list(completed_regions[-16:])
        state["target_sample_counts"] = dict(target_sample_counts)
        state["target_required_samples"] = dict(target_required_samples)
        target_required_samples = dict(state["target_required_samples"])
        state["primary_coupled_region_key"] = str(primary_coupled_region_key)
        state["secondary_coupled_region_key"] = str(secondary_coupled_region_key)
        state["simultaneous_changed_region_keys"] = [
            str(v) for v in simultaneous_changed_region_keys[:16]
        ]
        state["simultaneous_reachable_region_keys"] = [
            str(v) for v in simultaneous_reachable_region_keys[:16]
        ]
        state["simultaneous_unknown_region_keys"] = [
            str(v) for v in simultaneous_unknown_region_keys[:16]
        ]
        state["simultaneous_unreachable_region_keys"] = [
            str(v) for v in simultaneous_unreachable_region_keys[:16]
        ]
        state["simultaneous_anchor_region_key"] = str(simultaneous_anchor_region_key)
        state["simultaneous_changed_total_pixels"] = int(
            max(0, simultaneous_changed_total_pixels)
        )
        state["region_recent_change_pixels"] = {
            str(k): int(max(0, v))
            for (k, v) in region_recent_change_pixels.items()
            if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
        }
        state["region_change_magnitude"] = {
            str(k): float(max(0.0, min(1.0, v)))
            for (k, v) in region_change_magnitude_effective.items()
            if self._parse_region_key_v1(str(k)) is not None
        }
        state["region_change_magnitude_ema"] = {
            str(k): float(max(0.0, min(1.0, v)))
            for (k, v) in region_change_magnitude_ema.items()
            if self._parse_region_key_v1(str(k)) is not None
        }
        state["region_change_delta"] = {
            str(k): float(max(0.0, min(1.0, v)))
            for (k, v) in region_change_delta_now.items()
            if self._parse_region_key_v1(str(k)) is not None
        }
        state["region_sudden_spike_keys"] = [
            str(v) for v in sorted(region_sudden_spike_priority_keys)[:16]
        ]
        state["last_trigger_changed_pixels"] = int(max(0, changed_pixels))
        state["last_trigger_changed_region_diff_map"] = {
            str(k): int(max(0, v))
            for (k, v) in region_recent_change_pixels.items()
            if self._parse_region_key_v1(str(k)) is not None and int(v) > 0
        }
        state["cross_region_key"] = str(primary_coupled_region_key)
        state["gate_region_key"] = str(secondary_coupled_region_key)
        state["coupled_region_keys"] = [str(v) for v in sorted(coupled_region_keys)]
        state["interaction_chain_active"] = bool(interaction_chain_active)
        state["interaction_target_chain"] = list(interaction_target_chain[:16])
        state["interaction_target_index"] = int(max(0, interaction_target_index))
        state["chain_lock_active"] = bool(chain_lock_active)
        state["chain_lock_window_steps"] = int(chain_lock_window_steps)
        state["chain_lock_steps_remaining"] = int(max(0, chain_lock_steps_remaining))
        state["chain_lock_target_region_key"] = str(chain_lock_target_region_key)
        state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
        state["target_commit_active"] = bool(target_commit_active)
        state["target_commit_window_steps"] = int(target_commit_window_steps)
        state["target_commit_miss_limit"] = int(target_commit_miss_limit)
        state["chain_lock_last_status"] = str(chain_lock_last_status)
        state["priority_subqueue_active"] = bool(priority_subqueue_active)
        state["priority_subqueue_keys"] = [str(v) for v in priority_subqueue_keys[:16]]
        state["inner_loop_queue_isolated"] = bool(_inner_loop_hard_active_v1())
        state["reachable_diff_priority_active"] = bool(reachable_diff_priority_active)
        state["reachable_diff_priority_key"] = str(reachable_diff_priority_key)
        state["stage"] = "seek"
        opportunistic_target_key = "NA"
        nav_region_valid = self._parse_region_key_v1(str(nav_region_key)) is not None
        nav_in_coupled = bool(nav_region_valid and str(nav_region_key) in coupled_region_keys)
        simultaneous_subcycle_active = bool(priority_subqueue_active and priority_subqueue_keys)
        target_region_recent_change = int(
            max(0, region_recent_change_pixels.get(str(target_region_key), 0))
        )
        target_region_required_samples = int(
            max(
                1,
                target_required_samples.get(
                    str(target_region_key),
                    _required_samples(str(target_region_key)),
                ),
            )
        )
        target_region_sample_count = int(
            max(0, target_sample_counts.get(str(target_region_key), 0))
        )
        target_region_pending_samples = bool(
            target_region_sample_count < target_region_required_samples
        )
        target_is_event_priority = bool(
            str(target_region_key) in simultaneous_priority_set
            or str(target_region_key) in region_sudden_spike_priority_keys
        )
        allow_opportunistic_switch = bool(
            (not target_commit_active)
            and (not target_is_event_priority)
            and (
                (target_region_recent_change <= 0)
                or (not target_region_pending_samples)
            )
        )
        if (
            nav_in_coupled
            and str(nav_region_key) != str(target_region_key)
            and (not simultaneous_subcycle_active)
            and (not chain_lock_active)
            and (not hard_commit_mode)
            and (not reachable_diff_priority_active)
            and allow_opportunistic_switch
        ):
            nav_required_samples = int(_required_samples(str(nav_region_key)))
            nav_sample_count = int(max(0, target_sample_counts.get(str(nav_region_key), 0)))
            if (
                nav_sample_count < nav_required_samples
                and (
                    str(obs_change_type) != "NO_CHANGE"
                    or int(changed_pixels) >= 2
                )
            ):
                opportunistic_target_key = str(nav_region_key)
        if self._parse_region_key_v1(str(opportunistic_target_key)) is not None:
            target_region_key = str(opportunistic_target_key)
            queue = [str(target_region_key)] + [
                str(v) for v in queue if str(v) != str(target_region_key)
            ]
            state["target_miss_streak"] = 0
            state["last_status"] = "opportunistic_target_switch"
            if interaction_chain_active and interaction_target_chain:
                for idx, chain_key in enumerate(interaction_target_chain):
                    if str(chain_key) == str(target_region_key):
                        interaction_target_index = int(idx)
                        break
                state["interaction_target_index"] = int(max(0, interaction_target_index))
            state["current_target_region_key"] = str(target_region_key)
            state["target_region_queue"] = list(queue)
        _force_hard_lock_queue_v1("target_commit_hard_lock")
        if queue:
            target_region_key = str(queue[0])

        in_target_region = bool(str(nav_region_key) == str(target_region_key))
        if in_target_region:
            state["target_miss_streak"] = 0
            coupled_target = bool(str(target_region_key) in coupled_region_keys)
            sample_delta = (
                1
                if (
                    coupled_target
                    or str(obs_change_type) != "NO_CHANGE"
                    or int(changed_pixels) >= 2
                )
                else 0
            )
            target_sample_counts[str(target_region_key)] = int(
                max(0, target_sample_counts.get(str(target_region_key), 0)) + int(sample_delta)
            )
            sample_count = int(max(0, target_sample_counts.get(str(target_region_key), 0)))
            required_samples = int(_required_samples(str(target_region_key)))
            if str(target_region_key) in set(priority_subqueue_keys):
                required_samples = int(max(required_samples, 2))
            target_required_samples[str(target_region_key)] = int(required_samples)
            remaining_samples = int(max(0, int(required_samples) - sample_count))
            if (
                inner_loop_active
                and inner_loop_queue
                and str(target_region_key) == str(inner_loop_queue[0])
                and remaining_samples <= 0
            ):
                inner_loop_queue = [str(v) for v in inner_loop_queue[1:]]
                if inner_loop_queue:
                    next_inner_target = str(inner_loop_queue[0])
                    _arm_chain_lock(
                        str(next_inner_target),
                        "inner_loop_advance",
                        focus_commit=True,
                    )
                    priority_subqueue_active = True
                    priority_subqueue_keys = [str(next_inner_target)]
                    interaction_chain_active = False
                    interaction_target_chain = []
                    interaction_target_index = 0
                    state["last_status"] = "inner_loop_advance"
                else:
                    cycle_queue: list[str] = []
                    if novelty_protocol_active and self._parse_region_key_v1(
                        str(novelty_source_region_key)
                    ) is not None:
                        cycle_queue.append(str(novelty_source_region_key))
                        for related_key in novelty_related_region_keys:
                            key = str(related_key)
                            if self._parse_region_key_v1(key) is None:
                                continue
                            if key in cycle_queue:
                                continue
                            cycle_queue.append(str(key))
                    if cycle_queue:
                        _activate_inner_loop_v1(list(cycle_queue), "novelty_cycle_rearm")
                        state["last_status"] = "inner_loop_cycle_rearm"
                    else:
                        inner_loop_active = False
                        inner_loop_reason = "NA"
                        priority_subqueue_active = False
                        priority_subqueue_keys = []
                        _disable_chain_lock("inner_loop_completed")
                        state["last_status"] = "inner_loop_completed"
            novelty_pending_related: list[str] = []
            novelty_rotate_to_related = False
            if novelty_protocol_active:
                novelty_pending_related = [
                    str(region_key)
                    for region_key in novelty_related_region_keys
                    if self._parse_region_key_v1(str(region_key)) is not None
                    and int(max(0, target_sample_counts.get(str(region_key), 0)))
                    < int(
                        max(
                            1,
                            target_required_samples.get(
                                str(region_key),
                                _required_samples(str(region_key)),
                            ),
                        )
                    )
                ]
                novelty_rotate_to_related = bool(
                    remaining_samples <= 0
                    and novelty_pending_related
                    and (
                        str(target_region_key) == str(novelty_source_region_key)
                        or str(target_region_key) in set(novelty_related_region_keys)
                    )
                )
                if novelty_rotate_to_related and (not inner_loop_active):
                    _disable_chain_lock("novelty_rotate")
                    target_commit_active = False
                    priority_subqueue_active = True
                    priority_subqueue_keys = list(novelty_pending_related[:16])
                    score_memory[str(target_region_key)] = float(
                        min(float(score_memory.get(str(target_region_key), 0.0)), 0.06)
                    )
                    for idx, region_key in enumerate(novelty_pending_related):
                        score_memory[str(region_key)] = float(
                            max(
                                float(score_memory.get(str(region_key), 0.0)),
                                max(0.72, 0.94 - (0.07 * float(idx))),
                            )
                        )
                    state["last_status"] = "novelty_rotate_to_related"
            if remaining_samples <= 0:
                if not completed_regions or str(completed_regions[-1]) != str(target_region_key):
                    completed_regions.append(str(target_region_key))
                completed_regions = completed_regions[-16:]
                state["completed_target_regions"] = list(completed_regions)
            if str(target_region_key) in score_memory:
                if remaining_samples <= 0:
                    score_memory[str(target_region_key)] = float(
                        max(0.10, 0.72 * float(score_memory.get(str(target_region_key), 0.0)))
                    )
                else:
                    score_memory[str(target_region_key)] = float(
                        max(0.56, float(score_memory.get(str(target_region_key), 0.0)))
                    )
            next_interaction_target_key = "NA"
            if novelty_rotate_to_related and novelty_pending_related:
                next_interaction_target_key = str(novelty_pending_related[0])
            if (
                remaining_samples <= 0
                and interaction_chain_active
                and interaction_target_chain
                and (not target_commit_active)
            ):
                interaction_target_index = int(
                    max(0, min(len(interaction_target_chain) - 1, interaction_target_index))
                )
                chain_target_key = str(interaction_target_chain[interaction_target_index])
                if str(chain_target_key) == str(target_region_key):
                    if int(interaction_target_index + 1) < int(len(interaction_target_chain)):
                        interaction_target_index = int(interaction_target_index + 1)
                        next_interaction_target_key = str(
                            interaction_target_chain[interaction_target_index]
                        )
                        state["interaction_chain_active"] = True
                        state["interaction_target_chain"] = list(interaction_target_chain[:16])
                        state["interaction_target_index"] = int(interaction_target_index)
                        state["interaction_last_status"] = "advance"
                        _arm_chain_lock(str(next_interaction_target_key), "advance")
                    else:
                        interaction_chain_active = False
                        interaction_target_chain = []
                        interaction_target_index = 0
                        state["interaction_chain_active"] = False
                        state["interaction_target_chain"] = []
                        state["interaction_target_index"] = 0
                        state["interaction_last_status"] = "completed"
                        _disable_chain_lock("chain_completed")
            refreshed_queue = _rank_queue(
                score_memory,
                anchor_region_key=str(nav_region_key),
                completed_recent=set(str(v) for v in completed_regions[-8:]),
                sample_counts=target_sample_counts,
            )
            if remaining_samples <= 0:
                refreshed_queue = [str(v) for v in refreshed_queue if str(v) != str(target_region_key)]
            if self._parse_region_key_v1(str(next_interaction_target_key)) is not None:
                refreshed_queue = [str(next_interaction_target_key)] + [
                    str(v)
                    for v in refreshed_queue
                    if str(v) != str(next_interaction_target_key)
                ]
            if priority_subqueue_active and priority_subqueue_keys:
                active_priority_queue: list[str] = []
                for region_key in priority_subqueue_keys:
                    key = str(region_key)
                    if self._parse_region_key_v1(key) is None:
                        continue
                    req = int(
                        max(
                            target_required_samples.get(str(key), 0),
                            _required_samples(str(key)),
                            2,
                        )
                    )
                    target_required_samples[str(key)] = int(req)
                    if int(max(0, target_sample_counts.get(str(key), 0))) < int(req):
                        active_priority_queue.append(str(key))
                priority_subqueue_keys = list(active_priority_queue)
                priority_subqueue_active = bool(priority_subqueue_keys)
                if priority_subqueue_active:
                    refreshed_queue = [str(v) for v in priority_subqueue_keys] + [
                        str(v)
                        for v in refreshed_queue
                        if str(v) not in set(priority_subqueue_keys)
                    ]
            if (
                (not _inner_loop_hard_active_v1())
                and chain_lock_active
                and self._parse_region_key_v1(str(chain_lock_target_region_key)) is not None
            ):
                if str(chain_lock_target_region_key) in set(str(v) for v in refreshed_queue):
                    refreshed_queue = [str(chain_lock_target_region_key)] + [
                        str(v)
                        for v in refreshed_queue
                        if str(v) != str(chain_lock_target_region_key)
                    ]
                else:
                    refreshed_queue = [str(chain_lock_target_region_key)] + [
                        str(v) for v in refreshed_queue
                    ]
            if _commit_lock_hard_active_v1():
                lock_key = str(chain_lock_target_region_key)
                deferred_refresh = [str(v) for v in refreshed_queue if str(v) != str(lock_key)]
                if deferred_refresh:
                    _enqueue_pending_regions(
                        deferred_refresh,
                        score_hint={str(k): float(v) for (k, v) in score_memory.items()},
                    )
                refreshed_queue = [str(lock_key)]
                state["last_status"] = "target_commit_hard_lock"
            if refreshed_queue:
                state["target_region_queue"] = list(refreshed_queue)
                state["current_target_region_key"] = str(refreshed_queue[0])
                state["target_region_scores"] = dict(score_memory)
                state["target_sample_counts"] = dict(target_sample_counts)
                state["target_required_samples"] = dict(target_required_samples)
                _flush_pending_regions_to_state()
                _flush_inner_loop_state_to_state()
                _flush_novelty_regions_to_state()
                state["interaction_chain_active"] = bool(interaction_chain_active)
                state["interaction_target_chain"] = list(interaction_target_chain[:16])
                state["interaction_target_index"] = int(max(0, interaction_target_index))
                state["chain_lock_active"] = bool(chain_lock_active)
                state["chain_lock_window_steps"] = int(chain_lock_window_steps)
                state["chain_lock_steps_remaining"] = int(max(0, chain_lock_steps_remaining))
                state["chain_lock_target_region_key"] = str(chain_lock_target_region_key)
                state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
                state["target_commit_active"] = bool(target_commit_active)
                state["target_commit_window_steps"] = int(target_commit_window_steps)
                state["target_commit_miss_limit"] = int(target_commit_miss_limit)
                state["chain_lock_last_status"] = str(chain_lock_last_status)
                state["priority_subqueue_active"] = bool(priority_subqueue_active)
                state["priority_subqueue_keys"] = [str(v) for v in priority_subqueue_keys[:16]]
                if not bool(state.get("interaction_chain_active", False)):
                    state["interaction_last_status"] = "inactive"
                state["last_status"] = (
                    "target_sampled"
                    if remaining_samples <= 0
                    else "target_resample_pending"
                )
            else:
                state["active"] = False
                state["stage"] = "idle"
                state["steps_remaining"] = 0
                state["target_miss_streak"] = 0
                state["current_target_region_key"] = "NA"
                state["target_region_queue"] = []
                state["pending_region_queue"] = []
                state["pending_region_scores"] = {}
                state["target_region_scores"] = {}
                state["target_sample_counts"] = dict(target_sample_counts)
                state["target_required_samples"] = dict(target_required_samples)
                state["simultaneous_changed_region_keys"] = []
                state["simultaneous_reachable_region_keys"] = []
                state["simultaneous_unknown_region_keys"] = []
                state["simultaneous_unreachable_region_keys"] = []
                state["simultaneous_anchor_region_key"] = "NA"
                state["simultaneous_changed_total_pixels"] = 0
                state["interaction_chain_active"] = False
                state["interaction_target_chain"] = []
                state["interaction_target_index"] = 0
                state["interaction_last_status"] = "completed"
                _disable_chain_lock("completed")
                state["chain_lock_active"] = bool(chain_lock_active)
                state["chain_lock_window_steps"] = int(chain_lock_window_steps)
                state["chain_lock_steps_remaining"] = int(max(0, chain_lock_steps_remaining))
                state["chain_lock_target_region_key"] = str(chain_lock_target_region_key)
                state["chain_lock_miss_limit"] = int(chain_lock_miss_limit)
                state["target_commit_active"] = bool(target_commit_active)
                state["target_commit_window_steps"] = int(target_commit_window_steps)
                state["target_commit_miss_limit"] = int(target_commit_miss_limit)
                state["chain_lock_last_status"] = str(chain_lock_last_status)
                state["priority_subqueue_active"] = False
                state["priority_subqueue_keys"] = []
                novelty_protocol_active = False
                novelty_source_region_key = "NA"
                novelty_related_region_keys = []
                inner_loop_active = False
                inner_loop_reason = "NA"
                inner_loop_queue = []
                _flush_inner_loop_state_to_state()
                _flush_novelty_regions_to_state()
                state["completion_count"] = int(state.get("completion_count", 0) + 1)
                state["last_status"] = "completed"

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

    def _update_navigation_step_displacement_history_v1(
        self,
        navigation_state_estimate: dict[str, Any] | None,
    ) -> None:
        if not isinstance(navigation_state_estimate, dict):
            return
        if not bool(navigation_state_estimate.get("matched", False)):
            return
        action_id = int(navigation_state_estimate.get("action_id", 0))
        if action_id not in (1, 2, 3, 4):
            return
        displacement = int(max(0, navigation_state_estimate.get("displacement_manhattan", 0)))
        # Ignore sub-pixel / tracker jitter; we only keep meaningful movement quanta.
        if displacement < 2 or displacement > 32:
            return
        self._navigation_step_displacement_history.append(int(displacement))
        window = int(self._navigation_step_displacement_history_window)
        if len(self._navigation_step_displacement_history) > int(window):
            self._navigation_step_displacement_history = self._navigation_step_displacement_history[
                -int(window) :
            ]

    def _navigation_step_pixels_estimate_v1(self) -> int | None:
        if not self._navigation_step_displacement_history:
            return None
        histogram: dict[int, int] = {}
        for value in self._navigation_step_displacement_history:
            step = int(max(0, value))
            if step <= 0:
                continue
            histogram[step] = int(histogram.get(step, 0) + 1)
        if not histogram:
            return None
        best_step = sorted(
            histogram.items(),
            key=lambda item: (-int(item[1]), int(item[0])),
        )[0][0]
        return int(best_step)

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
        packet: ObservationPacketV1,
        representation: RepresentationStateV1,
    ) -> dict[str, Any]:
        agent_pos_xy = self._current_agent_position_xy_v1(representation)
        targets = self._navigation_key_targets_v1(
            representation,
            agent_pos_xy=agent_pos_xy,
        )
        orientation_alignment = self._orientation_alignment_state_from_targets_v1(
            packet=packet,
            representation=representation,
            targets=targets,
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
                "gate_like_enabled": False,
                "gate_like_target_region": {"x": -1, "y": -1},
                "gate_like_target_region_visit_count": 0,
                "orientation_alignment_enabled": bool(orientation_alignment.get("enabled", False)),
                "orientation_alignment_detected": bool(
                    orientation_alignment.get("detected", False)
                ),
                "orientation_alignment_aligned": bool(
                    orientation_alignment.get("aligned", False)
                ),
                "orientation_alignment_similarity": float(
                    max(0.0, min(1.0, orientation_alignment.get("similarity", 0.0)))
                ),
                "orientation_alignment_best_rotation_deg": int(
                    orientation_alignment.get("best_rotation_deg", -1)
                ),
                "orientation_alignment_rotation_bucket": str(
                    orientation_alignment.get("rotation_bucket", "rot_unknown")
                ),
                "orientation_alignment_v1": dict(orientation_alignment),
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

        primary_rx = int(max(0, min(7, int(primary.get("centroid_x", -1)) // 8)))
        primary_ry = int(max(0, min(7, int(primary.get("centroid_y", -1)) // 8)))
        primary_region_key = self._region_key_from_xy_v1(int(primary_rx), int(primary_ry))

        cross_region_key = "NA"
        gate_region_key = "NA"

        cross_target = next(
            (row for row in targets if str(row.get("kind", "")) == "cross_like"),
            None,
        )
        if isinstance(cross_target, dict):
            cross_rx = int(max(0, min(7, int(cross_target.get("centroid_x", -1)) // 8)))
            cross_ry = int(max(0, min(7, int(cross_target.get("centroid_y", -1)) // 8)))
            cross_region_key = self._region_key_from_xy_v1(int(cross_rx), int(cross_ry))
            cross_like_enabled = True
            cross_like_target_region = {"x": int(cross_rx), "y": int(cross_ry)}
            cross_like_target_region_visit_count = int(
                self._region_visit_counts.get(cross_region_key, 0)
            )
        else:
            cross_like_enabled = False
            cross_like_target_region = {"x": -1, "y": -1}
            cross_like_target_region_visit_count = 0
        gate_target = next(
            (row for row in targets if str(row.get("kind", "")) == "gate_like"),
            None,
        )
        if isinstance(gate_target, dict):
            gate_rx = int(max(0, min(7, int(gate_target.get("centroid_x", -1)) // 8)))
            gate_ry = int(max(0, min(7, int(gate_target.get("centroid_y", -1)) // 8)))
            gate_region_key = self._region_key_from_xy_v1(int(gate_rx), int(gate_ry))
            gate_like_enabled = True
            gate_like_target_region = {"x": int(gate_rx), "y": int(gate_ry)}
            gate_like_target_region_visit_count = int(
                self._region_visit_counts.get(gate_region_key, 0)
            )
        else:
            gate_like_enabled = False
            gate_like_target_region = {"x": -1, "y": -1}
            gate_like_target_region_visit_count = 0

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
            "target_region": {"x": int(primary_rx), "y": int(primary_ry)},
            "target_region_key": str(primary_region_key),
            "cross_like_enabled": bool(cross_like_enabled),
            "cross_like_target_region": dict(cross_like_target_region),
            "cross_like_target_region_key": str(cross_region_key),
            "cross_like_target_region_visit_count": int(cross_like_target_region_visit_count),
            "gate_like_enabled": bool(gate_like_enabled),
            "gate_like_target_region": dict(gate_like_target_region),
            "gate_like_target_region_key": str(gate_region_key),
            "gate_like_target_region_visit_count": int(gate_like_target_region_visit_count),
            "orientation_alignment_enabled": bool(orientation_alignment.get("enabled", False)),
            "orientation_alignment_detected": bool(
                orientation_alignment.get("detected", False)
            ),
            "orientation_alignment_aligned": bool(orientation_alignment.get("aligned", False)),
            "orientation_alignment_similarity": float(
                max(0.0, min(1.0, orientation_alignment.get("similarity", 0.0)))
            ),
            "orientation_alignment_best_rotation_deg": int(
                orientation_alignment.get("best_rotation_deg", -1)
            ),
            "orientation_alignment_rotation_bucket": str(
                orientation_alignment.get("rotation_bucket", "rot_unknown")
            ),
            "orientation_alignment_v1": dict(orientation_alignment),
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
            current_region_key = self._region_key_from_xy_v1(int(rx), int(ry))

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

        known_region_keys: set[str] = set()
        for region_key in self._region_visit_counts.keys():
            parsed = self._parse_region_key_v1(str(region_key))
            if parsed is None:
                continue
            known_region_keys.add(self._region_key_from_xy_v1(int(parsed[0]), int(parsed[1])))
        parsed_current_region = self._parse_region_key_v1(str(current_region_key))
        if parsed_current_region is not None:
            known_region_keys.add(
                self._region_key_from_xy_v1(
                    int(parsed_current_region[0]),
                    int(parsed_current_region[1]),
                )
            )
        for region_action_key in self._region_action_transition_counts.keys():
            try:
                source_region_key, _ = str(region_action_key).split("|a", 1)
            except Exception:
                continue
            parsed = self._parse_region_key_v1(str(source_region_key))
            if parsed is None:
                continue
            known_region_keys.add(self._region_key_from_xy_v1(int(parsed[0]), int(parsed[1])))
        for edge_key in self._edge_attempt_counts.keys():
            edge_token = str(edge_key)
            if not edge_token.startswith("region=") or "|action=" not in edge_token:
                continue
            region_token = edge_token.split("|action=", 1)[0]
            region_key = str(region_token).replace("region=", "", 1)
            parsed = self._parse_region_key_v1(str(region_key))
            if parsed is None:
                continue
            known_region_keys.add(self._region_key_from_xy_v1(int(parsed[0]), int(parsed[1])))

        activity_edges: list[dict[str, Any]] = []
        activity_status_histogram: dict[str, int] = {}
        for source_region_key in sorted(
            known_region_keys,
            key=lambda key: (
                int(self._parse_region_key_v1(str(key))[1]),
                int(self._parse_region_key_v1(str(key))[0]),
                str(key),
            ),
        ):
            for action_id in (1, 2, 3, 4):
                region_action_key = f"{str(source_region_key)}|a{int(action_id)}"
                edge_key = f"region={str(source_region_key)}|action={int(action_id)}"
                target_histogram_raw = self._region_action_transition_counts.get(
                    region_action_key,
                    {},
                )
                if not isinstance(target_histogram_raw, dict):
                    target_histogram_raw = {}
                plausible_target_histogram: dict[str, int] = {}
                for target_region_key_raw, count_raw in target_histogram_raw.items():
                    target_region_key = str(target_region_key_raw)
                    if self._parse_region_key_v1(target_region_key) is None:
                        continue
                    if not self._region_step_plausible_v1(
                        str(source_region_key),
                        str(target_region_key),
                        max_axis_step=1,
                    ):
                        continue
                    count = int(max(0, count_raw))
                    if count <= 0:
                        continue
                    plausible_target_histogram[target_region_key] = int(count)
                moved_count = int(
                    sum(
                        int(max(0, count))
                        for target_region_key, count in plausible_target_histogram.items()
                        if str(target_region_key) != str(source_region_key)
                    )
                )
                transition_total = int(
                    sum(int(max(0, count)) for count in plausible_target_histogram.values())
                )
                event_histogram = self._region_action_event_counts.get(region_action_key, {})
                if not isinstance(event_histogram, dict):
                    event_histogram = {}
                event_attempts = int(
                    sum(int(max(0, count)) for count in event_histogram.values())
                )
                edge_attempts = int(max(0, self._edge_attempt_counts.get(edge_key, 0)))
                blocked_count = int(max(0, self._blocked_edge_counts.get(edge_key, 0)))
                attempts = int(max(edge_attempts, event_attempts, moved_count + blocked_count))
                blocked_rate = float(blocked_count / float(max(1, attempts)))
                moved_rate = float(moved_count / float(max(1, attempts)))

                dominant_target_region_key = "NA"
                dominant_target_count = 0
                for target_region_key, count_raw in sorted(
                    plausible_target_histogram.items(),
                    key=lambda item: (
                        -int(max(0, item[1])),
                        str(item[0]),
                    ),
                ):
                    count = int(max(0, count_raw))
                    if count <= 0:
                        continue
                    if str(target_region_key) == str(source_region_key):
                        continue
                    dominant_target_region_key = str(target_region_key)
                    dominant_target_count = int(count)
                    break
                if dominant_target_region_key == "NA" and plausible_target_histogram:
                    fallback_target_key, fallback_target_count = max(
                        plausible_target_histogram.items(),
                        key=lambda item: (
                            int(max(0, item[1])),
                            str(item[0]),
                        ),
                    )
                    dominant_target_region_key = str(fallback_target_key)
                    dominant_target_count = int(max(0, fallback_target_count))

                status = "unknown"
                if attempts >= 2 and blocked_count >= 2 and blocked_rate >= 0.75:
                    status = "blocked"
                if moved_count >= 1 and moved_rate >= 0.20:
                    status = "walkable"
                if moved_count <= 0 and attempts >= 2 and blocked_rate >= 0.75:
                    status = "blocked"

                activity_edges.append(
                    {
                        "source_region_key": str(source_region_key),
                        "action_id": int(action_id),
                        "status": str(status),
                        "attempts": int(attempts),
                        "blocked_count": int(blocked_count),
                        "blocked_rate": float(blocked_rate),
                        "moved_count": int(moved_count),
                        "moved_rate": float(moved_rate),
                        "transition_total_count": int(transition_total),
                        "dominant_target_region_key": str(dominant_target_region_key),
                        "dominant_target_count": int(dominant_target_count),
                    }
                )
                activity_status_histogram[str(status)] = int(
                    activity_status_histogram.get(str(status), 0) + 1
                )
        activity_edges.sort(
            key=lambda row: (
                str(row.get("source_region_key", "")),
                int(row.get("action_id", 0)),
            )
        )
        max_activity_edges = int(max(max_edges, 256))
        if len(activity_edges) > max_activity_edges:
            activity_edges = activity_edges[:max_activity_edges]

        return {
            "schema_name": "active_inference_region_graph_snapshot_v1",
            "schema_version": 2,
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
            "activity_edge_count": int(len(activity_edges)),
            "activity_edge_status_histogram": {
                str(key): int(value)
                for (key, value) in sorted(activity_status_histogram.items())
            },
            "activity_edges": list(activity_edges),
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
                self._region_key_from_xy_v1(
                    int(self._last_known_agent_pos_region[0]),
                    int(self._last_known_agent_pos_region[1]),
                )
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
        high_info_allowed_pixel_mask, high_info_allowed_mask_meta = (
            self._high_info_allowed_pixel_mask_v1(
                frame_before=previous_packet.frame,
                frame_after=current_packet.frame,
                previous_representation=previous_representation,
                current_representation=current_representation,
                navigation_state_estimate=navigation_state_estimate,
                tracked_token_before=tracked_token_before,
            )
        )
        changed_region_diff_map = self._changed_region_diff_map_v1(
            frame_before=previous_packet.frame,
            frame_after=current_packet.frame,
            max_regions=64,
            allowed_pixel_mask=high_info_allowed_pixel_mask,
        )
        changed_region_total_pixels = int(
            sum(int(v) for v in changed_region_diff_map.values())
        )
        changed_region_topk = [
            {
                "region_key": str(region_key),
                "changed_pixels": int(count),
                "changed_ratio": float(
                    float(count) / float(max(1, changed_region_total_pixels))
                ),
            }
            for (region_key, count) in sorted(
                changed_region_diff_map.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[:8]
        ]
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
            "palette_delta_topk": {
                str(k): int(v)
                for (k, v) in dict(
                    getattr(observed_signature, "palette_delta_topk", {}) or {}
                ).items()
            },
            "palette_delta_total": int(
                sum(
                    abs(int(v))
                    for v in dict(
                        getattr(observed_signature, "palette_delta_topk", {}) or {}
                    ).values()
                    if isinstance(v, int) or str(v).lstrip("-").isdigit()
                )
            ),
            "navigation_state_estimate_v1": dict(navigation_state_estimate),
            "changed_region_diff_map_v1": {
                str(k): int(v) for (k, v) in changed_region_diff_map.items()
            },
            "changed_region_total_pixels_v1": int(changed_region_total_pixels),
            "changed_region_topk_v1": list(changed_region_topk),
            "high_info_allowed_mask_v1": dict(high_info_allowed_mask_meta),
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
            f"hit={hit}|boundary={boundary}|dist={dist_bucket}|region={coarse_y}:{coarse_x}"
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
            region_key = self._region_key_from_xy_v1(int(rx), int(ry))
            revisit_count_current = int(self._region_visit_counts.get(region_key, 0))
            edge_key = f"region={str(region_key)}|action={int(action_id)}"
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
        current_region_key_v1 = self._current_region_key_v1()
        parsed_current_region = self._parse_region_key_v1(str(current_region_key_v1))
        if parsed_current_region is not None:
            current_region = (
                int(parsed_current_region[0]),
                int(parsed_current_region[1]),
            )
            payload["current_region_source"] = "current_region_key_v1"
        if current_region is None and self._last_known_agent_pos_region is not None:
            current_region = (
                int(self._last_known_agent_pos_region[0]),
                int(self._last_known_agent_pos_region[1]),
            )
            payload["current_region_source"] = "last_known_region"
        if current_region is None and self._latest_observed_agent_pos_region is not None:
            current_region = (
                int(self._latest_observed_agent_pos_region[0]),
                int(self._latest_observed_agent_pos_region[1]),
            )
            payload["current_region_source"] = "latest_observed_region"
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
        current_region_key = self._region_key_from_xy_v1(int(current_rx), int(current_ry))
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
            plausible_empirical_counts: dict[str, int] = {}
            for candidate_region_key, candidate_count_raw in empirical_transition_counts.items():
                candidate_key = str(candidate_region_key)
                if self._parse_region_key_v1(candidate_key) is None:
                    continue
                if not self._region_step_plausible_v1(
                    str(current_region_key),
                    str(candidate_key),
                    max_axis_step=1,
                ):
                    continue
                candidate_count = int(max(0, candidate_count_raw))
                if candidate_count <= 0:
                    continue
                plausible_empirical_counts[candidate_key] = int(candidate_count)
            empirical_total = int(sum(plausible_empirical_counts.values()))
            payload["empirical_transition_total"] = int(empirical_total)
            if empirical_total > 0:
                empirical_target_key, empirical_target_count = max(
                    (
                        (str(region_key), int(max(0, count)))
                        for (region_key, count) in plausible_empirical_counts.items()
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
                parsed_target = self._parse_region_key_v1(str(empirical_target_key))
                if parsed_target is not None:
                    target_rx, target_ry = int(parsed_target[0]), int(parsed_target[1])
                else:
                    target_rx = -1
                    target_ry = -1
                payload["empirical_transition_target"] = {
                    "x": int(target_rx),
                    "y": int(target_ry),
                }
                frontier_key = "NA"
                frontier_count = 0
                frontier_visit = 10**9
                for (candidate_region_key, candidate_count_raw) in plausible_empirical_counts.items():
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
                    parsed_frontier = self._parse_region_key_v1(str(frontier_key))
                    if parsed_frontier is not None:
                        frontier_rx, frontier_ry = int(parsed_frontier[0]), int(parsed_frontier[1])
                    else:
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
        predicted_region_key = self._region_key_from_xy_v1(int(predicted_rx), int(predicted_ry))
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
        edge_key = f"region={str(current_region_key)}|action={int(action_id)}"
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
        changed_pixel_count = int(max(0, getattr(causal_signature, "changed_pixel_count", 0)))
        translation_delta_bucket = str(
            getattr(causal_signature, "translation_delta_bucket", "na")
        )
        palette_delta_topk = getattr(causal_signature, "palette_delta_topk", {})
        if not isinstance(palette_delta_topk, dict):
            palette_delta_topk = {}
        palette_delta_total = int(
            sum(
                abs(int(v))
                for v in palette_delta_topk.values()
                if isinstance(v, int) or str(v).lstrip("-").isdigit()
            )
        )
        event_tags_raw = getattr(causal_signature, "event_tags", [])
        if not isinstance(event_tags_raw, list):
            event_tags_raw = []
        event_tags = {
            str(tag)
            for tag in event_tags_raw
            if isinstance(tag, str) and str(tag).strip()
        }
        peripheral_ui_side_effect = bool("peripheral_ui_side_effect" in event_tags)
        terminal_failure = bool(
            "terminal_failure" in event_tags
            or str(getattr(causal_signature, "state_transition", "")).endswith("->GAME_OVER")
        )
        terminal_resource_depletion = bool(
            "terminal_resource_depletion" in event_tags
        )
        palette_changed = bool(palette_delta_total > 0)
        source_region_key = self._current_region_key_v1()
        edge_source_region_key = str(source_region_key)
        if action_id in (1, 2, 3, 4):
            self._navigation_attempt_count += 1
            action_key = str(action_id)
            action_stats = self._navigation_action_stats.setdefault(
                action_key,
                {"attempts": 0, "blocked": 0, "moved": 0},
            )
            action_stats["attempts"] = int(action_stats.get("attempts", 0) + 1)
            edge_key = None
            if (
                self._parse_region_key_v1(str(edge_source_region_key)) is None
                and self._last_known_agent_pos_region is not None
            ):
                rx, ry = self._last_known_agent_pos_region
                edge_source_region_key = self._region_key_from_xy_v1(int(rx), int(ry))
            if self._parse_region_key_v1(str(edge_source_region_key)) is not None:
                edge_key = f"region={str(edge_source_region_key)}|action={action_id}"
                self._edge_attempt_counts[edge_key] = int(
                    self._edge_attempt_counts.get(edge_key, 0) + 1
                )
            nav_matched = bool(navigation_state_estimate.get("matched", False))
            nav_delta = navigation_state_estimate.get("delta_pos_xy", {})
            if not isinstance(nav_delta, dict):
                nav_delta = {}
            nav_dx = int(nav_delta.get("dx", 0))
            nav_dy = int(nav_delta.get("dy", 0))
            nav_peripheral_ui = bool(
                navigation_state_estimate.get("peripheral_ui_candidate", False)
                or peripheral_ui_side_effect
            )
            if nav_peripheral_ui:
                self._navigation_ui_reject_count += 1
            nav_has_motion = bool(
                nav_matched and not nav_peripheral_ui and (abs(nav_dx) + abs(nav_dy) > 0)
            )
            if nav_matched and not nav_peripheral_ui:
                self._navigation_match_count += 1
            navigation_direction_bucket = self._navigation_direction_bucket_from_estimate_v1(
                navigation_state_estimate
            )
            if (
                nav_has_motion
                and navigation_direction_bucket in ("dir_l", "dir_r", "dir_u", "dir_d")
                and translation_delta_bucket in ("dir_l", "dir_r", "dir_u", "dir_d")
            ):
                self._navigation_semantic_compare_count += 1
                if str(navigation_direction_bucket) != str(translation_delta_bucket):
                    self._navigation_semantic_mismatch_count += 1
            moved = bool(nav_has_motion)
            if moved:
                self._navigation_moved_count += 1
                action_stats["moved"] = int(action_stats.get("moved", 0) + 1)
                movement_direction_bucket = str(translation_delta_bucket)
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
                        target_region_key = self._region_key_from_xy_v1(int(rx), int(ry))
                        plausible_transition = True
                        if self._parse_region_key_v1(str(edge_source_region_key)) is not None:
                            plausible_transition = bool(
                                self._region_step_plausible_v1(
                                    str(edge_source_region_key),
                                    str(target_region_key),
                                    max_axis_step=1,
                                )
                            )
                        if plausible_transition:
                            self._last_known_agent_pos_region = (rx, ry)
                            self._region_visit_counts[target_region_key] = int(
                                self._region_visit_counts.get(target_region_key, 0) + 1
                            )
                            if str(edge_source_region_key) != "NA":
                                region_action_key = (
                                    f"{str(edge_source_region_key)}|a{action_id}"
                                )
                                target_histogram = self._region_action_transition_counts.setdefault(
                                    region_action_key,
                                    {},
                                )
                                target_histogram[target_region_key] = int(
                                    target_histogram.get(target_region_key, 0) + 1
                                )
                        else:
                            self._navigation_implausible_transition_count = int(
                                self._navigation_implausible_transition_count + 1
                            )
            else:
                self._navigation_blocked_count += 1
                action_stats["blocked"] = int(action_stats.get("blocked", 0) + 1)
                if edge_key is not None:
                    self._blocked_edge_counts[edge_key] = (
                        int(self._blocked_edge_counts.get(edge_key, 0)) + 1
                    )

        if self._parse_region_key_v1(str(source_region_key)) is not None and action_id != 0:
            region_action_key = f"{str(source_region_key)}|a{int(action_id)}"
            histogram = self._region_action_event_counts.setdefault(region_action_key, {})
            histogram[str(obs_change_type)] = int(histogram.get(str(obs_change_type), 0) + 1)
            if str(obs_change_type) != "NO_CHANGE" and not peripheral_ui_side_effect:
                self._region_action_non_no_change_counts[region_action_key] = int(
                    self._region_action_non_no_change_counts.get(region_action_key, 0) + 1
                )
            strong_change = bool(
                str(obs_change_type) in ("CC_COUNT_CHANGE", "GLOBAL_PATTERN_CHANGE")
                or int(changed_pixel_count) >= int(self.high_info_strong_change_pixels)
            )
            if strong_change and not peripheral_ui_side_effect:
                self._region_action_strong_change_counts[region_action_key] = int(
                    self._region_action_strong_change_counts.get(region_action_key, 0) + 1
                )
            if int(level_delta) > 0:
                self._region_action_progress_counts[region_action_key] = int(
                    self._region_action_progress_counts.get(region_action_key, 0) + 1
                )
            if palette_changed and not peripheral_ui_side_effect:
                self._region_action_palette_change_counts[region_action_key] = int(
                    self._region_action_palette_change_counts.get(region_action_key, 0)
                    + 1
                )
            if not peripheral_ui_side_effect:
                self._region_action_palette_delta_total_sum[region_action_key] = int(
                    self._region_action_palette_delta_total_sum.get(region_action_key, 0)
                    + int(palette_delta_total)
                )
            if peripheral_ui_side_effect:
                self._region_action_ui_side_effect_counts[region_action_key] = int(
                    self._region_action_ui_side_effect_counts.get(region_action_key, 0) + 1
                )
            if terminal_failure or terminal_resource_depletion:
                self._region_action_terminal_failure_counts[region_action_key] = int(
                    self._region_action_terminal_failure_counts.get(region_action_key, 0)
                    + 1
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
        nav_ui_reject = int(self._navigation_ui_reject_count)
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
            "navigation_implausible_transition_count": int(
                self._navigation_implausible_transition_count
            ),
            "navigation_anchor_jump_reject_count": int(
                self._navigation_anchor_jump_reject_count
            ),
            "navigation_anchor_jump_streak": int(self._navigation_anchor_jump_streak),
            "navigation_blocked_rate": float(nav_blocked / float(max(1, nav_attempts))),
            "navigation_ui_reject_count": int(nav_ui_reject),
            "navigation_ui_reject_rate": float(nav_ui_reject / float(max(1, nav_attempts))),
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
            "region_action_palette_change_histogram": {
                str(key): int(max(0, value))
                for (key, value) in sorted(
                    self._region_action_palette_change_counts.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
            "region_action_palette_delta_total_histogram": {
                str(key): int(max(0, value))
                for (key, value) in sorted(
                    self._region_action_palette_delta_total_sum.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
            "region_action_ui_side_effect_histogram": {
                str(key): int(max(0, value))
                for (key, value) in sorted(
                    self._region_action_ui_side_effect_counts.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
            "region_action_terminal_failure_histogram": {
                str(key): int(max(0, value))
                for (key, value) in sorted(
                    self._region_action_terminal_failure_counts.items(),
                    key=lambda item: (-int(item[1]), item[0]),
                )[:128]
            },
            "click_bucket_effectiveness": click_summary,
            "click_subcluster_effectiveness": click_subcluster_summary,
            "navigation_semantic_features_v1": self._navigation_semantic_features_v1(),
            "orientation_alignment_state_v1": dict(
                self._latest_orientation_alignment_state_v1
            ),
            "orientation_action_stats": {
                str(action_key): {
                    "attempts": int(max(0, stats.get("attempts", 0))),
                    "improved": int(max(0, stats.get("improved", 0))),
                    "regressed": int(max(0, stats.get("regressed", 0))),
                    "aligned_hit": int(max(0, stats.get("aligned_hit", 0))),
                    "improve_rate": float(
                        float(max(0, stats.get("improved", 0)))
                        / float(max(1, int(max(0, stats.get("attempts", 0)))))
                    ),
                    "regress_rate": float(
                        float(max(0, stats.get("regressed", 0)))
                        / float(max(1, int(max(0, stats.get("attempts", 0)))))
                    ),
                    "aligned_hit_rate": float(
                        float(max(0, stats.get("aligned_hit", 0)))
                        / float(max(1, int(max(0, stats.get("attempts", 0)))))
                    ),
                }
                for (action_key, stats) in sorted(self._orientation_action_stats.items())
            },
            "sequence_causal_state_v1": self._sequence_causal_state_snapshot_v1(),
            "high_info_focus_state_v1": self._high_info_focus_state_snapshot_v1(),
            "high_info_region_scoreboard_v1": self._high_info_region_scoreboard_v1(
                max_regions=12
            ),
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

        action_id = int(executed_candidate.action_id)
        expected_dir = self._action_direction_vector_v1(action_id)
        tracked_digest = str(self._tracked_agent_token_digest or "")
        expected_step = 5
        frame_width = max(1, int(current_representation.frame_width))
        frame_height = max(1, int(current_representation.frame_height))
        peripheral_margin_x = max(1, int(round(float(frame_width) * 0.08)))
        peripheral_margin_y = max(1, int(round(float(frame_height) * 0.08)))
        ui_max_area = max(4, int(round(float(frame_width * frame_height) * 0.004)))
        ui_max_side = max(2, int(round(float(min(frame_width, frame_height)) * 0.08)))

        def peripheral_ui_likelihood(node: Any) -> float:
            try:
                area = int(getattr(node, "area", 0))
                cx = int(getattr(node, "centroid_x", -1))
                cy = int(getattr(node, "centroid_y", -1))
                min_x = int(getattr(node, "bbox_min_x", cx))
                max_x = int(getattr(node, "bbox_max_x", cx))
                min_y = int(getattr(node, "bbox_min_y", cy))
                max_y = int(getattr(node, "bbox_max_y", cy))
            except Exception:
                return 0.0
            if cx < 0 or cy < 0:
                return 0.0
            width = max(1, int(max_x - min_x + 1))
            height = max(1, int(max_y - min_y + 1))
            near_periphery = bool(
                cx < peripheral_margin_x
                or cx >= max(0, frame_width - peripheral_margin_x)
                or cy < peripheral_margin_y
                or cy >= max(0, frame_height - peripheral_margin_y)
            )
            if not near_periphery:
                return 0.0
            area_score = 0.0
            if area <= ui_max_area:
                area_score = 1.0
            else:
                area_score = max(
                    0.0,
                    1.0 - (float(area - ui_max_area) / float(max(1, ui_max_area * 2))),
                )
            side = max(width, height)
            side_score = 1.0 if side <= ui_max_side else 0.0
            boundary_score = 0.25 if bool(getattr(node, "touches_boundary", False)) else 0.0
            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        (0.65 * area_score) + (0.25 * side_score) + float(boundary_score),
                    ),
                )
            )

        def pair_metrics(previous: Any, current: Any) -> dict[str, float] | None:
            if int(previous.color) != int(current.color):
                return None
            area_gap = abs(int(previous.area) - int(current.area))
            if area_gap > max(2, int(previous.area * 0.3)):
                return None

            delta_x = int(current.centroid_x) - int(previous.centroid_x)
            delta_y = int(current.centroid_y) - int(previous.centroid_y)
            shift = abs(int(delta_x)) + abs(int(delta_y))
            if shift <= 0:
                return None

            shift_error = abs(int(shift) - int(expected_step))
            if shift < 2:
                shift_error += 8
            elif shift > 14:
                shift_error += int(shift - 14) * 3

            direction_penalty = 0
            projection_error = 0
            direction_alignment = 0.0
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

                projected_x = int(previous.centroid_x) + (int(expected_dx_sign) * int(expected_step))
                projected_y = int(previous.centroid_y) + (int(expected_dy_sign) * int(expected_step))
                projection_error = int(
                    abs(int(current.centroid_x) - int(projected_x))
                    + abs(int(current.centroid_y) - int(projected_y))
                )
                if shift > 0:
                    direction_alignment = float(
                        (
                            (int(delta_x) * int(expected_dx_sign))
                            + (int(delta_y) * int(expected_dy_sign))
                        )
                        / float(max(1, int(shift)))
                    )

            track_penalty = 0
            if tracked_digest:
                if str(previous.digest) != tracked_digest:
                    track_penalty += 10
                else:
                    track_penalty -= 4
                if str(current.digest) == tracked_digest:
                    track_penalty -= 2

            # Prefer non-HUD, non-boundary objects for navigation tracking.
            boundary_penalty = 0
            if bool(getattr(previous, "touches_boundary", False)):
                boundary_penalty += 18
            if bool(getattr(current, "touches_boundary", False)):
                boundary_penalty += 18

            continuity_bonus = 0
            if str(previous.object_id) == str(current.object_id):
                continuity_bonus = 6
            previous_ui_likelihood = float(peripheral_ui_likelihood(previous))
            current_ui_likelihood = float(peripheral_ui_likelihood(current))
            peripheral_ui_likelihood_pair = float(
                max(previous_ui_likelihood, current_ui_likelihood)
            )
            if previous_ui_likelihood > 0.0 and current_ui_likelihood > 0.0:
                peripheral_ui_likelihood_pair = float(
                    max(
                        peripheral_ui_likelihood_pair,
                        min(
                            1.0,
                            (0.50 * previous_ui_likelihood)
                            + (0.50 * current_ui_likelihood)
                            + 0.20,
                        ),
                    )
                )
            ui_penalty = int(round(90.0 * float(peripheral_ui_likelihood_pair)))

            score = (
                (int(area_gap) * 10)
                + (int(shift_error) * 6)
                + int(direction_penalty)
                + (int(projection_error) * 2)
                + int(track_penalty)
                + int(boundary_penalty)
                + int(ui_penalty)
                - int(continuity_bonus)
            )
            return {
                "score": float(score),
                "delta_x": float(delta_x),
                "delta_y": float(delta_y),
                "shift": float(shift),
                "direction_alignment": float(max(-1.0, min(1.0, direction_alignment))),
                "projection_error": float(projection_error),
                "area_gap": float(area_gap),
                "boundary_penalty": float(boundary_penalty),
                "ui_penalty": float(ui_penalty),
                "peripheral_ui_likelihood": float(peripheral_ui_likelihood_pair),
            }

        def bbox_gap_with_pad(a: Any, b: Any, pad: int = 0) -> tuple[int, int]:
            try:
                a_min_x = int(getattr(a, "bbox_min_x", int(getattr(a, "centroid_x", 0))))
                a_max_x = int(getattr(a, "bbox_max_x", int(getattr(a, "centroid_x", 0))))
                a_min_y = int(getattr(a, "bbox_min_y", int(getattr(a, "centroid_y", 0))))
                a_max_y = int(getattr(a, "bbox_max_y", int(getattr(a, "centroid_y", 0))))
                b_min_x = int(getattr(b, "bbox_min_x", int(getattr(b, "centroid_x", 0))))
                b_max_x = int(getattr(b, "bbox_max_x", int(getattr(b, "centroid_x", 0))))
                b_min_y = int(getattr(b, "bbox_min_y", int(getattr(b, "centroid_y", 0))))
                b_max_y = int(getattr(b, "bbox_max_y", int(getattr(b, "centroid_y", 0))))
            except Exception:
                return (10**9, 10**9)
            if a_max_x + int(pad) < b_min_x:
                gap_x = int(b_min_x - (a_max_x + int(pad)))
            elif b_max_x + int(pad) < a_min_x:
                gap_x = int(a_min_x - (b_max_x + int(pad)))
            else:
                gap_x = 0
            if a_max_y + int(pad) < b_min_y:
                gap_y = int(b_min_y - (a_max_y + int(pad)))
            elif b_max_y + int(pad) < a_min_y:
                gap_y = int(a_min_y - (b_max_y + int(pad)))
            else:
                gap_y = 0
            return (int(gap_x), int(gap_y))

        def nodes_adjacent(a: Any, b: Any, pad: int = 1) -> bool:
            gap_x, gap_y = bbox_gap_with_pad(a, b, pad=pad)
            return bool(gap_x <= 0 and gap_y <= 0)

        def composite_component(seed: Any, nodes: list[Any]) -> list[Any]:
            seed_area = int(max(1, getattr(seed, "area", 1)))
            area_limit = int(max(16, (seed_area * 3)))
            side_limit = int(max(4, round(float(min(frame_width, frame_height)) * 0.22)))
            component: list[Any] = [seed]
            stack: list[Any] = [seed]
            seen = {str(getattr(seed, "digest", ""))}
            while stack:
                anchor = stack.pop()
                for candidate in nodes:
                    candidate_digest = str(getattr(candidate, "digest", ""))
                    if candidate_digest in seen:
                        continue
                    try:
                        candidate_area = int(max(1, getattr(candidate, "area", 1)))
                        candidate_w = int(
                            max(
                                1,
                                int(getattr(candidate, "bbox_max_x", 0))
                                - int(getattr(candidate, "bbox_min_x", 0))
                                + 1,
                            )
                        )
                        candidate_h = int(
                            max(
                                1,
                                int(getattr(candidate, "bbox_max_y", 0))
                                - int(getattr(candidate, "bbox_min_y", 0))
                                + 1,
                            )
                        )
                    except Exception:
                        continue
                    if candidate_area > area_limit:
                        continue
                    if max(candidate_w, candidate_h) > side_limit:
                        continue
                    if float(peripheral_ui_likelihood(candidate)) >= 0.70:
                        continue
                    if not nodes_adjacent(anchor, candidate, pad=1):
                        continue
                    seen.add(candidate_digest)
                    component.append(candidate)
                    stack.append(candidate)
            return component

        def weighted_centroid(nodes: list[Any], fallback: Any) -> tuple[int, int]:
            total_weight = 0.0
            weighted_x = 0.0
            weighted_y = 0.0
            for node in nodes:
                try:
                    node_area = float(max(1, int(getattr(node, "area", 1))))
                    node_x = float(int(getattr(node, "centroid_x", 0)))
                    node_y = float(int(getattr(node, "centroid_y", 0)))
                except Exception:
                    continue
                total_weight += node_area
                weighted_x += (node_x * node_area)
                weighted_y += (node_y * node_area)
            if total_weight <= 0.0:
                return (
                    int(getattr(fallback, "centroid_x", 0)),
                    int(getattr(fallback, "centroid_y", 0)),
                )
            return (
                int(round(weighted_x / total_weight)),
                int(round(weighted_y / total_weight)),
            )

        def search(previous_pool: list[Any]) -> tuple[tuple[Any, Any] | None, dict[str, float]]:
            best_pair_local: tuple[Any, Any] | None = None
            best_metrics_local: dict[str, float] = {}
            best_score_local = 10**9
            for previous in previous_pool:
                for current in current_nodes:
                    metrics = pair_metrics(previous, current)
                    if metrics is None:
                        continue
                    score = float(metrics.get("score", 10**9))
                    if score < float(best_score_local):
                        best_score_local = score
                        best_pair_local = (previous, current)
                        best_metrics_local = dict(metrics)
            return best_pair_local, best_metrics_local

        tracked_previous_nodes = (
            [node for node in previous_nodes if str(node.digest) == tracked_digest]
            if tracked_digest
            else []
        )
        best_pair: tuple[Any, Any] | None = None
        best_metrics: dict[str, float] = {}
        if tracked_previous_nodes:
            best_pair, best_metrics = search(tracked_previous_nodes)
            if best_pair is None:
                fallback_pair, fallback_metrics = search(previous_nodes)
                if fallback_pair is not None:
                    best_pair = fallback_pair
                    best_metrics = fallback_metrics
        else:
            best_pair, best_metrics = search(previous_nodes)

        if best_pair is None:
            return {
                "schema_name": "active_inference_navigation_state_estimate_v1",
                "schema_version": 1,
                "matched": False,
                "reason": "no_translation_match",
                "peripheral_ui_candidate": False,
                "peripheral_ui_likelihood": 0.0,
                "control_schema_posterior": self._control_schema_posterior(),
            }

        previous, current = best_pair
        previous_component = composite_component(previous, previous_nodes)
        current_component = composite_component(current, current_nodes)
        previous_centroid_x, previous_centroid_y = weighted_centroid(
            previous_component,
            previous,
        )
        current_centroid_x, current_centroid_y = weighted_centroid(
            current_component,
            current,
        )
        peripheral_ui_score = float(
            max(0.0, min(1.0, best_metrics.get("peripheral_ui_likelihood", 0.0)))
        )
        peripheral_ui_candidate = bool(peripheral_ui_score >= 0.75)
        if peripheral_ui_candidate:
            anchor_x = -1
            anchor_y = -1
            if self._tracked_agent_anchor_xy is not None:
                anchor_x = int(self._tracked_agent_anchor_xy[0])
                anchor_y = int(self._tracked_agent_anchor_xy[1])
            return {
                "schema_name": "active_inference_navigation_state_estimate_v1",
                "schema_version": 1,
                "matched": False,
                "reason": "peripheral_ui_motion",
                "peripheral_ui_candidate": True,
                "peripheral_ui_likelihood": float(peripheral_ui_score),
                "agent_pos_xy": {
                    "x": int(anchor_x),
                    "y": int(anchor_y),
                },
                "agent_pos_region": {
                    "x": int(max(0, min(7, int(anchor_x) // 8))) if anchor_x >= 0 else -1,
                    "y": int(max(0, min(7, int(anchor_y) // 8))) if anchor_y >= 0 else -1,
                },
                "candidate_centroid_xy": {
                    "x": int(getattr(current, "centroid_x", -1)),
                    "y": int(getattr(current, "centroid_y", -1)),
                },
                "control_schema_posterior": self._control_schema_posterior(),
            }

        if action_id in (1, 2, 3, 4) and self._tracked_agent_anchor_xy is not None:
            anchor_x, anchor_y = self._tracked_agent_anchor_xy
            anchor_jump = int(
                abs(int(current_centroid_x) - int(anchor_x))
                + abs(int(current_centroid_y) - int(anchor_y))
            )
            anchor_jump_max = 24
            if anchor_jump > int(anchor_jump_max):
                self._navigation_anchor_jump_reject_count = int(
                    self._navigation_anchor_jump_reject_count + 1
                )
                self._navigation_anchor_jump_streak = int(
                    self._navigation_anchor_jump_streak + 1
                )
                if self._navigation_anchor_jump_streak < 3:
                    anchor_region_x = int(max(0, min(7, int(anchor_x) // 8)))
                    anchor_region_y = int(max(0, min(7, int(anchor_y) // 8)))
                    return {
                        "schema_name": "active_inference_navigation_state_estimate_v1",
                        "schema_version": 1,
                        "matched": False,
                        "reason": "anchor_jump_guard",
                        "peripheral_ui_candidate": False,
                        "peripheral_ui_likelihood": float(peripheral_ui_score),
                        "agent_pos_xy": {
                            "x": int(anchor_x),
                            "y": int(anchor_y),
                        },
                        "agent_pos_region": {
                            "x": int(anchor_region_x),
                            "y": int(anchor_region_y),
                        },
                        "anchor_jump_distance": int(anchor_jump),
                        "anchor_jump_max": int(anchor_jump_max),
                        "anchor_jump_streak": int(self._navigation_anchor_jump_streak),
                        "tracked_pair_source": (
                            "tracked_first_pass"
                            if tracked_previous_nodes
                            and str(previous.digest) == tracked_digest
                            else "global_pair_search"
                        ),
                        "control_schema_posterior": self._control_schema_posterior(),
                    }
                # Allow controlled re-anchor after repeated rejects.
                self._navigation_anchor_jump_streak = 0

        delta_x = int(current_centroid_x) - int(previous_centroid_x)
        delta_y = int(current_centroid_y) - int(previous_centroid_y)
        shift = int(abs(delta_x) + abs(delta_y))
        delta_key = f"dx={delta_x}|dy={delta_y}"
        action_key = str(int(executed_candidate.action_id))
        per_action = self._control_schema_counts.setdefault(action_key, {})
        per_action[delta_key] = int(per_action.get(delta_key, 0) + 1)
        self._tracked_agent_token_digest = str(current.digest)
        self._tracked_agent_anchor_xy = (int(current_centroid_x), int(current_centroid_y))
        try:
            self._tracked_agent_color = int(getattr(current, "color", -1))
            tracked_area_now = float(max(1, int(getattr(current, "area", 1))))
            if self._tracked_agent_area_ema is None:
                self._tracked_agent_area_ema = float(tracked_area_now)
            else:
                self._tracked_agent_area_ema = float(
                    (0.75 * float(self._tracked_agent_area_ema)) + (0.25 * tracked_area_now)
                )
        except Exception:
            pass
        self._navigation_anchor_jump_streak = 0

        return {
            "schema_name": "active_inference_navigation_state_estimate_v1",
            "schema_version": 1,
            "matched": True,
            "tracked_agent_token_id": str(current.digest),
            "tracked_agent_token_pair": {
                "previous_digest": str(previous.digest),
                "current_digest": str(current.digest),
            },
            "tracked_agent_component_v1": {
                "mode": "spatial_merge_single_track_v1",
                "previous_member_count": int(len(previous_component)),
                "current_member_count": int(len(current_component)),
                "previous_member_digests": [
                    str(getattr(node, "digest", "NA")) for node in previous_component[:8]
                ],
                "current_member_digests": [
                    str(getattr(node, "digest", "NA")) for node in current_component[:8]
                ],
            },
            "agent_pos_xy": {
                "x": int(current_centroid_x),
                "y": int(current_centroid_y),
            },
            "agent_pos_region": {
                "x": int(max(0, min(7, int(current_centroid_x) // 8))),
                "y": int(max(0, min(7, int(current_centroid_y) // 8))),
            },
            "delta_pos_xy": {"dx": int(delta_x), "dy": int(delta_y)},
            "displacement_manhattan": int(shift),
            "match_score": float(best_metrics.get("score", 0.0)),
            "direction_alignment": float(best_metrics.get("direction_alignment", 0.0)),
            "projection_error": float(best_metrics.get("projection_error", 0.0)),
            "peripheral_ui_candidate": False,
            "peripheral_ui_likelihood": float(peripheral_ui_score),
            "tracked_pair_source": (
                "tracked_first_pass"
                if tracked_previous_nodes and str(previous.digest) == tracked_digest
                else "global_pair_search"
            ),
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
            self._update_observed_agent_region_from_representation_v1(representation)
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
                previous_targets = self._navigation_key_targets_v1(
                    self._previous_representation,
                    agent_pos_xy=self._current_agent_position_xy_v1(
                        self._previous_representation
                    ),
                )
                previous_orientation_alignment = (
                    self._orientation_alignment_state_from_targets_v1(
                        packet=self._previous_packet,
                        representation=self._previous_representation,
                        targets=previous_targets,
                    )
                )
                current_targets = self._navigation_key_targets_v1(
                    representation,
                    agent_pos_xy=self._current_agent_position_xy_v1(representation),
                )
                current_orientation_alignment = (
                    self._orientation_alignment_state_from_targets_v1(
                        packet=packet,
                        representation=representation,
                        targets=current_targets,
                    )
                )
                self._update_orientation_action_stats_v1(
                    executed_action_id=int(self._previous_action_candidate.action_id),
                    previous_alignment=previous_orientation_alignment,
                    current_alignment=current_orientation_alignment,
                )
                self._latest_orientation_alignment_state_v1 = dict(
                    current_orientation_alignment
                )
                self._latest_navigation_target_features_v1 = dict(
                    self._navigation_target_features_v1(
                        packet,
                        representation,
                    )
                )
                self._update_operability_stats_v1(
                    executed_candidate=self._previous_action_candidate,
                    causal_signature=causal_signature,
                    navigation_state_estimate=navigation_state_estimate,
                )
                self._update_sequence_causal_state_v1(
                    current_packet=packet,
                    transition_record=transition_record,
                    causal_signature=causal_signature,
                    navigation_state_estimate=navigation_state_estimate,
                    executed_candidate=self._previous_action_candidate,
                )
                self._update_high_info_focus_state_v1(
                    current_packet=packet,
                    transition_record=transition_record,
                    causal_signature=causal_signature,
                    navigation_state_estimate=navigation_state_estimate,
                    executed_candidate=self._previous_action_candidate,
                )
                self._latest_navigation_state_estimate = dict(navigation_state_estimate)
                self._update_navigation_step_displacement_history_v1(navigation_state_estimate)
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
                        "orientation_alignment_state_v1": dict(
                            self._latest_orientation_alignment_state_v1
                        ),
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
            tb = exc.__traceback__
            while tb is not None and tb.tb_next is not None:
                tb = tb.tb_next
            location = "unknown"
            if tb is not None and tb.tb_frame is not None:
                location = f"{tb.tb_frame.f_code.co_name}:{int(tb.tb_lineno)}"
            diagnostics.finish_rejected(
                "causal_update",
                f"causal_update_error::{type(exc).__name__}::{location}::{str(exc)[:120]}",
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
                            packet,
                            representation,
                        )
                        self._latest_navigation_target_features_v1 = dict(
                            navigation_target_features
                        )
                        if isinstance(
                            navigation_target_features.get("orientation_alignment_v1"),
                            dict,
                        ):
                            self._latest_orientation_alignment_state_v1 = dict(
                                navigation_target_features.get(
                                    "orientation_alignment_v1",
                                    {},
                                )
                            )
                        navigation_semantic_features = self._navigation_semantic_features_v1()
                        sequence_causal_state = self._sequence_causal_state_snapshot_v1()
                        high_info_focus_state = self._high_info_focus_state_snapshot_v1()
                        high_info_region_scoreboard = self._high_info_region_scoreboard_v1(
                            max_regions=12
                        )
                        region_graph_snapshot = self._region_graph_snapshot_v1()
                        agent_pos_xy_for_nav_map = self._current_agent_position_xy_v1(
                            representation
                        )
                        navigation_map_snapshot_v1 = build_navigation_map_snapshot_v1(
                            packet.frame,
                            agent_pos_xy=agent_pos_xy_for_nav_map,
                            movement_step_pixels=self._navigation_step_pixels_estimate_v1(),
                            region_size=8,
                            walkable_ratio_threshold=0.02,
                        )
                        self._latest_navigation_map_snapshot_v1 = dict(
                            navigation_map_snapshot_v1
                        )
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
                            candidate.metadata["navigation_semantic_features_v1"] = dict(
                                navigation_semantic_features
                            )
                            candidate.metadata["sequence_causal_state_v1"] = dict(
                                sequence_causal_state
                            )
                            candidate.metadata["high_info_focus_state_v1"] = dict(
                                high_info_focus_state
                            )
                            candidate.metadata["high_info_region_scoreboard_v1"] = dict(
                                high_info_region_scoreboard
                            )
                            predicted_region_features: dict[str, Any] | None = None
                            if int(candidate.action_id) in (1, 2, 3, 4):
                                candidate.metadata["blocked_edge_observed_stats"] = (
                                    self._navigation_candidate_stats(int(candidate.action_id))
                                )
                                predicted_region_features = self._predicted_region_features_v1(
                                    action_id=int(candidate.action_id),
                                    action_posterior=action_posterior,
                                )
                                candidate.metadata["predicted_region_features_v1"] = dict(
                                    predicted_region_features
                                )
                                target_payload = dict(navigation_target_features)
                                target_payload["candidate_action_id"] = int(candidate.action_id)
                                candidate.metadata["navigation_target_features_v1"] = target_payload
                                candidate.metadata["navigation_step_projection_features_v1"] = (
                                    self._navigation_step_projection_features_v1(
                                        action_id=int(candidate.action_id),
                                        action_posterior=action_posterior,
                                        navigation_target_features=target_payload,
                                        predicted_region_features=predicted_region_features,
                                    )
                                )
                                candidate.metadata["orientation_alignment_features_v1"] = (
                                    self._orientation_alignment_candidate_features_v1(
                                        candidate=candidate,
                                        orientation_state=target_payload.get(
                                            "orientation_alignment_v1",
                                            {},
                                        ),
                                    )
                                )
                                candidate.metadata["region_action_semantics_v1"] = (
                                    self._region_action_semantics_v1(
                                        action_id=int(candidate.action_id),
                                        current_region_key=str(
                                            predicted_region_features.get(
                                                "current_region_key",
                                                self._current_region_key_v1(),
                                            )
                                            if isinstance(predicted_region_features, dict)
                                            else self._current_region_key_v1()
                                        ),
                                    )
                                )
                                candidate.metadata["region_graph_snapshot_v1"] = dict(
                                    region_graph_snapshot
                                )
                                candidate.metadata["navigation_map_snapshot_v1"] = dict(
                                    navigation_map_snapshot_v1
                                )
                            else:
                                candidate.metadata["orientation_alignment_features_v1"] = (
                                    self._orientation_alignment_candidate_features_v1(
                                        candidate=candidate,
                                        orientation_state=navigation_target_features.get(
                                            "orientation_alignment_v1",
                                            {},
                                        ),
                                    )
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
                            candidate.metadata["sequence_causal_features_v1"] = (
                                self._sequence_causal_candidate_features_v1(
                                    candidate=candidate,
                                    predicted_region_features=predicted_region_features,
                                )
                            )
                            candidate.metadata["high_info_focus_features_v1"] = (
                                self._high_info_focus_candidate_features_v1(
                                    candidate=candidate,
                                    predicted_region_features=predicted_region_features,
                                )
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
                    error_location = "NA"
                    try:
                        tb = traceback.extract_tb(exc.__traceback__)
                        if tb:
                            last = tb[-1]
                            error_location = f"{os.path.basename(last.filename)}:{int(last.lineno)}"
                    except Exception:
                        error_location = "NA"
                    diagnostics.finish_rejected(
                        "candidate_generation",
                        (
                            "candidate_generation_error::"
                            f"{type(exc).__name__}::"
                            f"{error_location}::"
                            f"{str(exc)[:120]}"
                        ),
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
            trace_object_snapshot = self._trace_object_snapshot_v1(representation)
            trace_neighborhood_context = self._trace_neighborhood_context_v1(
                packet=packet,
                representation=representation,
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
                    "trace_object_snapshot_v1": trace_object_snapshot,
                    "trace_neighborhood_context_v1": trace_neighborhood_context,
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
        if selected_action_id == 0:
            self._tracked_agent_token_digest = None
            self._tracked_agent_anchor_xy = None
            self._tracked_agent_color = None
            self._tracked_agent_area_ema = None
            self._navigation_anchor_jump_streak = 0

        self._previous_packet = packet
        self._previous_representation = representation
        self._previous_action_candidate = selected_candidate

        return action

    def _navigation_map_audit_run_name_v1(self) -> str:
        if self.trace_recorder is not None:
            try:
                trace_path = Path(str(self.trace_recorder.path)).resolve()
                filename = str(trace_path.name)
                if filename.endswith(".trace.jsonl"):
                    filename = filename[: -len(".trace.jsonl")]
                elif filename.endswith(".jsonl"):
                    filename = filename[: -len(".jsonl")]
                if filename:
                    safe = "".join(
                        ch if (ch.isalnum() or ch in "._-") else "_"
                        for ch in str(filename)
                    ).strip("._-")
                    if safe:
                        return str(safe)
            except Exception:
                pass
        fallback = f"{str(self.game_id)}.{str(self.card_id)}.{int(self.action_counter)}"
        return "".join(
            ch if (ch.isalnum() or ch in "._-") else "_"
            for ch in str(fallback)
        ).strip("._-") or "navigation_map_audit"

    def _navigation_map_audit_anchor_xy_v1(self) -> tuple[int, int] | None:
        if self._previous_representation is not None:
            try:
                anchor_xy = self._current_agent_position_xy_v1(
                    self._previous_representation
                )
                if anchor_xy is not None:
                    return (int(anchor_xy[0]), int(anchor_xy[1]))
            except Exception:
                pass
        latest = (
            self._latest_navigation_state_estimate
            if isinstance(self._latest_navigation_state_estimate, dict)
            else {}
        )
        pos = latest.get("agent_pos_xy", {})
        if isinstance(pos, dict):
            x = int(pos.get("x", -1))
            y = int(pos.get("y", -1))
            if x >= 0 and y >= 0:
                return (int(x), int(y))
        return None

    def _run_navigation_map_audit_on_cleanup_v1(self) -> dict[str, Any]:
        disabled_summary: dict[str, Any] = {
            "schema_name": "active_inference_navigation_map_audit_v1",
            "schema_version": 1,
            "enabled": False,
            "reason": "disabled",
        }
        if not bool(self.navigation_map_audit_enabled):
            self._latest_navigation_map_audit_v1 = dict(disabled_summary)
            self._latest_navigation_map_audit_v1["reason"] = "disabled_by_config"
            return dict(self._latest_navigation_map_audit_v1)
        packet = self._previous_packet
        if packet is None or not packet.frame:
            self._latest_navigation_map_audit_v1 = dict(disabled_summary)
            self._latest_navigation_map_audit_v1["reason"] = "missing_frame"
            return dict(self._latest_navigation_map_audit_v1)
        anchor_xy = self._navigation_map_audit_anchor_xy_v1()
        recordings_root = Path(
            get_runtime_str("RECORDINGS_DIR", "recordings", section="runtime")
        ).resolve()
        subdir_raw = str(self.navigation_map_audit_subdir).strip()
        subdir_clean = subdir_raw.strip("/\\") or "navigation_checks"
        output_dir = (recordings_root / subdir_clean).resolve()
        summary = run_navigation_map_audit_v1(
            frame_any=packet.frame,
            anchor_xy=anchor_xy,
            navigation_map_snapshot_v1=dict(self._latest_navigation_map_snapshot_v1),
            output_dir=output_dir,
            run_name=self._navigation_map_audit_run_name_v1(),
        )
        self._latest_navigation_map_audit_v1 = dict(summary)
        return dict(self._latest_navigation_map_audit_v1)

    def cleanup(self, scorecard: Any = None) -> None:
        final_navigation_map_audit_v1 = self._run_navigation_map_audit_on_cleanup_v1()
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
                    "final_navigation_map_snapshot_v1": dict(
                        self._latest_navigation_map_snapshot_v1
                    ),
                    "final_navigation_map_audit_v1": dict(
                        final_navigation_map_audit_v1
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
