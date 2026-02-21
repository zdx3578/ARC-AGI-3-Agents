from __future__ import annotations

import math
from collections import deque
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
        coverage_sweep_min_region_visits: int = 2,
        coverage_prepass_steps: int = 192,
        coverage_prepass_relaxed_completion_ratio: float = 0.78,
        coverage_prepass_relaxed_min_known_regions: int = 16,
        coverage_prepass_relaxed_start_step_fraction: float = 0.55,
        coverage_sweep_score_margin: float = 0.42,
        coverage_resweep_interval: int = 96,
        coverage_resweep_span: int = 24,
        coverage_sweep_direction_retry_limit: int = 8,
        coverage_matrix_sweep_enabled: bool = True,
        coverage_sweep_force_in_exploit: bool = True,
        coverage_blocked_action_attempt_threshold: int = 12,
        coverage_blocked_action_rate_threshold: float = 0.65,
        coverage_blocked_edge_attempt_threshold: int = 6,
        coverage_blocked_edge_rate_threshold: float = 0.60,
        coverage_ui_side_effect_attempt_threshold: int = 8,
        coverage_ui_side_effect_rate_threshold: float = 0.60,
        coverage_low_yield_attempt_threshold: int = 10,
        coverage_low_yield_moved_rate_threshold: float = 0.70,
        coverage_low_yield_strong_change_rate_threshold: float = 0.10,
        coverage_low_yield_palette_delta_norm_threshold: float = 0.30,
        coverage_low_yield_cc_count_change_rate_threshold: float = 0.10,
        coverage_repetition_no_progress_attempt_threshold: int = 18,
        coverage_repetition_no_progress_high_info_attempt_threshold: int = 26,
        coverage_repetition_no_progress_moved_rate_threshold: float = 0.45,
        coverage_bidirectional_loop_min_edge_count: int = 3,
        coverage_bidirectional_loop_min_total_count: int = 8,
        coverage_bidirectional_loop_moved_rate_threshold: float = 0.55,
        high_info_recoverable_skip_enabled: bool = True,
        high_info_recover_edge_declared_blocked_moved_rate_threshold: float = 0.60,
        high_info_focus_release_after_first_pass: bool = True,
        high_info_focus_release_action_counter: int = -1,
        high_info_release_min_target_score: float = 0.62,
        high_info_release_min_remaining_samples: int = 1,
        high_info_release_min_queue_length: int = 1,
        navigation_confidence_gating_enabled: bool = True,
        sequence_causal_term_enabled: bool = True,
        sequence_causal_bonus_weight: float = 0.55,
        sequence_causal_penalty_weight: float = 0.35,
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
        self.coverage_sweep_min_region_visits = int(max(1, coverage_sweep_min_region_visits))
        self.coverage_prepass_steps = int(max(0, coverage_prepass_steps))
        self.coverage_prepass_relaxed_completion_ratio = self._clamp01(
            float(coverage_prepass_relaxed_completion_ratio)
        )
        self.coverage_prepass_relaxed_min_known_regions = int(
            max(4, coverage_prepass_relaxed_min_known_regions)
        )
        self.coverage_prepass_relaxed_start_step_fraction = float(
            max(0.0, min(1.0, coverage_prepass_relaxed_start_step_fraction))
        )
        self.coverage_sweep_score_margin = float(max(0.0, coverage_sweep_score_margin))
        self.coverage_resweep_interval = int(max(0, coverage_resweep_interval))
        self.coverage_resweep_span = int(max(0, coverage_resweep_span))
        self.coverage_sweep_direction_retry_limit = int(
            max(1, coverage_sweep_direction_retry_limit)
        )
        self.coverage_matrix_sweep_enabled = bool(coverage_matrix_sweep_enabled)
        self.coverage_sweep_force_in_exploit = bool(coverage_sweep_force_in_exploit)
        self.coverage_blocked_action_attempt_threshold = int(
            max(1, coverage_blocked_action_attempt_threshold)
        )
        self.coverage_blocked_action_rate_threshold = self._clamp01(
            float(coverage_blocked_action_rate_threshold)
        )
        self.coverage_blocked_edge_attempt_threshold = int(
            max(1, coverage_blocked_edge_attempt_threshold)
        )
        self.coverage_blocked_edge_rate_threshold = self._clamp01(
            float(coverage_blocked_edge_rate_threshold)
        )
        self.coverage_ui_side_effect_attempt_threshold = int(
            max(1, coverage_ui_side_effect_attempt_threshold)
        )
        self.coverage_ui_side_effect_rate_threshold = self._clamp01(
            float(coverage_ui_side_effect_rate_threshold)
        )
        self.coverage_low_yield_attempt_threshold = int(
            max(1, coverage_low_yield_attempt_threshold)
        )
        self.coverage_low_yield_moved_rate_threshold = self._clamp01(
            float(coverage_low_yield_moved_rate_threshold)
        )
        self.coverage_low_yield_strong_change_rate_threshold = self._clamp01(
            float(coverage_low_yield_strong_change_rate_threshold)
        )
        self.coverage_low_yield_palette_delta_norm_threshold = self._clamp01(
            float(coverage_low_yield_palette_delta_norm_threshold)
        )
        self.coverage_low_yield_cc_count_change_rate_threshold = self._clamp01(
            float(coverage_low_yield_cc_count_change_rate_threshold)
        )
        self.coverage_repetition_no_progress_attempt_threshold = int(
            max(1, coverage_repetition_no_progress_attempt_threshold)
        )
        self.coverage_repetition_no_progress_high_info_attempt_threshold = int(
            max(
                int(self.coverage_repetition_no_progress_attempt_threshold),
                coverage_repetition_no_progress_high_info_attempt_threshold,
            )
        )
        self.coverage_repetition_no_progress_moved_rate_threshold = self._clamp01(
            float(coverage_repetition_no_progress_moved_rate_threshold)
        )
        self.coverage_bidirectional_loop_min_edge_count = int(
            max(1, coverage_bidirectional_loop_min_edge_count)
        )
        self.coverage_bidirectional_loop_min_total_count = int(
            max(
                int(self.coverage_bidirectional_loop_min_edge_count) * 2,
                coverage_bidirectional_loop_min_total_count,
            )
        )
        self.coverage_bidirectional_loop_moved_rate_threshold = self._clamp01(
            float(coverage_bidirectional_loop_moved_rate_threshold)
        )
        self.high_info_recoverable_skip_enabled = bool(high_info_recoverable_skip_enabled)
        self.high_info_recover_edge_declared_blocked_moved_rate_threshold = self._clamp01(
            float(high_info_recover_edge_declared_blocked_moved_rate_threshold)
        )
        self.high_info_focus_release_after_first_pass = bool(
            high_info_focus_release_after_first_pass
        )
        self.high_info_focus_release_action_counter = int(
            high_info_focus_release_action_counter
        )
        self.high_info_release_min_target_score = self._clamp01(
            float(high_info_release_min_target_score)
        )
        self.high_info_release_min_remaining_samples = int(
            max(0, high_info_release_min_remaining_samples)
        )
        self.high_info_release_min_queue_length = int(
            max(0, high_info_release_min_queue_length)
        )
        self.navigation_confidence_gating_enabled = bool(
            navigation_confidence_gating_enabled
        )
        self.sequence_causal_term_enabled = bool(sequence_causal_term_enabled)
        self.sequence_causal_bonus_weight = float(max(0.0, sequence_causal_bonus_weight))
        self.sequence_causal_penalty_weight = float(max(0.0, sequence_causal_penalty_weight))
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

    @staticmethod
    def _navigation_action_direction(action_id: int) -> str:
        mapping = {
            1: "dir_u",
            2: "dir_d",
            3: "dir_l",
            4: "dir_r",
        }
        return str(mapping.get(int(action_id), "na"))

    @staticmethod
    def _parse_region_key(region_key: str) -> tuple[int, int] | None:
        try:
            sx, sy = str(region_key).split(":", 1)
            rx = int(sx)
            ry = int(sy)
        except Exception:
            return None
        if rx < 0 or ry < 0:
            return None
        return (int(rx), int(ry))

    @classmethod
    def _region_distance_from_keys(
        cls,
        source_region_key: str,
        target_region_key: str,
    ) -> int:
        source = cls._parse_region_key(source_region_key)
        target = cls._parse_region_key(target_region_key)
        if source is None or target is None:
            return 10**6
        sx, sy = source
        tx, ty = target
        return int(abs(int(sx) - int(tx)) + abs(int(sy) - int(ty)))

    def _region_graph_adjacency(
        self,
        snapshot: dict[str, Any] | None,
    ) -> dict[str, dict[str, int]]:
        adjacency: dict[str, dict[str, int]] = {}
        if not isinstance(snapshot, dict):
            return adjacency
        edges = snapshot.get("edges", [])
        if not isinstance(edges, list):
            return adjacency
        for row in edges:
            if not isinstance(row, dict):
                continue
            source = str(row.get("source_region_key", "NA"))
            target = str(row.get("target_region_key", "NA"))
            if self._parse_region_key(source) is None:
                continue
            if self._parse_region_key(target) is None:
                continue
            count = int(max(0, row.get("count", 0)))
            if count <= 0:
                continue
            source_hist = adjacency.setdefault(source, {})
            source_hist[target] = max(int(source_hist.get(target, 0)), int(count))
        return adjacency

    def _bfs_next_region_key(
        self,
        adjacency: dict[str, dict[str, int]],
        *,
        start_region_key: str,
        goal_region_key: str,
    ) -> str:
        start = str(start_region_key)
        goal = str(goal_region_key)
        if start == goal:
            return str(start)
        if start not in adjacency:
            return "NA"
        queue: deque[str] = deque([start])
        parent: dict[str, str | None] = {start: None}
        while queue:
            node = str(queue.popleft())
            neighbors = adjacency.get(node, {})
            for neighbor, _ in sorted(
                neighbors.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            ):
                next_node = str(neighbor)
                if next_node in parent:
                    continue
                parent[next_node] = node
                if next_node == goal:
                    cursor = str(next_node)
                    while parent.get(cursor) not in (None, start):
                        cursor = str(parent.get(cursor))
                    return str(cursor)
                queue.append(next_node)
        return "NA"

    @classmethod
    def _serpentine_region_rank(cls, region_key: str) -> int:
        parsed = cls._parse_region_key(region_key)
        if parsed is None:
            return 10**6
        rx, ry = parsed
        if int(ry) % 2 == 0:
            col = int(rx)
        else:
            col = int(7 - int(rx))
        return int((int(ry) * 8) + col)

    @classmethod
    def _direction_toward_region(
        cls,
        *,
        source_region_key: str,
        target_region_key: str,
    ) -> str:
        source = cls._parse_region_key(source_region_key)
        target = cls._parse_region_key(target_region_key)
        if source is None or target is None:
            return "na"
        sx, sy = source
        tx, ty = target
        dx = int(tx) - int(sx)
        dy = int(ty) - int(sy)
        if abs(dx) >= abs(dy):
            if dx < 0:
                return "dir_l"
            if dx > 0:
                return "dir_r"
        if dy < 0:
            return "dir_u"
        if dy > 0:
            return "dir_d"
        return "na"

    @staticmethod
    def _serpentine_single_pass_length() -> int:
        row_count = 8
        horizontal_span = 12
        vertical_span = 1
        return int((row_count * horizontal_span) + ((row_count - 1) * vertical_span))

    def _two_pass_serpentine_action_id(self, action_counter: int) -> int | None:
        row_count = 8
        horizontal_span = 12
        vertical_span = 1
        pass_length = int(self._serpentine_single_pass_length())
        total_length = int(2 * pass_length)
        step = int(max(0, action_counter))
        if step >= int(total_length):
            return None
        pass_index = int(step // pass_length)
        offset = int(step % pass_length)
        for row in range(row_count):
            horizontal_direction = 3 if ((row + pass_index) % 2 == 0) else 4
            if offset < horizontal_span:
                return int(horizontal_direction)
            offset -= int(horizontal_span)
            if row >= (row_count - 1):
                break
            if offset < vertical_span:
                return int(1 if pass_index == 0 else 2)
            offset -= int(vertical_span)
        return None

    def _select_fixed_two_pass_traversal_entry(
        self,
        *,
        packet: ObservationPacketV1,
        entries: list[FreeEnergyLedgerEntryV1],
        action_count_map: dict[int, int],
    ) -> tuple[FreeEnergyLedgerEntryV1 | None, dict[str, Any]]:
        diagnostics: dict[str, Any] = {
            "enabled": False,
            "mode": "inactive",
            "visit_target": int(max(1, self.coverage_sweep_min_region_visits)),
            "known_region_count": 0,
            "min_region_visit_count": 0,
            "regions_visited_at_least_target": 0,
            "current_region_key": "NA",
            "goal_region_key": "NA",
            "next_region_key": "NA",
            "desired_direction": "na",
            "cross_region_key": "NA",
            "cross_region_visit_count": 0,
            "cross_visit_target": int(max(1, self.coverage_sweep_min_region_visits)),
            "prepass_complete": False,
            "candidate_pool_size": 0,
            "relaxed_completion_ratio": 0.0,
            "relaxed_completion_threshold": float(self.coverage_prepass_relaxed_completion_ratio),
            "relaxed_completion_min_known_regions": int(
                self.coverage_prepass_relaxed_min_known_regions
            ),
            "relaxed_completion_start_step": 0,
        }
        if int(packet.levels_completed) > 0:
            diagnostics["mode"] = "levels_progressed"
            return None, diagnostics
        if int(packet.action_counter) >= int(self.coverage_prepass_steps):
            diagnostics["mode"] = "prepass_window_exhausted"
            return None, diagnostics

        navigation_entries = [
            entry
            for entry in entries
            if int(entry.candidate.action_id) in (1, 2, 3, 4)
        ]
        diagnostics["candidate_pool_size"] = int(len(navigation_entries))
        if not navigation_entries:
            diagnostics["mode"] = "no_navigation_candidates"
            return None, diagnostics

        predicted_stats_by_candidate_id: dict[str, dict[str, Any]] = {
            str(entry.candidate.candidate_id): self._candidate_predicted_region_stats(
                entry.candidate
            )
            for entry in navigation_entries
        }
        coverage_block_profile_by_candidate_id: dict[str, dict[str, Any]] = {
            str(entry.candidate.candidate_id): self._candidate_coverage_block_profile(
                entry.candidate,
                predicted_region_stats=predicted_stats_by_candidate_id.get(
                    str(entry.candidate.candidate_id),
                    {},
                ),
            )
            for entry in navigation_entries
        }
        blocked_hard_skip_count = int(
            sum(
                1
                for entry in navigation_entries
                if self._coverage_hard_skip(
                    profile=coverage_block_profile_by_candidate_id.get(
                        str(entry.candidate.candidate_id),
                        {},
                    ),
                    predicted_region_stats=predicted_stats_by_candidate_id.get(
                        str(entry.candidate.candidate_id),
                        {},
                    ),
                )
            )
        )
        diagnostics["blocked_hard_skip_count"] = int(blocked_hard_skip_count)
        diagnostics["blocked_hard_skip_applied"] = False
        scripted_action_id = self._two_pass_serpentine_action_id(
            int(packet.action_counter)
        )
        if scripted_action_id is not None:
            scripted_direction = self._navigation_action_direction(int(scripted_action_id))

            def _script_sort_key(entry: FreeEnergyLedgerEntryV1) -> tuple[Any, ...]:
                stats = predicted_stats_by_candidate_id.get(
                    str(entry.candidate.candidate_id),
                    {},
                )
                profile = coverage_block_profile_by_candidate_id.get(
                    str(entry.candidate.candidate_id),
                    {},
                )
                edge_attempts = int(stats.get("edge_attempts", 0))
                edge_blocked_rate = float(stats.get("edge_blocked_rate", 0.0))
                retry_saturated = bool(
                    edge_attempts >= int(self.coverage_sweep_direction_retry_limit)
                    and edge_blocked_rate >= 0.5
                )
                hard_skip = self._coverage_hard_skip(
                    profile=profile,
                    predicted_region_stats=stats,
                )
                soft_penalty = float(profile.get("soft_penalty", 0.0))
                predicted_region_key = str(stats.get("predicted_region_key", "NA"))
                current_region_key = str(stats.get("current_region_key", "NA"))
                predicted_stays = bool(
                    predicted_region_key == "NA"
                    or predicted_region_key == str(current_region_key)
                )
                return (
                    1 if hard_skip else 0,
                    0
                    if (
                        int(entry.candidate.action_id) == int(scripted_action_id)
                        and not hard_skip
                    )
                    else 1,
                    1 if retry_saturated else 0,
                    1 if predicted_stays else 0,
                    float(soft_penalty),
                    int(action_count_map.get(int(entry.candidate.action_id), 0)),
                    int(entry.candidate.action_id),
                    str(entry.candidate.candidate_id),
                )

            candidate_entries = list(navigation_entries)
            safe_entries = [
                entry
                for entry in navigation_entries
                if not bool(
                    self._coverage_hard_skip(
                        profile=coverage_block_profile_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            {},
                        ),
                        predicted_region_stats=predicted_stats_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            {},
                        ),
                    )
                )
            ]
            if (
                (not safe_entries)
                and int(packet.action_counter)
                >= int(max(12, int(self.coverage_sweep_direction_retry_limit) * 2))
            ):
                diagnostics["enabled"] = True
                diagnostics["mode"] = "deterministic_serpentine_blocked_escape"
                diagnostics["desired_direction"] = str(scripted_direction)
                diagnostics["blocked_hard_skip_applied"] = True
                diagnostics["prepass_complete"] = False
                return None, diagnostics
            if safe_entries:
                candidate_entries = list(safe_entries)
            ordered_entries = sorted(candidate_entries, key=_script_sort_key)
            diagnostics["enabled"] = True
            diagnostics["mode"] = "deterministic_serpentine"
            diagnostics["desired_direction"] = str(scripted_direction)
            diagnostics["blocked_hard_skip_applied"] = bool(
                len(candidate_entries) < len(navigation_entries)
            )
            return ordered_entries[0], diagnostics

        diagnostics["enabled"] = True
        diagnostics["mode"] = "serpentine_complete_bfs_fallback"
        diagnostics["prepass_complete"] = False

        sample_entry = navigation_entries[0]
        sample_candidate_id = str(sample_entry.candidate.candidate_id)
        sample_stats = predicted_stats_by_candidate_id.get(sample_candidate_id, {})
        current_region_key = str(sample_stats.get("current_region_key", "NA"))
        diagnostics["current_region_key"] = str(current_region_key)

        navigation_target = sample_entry.candidate.metadata.get(
            "navigation_target_features_v1",
            {},
        )
        if not isinstance(navigation_target, dict):
            navigation_target = {}
        region_graph_snapshot = sample_entry.candidate.metadata.get(
            "region_graph_snapshot_v1",
            {},
        )
        if not isinstance(region_graph_snapshot, dict):
            region_graph_snapshot = {}
        region_visit_histogram = region_graph_snapshot.get("region_visit_histogram", {})
        if not isinstance(region_visit_histogram, dict):
            region_visit_histogram = {}

        visit_target = int(max(1, self.coverage_sweep_min_region_visits))
        diagnostics["visit_target"] = int(visit_target)

        cross_region = navigation_target.get("cross_like_target_region", {})
        if not isinstance(cross_region, dict):
            cross_region = {}
        cross_rx = int(cross_region.get("x", -1))
        cross_ry = int(cross_region.get("y", -1))
        cross_region_key = (
            f"{cross_rx}:{cross_ry}"
            if cross_rx >= 0 and cross_ry >= 0
            else "NA"
        )
        diagnostics["cross_region_key"] = str(cross_region_key)
        cross_region_visit_count = int(
            max(
                0,
                navigation_target.get(
                    "cross_like_target_region_visit_count",
                    region_visit_histogram.get(cross_region_key, 0),
                ),
            )
        )
        diagnostics["cross_region_visit_count"] = int(cross_region_visit_count)

        normalized_region_visits: dict[str, int] = {}
        for region_key_raw, count_raw in region_visit_histogram.items():
            region_key = str(region_key_raw)
            if self._parse_region_key(region_key) is None:
                continue
            normalized_region_visits[region_key] = int(max(0, count_raw))
        if current_region_key != "NA" and self._parse_region_key(current_region_key) is not None:
            normalized_region_visits.setdefault(str(current_region_key), 0)
        if cross_region_key != "NA" and self._parse_region_key(cross_region_key) is not None:
            normalized_region_visits.setdefault(str(cross_region_key), int(cross_region_visit_count))

        known_region_count = int(len(normalized_region_visits))
        min_region_visit_count = int(min(normalized_region_visits.values(), default=0))
        regions_visited_at_least_target = int(
            sum(1 for count in normalized_region_visits.values() if int(count) >= int(visit_target))
        )
        relaxed_completion_ratio = float(
            regions_visited_at_least_target / float(max(1, known_region_count))
        )
        relaxed_completion_start_step = int(
            round(
                float(self.coverage_prepass_relaxed_start_step_fraction)
                * float(max(1, int(self.coverage_prepass_steps)))
            )
        )
        diagnostics["known_region_count"] = int(known_region_count)
        diagnostics["min_region_visit_count"] = int(min_region_visit_count)
        diagnostics["regions_visited_at_least_target"] = int(regions_visited_at_least_target)
        diagnostics["relaxed_completion_ratio"] = float(relaxed_completion_ratio)
        diagnostics["relaxed_completion_start_step"] = int(relaxed_completion_start_step)

        under_visited_regions = sorted(
            (
                (region_key, int(count))
                for region_key, count in normalized_region_visits.items()
                if int(count) < int(visit_target)
            ),
            key=lambda item: (
                int(item[1]),
                int(self._serpentine_region_rank(str(item[0]))),
                str(item[0]),
            ),
        )

        cross_under_visited = bool(
            cross_region_key != "NA"
            and self._parse_region_key(cross_region_key) is not None
            and int(cross_region_visit_count) < int(visit_target)
        )
        if (
            int(packet.action_counter) >= int(relaxed_completion_start_step)
            and int(known_region_count) >= int(self.coverage_prepass_relaxed_min_known_regions)
            and not bool(cross_under_visited)
            and float(relaxed_completion_ratio)
            >= float(self.coverage_prepass_relaxed_completion_ratio)
        ):
            diagnostics["enabled"] = True
            diagnostics["mode"] = "prepass_relaxed_complete"
            diagnostics["prepass_complete"] = True
            return None, diagnostics
        if cross_under_visited:
            goal_region_key = str(cross_region_key)
        elif under_visited_regions:
            goal_region_key = str(under_visited_regions[0][0])
        else:
            diagnostics["enabled"] = True
            diagnostics["mode"] = "prepass_complete"
            diagnostics["prepass_complete"] = True
            return None, diagnostics
        diagnostics["goal_region_key"] = str(goal_region_key)

        adjacency = self._region_graph_adjacency(region_graph_snapshot)
        next_region_key = "NA"
        if (
            self._parse_region_key(current_region_key) is not None
            and self._parse_region_key(goal_region_key) is not None
        ):
            next_region_key = self._bfs_next_region_key(
                adjacency,
                start_region_key=str(current_region_key),
                goal_region_key=str(goal_region_key),
            )
        if (
            next_region_key == "NA"
            and self._parse_region_key(goal_region_key) is not None
        ):
            next_region_key = str(goal_region_key)
        diagnostics["next_region_key"] = str(next_region_key)

        desired_direction = self._direction_toward_region(
            source_region_key=str(current_region_key),
            target_region_key=str(next_region_key),
        )
        if desired_direction not in ("dir_l", "dir_r", "dir_u", "dir_d"):
            desired_direction = self._direction_toward_region(
                source_region_key=str(current_region_key),
                target_region_key=str(goal_region_key),
            )
        diagnostics["desired_direction"] = str(desired_direction)

        if self._parse_region_key(current_region_key) is None:
            bootstrap_cycle = [1, 3, 2, 4]
            desired_action = int(
                bootstrap_cycle[int(packet.action_counter) % int(len(bootstrap_cycle))]
            )
            ordered = sorted(
                navigation_entries,
                key=lambda entry: (
                    0 if int(entry.candidate.action_id) == int(desired_action) else 1,
                    int(action_count_map.get(int(entry.candidate.action_id), 0)),
                    int(entry.candidate.action_id),
                    str(entry.candidate.candidate_id),
                ),
            )
            diagnostics["enabled"] = True
            diagnostics["mode"] = "bootstrap_cycle"
            return ordered[0], diagnostics

        def _hard_sort_key(entry: FreeEnergyLedgerEntryV1) -> tuple[Any, ...]:
            stats = predicted_stats_by_candidate_id.get(
                str(entry.candidate.candidate_id),
                {},
            )
            profile = coverage_block_profile_by_candidate_id.get(
                str(entry.candidate.candidate_id),
                {},
            )
            entry_direction = self._entry_navigation_direction(entry)
            edge_attempts = int(stats.get("edge_attempts", 0))
            edge_blocked_rate = float(stats.get("edge_blocked_rate", 0.0))
            direction_retry_saturated = bool(
                edge_attempts >= int(self.coverage_sweep_direction_retry_limit)
                and edge_blocked_rate >= 0.5
            )
            predicted_region_key = str(stats.get("predicted_region_key", "NA"))
            current_key = str(stats.get("current_region_key", "NA"))
            hard_skip = self._coverage_hard_skip(
                profile=profile,
                predicted_region_stats=stats,
            )
            soft_penalty = float(profile.get("soft_penalty", 0.0))
            predicted_stays = bool(
                predicted_region_key == "NA"
                or predicted_region_key == str(current_key)
            )
            return (
                1 if hard_skip else 0,
                0
                if (
                    desired_direction in ("dir_l", "dir_r", "dir_u", "dir_d")
                    and str(entry_direction) == str(desired_direction)
                    and not direction_retry_saturated
                    and not hard_skip
                )
                else 1,
                0
                if (
                    predicted_region_key != "NA"
                    and predicted_region_key != str(current_key)
                )
                else 1,
                1 if predicted_stays else 0,
                float(soft_penalty),
                int(
                    self._region_distance_from_keys(
                        str(predicted_region_key),
                        str(goal_region_key),
                    )
                ),
                int(stats.get("predicted_region_visit_count", 10**6)),
                int(stats.get("edge_attempts", 10**6)),
                float(stats.get("edge_blocked_rate", 1.0)),
                int(action_count_map.get(int(entry.candidate.action_id), 0)),
                int(entry.candidate.action_id),
                str(entry.candidate.candidate_id),
            )

        candidate_entries = list(navigation_entries)
        safe_entries = [
            entry
            for entry in navigation_entries
            if not bool(
                self._coverage_hard_skip(
                    profile=coverage_block_profile_by_candidate_id.get(
                        str(entry.candidate.candidate_id),
                        {},
                    ),
                    predicted_region_stats=predicted_stats_by_candidate_id.get(
                        str(entry.candidate.candidate_id),
                        {},
                    ),
                )
            )
        ]
        if (
            (not safe_entries)
            and int(packet.action_counter)
            >= int(max(12, int(self.coverage_sweep_direction_retry_limit) * 2))
        ):
            diagnostics["enabled"] = True
            diagnostics["mode"] = "bfs_two_pass_blocked_escape"
            diagnostics["blocked_hard_skip_applied"] = True
            return None, diagnostics
        if safe_entries:
            candidate_entries = list(safe_entries)
        ordered_entries = sorted(candidate_entries, key=_hard_sort_key)
        diagnostics["enabled"] = True
        diagnostics["mode"] = "bfs_two_pass"
        diagnostics["blocked_hard_skip_applied"] = bool(
            len(candidate_entries) < len(navigation_entries)
        )
        return ordered_entries[0], diagnostics

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
        navigation_semantic = candidate.metadata.get("navigation_semantic_features_v1", {})
        if not isinstance(navigation_semantic, dict):
            navigation_semantic = {}
        nav_semantic_enabled = bool(navigation_semantic.get("enabled", False))
        nav_semantic_compare_count = int(max(0, navigation_semantic.get("compare_count", 0)))
        nav_semantic_mismatch_rate = self._clamp01(
            float(navigation_semantic.get("mismatch_rate", 0.0))
        )
        nav_semantic_confidence = self._clamp01(
            float(navigation_semantic.get("confidence", 1.0))
        )
        navigation_confidence_gate = 1.0
        if self.navigation_confidence_gating_enabled and nav_semantic_enabled:
            navigation_confidence_gate = float(nav_semantic_confidence)
            if nav_semantic_compare_count >= 12 and nav_semantic_mismatch_rate >= 0.40:
                navigation_confidence_gate = float(
                    max(0.0, min(1.0, navigation_confidence_gate * 0.75))
                )
        navigation_relocalization_bonus = float(
            max(
                0.0,
                (1.0 - navigation_confidence_gate)
                * max(action_frontier_novelty, state_action_frontier_novelty)
                * max(translation_probability, action_moved_rate),
            )
        )

        sequence_causal = candidate.metadata.get("sequence_causal_features_v1", {})
        if not isinstance(sequence_causal, dict):
            sequence_causal = {}
        sequence_causal_enabled = bool(
            self.sequence_causal_term_enabled and sequence_causal.get("enabled", False)
        )
        sequence_causal_active = bool(sequence_causal.get("active", False))
        sequence_causal_stage = str(sequence_causal.get("stage", "idle"))
        sequence_causal_bonus_hint = float(max(0.0, sequence_causal.get("bonus_hint", 0.0)))
        sequence_causal_penalty_hint = float(max(0.0, sequence_causal.get("penalty_hint", 0.0)))
        sequence_causal_bonus = 0.0
        sequence_causal_penalty = 0.0
        if sequence_causal_enabled and sequence_causal_active:
            sequence_causal_bonus = float(
                self.sequence_causal_bonus_weight * sequence_causal_bonus_hint
            )
            sequence_causal_penalty = float(
                self.sequence_causal_penalty_weight * sequence_causal_penalty_hint
            )
            if bool(sequence_causal.get("verify_action_candidate", False)):
                sequence_causal_bonus = float(
                    sequence_causal_bonus + (0.30 * self.sequence_causal_bonus_weight)
                )
            elif (
                int(candidate.action_id) in (1, 2, 3, 4)
                and bool(sequence_causal.get("moves_away_from_target", False))
            ):
                sequence_causal_penalty = float(
                    sequence_causal_penalty + (0.20 * self.sequence_causal_penalty_weight)
                )
        high_info_focus = self._candidate_high_info_focus_features(candidate)
        high_info_focus_enabled = bool(high_info_focus.get("enabled", False))
        high_info_focus_active = bool(high_info_focus.get("active", False))
        high_info_focus_stage = str(high_info_focus.get("stage", "idle"))
        high_info_target_score = self._clamp01(float(high_info_focus.get("target_score", 0.0)))
        high_info_bonus_hint = float(max(0.0, high_info_focus.get("bonus_hint", 0.0)))
        high_info_penalty_hint = float(max(0.0, high_info_focus.get("penalty_hint", 0.0)))
        high_info_verify_action_candidate = bool(
            high_info_focus.get("verify_action_candidate", False)
        )
        high_info_bonus = 0.0
        high_info_penalty = 0.0
        if high_info_focus_enabled and high_info_focus_active:
            high_info_bonus = float(
                (0.55 + (0.45 * high_info_target_score)) * high_info_bonus_hint
            )
            high_info_penalty = float(
                (0.40 + (0.60 * high_info_target_score)) * high_info_penalty_hint
            )
            if high_info_verify_action_candidate:
                high_info_bonus = float(high_info_bonus + (0.45 * high_info_target_score))
        orientation_alignment = self._candidate_orientation_alignment_features(candidate)
        orientation_alignment_enabled = bool(
            int(candidate.action_id) in (1, 2, 3, 4)
            and orientation_alignment.get("enabled", False)
            and orientation_alignment.get("detected", False)
        )
        orientation_alignment_aligned = bool(orientation_alignment.get("aligned", False))
        orientation_alignment_similarity = self._clamp01(
            float(orientation_alignment.get("similarity", 0.0))
        )
        orientation_alignment_mismatch = float(max(0.0, 1.0 - orientation_alignment_similarity))
        orientation_alignment_bonus_hint = self._clamp01(
            float(orientation_alignment.get("bonus_hint", 0.0))
        )
        orientation_alignment_penalty_hint = self._clamp01(
            float(orientation_alignment.get("penalty_hint", 0.0))
        )
        orientation_action_improve_rate = self._clamp01(
            float(orientation_alignment.get("action_improve_rate", 0.0))
        )
        orientation_action_regress_rate = self._clamp01(
            float(orientation_alignment.get("action_regress_rate", 0.0))
        )
        orientation_action_aligned_hit_rate = self._clamp01(
            float(orientation_alignment.get("action_aligned_hit_rate", 0.0))
        )
        orientation_alignment_bonus = 0.0
        orientation_alignment_penalty = 0.0
        if orientation_alignment_enabled:
            orientation_alignment_bonus = float(
                orientation_alignment_bonus_hint
                * (0.45 + (0.55 * orientation_alignment_mismatch))
            )
            orientation_alignment_penalty = float(
                orientation_alignment_penalty_hint
                * (0.30 + (0.70 * orientation_alignment_mismatch))
            )
            orientation_alignment_bonus += float(
                0.20 * orientation_action_improve_rate * orientation_alignment_mismatch
            )
            orientation_alignment_penalty += float(
                0.15 * orientation_action_regress_rate * orientation_alignment_mismatch
            )
            if orientation_alignment_aligned:
                orientation_alignment_bonus = float(
                    max(
                        orientation_alignment_bonus,
                        0.18 * orientation_action_aligned_hit_rate,
                    )
                )
        region_action_semantics = self._candidate_region_action_semantics(candidate)
        region_action_semantics_enabled = bool(
            int(candidate.action_id) in (1, 2, 3, 4)
            and region_action_semantics.get("enabled", False)
        )
        region_info_trigger_score = self._clamp01(
            float(region_action_semantics.get("info_trigger_score", 0.0))
        )
        region_palette_change_rate = self._clamp01(
            float(region_action_semantics.get("palette_change_rate", 0.0))
        )
        region_palette_delta_mean_norm = self._clamp01(
            float(region_action_semantics.get("palette_delta_mean_norm", 0.0))
        )
        region_cc_count_change_rate = self._clamp01(
            float(region_action_semantics.get("cc_count_change_rate", 0.0))
        )
        region_strong_change_rate = self._clamp01(
            float(region_action_semantics.get("strong_change_rate", 0.0))
        )
        region_progress_rate = self._clamp01(
            float(region_action_semantics.get("progress_rate", 0.0))
        )
        region_ui_side_effect_rate = self._clamp01(
            float(region_action_semantics.get("ui_side_effect_rate", 0.0))
        )
        region_terminal_failure_rate = self._clamp01(
            float(region_action_semantics.get("terminal_failure_rate", 0.0))
        )
        region_coupling_signal_score = self._clamp01(
            float(region_action_semantics.get("coupling_signal_score", 0.0))
        )
        region_coupling_signal_kind = str(
            region_action_semantics.get("coupling_signal_kind", "unknown")
        )
        region_semantics_edge_status = str(
            region_action_semantics.get("edge_status", "unknown")
        )
        region_semantics_moved_rate = self._clamp01(
            float(region_action_semantics.get("moved_rate", 0.0))
        )
        color_coupling_signal = 0.0
        color_coupling_bonus = 0.0
        color_coupling_penalty = 0.0
        ui_side_effect_penalty = 0.0
        ui_suppression = 0.0
        if region_action_semantics_enabled:
            region_channel_signal = self._clamp01(
                (0.35 * region_info_trigger_score)
                + (0.20 * region_palette_change_rate)
                + (0.15 * region_palette_delta_mean_norm)
                + (0.15 * region_cc_count_change_rate)
                + (0.15 * region_strong_change_rate)
            )
            color_coupling_signal = self._clamp01(
                max(
                    float(region_coupling_signal_score),
                    float(region_channel_signal),
                    float(region_progress_rate),
                )
            )
            color_coupling_bonus = float(
                color_coupling_signal
                * (0.30 + (0.70 * frontier_novelty))
                * (0.30 + (0.70 * region_novelty))
            )
            if region_semantics_edge_status == "blocked":
                color_coupling_penalty = float(
                    0.30 * color_coupling_signal * (0.55 + (0.45 * blocked_probability))
                )
            elif region_semantics_moved_rate <= 0.20:
                color_coupling_penalty = float(0.12 * color_coupling_signal)
            ui_suppression = self._clamp01(
                (0.85 * region_ui_side_effect_rate)
                + (0.55 * region_terminal_failure_rate)
            )
            if ui_suppression > 0.0:
                color_coupling_signal = self._clamp01(
                    color_coupling_signal * (1.0 - ui_suppression)
                )
                color_coupling_bonus = float(color_coupling_bonus * (1.0 - ui_suppression))
                ui_side_effect_penalty = float(
                    0.22 * ui_suppression * (0.55 + (0.45 * blocked_probability))
                )
                color_coupling_penalty = float(
                    max(0.0, color_coupling_penalty + ui_side_effect_penalty)
                )
        navigation_projection = candidate.metadata.get(
            "navigation_step_projection_features_v1",
            {},
        )
        if not isinstance(navigation_projection, dict):
            navigation_projection = {}
        projection_enabled = bool(navigation_projection.get("enabled", False))
        projection_source = str(navigation_projection.get("projection_source", "none"))
        projection_distance_before = float(navigation_projection.get("distance_before", -1.0))
        projection_distance_after = float(navigation_projection.get("distance_after", -1.0))
        projection_distance_delta = float(navigation_projection.get("distance_delta", 0.0))
        projection_distance_delta_normalized = float(
            navigation_projection.get("distance_delta_normalized", 0.0)
        )
        projection_alignment = float(
            max(-1.0, min(1.0, navigation_projection.get("alignment", 0.0)))
        )
        projection_confidence = self._clamp01(
            float(navigation_projection.get("confidence", 0.0))
        )
        projection_bonus_hint = float(max(0.0, navigation_projection.get("bonus_hint", 0.0)))
        projection_penalty_hint = float(
            max(0.0, navigation_projection.get("penalty_hint", 0.0))
        )
        projection_toward_bonus = float(
            max(0.0, projection_bonus_hint)
            * (0.70 + (0.30 * max(0.0, projection_alignment)))
        )
        projection_away_penalty = float(
            max(0.0, projection_penalty_hint)
            * (0.70 + (0.30 * max(0.0, -projection_alignment)))
        )
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
        geometry_term_gate = (
            float(navigation_confidence_gate)
            if self.navigation_confidence_gating_enabled
            else 1.0
        )
        key_target_escape_bonus_effective = float(
            key_target_escape_bonus * geometry_term_gate
        )
        key_target_away_penalty_effective = float(
            key_target_away_penalty * geometry_term_gate
        )
        projection_toward_bonus_effective = float(
            projection_toward_bonus * projection_confidence * geometry_term_gate
        )
        projection_away_penalty_effective = float(
            projection_away_penalty * projection_confidence * geometry_term_gate
        )
        high_info_bonus_effective = float(
            high_info_bonus * geometry_term_gate
        )
        high_info_penalty_effective = float(
            high_info_penalty * geometry_term_gate
        )
        orientation_alignment_bonus_effective = float(
            orientation_alignment_bonus * geometry_term_gate
        )
        orientation_alignment_penalty_effective = float(
            orientation_alignment_penalty * geometry_term_gate
        )
        color_coupling_bonus_effective = float(
            color_coupling_bonus * (0.65 + (0.35 * geometry_term_gate))
        )
        color_coupling_penalty_effective = float(
            color_coupling_penalty * (0.65 + (0.35 * geometry_term_gate))
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
                + (0.25 * key_target_away_penalty_effective)
                + (0.35 * projection_away_penalty_effective)
                + (0.40 * high_info_penalty_effective)
                + (0.28 * orientation_alignment_penalty_effective)
                + (0.18 * color_coupling_penalty_effective)
                + (0.10 * coverage_repeat_penalty)
                - (
                    0.30
                    * key_target_escape_bonus_effective
                    * max(0.25, float(translation_probability))
                ),
            )
        )
        operability_risk = float(
            max(
                0.0,
                operability_risk
                - (
                    0.45
                    * projection_toward_bonus_effective
                    * max(0.25, float(translation_probability))
                ),
            )
        )
        operability_risk = float(
            max(
                0.0,
                operability_risk
                - (0.40 * orientation_alignment_bonus_effective)
                - (0.28 * color_coupling_bonus_effective),
            )
        )
        operability_risk = float(
            max(0.0, operability_risk - (0.55 * high_info_bonus_effective))
        )
        operability_risk = float(
            max(
                0.0,
                operability_risk
                + (0.18 * sequence_causal_penalty)
                - (0.22 * sequence_causal_bonus),
            )
        )
        operability_risk = float(
            max(
                0.0,
                operability_risk
                - (0.12 * coverage_frontier_bonus)
                - (0.18 * navigation_relocalization_bonus),
            )
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
            "projection_enabled": bool(projection_enabled),
            "projection_source": str(projection_source),
            "projection_confidence": float(projection_confidence),
            "projection_distance_before": float(projection_distance_before),
            "projection_distance_after": float(projection_distance_after),
            "projection_distance_delta": float(projection_distance_delta),
            "projection_distance_delta_normalized": float(
                projection_distance_delta_normalized
            ),
            "projection_alignment": float(projection_alignment),
            "projection_toward_bonus": float(projection_toward_bonus),
            "projection_away_penalty": float(projection_away_penalty),
            "projection_toward_bonus_effective": float(
                projection_toward_bonus_effective
            ),
            "projection_away_penalty_effective": float(
                projection_away_penalty_effective
            ),
            "high_info_focus_enabled": bool(high_info_focus_enabled),
            "high_info_focus_active": bool(high_info_focus_active),
            "high_info_focus_stage": str(high_info_focus_stage),
            "high_info_target_score": float(high_info_target_score),
            "high_info_verify_action_candidate": bool(high_info_verify_action_candidate),
            "high_info_bonus": float(high_info_bonus),
            "high_info_penalty": float(high_info_penalty),
            "high_info_bonus_effective": float(high_info_bonus_effective),
            "high_info_penalty_effective": float(high_info_penalty_effective),
            "orientation_alignment_enabled": bool(orientation_alignment_enabled),
            "orientation_alignment_aligned": bool(orientation_alignment_aligned),
            "orientation_alignment_similarity": float(orientation_alignment_similarity),
            "orientation_alignment_bonus": float(orientation_alignment_bonus),
            "orientation_alignment_penalty": float(orientation_alignment_penalty),
            "orientation_alignment_bonus_effective": float(
                orientation_alignment_bonus_effective
            ),
            "orientation_alignment_penalty_effective": float(
                orientation_alignment_penalty_effective
            ),
            "orientation_rotation_bucket": str(
                orientation_alignment.get("rotation_bucket", "rot_unknown")
            ),
            "orientation_best_rotation_deg": int(
                orientation_alignment.get("best_rotation_deg", -1)
            ),
            "region_action_semantics_enabled": bool(region_action_semantics_enabled),
            "region_info_trigger_score": float(region_info_trigger_score),
            "region_palette_change_rate": float(region_palette_change_rate),
            "region_palette_delta_mean_norm": float(region_palette_delta_mean_norm),
            "region_cc_count_change_rate": float(region_cc_count_change_rate),
            "region_strong_change_rate": float(region_strong_change_rate),
            "region_progress_rate": float(region_progress_rate),
            "region_coupling_signal_score": float(region_coupling_signal_score),
            "region_coupling_signal_kind": str(region_coupling_signal_kind),
            "region_semantics_edge_status": str(region_semantics_edge_status),
            "color_coupling_signal": float(color_coupling_signal),
            "color_coupling_bonus": float(color_coupling_bonus),
            "color_coupling_penalty": float(color_coupling_penalty),
            "color_coupling_bonus_effective": float(color_coupling_bonus_effective),
            "color_coupling_penalty_effective": float(color_coupling_penalty_effective),
            "navigation_confidence_gating_enabled": bool(
                self.navigation_confidence_gating_enabled
            ),
            "navigation_confidence_gate": float(navigation_confidence_gate),
            "navigation_semantic_confidence": float(nav_semantic_confidence),
            "navigation_semantic_compare_count": int(nav_semantic_compare_count),
            "navigation_semantic_mismatch_rate": float(nav_semantic_mismatch_rate),
            "navigation_relocalization_bonus": float(navigation_relocalization_bonus),
            "sequence_causal_enabled": bool(sequence_causal_enabled),
            "sequence_causal_active": bool(sequence_causal_active),
            "sequence_causal_stage": str(sequence_causal_stage),
            "sequence_causal_bonus": float(sequence_causal_bonus),
            "sequence_causal_penalty": float(sequence_causal_penalty),
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
                    + (0.18 * key_target_away_penalty_effective)
                    + (0.24 * projection_away_penalty_effective)
                    + (0.30 * high_info_penalty_effective)
                    + (0.24 * orientation_alignment_penalty_effective)
                    + (0.20 * color_coupling_penalty_effective)
                    + (0.20 * coverage_repeat_penalty)
                    + (0.28 * sequence_causal_penalty)
                    - (0.25 * leave_high_revisit_potential)
                    - (0.30 * key_target_escape_bonus_effective)
                    - (0.34 * projection_toward_bonus_effective)
                    - (0.38 * high_info_bonus_effective)
                    - (0.30 * orientation_alignment_bonus_effective)
                    - (0.32 * color_coupling_bonus_effective)
                    - (0.35 * coverage_frontier_bonus)
                    - (0.45 * sequence_causal_bonus)
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
        habit_risk += float(max(0.0, 0.25 * key_target_away_penalty_effective))
        habit_risk += float(max(0.0, 0.30 * projection_away_penalty_effective))
        habit_risk += float(max(0.0, 0.34 * high_info_penalty_effective))
        habit_risk += float(max(0.0, 0.26 * orientation_alignment_penalty_effective))
        habit_risk += float(max(0.0, 0.22 * color_coupling_penalty_effective))
        habit_risk += float(max(0.0, 0.24 * ui_side_effect_penalty))
        habit_risk += float(max(0.0, 0.30 * coverage_repeat_penalty))
        habit_risk += float(max(0.0, 0.24 * sequence_causal_penalty))
        habit_risk = float(max(0.0, habit_risk - (0.22 * key_target_escape_bonus_effective)))
        habit_risk = float(max(0.0, habit_risk - (0.28 * projection_toward_bonus_effective)))
        habit_risk = float(max(0.0, habit_risk - (0.34 * high_info_bonus_effective)))
        habit_risk = float(max(0.0, habit_risk - (0.25 * orientation_alignment_bonus_effective)))
        habit_risk = float(max(0.0, habit_risk - (0.26 * color_coupling_bonus_effective)))
        habit_risk = float(max(0.0, habit_risk - (0.26 * coverage_frontier_bonus)))
        habit_risk = float(max(0.0, habit_risk - (0.30 * sequence_causal_bonus)))
        progress_information_gain = float(
            max(
                0.0,
                ig_causal_mapping
                * (0.5 + (0.5 * progress_gap_ratio))
                * (
                    0.30
                    + (0.45 * frontier_novelty)
                    + (0.25 * evidence_novelty)
                    + (0.20 * key_target_escape_bonus_effective)
                    + (0.30 * projection_toward_bonus_effective)
                    + (0.34 * high_info_bonus_effective)
                    + (0.28 * orientation_alignment_bonus_effective)
                    + (0.34 * color_coupling_bonus_effective)
                    + (0.20 * color_coupling_signal)
                    + (0.28 * coverage_frontier_bonus)
                    + (0.35 * sequence_causal_bonus)
                    + (0.20 * navigation_relocalization_bonus)
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
            "key_target_escape_bonus_effective": float(key_target_escape_bonus_effective),
            "key_target_away_penalty_effective": float(key_target_away_penalty_effective),
            "projection_enabled": bool(projection_enabled),
            "projection_source": str(projection_source),
            "projection_confidence": float(projection_confidence),
            "projection_distance_before": float(projection_distance_before),
            "projection_distance_after": float(projection_distance_after),
            "projection_distance_delta": float(projection_distance_delta),
            "projection_distance_delta_normalized": float(
                projection_distance_delta_normalized
            ),
            "projection_alignment": float(projection_alignment),
            "projection_toward_bonus": float(projection_toward_bonus),
            "projection_away_penalty": float(projection_away_penalty),
            "projection_toward_bonus_effective": float(
                projection_toward_bonus_effective
            ),
            "projection_away_penalty_effective": float(
                projection_away_penalty_effective
            ),
            "high_info_focus_enabled": bool(high_info_focus_enabled),
            "high_info_focus_active": bool(high_info_focus_active),
            "high_info_focus_stage": str(high_info_focus_stage),
            "high_info_target_score": float(high_info_target_score),
            "high_info_verify_action_candidate": bool(high_info_verify_action_candidate),
            "high_info_bonus": float(high_info_bonus),
            "high_info_penalty": float(high_info_penalty),
            "high_info_bonus_effective": float(high_info_bonus_effective),
            "high_info_penalty_effective": float(high_info_penalty_effective),
            "orientation_alignment_enabled": bool(orientation_alignment_enabled),
            "orientation_alignment_aligned": bool(orientation_alignment_aligned),
            "orientation_alignment_similarity": float(orientation_alignment_similarity),
            "orientation_alignment_bonus": float(orientation_alignment_bonus),
            "orientation_alignment_penalty": float(orientation_alignment_penalty),
            "orientation_alignment_bonus_effective": float(
                orientation_alignment_bonus_effective
            ),
            "orientation_alignment_penalty_effective": float(
                orientation_alignment_penalty_effective
            ),
            "orientation_rotation_bucket": str(
                orientation_alignment.get("rotation_bucket", "rot_unknown")
            ),
            "orientation_best_rotation_deg": int(
                orientation_alignment.get("best_rotation_deg", -1)
            ),
            "geometry_term_gate": float(geometry_term_gate),
            "region_action_semantics_enabled": bool(region_action_semantics_enabled),
            "region_info_trigger_score": float(region_info_trigger_score),
            "region_palette_change_rate": float(region_palette_change_rate),
            "region_palette_delta_mean_norm": float(region_palette_delta_mean_norm),
            "region_cc_count_change_rate": float(region_cc_count_change_rate),
            "region_strong_change_rate": float(region_strong_change_rate),
            "region_progress_rate": float(region_progress_rate),
            "region_ui_side_effect_rate": float(region_ui_side_effect_rate),
            "region_terminal_failure_rate": float(region_terminal_failure_rate),
            "region_ui_suppression": float(ui_suppression),
            "region_coupling_signal_score": float(region_coupling_signal_score),
            "region_coupling_signal_kind": str(region_coupling_signal_kind),
            "region_semantics_edge_status": str(region_semantics_edge_status),
            "color_coupling_signal": float(color_coupling_signal),
            "color_coupling_bonus": float(color_coupling_bonus),
            "color_coupling_penalty": float(color_coupling_penalty),
            "ui_side_effect_penalty": float(ui_side_effect_penalty),
            "color_coupling_bonus_effective": float(color_coupling_bonus_effective),
            "color_coupling_penalty_effective": float(color_coupling_penalty_effective),
            "coverage_enabled": bool(coverage_enabled),
            "coverage_confidence": float(coverage_confidence),
            "coverage_next_region_key": str(coverage_next_region_key),
            "coverage_next_region_visit_count": int(coverage_next_region_visit_count),
            "coverage_region_novelty": float(coverage_region_novelty),
            "coverage_frontier_bonus": float(coverage_frontier_bonus),
            "coverage_repeat_penalty": float(coverage_repeat_penalty),
            "navigation_confidence_gating_enabled": bool(
                self.navigation_confidence_gating_enabled
            ),
            "navigation_confidence_gate": float(navigation_confidence_gate),
            "navigation_semantic_confidence": float(nav_semantic_confidence),
            "navigation_semantic_compare_count": int(nav_semantic_compare_count),
            "navigation_semantic_mismatch_rate": float(nav_semantic_mismatch_rate),
            "navigation_relocalization_bonus": float(navigation_relocalization_bonus),
            "sequence_causal_enabled": bool(sequence_causal_enabled),
            "sequence_causal_active": bool(sequence_causal_active),
            "sequence_causal_stage": str(sequence_causal_stage),
            "sequence_causal_bonus": float(sequence_causal_bonus),
            "sequence_causal_penalty": float(sequence_causal_penalty),
            "sequence_causal_features_v1": {
                str(k): v
                for (k, v) in sequence_causal.items()
            },
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
        navigation_confidence_gate = self._clamp01(
            float(level1_terms.get("navigation_confidence_gate", 1.0))
        )
        navigation_relocalization_bonus = self._clamp01(
            float(level1_terms.get("navigation_relocalization_bonus", 0.0))
        )
        sequence_causal_bonus = self._clamp01(
            float(level2_terms.get("sequence_causal_bonus", 0.0))
        )
        sequence_causal_penalty = self._clamp01(
            float(level2_terms.get("sequence_causal_penalty", 0.0))
        )
        orientation_alignment_bonus_effective = self._clamp01(
            float(level2_terms.get("orientation_alignment_bonus_effective", 0.0))
        )
        color_coupling_signal = self._clamp01(
            float(level2_terms.get("color_coupling_signal", 0.0))
        )
        sequence_causal_active = bool(level2_terms.get("sequence_causal_active", False))
        stuck_score = self._clamp01(
            (0.30 * stagnation_context)
            + (0.30 * region_revisit_ratio)
            + (0.25 * translation_no_progress_amplified)
            + (0.15 * habit_pressure)
            + (0.10 * sequence_causal_penalty)
            + (0.10 * (1.0 - navigation_confidence_gate))
            - (0.12 * orientation_alignment_bonus_effective)
            - (0.10 * color_coupling_signal)
        )

        phase_focus = "balanced"
        explain_escape_pressure = 0.0
        low_navigation_confidence = bool(
            self.navigation_confidence_gating_enabled and navigation_confidence_gate < 0.60
        )
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
            if low_navigation_confidence:
                hierarchy_weights["progress_risk"] = self._scaled_weight(
                    hierarchy_weights["progress_risk"],
                    0.84,
                )
                hierarchy_weights["progress_information_gain"] = self._scaled_weight(
                    hierarchy_weights["progress_information_gain"],
                    1.0 + (0.45 * (1.0 - navigation_confidence_gate)),
                )
                weights["information_gain_mechanism_dynamics"] = self._scaled_weight(
                    weights["information_gain_mechanism_dynamics"],
                    1.0 + (0.35 * (1.0 - navigation_confidence_gate)),
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
            if low_navigation_confidence:
                hierarchy_weights["progress_risk"] = self._scaled_weight(
                    hierarchy_weights["progress_risk"],
                    0.82,
                )
                hierarchy_weights["progress_information_gain"] = self._scaled_weight(
                    hierarchy_weights["progress_information_gain"],
                    1.0 + (0.60 * (1.0 - navigation_confidence_gate)),
                )
                weights["information_gain_mechanism_dynamics"] = self._scaled_weight(
                    weights["information_gain_mechanism_dynamics"],
                    1.0 + (0.40 * (1.0 - navigation_confidence_gate)),
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
            if low_navigation_confidence:
                hierarchy_weights["progress_risk"] = self._scaled_weight(
                    hierarchy_weights["progress_risk"],
                    0.78,
                )
                hierarchy_weights["progress_information_gain"] = self._scaled_weight(
                    hierarchy_weights["progress_information_gain"],
                    1.0 + (0.75 * (1.0 - navigation_confidence_gate)),
                )
                weights["information_gain_mechanism_dynamics"] = self._scaled_weight(
                    weights["information_gain_mechanism_dynamics"],
                    1.0 + (0.55 * (1.0 - navigation_confidence_gate)),
                )
            if self.sequence_causal_term_enabled and sequence_causal_active:
                hierarchy_weights["progress_risk"] = self._scaled_weight(
                    hierarchy_weights["progress_risk"],
                    max(0.72, 1.0 - (0.28 * sequence_causal_bonus)),
                )
                hierarchy_weights["progress_information_gain"] = self._scaled_weight(
                    hierarchy_weights["progress_information_gain"],
                    1.0 + (0.40 * sequence_causal_bonus),
                )
                hierarchy_weights["habit_risk"] = self._scaled_weight(
                    hierarchy_weights["habit_risk"],
                    max(0.78, 1.0 - (0.25 * sequence_causal_bonus)),
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
            "navigation_confidence_gate": float(navigation_confidence_gate),
            "navigation_relocalization_bonus": float(navigation_relocalization_bonus),
            "low_navigation_confidence": bool(low_navigation_confidence),
            "sequence_causal_active": bool(sequence_causal_active),
            "sequence_causal_bonus": float(sequence_causal_bonus),
            "sequence_causal_penalty": float(sequence_causal_penalty),
            "orientation_alignment_bonus_effective": float(
                orientation_alignment_bonus_effective
            ),
            "color_coupling_signal": float(color_coupling_signal),
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
            "min_region_visit_count": int(max(0, raw.get("min_region_visit_count", 0))),
            "regions_visited_at_least_twice": int(
                max(0, raw.get("regions_visited_at_least_twice", 0))
            ),
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
            "empirical_transition_frontier_key": str(
                raw.get("empirical_transition_frontier_key", "NA")
            ),
            "empirical_transition_frontier_visit_count": int(
                max(0, raw.get("empirical_transition_frontier_visit_count", 0))
            ),
            "empirical_transition_frontier_confidence": self._clamp01(
                float(raw.get("empirical_transition_frontier_confidence", 0.0))
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

    def _candidate_sequence_causal_features(
        self,
        candidate: ActionCandidateV1,
    ) -> dict[str, Any]:
        raw = candidate.metadata.get("sequence_causal_features_v1", {})
        if not isinstance(raw, dict):
            raw = {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "active": bool(raw.get("active", False)),
            "stage": str(raw.get("stage", "idle")),
            "trigger_region_key": str(raw.get("trigger_region_key", "2:4")),
            "target_region_key": str(raw.get("target_region_key", "4:1")),
            "current_region_key": str(raw.get("current_region_key", "NA")),
            "predicted_region_key": str(raw.get("predicted_region_key", "NA")),
            "steps_remaining": int(max(0, raw.get("steps_remaining", 0))),
            "advances_to_target": bool(raw.get("advances_to_target", False)),
            "moves_away_from_target": bool(raw.get("moves_away_from_target", False)),
            "reaches_target": bool(raw.get("reaches_target", False)),
            "verify_action_candidate": bool(raw.get("verify_action_candidate", False)),
            "bonus_hint": float(max(0.0, raw.get("bonus_hint", 0.0))),
            "penalty_hint": float(max(0.0, raw.get("penalty_hint", 0.0))),
            "predicted_distance_to_target": int(
                max(0, raw.get("predicted_distance_to_target", 10**6))
            ),
            "distance_delta_to_target": int(raw.get("distance_delta_to_target", 0)),
        }

    def _candidate_high_info_focus_features(
        self,
        candidate: ActionCandidateV1,
    ) -> dict[str, Any]:
        raw = candidate.metadata.get("high_info_focus_features_v1", {})
        if not isinstance(raw, dict):
            raw = {}
        verify_action_ids = raw.get("verify_action_ids", [])
        if not isinstance(verify_action_ids, list):
            verify_action_ids = []
        return {
            "enabled": bool(raw.get("enabled", False)),
            "active": bool(raw.get("active", False)),
            "stage": str(raw.get("stage", "idle")),
            "current_region_key": str(raw.get("current_region_key", "NA")),
            "target_region_key": str(raw.get("target_region_key", "NA")),
            "predicted_region_key": str(raw.get("predicted_region_key", "NA")),
            "queue_length": int(max(0, raw.get("queue_length", 0))),
            "steps_remaining": int(max(0, raw.get("steps_remaining", 0))),
            "target_score": self._clamp01(float(raw.get("target_score", 0.0))),
            "target_sample_count": int(max(0, raw.get("target_sample_count", 0))),
            "remaining_samples": int(max(0, raw.get("remaining_samples", 0))),
            "distance_before": int(raw.get("distance_before", 10**6)),
            "distance_after": int(raw.get("distance_after", 10**6)),
            "distance_delta": int(raw.get("distance_delta", 0)),
            "moves_toward_target_region": bool(raw.get("moves_toward_target_region", False)),
            "moves_away_target_region": bool(raw.get("moves_away_target_region", False)),
            "reaches_target_region": bool(raw.get("reaches_target_region", False)),
            "verify_action_candidate": bool(raw.get("verify_action_candidate", False)),
            "verify_action_ids": [
                int(v)
                for v in verify_action_ids
                if isinstance(v, int) or str(v).isdigit()
            ],
            "bonus_hint": float(max(0.0, raw.get("bonus_hint", 0.0))),
            "penalty_hint": float(max(0.0, raw.get("penalty_hint", 0.0))),
        }

    def _candidate_orientation_alignment_features(
        self,
        candidate: ActionCandidateV1,
    ) -> dict[str, Any]:
        raw = candidate.metadata.get("orientation_alignment_features_v1", {})
        if not isinstance(raw, dict):
            raw = {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "detected": bool(raw.get("detected", False)),
            "aligned": bool(raw.get("aligned", False)),
            "similarity": self._clamp01(float(raw.get("similarity", 0.0))),
            "best_rotation_deg": int(raw.get("best_rotation_deg", -1)),
            "rotation_bucket": str(raw.get("rotation_bucket", "rot_unknown")),
            "action_attempts": int(max(0, raw.get("action_attempts", 0))),
            "action_improve_rate": self._clamp01(float(raw.get("action_improve_rate", 0.0))),
            "action_regress_rate": self._clamp01(float(raw.get("action_regress_rate", 0.0))),
            "action_aligned_hit_rate": self._clamp01(
                float(raw.get("action_aligned_hit_rate", 0.0))
            ),
            "bonus_hint": self._clamp01(float(raw.get("bonus_hint", 0.0))),
            "penalty_hint": self._clamp01(float(raw.get("penalty_hint", 0.0))),
        }

    def _candidate_region_action_semantics(
        self,
        candidate: ActionCandidateV1,
    ) -> dict[str, Any]:
        raw = candidate.metadata.get("region_action_semantics_v1", {})
        if not isinstance(raw, dict):
            raw = {}
        return {
            "enabled": bool(raw.get("enabled", False)),
            "action_id": int(raw.get("action_id", int(candidate.action_id))),
            "current_region_key": str(raw.get("current_region_key", "NA")),
            "attempts": int(max(0, raw.get("attempts", 0))),
            "moved_count": int(max(0, raw.get("moved_count", 0))),
            "blocked_count": int(max(0, raw.get("blocked_count", 0))),
            "moved_rate": self._clamp01(float(raw.get("moved_rate", 0.0))),
            "blocked_rate": self._clamp01(float(raw.get("blocked_rate", 0.0))),
            "non_no_change_rate": self._clamp01(float(raw.get("non_no_change_rate", 0.0))),
            "strong_change_rate": self._clamp01(float(raw.get("strong_change_rate", 0.0))),
            "progress_rate": self._clamp01(float(raw.get("progress_rate", 0.0))),
            "ui_side_effect_rate": self._clamp01(float(raw.get("ui_side_effect_rate", 0.0))),
            "terminal_failure_rate": self._clamp01(
                float(raw.get("terminal_failure_rate", 0.0))
            ),
            "ui_suppression": self._clamp01(float(raw.get("ui_suppression", 0.0))),
            "cc_count_change_rate": self._clamp01(float(raw.get("cc_count_change_rate", 0.0))),
            "palette_change_rate": self._clamp01(float(raw.get("palette_change_rate", 0.0))),
            "palette_delta_mean": float(max(0.0, raw.get("palette_delta_mean", 0.0))),
            "palette_delta_mean_norm": self._clamp01(
                float(raw.get("palette_delta_mean_norm", 0.0))
            ),
            "event_entropy_norm": self._clamp01(float(raw.get("event_entropy_norm", 0.0))),
            "info_trigger_score": self._clamp01(float(raw.get("info_trigger_score", 0.0))),
            "coupling_signal_score": self._clamp01(
                float(raw.get("coupling_signal_score", raw.get("info_trigger_score", 0.0)))
            ),
            "coupling_signal_kind": str(raw.get("coupling_signal_kind", "unknown")),
            "edge_status": str(raw.get("edge_status", "unknown")),
        }

    def _candidate_coverage_block_profile(
        self,
        candidate: ActionCandidateV1,
        *,
        predicted_region_stats: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        blocked_stats = self._candidate_blocked_edge_stats(candidate)
        region_semantics = self._candidate_region_action_semantics(candidate)
        predicted_stats = (
            predicted_region_stats
            if isinstance(predicted_region_stats, dict)
            else self._candidate_predicted_region_stats(candidate)
        )

        action_attempts = int(max(0, blocked_stats.get("action_attempts", 0)))
        action_blocked_rate = self._clamp01(float(blocked_stats.get("action_blocked_rate", 0.0)))
        action_moved_rate = self._clamp01(float(blocked_stats.get("action_moved_rate", 0.0)))
        edge_attempts = int(
            max(
                0,
                max(
                    int(blocked_stats.get("edge_attempts", 0)),
                    int(predicted_stats.get("edge_attempts", 0)),
                ),
            )
        )
        edge_blocked_rate = self._clamp01(
            max(
                float(blocked_stats.get("edge_blocked_rate", 0.0)),
                float(predicted_stats.get("edge_blocked_rate", 0.0)),
            )
        )
        region_attempts = int(max(0, region_semantics.get("attempts", 0)))
        ui_side_effect_rate = self._clamp01(float(region_semantics.get("ui_side_effect_rate", 0.0)))
        terminal_failure_rate = self._clamp01(
            float(region_semantics.get("terminal_failure_rate", 0.0))
        )
        strong_change_rate = self._clamp01(float(region_semantics.get("strong_change_rate", 0.0)))
        palette_delta_mean_norm = self._clamp01(
            float(region_semantics.get("palette_delta_mean_norm", 0.0))
        )
        cc_count_change_rate = self._clamp01(
            float(region_semantics.get("cc_count_change_rate", 0.0))
        )
        progress_rate = self._clamp01(float(region_semantics.get("progress_rate", 0.0)))
        edge_status = str(region_semantics.get("edge_status", "unknown"))
        current_region_key = str(predicted_stats.get("current_region_key", "NA"))
        predicted_region_key = str(predicted_stats.get("predicted_region_key", "NA"))
        empirical_transition_target_key = str(
            predicted_stats.get("empirical_transition_target_key", "NA")
        )
        empirical_transition_total = int(
            max(0, predicted_stats.get("empirical_transition_total", 0))
        )
        empirical_transition_confidence = self._clamp01(
            float(predicted_stats.get("empirical_transition_confidence", 0.0))
        )

        hard_blocked_edge = bool(
            edge_attempts >= int(self.coverage_blocked_edge_attempt_threshold)
            and edge_blocked_rate >= float(self.coverage_blocked_edge_rate_threshold)
        )
        hard_blocked_action = bool(
            action_attempts >= int(self.coverage_blocked_action_attempt_threshold)
            and action_blocked_rate >= float(self.coverage_blocked_action_rate_threshold)
            and action_moved_rate <= 0.40
        )
        ui_side_effect_trap = bool(
            region_attempts >= int(self.coverage_ui_side_effect_attempt_threshold)
            and ui_side_effect_rate >= float(self.coverage_ui_side_effect_rate_threshold)
            and progress_rate <= 0.01
        )
        terminal_failure_trap = bool(
            region_attempts >= int(self.coverage_ui_side_effect_attempt_threshold)
            and terminal_failure_rate >= 0.10
            and progress_rate <= 0.01
        )
        edge_declared_blocked = bool(
            edge_status in ("blocked", "ui_blocked", "terminal_failure")
            and action_attempts >= max(2, int(self.coverage_blocked_action_attempt_threshold // 2))
            and (action_moved_rate <= 0.35 or edge_blocked_rate >= 0.85)
        )
        low_yield_translation_trap = bool(
            region_attempts >= int(self.coverage_low_yield_attempt_threshold)
            and action_moved_rate >= float(self.coverage_low_yield_moved_rate_threshold)
            and progress_rate <= 0.01
            and strong_change_rate <= float(self.coverage_low_yield_strong_change_rate_threshold)
            and palette_delta_mean_norm
            <= float(self.coverage_low_yield_palette_delta_norm_threshold)
            and cc_count_change_rate
            <= float(self.coverage_low_yield_cc_count_change_rate_threshold)
        )
        repeated_no_progress_attempt_threshold = int(
            self.coverage_repetition_no_progress_attempt_threshold
        )
        if strong_change_rate >= 0.10 or cc_count_change_rate >= 0.10:
            repeated_no_progress_attempt_threshold = int(
                self.coverage_repetition_no_progress_high_info_attempt_threshold
            )
        repeated_no_progress_trap = bool(
            region_attempts >= int(repeated_no_progress_attempt_threshold)
            and action_moved_rate
            >= float(self.coverage_repetition_no_progress_moved_rate_threshold)
            and progress_rate <= 0.01
        )
        empirical_self_loop_no_progress_trap = bool(
            progress_rate <= 0.01
            and action_moved_rate
            >= float(self.coverage_repetition_no_progress_moved_rate_threshold)
            and edge_attempts >= int(max(4, int(self.coverage_sweep_direction_retry_limit // 2)))
            and current_region_key != "NA"
            and empirical_transition_target_key == current_region_key
            and empirical_transition_total >= int(max(3, int(round(0.60 * edge_attempts))))
            and empirical_transition_confidence >= 0.65
        )
        bidirectional_loop_forward_count = 0
        bidirectional_loop_reverse_count = 0
        bidirectional_loop_total_count = 0
        region_graph_snapshot = candidate.metadata.get("region_graph_snapshot_v1", {})
        if not isinstance(region_graph_snapshot, dict):
            region_graph_snapshot = {}
        edge_rows = region_graph_snapshot.get("edges", [])
        if not isinstance(edge_rows, list):
            edge_rows = []
        if (
            self._parse_region_key(current_region_key) is not None
            and self._parse_region_key(predicted_region_key) is not None
            and current_region_key != predicted_region_key
        ):
            for edge_row in edge_rows:
                if not isinstance(edge_row, dict):
                    continue
                src = str(edge_row.get("source_region_key", "NA"))
                dst = str(edge_row.get("target_region_key", "NA"))
                count = int(max(0, edge_row.get("count", 0)))
                if count <= 0:
                    continue
                if src == current_region_key and dst == predicted_region_key:
                    bidirectional_loop_forward_count += int(count)
                elif src == predicted_region_key and dst == current_region_key:
                    bidirectional_loop_reverse_count += int(count)
            bidirectional_loop_total_count = int(
                bidirectional_loop_forward_count + bidirectional_loop_reverse_count
            )
        bidirectional_no_progress_loop_trap = bool(
            progress_rate <= 0.01
            and action_moved_rate >= float(self.coverage_bidirectional_loop_moved_rate_threshold)
            and min(
                int(bidirectional_loop_forward_count),
                int(bidirectional_loop_reverse_count),
            )
            >= int(self.coverage_bidirectional_loop_min_edge_count)
            and int(bidirectional_loop_total_count)
            >= int(self.coverage_bidirectional_loop_min_total_count)
        )
        hard_skip = bool(
            hard_blocked_edge
            or hard_blocked_action
            or ui_side_effect_trap
            or terminal_failure_trap
            or edge_declared_blocked
            or low_yield_translation_trap
            or repeated_no_progress_trap
            or empirical_self_loop_no_progress_trap
            or bidirectional_no_progress_loop_trap
        )

        evidence_scale = min(
            1.0,
            max(
                float(action_attempts)
                / float(max(1, int(self.coverage_blocked_action_attempt_threshold))),
                float(edge_attempts)
                / float(max(1, int(self.coverage_blocked_edge_attempt_threshold))),
                float(region_attempts)
                / float(max(1, int(self.coverage_ui_side_effect_attempt_threshold))),
            ),
        )
        soft_penalty = self._clamp01(
            evidence_scale
            * (
                (0.45 * action_blocked_rate)
                + (0.35 * edge_blocked_rate)
                + (0.15 * ui_side_effect_rate)
                + (0.05 * terminal_failure_rate)
            )
        )

        reasons: list[str] = []
        if hard_blocked_edge:
            reasons.append("hard_blocked_edge")
        if hard_blocked_action:
            reasons.append("hard_blocked_action")
        if ui_side_effect_trap:
            reasons.append("ui_side_effect_trap")
        if terminal_failure_trap:
            reasons.append("terminal_failure_trap")
        if edge_declared_blocked:
            reasons.append("edge_declared_blocked")
        if low_yield_translation_trap:
            reasons.append("low_yield_translation_trap")
        if repeated_no_progress_trap:
            reasons.append("repeated_no_progress_trap")
        if empirical_self_loop_no_progress_trap:
            reasons.append("empirical_self_loop_no_progress_trap")
        if bidirectional_no_progress_loop_trap:
            reasons.append("bidirectional_no_progress_loop_trap")

        return {
            "hard_skip": bool(hard_skip),
            "soft_penalty": float(soft_penalty),
            "reasons": list(reasons),
            "action_attempts": int(action_attempts),
            "action_blocked_rate": float(action_blocked_rate),
            "action_moved_rate": float(action_moved_rate),
            "edge_attempts": int(edge_attempts),
            "edge_blocked_rate": float(edge_blocked_rate),
            "region_attempts": int(region_attempts),
            "ui_side_effect_rate": float(ui_side_effect_rate),
            "terminal_failure_rate": float(terminal_failure_rate),
            "strong_change_rate": float(strong_change_rate),
            "palette_delta_mean_norm": float(palette_delta_mean_norm),
            "cc_count_change_rate": float(cc_count_change_rate),
            "progress_rate": float(progress_rate),
            "edge_status": str(edge_status),
            "current_region_key": str(current_region_key),
            "predicted_region_key": str(predicted_region_key),
            "empirical_transition_target_key": str(empirical_transition_target_key),
            "empirical_transition_total": int(empirical_transition_total),
            "empirical_transition_confidence": float(empirical_transition_confidence),
            "predicted_region_visit_count": int(
                max(0, predicted_stats.get("predicted_region_visit_count", 0))
            ),
            "bidirectional_loop_forward_count": int(bidirectional_loop_forward_count),
            "bidirectional_loop_reverse_count": int(bidirectional_loop_reverse_count),
            "bidirectional_loop_total_count": int(bidirectional_loop_total_count),
            "repeated_no_progress_attempt_threshold": int(
                repeated_no_progress_attempt_threshold
            ),
        }

    def _coverage_hard_skip(
        self,
        *,
        profile: dict[str, Any] | None = None,
        predicted_region_stats: dict[str, Any] | None = None,
    ) -> bool:
        profile_dict = profile if isinstance(profile, dict) else {}
        stats = (
            predicted_region_stats
            if isinstance(predicted_region_stats, dict)
            else {}
        )
        profile_hard_skip = bool(profile_dict.get("hard_skip", False))
        edge_attempts = int(max(0, stats.get("edge_attempts", 0)))
        edge_blocked_rate = self._clamp01(float(stats.get("edge_blocked_rate", 0.0)))
        edge_saturated = bool(
            edge_attempts >= int(self.coverage_blocked_edge_attempt_threshold)
            and edge_blocked_rate >= float(self.coverage_blocked_edge_rate_threshold)
        )
        retry_saturated = bool(
            edge_attempts >= int(self.coverage_sweep_direction_retry_limit)
            and edge_blocked_rate >= 0.5
        )
        return bool(profile_hard_skip or edge_saturated or retry_saturated)

    def _high_info_recoverable_hard_skip(
        self,
        *,
        profile: dict[str, Any] | None = None,
        focus_features: dict[str, Any] | None = None,
    ) -> bool:
        if not bool(self.high_info_recoverable_skip_enabled):
            return False
        profile_dict = profile if isinstance(profile, dict) else {}
        if not bool(profile_dict.get("hard_skip", False)):
            return False
        features = focus_features if isinstance(focus_features, dict) else {}
        if not (
            bool(features.get("active", False))
            and str(features.get("stage", "idle")) in ("seek", "verify")
        ):
            return False
        if not (
            bool(features.get("moves_toward_target_region", False))
            or bool(features.get("reaches_target_region", False))
        ):
            return False
        current_region_key = str(features.get("current_region_key", "NA"))
        predicted_region_key = str(features.get("predicted_region_key", "NA"))
        if (
            predicted_region_key == "NA"
            or current_region_key == "NA"
            or predicted_region_key == current_region_key
        ):
            return False
        reasons = {
            str(reason)
            for reason in profile_dict.get("reasons", [])
            if isinstance(reason, str)
        }
        if not reasons:
            return False
        critical_reasons = {
            "hard_blocked_edge",
            "hard_blocked_action",
            "ui_side_effect_trap",
            "terminal_failure_trap",
        }
        if reasons.intersection(critical_reasons):
            return False
        recoverable_reasons = {
            "low_yield_translation_trap",
            "edge_declared_blocked",
        }
        if not reasons.issubset(recoverable_reasons):
            return False
        empirical_transition_target_key = str(
            profile_dict.get("empirical_transition_target_key", "NA")
        )
        empirical_transition_total = int(
            max(0, profile_dict.get("empirical_transition_total", 0))
        )
        empirical_transition_confidence = self._clamp01(
            float(profile_dict.get("empirical_transition_confidence", 0.0))
        )
        if (
            empirical_transition_target_key == current_region_key
            and empirical_transition_total >= max(3, int(self.coverage_sweep_direction_retry_limit // 2))
            and empirical_transition_confidence >= 0.60
        ):
            return False
        if "edge_declared_blocked" in reasons:
            action_moved_rate = self._clamp01(float(profile_dict.get("action_moved_rate", 0.0)))
            edge_blocked_rate = self._clamp01(float(profile_dict.get("edge_blocked_rate", 0.0)))
            if (
                action_moved_rate
                < float(self.high_info_recover_edge_declared_blocked_moved_rate_threshold)
                or edge_blocked_rate >= 0.85
            ):
                return False
        return True

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

    def _entry_navigation_direction(self, entry: FreeEnergyLedgerEntryV1) -> str:
        canonical = self._navigation_action_direction(int(entry.candidate.action_id))
        if canonical in ("dir_l", "dir_r", "dir_u", "dir_d"):
            return canonical
        predicted = self._dominant_predicted_direction(entry)
        if predicted in ("dir_l", "dir_r", "dir_u", "dir_d"):
            return predicted
        return "na"

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
                current_direction = self._entry_navigation_direction(entry)
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
            current_direction = self._entry_navigation_direction(entry)
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
                            future_direction = self._entry_navigation_direction(future_entry)
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
        # Apply loop/block penalties to global selection scores so non-coverage paths
        # also de-prioritize high-repeat no-progress edges.
        for entry in entries:
            candidate_id = str(entry.candidate.candidate_id)
            predicted_stats = self._candidate_predicted_region_stats(entry.candidate)
            coverage_profile = self._candidate_coverage_block_profile(
                entry.candidate,
                predicted_region_stats=predicted_stats,
            )
            hard_skip_raw = bool(
                self._coverage_hard_skip(
                    profile=coverage_profile,
                    predicted_region_stats=predicted_stats,
                )
            )
            hard_skip_recovered = False
            if hard_skip_raw:
                focus_features = self._candidate_high_info_focus_features(entry.candidate)
                hard_skip_recovered = bool(
                    self._high_info_recoverable_hard_skip(
                        profile=coverage_profile,
                        focus_features=focus_features,
                    )
                )
            hard_skip_effective = bool(hard_skip_raw and (not hard_skip_recovered))
            blocked_soft_penalty = self._clamp01(
                float(coverage_profile.get("soft_penalty", 0.0))
            )
            reasons = {
                str(reason)
                for reason in coverage_profile.get("reasons", [])
                if isinstance(reason, str)
            }
            loop_repetition_penalty = 0.0
            if reasons.intersection(
                {
                    "bidirectional_no_progress_loop_trap",
                    "repeated_no_progress_trap",
                    "empirical_self_loop_no_progress_trap",
                }
            ):
                loop_repetition_penalty = 0.28
            adjusted_penalty = float(
                (0.36 * blocked_soft_penalty)
                + (0.72 if hard_skip_effective else 0.0)
                + loop_repetition_penalty
            )
            selection_score_by_candidate[candidate_id] = float(
                selection_score_by_candidate.get(candidate_id, entry.total_efe)
                + adjusted_penalty
            )
        entries.sort(
            key=lambda entry: (
                float(
                    selection_score_by_candidate.get(
                        entry.candidate.candidate_id,
                        entry.total_efe,
                    )
                ),
                float(entry.total_efe),
                int(entry.candidate.action_id),
                str(entry.candidate.candidate_id),
            )
        )
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
        coverage_min_region_visit_count = 0
        coverage_regions_visited_at_least_twice = 0
        coverage_target_region_count = int(self.coverage_sweep_target_regions)
        coverage_periodic_resweep_active = False
        coverage_hard_prepass_active = False
        coverage_prepass_goal_kind = "none"
        coverage_prepass_goal_region = {"x": -1, "y": -1}
        coverage_prepass_bfs_next_region_key = "NA"
        coverage_prepass_bfs_applied = False
        coverage_score_margin_used = 0.0
        coverage_sweep_pattern = "disabled"
        coverage_sweep_target_region = {"x": -1, "y": -1}
        coverage_sweep_target_direction = "na"
        fixed_two_pass_traversal_applied = False
        fixed_two_pass_traversal_v1: dict[str, Any] = {
            "enabled": False,
            "mode": "inactive",
            "visit_target": int(max(1, self.coverage_sweep_min_region_visits)),
            "known_region_count": 0,
            "min_region_visit_count": 0,
            "regions_visited_at_least_target": 0,
            "current_region_key": "NA",
            "goal_region_key": "NA",
            "next_region_key": "NA",
            "desired_direction": "na",
            "cross_region_key": "NA",
            "cross_region_visit_count": 0,
            "cross_visit_target": int(max(1, self.coverage_sweep_min_region_visits)),
            "prepass_complete": False,
            "candidate_pool_size": 0,
        }
        direction_sequence_probe_applied = False
        direction_sequence_probe_candidates: list[dict[str, Any]] = []
        high_info_focus_probe_applied = False
        high_info_focus_probe_reason = "inactive"
        high_info_focus_probe_candidates: list[dict[str, Any]] = []
        high_info_focus_active_present = False
        high_info_focus_priority_available = False
        high_info_focus_strong_signal_available = False
        high_info_focus_strong_signal_target_score = 0.0
        high_info_focus_strong_signal_remaining_samples = 0
        high_info_focus_strong_signal_queue_length = 0
        high_info_release_action_counter = int(
            self.high_info_focus_release_action_counter
        )
        if high_info_release_action_counter < 0:
            high_info_release_action_counter = int(
                self._serpentine_single_pass_length()
            )
        for entry in entries:
            focus_features = self._candidate_high_info_focus_features(entry.candidate)
            if not (
                bool(focus_features.get("enabled", False))
                and bool(focus_features.get("active", False))
            ):
                continue
            high_info_focus_active_present = True
            target_score = self._clamp01(float(focus_features.get("target_score", 0.0)))
            remaining_samples = int(max(0, focus_features.get("remaining_samples", 0)))
            queue_length = int(max(0, focus_features.get("queue_length", 0)))
            if target_score >= float(self.high_info_release_min_target_score) and (
                remaining_samples >= int(self.high_info_release_min_remaining_samples)
                or queue_length >= int(self.high_info_release_min_queue_length)
            ):
                high_info_focus_strong_signal_available = True
                if (
                    target_score > float(high_info_focus_strong_signal_target_score)
                    or (
                        target_score
                        >= float(high_info_focus_strong_signal_target_score) - 1.0e-9
                        and remaining_samples
                        > int(high_info_focus_strong_signal_remaining_samples)
                    )
                ):
                    high_info_focus_strong_signal_target_score = float(target_score)
                    high_info_focus_strong_signal_remaining_samples = int(remaining_samples)
                    high_info_focus_strong_signal_queue_length = int(queue_length)
            if bool(focus_features.get("verify_action_candidate", False)):
                high_info_focus_priority_available = True
                break
            if bool(focus_features.get("reaches_target_region", False)):
                high_info_focus_priority_available = True
                break
            if bool(focus_features.get("moves_toward_target_region", False)):
                high_info_focus_priority_available = True
                break
        high_info_after_first_pass_gate_open = bool(
            self.high_info_focus_release_after_first_pass
            and (
                high_info_focus_priority_available
                or high_info_focus_strong_signal_available
            )
            and int(packet.action_counter) >= int(high_info_release_action_counter)
        )
        fixed_two_pass_suppressed_by_high_info = False
        sequence_causal_probe_applied = False
        sequence_causal_probe_reason = "inactive"
        sequence_causal_probe_candidates: list[dict[str, Any]] = []

        fixed_prepass_entry, fixed_two_pass_traversal_v1 = (
            self._select_fixed_two_pass_traversal_entry(
                packet=packet,
                entries=entries,
                action_count_map=action_count_map,
            )
        )
        if fixed_prepass_entry is not None and not high_info_after_first_pass_gate_open:
            selected_entry = fixed_prepass_entry
            fixed_two_pass_traversal_applied = True
            least_tried_probe_applied = True
            coverage_region_probe_applied = True
            coverage_sweep_active = True
            coverage_hard_prepass_active = True
            coverage_sweep_reason = "fixed_two_pass_prepass"
            coverage_known_region_count = int(
                fixed_two_pass_traversal_v1.get("known_region_count", 0)
            )
            coverage_min_region_visit_count = int(
                fixed_two_pass_traversal_v1.get("min_region_visit_count", 0)
            )
            coverage_regions_visited_at_least_twice = int(
                fixed_two_pass_traversal_v1.get("regions_visited_at_least_target", 0)
            )
            coverage_prepass_goal_kind = str(
                fixed_two_pass_traversal_v1.get("mode", "two_pass")
            )
            coverage_prepass_bfs_applied = bool(
                str(fixed_two_pass_traversal_v1.get("mode", ""))
                == "bfs_two_pass"
            )
            coverage_prepass_bfs_next_region_key = str(
                fixed_two_pass_traversal_v1.get("next_region_key", "NA")
            )
            coverage_sweep_target_direction = str(
                fixed_two_pass_traversal_v1.get("desired_direction", "na")
            )
            goal_region_key = str(
                fixed_two_pass_traversal_v1.get("goal_region_key", "NA")
            )
            goal_parsed = self._parse_region_key(goal_region_key)
            if goal_parsed is not None:
                goal_rx, goal_ry = goal_parsed
                coverage_prepass_goal_region = {"x": int(goal_rx), "y": int(goal_ry)}
                coverage_sweep_target_region = {"x": int(goal_rx), "y": int(goal_ry)}
            if selected_entry is not entries[0]:
                entries.remove(selected_entry)
                entries.insert(0, selected_entry)
        elif fixed_prepass_entry is not None and high_info_after_first_pass_gate_open:
            fixed_two_pass_suppressed_by_high_info = True
            fixed_two_pass_traversal_v1["enabled"] = True
            fixed_two_pass_traversal_v1["mode"] = (
                "suppressed_by_high_info_after_first_pass"
            )

        early_probe_active = bool(
            (not fixed_two_pass_traversal_applied)
            and (not bool(fixed_two_pass_traversal_v1.get("prepass_complete", False)))
            and int(early_probe_budget_remaining) > 0
            and phase in ("explore", "explain")
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
            (
                phase in ("explore", "explain")
                or (self.coverage_sweep_force_in_exploit and phase == "exploit")
            )
            and (not high_info_after_first_pass_gate_open)
        )
        if (
            not fixed_two_pass_traversal_applied
            and
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
                coverage_block_profile_by_candidate_id: dict[str, dict[str, Any]] = {
                    str(entry.candidate.candidate_id): self._candidate_coverage_block_profile(
                        entry.candidate,
                        predicted_region_stats=predicted_stats_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            {},
                        ),
                    )
                    for entry in navigation_entries
                }
                def _coverage_hard_skip_for_entry(
                    entry: FreeEnergyLedgerEntryV1,
                ) -> bool:
                    candidate_id = str(entry.candidate.candidate_id)
                    return self._coverage_hard_skip(
                        profile=coverage_block_profile_by_candidate_id.get(
                            candidate_id,
                            {},
                        ),
                        predicted_region_stats=predicted_stats_by_candidate_id.get(
                            candidate_id,
                            {},
                        ),
                    )

                def _coverage_soft_penalty_for_entry(
                    entry: FreeEnergyLedgerEntryV1,
                ) -> float:
                    candidate_id = str(entry.candidate.candidate_id)
                    return float(
                        coverage_block_profile_by_candidate_id.get(
                            candidate_id,
                            {},
                        ).get("soft_penalty", 0.0)
                    )

                def _coverage_effective_hard_skip_for_entry(
                    entry: FreeEnergyLedgerEntryV1,
                ) -> bool:
                    if bool(_coverage_hard_skip_for_entry(entry)):
                        return True
                    if not bool(coverage_hard_prepass_active):
                        return False
                    candidate_id = str(entry.candidate.candidate_id)
                    stats = predicted_stats_by_candidate_id.get(candidate_id, {})
                    profile = coverage_block_profile_by_candidate_id.get(candidate_id, {})
                    edge_attempts = int(max(0, stats.get("edge_attempts", 0)))
                    edge_blocked_rate = self._clamp01(float(stats.get("edge_blocked_rate", 0.0)))
                    ui_side_effect_rate = self._clamp01(
                        float(profile.get("ui_side_effect_rate", 0.0))
                    )
                    progress_rate = self._clamp01(float(profile.get("progress_rate", 0.0)))
                    return bool(
                        edge_attempts >= 1
                        and edge_blocked_rate >= 0.95
                        and ui_side_effect_rate >= 0.85
                        and progress_rate <= 0.01
                    )
                coverage_known_region_count = int(
                    max(
                        (
                            int(stats.get("known_region_count", 0))
                            for stats in predicted_stats_by_candidate_id.values()
                        ),
                        default=0,
                    )
                )
                coverage_min_region_visit_count = int(
                    min(
                        (
                            int(stats.get("min_region_visit_count", 0))
                            for stats in predicted_stats_by_candidate_id.values()
                        ),
                        default=0,
                    )
                )
                coverage_regions_visited_at_least_twice = int(
                    max(
                        (
                            int(stats.get("regions_visited_at_least_twice", 0))
                            for stats in predicted_stats_by_candidate_id.values()
                        ),
                        default=0,
                    )
                )
                coverage_region_goal_unmet = bool(
                    coverage_known_region_count <= 0
                )
                coverage_revisit_goal_unmet = bool(
                    int(self.coverage_sweep_min_region_visits) > 1
                    and coverage_known_region_count > 0
                    and coverage_regions_visited_at_least_twice < coverage_known_region_count
                    and coverage_min_region_visit_count
                    < int(self.coverage_sweep_min_region_visits)
                )
                coverage_goal_unmet = bool(
                    coverage_region_goal_unmet or coverage_revisit_goal_unmet
                )
                coverage_hard_prepass_active = bool(
                    coverage_goal_unmet
                    and int(packet.action_counter) < int(self.coverage_prepass_steps)
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
                if coverage_region_goal_unmet:
                    coverage_sweep_reason = "region_goal_unmet"
                elif coverage_revisit_goal_unmet:
                    coverage_sweep_reason = "min_visit_goal_unmet"
                elif periodic_window_active:
                    coverage_sweep_reason = "periodic_resweep"
                else:
                    coverage_sweep_reason = "goal_reached"
                if coverage_hard_prepass_active:
                    coverage_sweep_reason = f"{coverage_sweep_reason}|hard_prepass"
                if coverage_sweep_active:
                    target_rx = -1
                    target_ry = -1
                    target_direction = "na"
                    current_region_visit_reference = 0
                    prepass_goal_region_key = "NA"
                    prepass_current_region_key = "NA"
                    if self.coverage_matrix_sweep_enabled:
                        coverage_sweep_pattern = "row_serpentine_up"
                        sample_stats = predicted_stats_by_candidate_id.get(
                            str(navigation_entries[0].candidate.candidate_id),
                            {},
                        )
                        current_rx = int(sample_stats.get("current_region_x", -1))
                        current_ry = int(sample_stats.get("current_region_y", -1))
                        current_region_visit_reference = int(
                            max(0, sample_stats.get("current_region_visit_count", 0))
                        )
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
                                elif current_ry < 7:
                                    target_ry = int(current_ry + 1)
                                    target_direction = "dir_d"
                            else:
                                if current_rx < 7:
                                    target_rx = int(current_rx + 1)
                                    target_direction = "dir_r"
                                elif current_ry > 0:
                                    target_ry = int(current_ry - 1)
                                    target_direction = "dir_u"
                                elif current_ry < 7:
                                    target_ry = int(current_ry + 1)
                                    target_direction = "dir_d"
                    coverage_sweep_target_region = {
                        "x": int(target_rx),
                        "y": int(target_ry),
                    }
                    coverage_sweep_target_direction = str(target_direction)
                    if coverage_hard_prepass_active and navigation_entries:
                        sample_entry = navigation_entries[0]
                        sample_candidate_id = str(sample_entry.candidate.candidate_id)
                        sample_stats = predicted_stats_by_candidate_id.get(
                            sample_candidate_id,
                            {},
                        )
                        prepass_current_region_key = str(
                            sample_stats.get("current_region_key", "NA")
                        )
                        navigation_target = sample_entry.candidate.metadata.get(
                            "navigation_target_features_v1",
                            {},
                        )
                        if not isinstance(navigation_target, dict):
                            navigation_target = {}
                        region_graph_snapshot = sample_entry.candidate.metadata.get(
                            "region_graph_snapshot_v1",
                            {},
                        )
                        if not isinstance(region_graph_snapshot, dict):
                            region_graph_snapshot = {}

                        region_visit_histogram = region_graph_snapshot.get(
                            "region_visit_histogram",
                            {},
                        )
                        if not isinstance(region_visit_histogram, dict):
                            region_visit_histogram = {}

                        least_region_key = "NA"
                        least_region_count = 10**9
                        for region_key_raw, count_raw in region_visit_histogram.items():
                            region_key = str(region_key_raw)
                            if self._parse_region_key(region_key) is None:
                                continue
                            count = int(max(0, count_raw))
                            if count < least_region_count:
                                least_region_count = int(count)
                                least_region_key = str(region_key)

                        cross_enabled = bool(navigation_target.get("cross_like_enabled", False))
                        cross_region = navigation_target.get("cross_like_target_region", {})
                        if not isinstance(cross_region, dict):
                            cross_region = {}
                        cross_rx = int(cross_region.get("x", -1))
                        cross_ry = int(cross_region.get("y", -1))
                        cross_region_key = (
                            f"{cross_rx}:{cross_ry}"
                            if cross_rx >= 0 and cross_ry >= 0
                            else "NA"
                        )
                        cross_region_visit_count = int(
                            max(
                                0,
                                navigation_target.get(
                                    "cross_like_target_region_visit_count",
                                    region_visit_histogram.get(cross_region_key, 0),
                                ),
                            )
                        )

                        if (
                            cross_enabled
                            and cross_region_key != "NA"
                            and cross_region_visit_count <= 0
                        ):
                            prepass_goal_region_key = str(cross_region_key)
                            coverage_prepass_goal_kind = "cross_like_unvisited"
                            coverage_sweep_reason = f"{coverage_sweep_reason}|cross_goal"
                        elif least_region_key != "NA":
                            prepass_goal_region_key = str(least_region_key)
                            coverage_prepass_goal_kind = "least_visited_region"
                            coverage_sweep_reason = f"{coverage_sweep_reason}|coverage_goal"

                        parsed_goal_region = self._parse_region_key(prepass_goal_region_key)
                        if parsed_goal_region is not None:
                            goal_rx, goal_ry = parsed_goal_region
                            target_rx = int(goal_rx)
                            target_ry = int(goal_ry)
                            coverage_prepass_goal_region = {
                                "x": int(goal_rx),
                                "y": int(goal_ry),
                            }
                            coverage_sweep_target_region = {
                                "x": int(goal_rx),
                                "y": int(goal_ry),
                            }

                        adjacency = self._region_graph_adjacency(region_graph_snapshot)
                        if (
                            self._parse_region_key(prepass_current_region_key) is not None
                            and self._parse_region_key(prepass_goal_region_key) is not None
                        ):
                            bfs_next_region_key = self._bfs_next_region_key(
                                adjacency,
                                start_region_key=str(prepass_current_region_key),
                                goal_region_key=str(prepass_goal_region_key),
                            )
                            if (
                                bfs_next_region_key != "NA"
                                and bfs_next_region_key != str(prepass_current_region_key)
                            ):
                                coverage_prepass_bfs_next_region_key = str(
                                    bfs_next_region_key
                                )
                                coverage_prepass_bfs_applied = True
                                coverage_sweep_reason = f"{coverage_sweep_reason}|bfs_path"
                            elif prepass_goal_region_key != "NA":
                                coverage_sweep_reason = f"{coverage_sweep_reason}|bfs_fallback"
                    best_navigation_score = min(
                        float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        )
                        for entry in navigation_entries
                    )
                    coverage_margin = 0.0
                    if not coverage_hard_prepass_active:
                        coverage_margin = float(
                            max(
                                self.coverage_sweep_score_margin,
                                self.stagnation_probe_score_margin,
                            )
                        )
                    coverage_score_margin_used = float(coverage_margin)
                    def _coverage_frontier_visit(entry: FreeEnergyLedgerEntryV1) -> int:
                        stats = predicted_stats_by_candidate_id.get(
                            str(entry.candidate.candidate_id),
                            {},
                        )
                        frontier_key = str(
                            stats.get("empirical_transition_frontier_key", "NA")
                        )
                        if frontier_key != "NA":
                            return int(
                                max(
                                    0,
                                    stats.get(
                                        "empirical_transition_frontier_visit_count",
                                        0,
                                    ),
                                )
                            )
                        return int(max(0, stats.get("predicted_region_visit_count", 0)))

                    high_revisit_escape_required = bool(
                        int(current_region_visit_reference)
                        >= int(self.region_revisit_hard_threshold)
                    )
                    if coverage_hard_prepass_active:
                        coverage_pool = list(navigation_entries)
                    else:
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
                    safe_coverage_pool = [
                        entry
                        for entry in coverage_pool
                        if not bool(_coverage_effective_hard_skip_for_entry(entry))
                    ]
                    coverage_all_blocked = False
                    if safe_coverage_pool:
                        coverage_pool = list(safe_coverage_pool)
                        coverage_sweep_reason = (
                            f"{coverage_sweep_reason}|blocked_edge_auto_skip"
                        )
                    else:
                        coverage_all_blocked = True
                        coverage_pool = []
                        coverage_sweep_reason = (
                            f"{coverage_sweep_reason}|blocked_edge_auto_skip_all_blocked_escape"
                        )
                    if not coverage_hard_prepass_active and len(coverage_pool) < 2:
                        coverage_pool = sorted(
                            navigation_entries,
                            key=lambda entry: (
                                bool(_coverage_effective_hard_skip_for_entry(entry)),
                                float(_coverage_soft_penalty_for_entry(entry)),
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
                    if bool(coverage_all_blocked):
                        coverage_pool = []
                    if (
                        not coverage_hard_prepass_active
                        and (not bool(coverage_all_blocked))
                        and
                        self.coverage_matrix_sweep_enabled
                        and str(coverage_sweep_target_direction)
                        in ("dir_l", "dir_r", "dir_u", "dir_d")
                    ):
                        directed_pool = [
                            entry
                            for entry in navigation_entries
                            if (
                                not bool(_coverage_effective_hard_skip_for_entry(entry))
                                and
                                self._navigation_action_direction(int(entry.candidate.action_id))
                                == str(coverage_sweep_target_direction)
                                and int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("edge_attempts", 0)
                                )
                                < int(self.coverage_sweep_direction_retry_limit)
                            )
                        ]
                        if high_revisit_escape_required:
                            directed_pool = [
                                entry
                                for entry in directed_pool
                                if _coverage_frontier_visit(entry)
                                < int(current_region_visit_reference)
                            ]
                        if directed_pool:
                            coverage_pool = list(directed_pool)
                            coverage_sweep_reason = (
                                f"{coverage_sweep_reason}|matrix_direction_forced"
                            )
                        else:
                            escaped_pool = sorted(
                                navigation_entries,
                                key=lambda entry: (
                                    bool(_coverage_effective_hard_skip_for_entry(entry)),
                                    float(_coverage_soft_penalty_for_entry(entry)),
                                    _coverage_frontier_visit(entry),
                                    -float(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get(
                                            "empirical_transition_frontier_confidence",
                                            0.0,
                                        )
                                    ),
                                    int(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("edge_attempts", 0)
                                    ),
                                    float(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("edge_blocked_rate", 0.0)
                                    ),
                                    int(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("predicted_region_visit_count", 10**6)
                                    ),
                                    float(
                                        selection_score_by_candidate.get(
                                            entry.candidate.candidate_id,
                                            entry.total_efe,
                                        )
                                    ),
                                    int(entry.candidate.action_id),
                                    str(entry.candidate.candidate_id),
                                ),
                            )
                            escaped_pool = [
                                entry
                                for entry in escaped_pool
                                if not bool(_coverage_effective_hard_skip_for_entry(entry))
                            ]
                            if escaped_pool:
                                escape_direction = self._navigation_action_direction(
                                    int(escaped_pool[0].candidate.action_id)
                                )
                                if escape_direction in ("dir_l", "dir_r", "dir_u", "dir_d"):
                                    coverage_sweep_target_direction = str(escape_direction)
                                    coverage_sweep_reason = (
                                        f"{coverage_sweep_reason}|target_retry_exceeded_escape"
                                    )
                                coverage_pool = list(escaped_pool[: max(2, min(6, len(escaped_pool)))])
                            else:
                                coverage_pool = []
                                coverage_sweep_reason = (
                                    f"{coverage_sweep_reason}|all_hard_skip_no_probe"
                                )
                    if (
                        not coverage_hard_prepass_active
                        and (not bool(coverage_all_blocked))
                        and
                        self.coverage_matrix_sweep_enabled
                        and str(coverage_sweep_target_direction)
                        in ("dir_l", "dir_r", "dir_u", "dir_d")
                        and coverage_pool
                    ):
                        filtered_pool = [
                            entry
                            for entry in coverage_pool
                            if not (
                                bool(_coverage_effective_hard_skip_for_entry(entry))
                                or
                                self._navigation_action_direction(
                                    int(entry.candidate.action_id)
                                )
                                == str(coverage_sweep_target_direction)
                                and int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("edge_attempts", 0)
                                )
                                >= int(self.coverage_sweep_direction_retry_limit)
                                and float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("edge_blocked_rate", 0.0)
                                )
                                >= 0.5
                            )
                        ]
                        if filtered_pool:
                            coverage_pool = filtered_pool
                    if coverage_pool:
                        if coverage_hard_prepass_active:
                            coverage_pool.sort(
                                key=lambda entry: (
                                    bool(_coverage_effective_hard_skip_for_entry(entry)),
                                    float(_coverage_soft_penalty_for_entry(entry)),
                                    0
                                    if (
                                        bool(coverage_prepass_bfs_applied)
                                        and str(coverage_prepass_bfs_next_region_key) != "NA"
                                        and str(
                                            predicted_stats_by_candidate_id.get(
                                                str(entry.candidate.candidate_id),
                                                {},
                                            ).get("predicted_region_key", "NA")
                                        )
                                        == str(coverage_prepass_bfs_next_region_key)
                                    )
                                    else 1,
                                    int(
                                        self._region_distance_from_keys(
                                            str(
                                                predicted_stats_by_candidate_id.get(
                                                    str(entry.candidate.candidate_id),
                                                    {},
                                                ).get("predicted_region_key", "NA")
                                            ),
                                            str(prepass_goal_region_key),
                                        )
                                    ),
                                    int(_coverage_frontier_visit(entry)),
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
                                        ).get("edge_attempts", 10**6)
                                    ),
                                    float(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("edge_blocked_rate", 1.0)
                                    ),
                                    int(action_count_map.get(int(entry.candidate.action_id), 0)),
                                    int(entry.candidate.action_id),
                                    str(entry.candidate.candidate_id),
                                )
                            )
                        else:
                            coverage_pool.sort(
                                key=lambda entry: (
                                    bool(_coverage_effective_hard_skip_for_entry(entry)),
                                    float(_coverage_soft_penalty_for_entry(entry)),
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
                                        and self._entry_navigation_direction(entry)
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
                                    int(_coverage_frontier_visit(entry)),
                                    -float(
                                        predicted_stats_by_candidate_id.get(
                                            str(entry.candidate.candidate_id),
                                            {},
                                        ).get("empirical_transition_frontier_confidence", 0.0)
                                    ),
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
                                "empirical_transition_frontier_key": str(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_frontier_key", "NA")
                                ),
                                "empirical_transition_frontier_visit_count": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_frontier_visit_count", 0)
                                ),
                                "empirical_transition_frontier_confidence": float(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("empirical_transition_frontier_confidence", 0.0)
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
                                "min_region_visit_count": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("min_region_visit_count", 0)
                                ),
                                "regions_visited_at_least_twice": int(
                                    predicted_stats_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("regions_visited_at_least_twice", 0)
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
                                    self._entry_navigation_direction(entry)
                                ),
                                "predicted_direction_model_only": str(
                                    self._dominant_predicted_direction(entry)
                                ),
                                "target_direction_match": bool(
                                    str(coverage_sweep_target_direction)
                                    in ("dir_l", "dir_r", "dir_u", "dir_d")
                                    and self._entry_navigation_direction(entry)
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
                                "blocked_hard_skip": bool(
                                    _coverage_effective_hard_skip_for_entry(entry)
                                ),
                                "blocked_soft_penalty": float(
                                    coverage_block_profile_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("soft_penalty", 0.0)
                                ),
                                "blocked_skip_reasons": list(
                                    coverage_block_profile_by_candidate_id.get(
                                        str(entry.candidate.candidate_id),
                                        {},
                                    ).get("reasons", [])
                                ),
                            }
                            for entry in coverage_pool[:10]
                        ]

        if not early_probe_applied and fixed_two_pass_traversal_applied:
            high_info_focus_probe_reason = "suppressed_by_fixed_two_pass_prepass"

        if not early_probe_applied and not fixed_two_pass_traversal_applied:
            high_info_rows: list[dict[str, Any]] = []
            for entry in entries:
                focus_features = self._candidate_high_info_focus_features(entry.candidate)
                if not (
                    bool(focus_features.get("enabled", False))
                    and bool(focus_features.get("active", False))
                ):
                    continue
                predicted_region_stats = self._candidate_predicted_region_stats(
                    entry.candidate
                )
                coverage_profile = self._candidate_coverage_block_profile(
                    entry.candidate,
                    predicted_region_stats=predicted_region_stats,
                )
                blocked_hard_skip_raw = self._coverage_hard_skip(
                    profile=coverage_profile,
                    predicted_region_stats=predicted_region_stats,
                )
                blocked_hard_skip_recovered = bool(
                    blocked_hard_skip_raw
                    and self._high_info_recoverable_hard_skip(
                        profile=coverage_profile,
                        focus_features=focus_features,
                    )
                )
                blocked_hard_skip = bool(
                    blocked_hard_skip_raw and (not blocked_hard_skip_recovered)
                )
                high_info_rows.append(
                    {
                        "entry": entry,
                        "features": focus_features,
                        "score": float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        ),
                        "blocked_hard_skip": bool(blocked_hard_skip),
                        "blocked_hard_skip_raw": bool(blocked_hard_skip_raw),
                        "blocked_hard_skip_recovered": bool(
                            blocked_hard_skip_recovered
                        ),
                        "blocked_soft_penalty": float(
                            coverage_profile.get("soft_penalty", 0.0)
                        ),
                        "blocked_reasons": list(coverage_profile.get("reasons", [])),
                    }
                )
            if high_info_rows:
                high_info_focus_probe_reason = "active_no_override"
                safe_high_info_rows = [
                    row for row in high_info_rows if not bool(row.get("blocked_hard_skip", False))
                ]
                candidate_high_info_rows = (
                    safe_high_info_rows if safe_high_info_rows else high_info_rows
                )
                recovered_block_rows = [
                    row
                    for row in high_info_rows
                    if bool(row.get("blocked_hard_skip_recovered", False))
                ]
                if len(candidate_high_info_rows) < len(high_info_rows):
                    high_info_focus_probe_reason = "active_blocked_edge_auto_skip"
                elif recovered_block_rows:
                    high_info_focus_probe_reason = "active_recoverable_block_override"
                best_high_info_score = min(
                    float(row["score"]) for row in candidate_high_info_rows
                )
                verify_pool = [
                    row
                    for row in candidate_high_info_rows
                    if bool(row["features"].get("verify_action_candidate", False))
                ]
                if verify_pool:
                    verify_pool.sort(
                        key=lambda row: (
                            -float(row["features"].get("bonus_hint", 0.0)),
                            float(row["score"]),
                            int(action_count_map.get(int(row["entry"].candidate.action_id), 0)),
                            int(row["entry"].candidate.action_id),
                            str(row["entry"].candidate.candidate_id),
                        )
                    )
                    best_verify = verify_pool[0]
                    verify_margin = float(max(self.sequence_probe_score_margin, 0.55))
                    if float(best_verify["score"]) <= (best_high_info_score + verify_margin):
                        selected_entry = best_verify["entry"]
                        high_info_focus_probe_applied = True
                        high_info_focus_probe_reason = "verify_action_priority"
                if not high_info_focus_probe_applied:
                    seek_pool = [
                        row
                        for row in candidate_high_info_rows
                        if int(row["entry"].candidate.action_id) in (1, 2, 3, 4)
                        and (
                            bool(row["features"].get("moves_toward_target_region", False))
                            or bool(row["features"].get("reaches_target_region", False))
                        )
                    ]
                    if seek_pool:
                        seek_pool.sort(
                            key=lambda row: (
                                0 if bool(row["features"].get("reaches_target_region", False)) else 1,
                                -int(row["features"].get("remaining_samples", 0)),
                                int(row["features"].get("distance_after", 10**6)),
                                -float(row["features"].get("target_score", 0.0)),
                                -float(row["features"].get("bonus_hint", 0.0)),
                                float(row["score"]),
                                int(action_count_map.get(int(row["entry"].candidate.action_id), 0)),
                                int(row["entry"].candidate.action_id),
                                str(row["entry"].candidate.candidate_id),
                            )
                        )
                        best_seek = seek_pool[0]
                        seek_margin = float(max(self.sequence_probe_score_margin, 0.40))
                        if bool(best_seek["features"].get("reaches_target_region", False)) or float(
                            best_seek["score"]
                        ) <= (best_high_info_score + seek_margin):
                            selected_entry = best_seek["entry"]
                            high_info_focus_probe_applied = True
                            high_info_focus_probe_reason = "seek_target_priority"
                if not high_info_focus_probe_applied:
                    value_pool = [
                        row
                        for row in candidate_high_info_rows
                        if int(row["entry"].candidate.action_id) in (1, 2, 3, 4)
                        and float(row["features"].get("target_score", 0.0)) >= 0.32
                        and (
                            str(row["features"].get("stage", "idle")) != "seek"
                            or bool(row["features"].get("moves_toward_target_region", False))
                            or bool(row["features"].get("reaches_target_region", False))
                            or (
                                str(row["features"].get("predicted_region_key", "NA"))
                                != str(row["features"].get("current_region_key", "NA"))
                                and not bool(
                                    row["features"].get("moves_away_target_region", False)
                                )
                            )
                        )
                    ]
                    if value_pool:
                        value_pool.sort(
                            key=lambda row: (
                                0 if bool(row["features"].get("reaches_target_region", False)) else 1,
                                0
                                if bool(row["features"].get("moves_toward_target_region", False))
                                else 1,
                                0
                                if str(row["features"].get("predicted_region_key", "NA"))
                                != str(row["features"].get("current_region_key", "NA"))
                                else 1,
                                1
                                if bool(row["features"].get("moves_away_target_region", False))
                                else 0,
                                -int(row["features"].get("remaining_samples", 0)),
                                -float(row["features"].get("target_score", 0.0)),
                                -float(row["features"].get("bonus_hint", 0.0)),
                                int(row["features"].get("distance_after", 10**6)),
                                float(row["score"]),
                                int(action_count_map.get(int(row["entry"].candidate.action_id), 0)),
                                int(row["entry"].candidate.action_id),
                                str(row["entry"].candidate.candidate_id),
                            )
                        )
                        best_value = value_pool[0]
                        value_margin = float(max(0.22, 0.75 * self.sequence_probe_score_margin))
                        if (
                            float(best_value["features"].get("target_score", 0.0)) >= 0.70
                            or float(best_value["score"]) <= (best_high_info_score + value_margin)
                        ):
                            selected_entry = best_value["entry"]
                            high_info_focus_probe_applied = True
                            high_info_focus_probe_reason = "high_value_region_priority"
                if high_info_focus_probe_applied:
                    least_tried_probe_applied = True
                    if selected_entry is not entries[0]:
                        entries.remove(selected_entry)
                        entries.insert(0, selected_entry)
                high_info_focus_probe_candidates = [
                    {
                        "candidate_id": str(row["entry"].candidate.candidate_id),
                        "action_id": int(row["entry"].candidate.action_id),
                        "score": float(row["score"]),
                        "blocked_hard_skip": bool(row.get("blocked_hard_skip", False)),
                        "blocked_hard_skip_raw": bool(
                            row.get("blocked_hard_skip_raw", False)
                        ),
                        "blocked_hard_skip_recovered": bool(
                            row.get("blocked_hard_skip_recovered", False)
                        ),
                        "blocked_soft_penalty": float(row.get("blocked_soft_penalty", 0.0)),
                        "blocked_reasons": list(row.get("blocked_reasons", [])),
                        "stage": str(row["features"].get("stage", "idle")),
                        "current_region_key": str(
                            row["features"].get("current_region_key", "NA")
                        ),
                        "target_region_key": str(
                            row["features"].get("target_region_key", "NA")
                        ),
                        "predicted_region_key": str(
                            row["features"].get("predicted_region_key", "NA")
                        ),
                        "distance_before": int(row["features"].get("distance_before", 10**6)),
                        "distance_after": int(row["features"].get("distance_after", 10**6)),
                        "target_score": float(row["features"].get("target_score", 0.0)),
                        "target_sample_count": int(
                            row["features"].get("target_sample_count", 0)
                        ),
                        "remaining_samples": int(
                            row["features"].get("remaining_samples", 0)
                        ),
                        "moves_toward_target_region": bool(
                            row["features"].get("moves_toward_target_region", False)
                        ),
                        "reaches_target_region": bool(
                            row["features"].get("reaches_target_region", False)
                        ),
                        "verify_action_candidate": bool(
                            row["features"].get("verify_action_candidate", False)
                        ),
                        "bonus_hint": float(row["features"].get("bonus_hint", 0.0)),
                        "penalty_hint": float(row["features"].get("penalty_hint", 0.0)),
                        "steps_remaining": int(row["features"].get("steps_remaining", 0)),
                    }
                    for row in high_info_rows[:12]
                ]

        if (
            self.sequence_causal_term_enabled
            and not fixed_two_pass_traversal_applied
            and not early_probe_applied
            and not high_info_focus_probe_applied
        ):
            sequence_rows: list[dict[str, Any]] = []
            for entry in entries:
                seq_features = self._candidate_sequence_causal_features(entry.candidate)
                if not (
                    bool(seq_features.get("enabled", False))
                    and bool(seq_features.get("active", False))
                ):
                    continue
                predicted_region_stats = self._candidate_predicted_region_stats(
                    entry.candidate
                )
                coverage_profile = self._candidate_coverage_block_profile(
                    entry.candidate,
                    predicted_region_stats=predicted_region_stats,
                )
                blocked_hard_skip = self._coverage_hard_skip(
                    profile=coverage_profile,
                    predicted_region_stats=predicted_region_stats,
                )
                sequence_rows.append(
                    {
                        "entry": entry,
                        "features": seq_features,
                        "score": float(
                            selection_score_by_candidate.get(
                                entry.candidate.candidate_id,
                                entry.total_efe,
                            )
                        ),
                        "blocked_hard_skip": bool(blocked_hard_skip),
                        "blocked_soft_penalty": float(
                            coverage_profile.get("soft_penalty", 0.0)
                        ),
                        "blocked_reasons": list(coverage_profile.get("reasons", [])),
                    }
                )
            if sequence_rows:
                sequence_causal_probe_reason = "active_no_override"
                safe_sequence_rows = [
                    row for row in sequence_rows if not bool(row.get("blocked_hard_skip", False))
                ]
                if not safe_sequence_rows:
                    sequence_causal_probe_reason = "blocked_edge_auto_skip"
                else:
                    if len(safe_sequence_rows) < len(sequence_rows):
                        sequence_causal_probe_reason = "active_blocked_edge_auto_skip"
                    sequence_rows = safe_sequence_rows
                stage = str(sequence_rows[0]["features"].get("stage", "idle"))
                if sequence_rows:
                    best_sequence_score = min(float(row["score"]) for row in sequence_rows)
                    if stage == "verify":
                        verify_pool = [
                            row
                            for row in sequence_rows
                            if bool(row["features"].get("verify_action_candidate", False))
                        ]
                        if verify_pool:
                            verify_pool.sort(
                                key=lambda row: (
                                    -float(row["features"].get("bonus_hint", 0.0)),
                                    float(row["score"]),
                                    int(action_count_map.get(int(row["entry"].candidate.action_id), 0)),
                                    int(row["entry"].candidate.action_id),
                                    str(row["entry"].candidate.candidate_id),
                                )
                            )
                            best_verify = verify_pool[0]
                            verify_margin = float(max(self.sequence_probe_score_margin, 0.55))
                            if float(best_verify["score"]) <= (best_sequence_score + verify_margin):
                                selected_entry = best_verify["entry"]
                                sequence_causal_probe_applied = True
                                sequence_causal_probe_reason = "verify_action_priority"
                    if not sequence_causal_probe_applied:
                        seek_pool = [
                            row
                            for row in sequence_rows
                            if int(row["entry"].candidate.action_id) in (1, 2, 3, 4)
                            and (
                                bool(row["features"].get("advances_to_target", False))
                                or bool(row["features"].get("reaches_target", False))
                            )
                        ]
                        if seek_pool:
                            seek_pool.sort(
                                key=lambda row: (
                                    0 if bool(row["features"].get("reaches_target", False)) else 1,
                                    int(row["features"].get("predicted_distance_to_target", 10**6)),
                                    -float(row["features"].get("bonus_hint", 0.0)),
                                    float(row["score"]),
                                    int(action_count_map.get(int(row["entry"].candidate.action_id), 0)),
                                    int(row["entry"].candidate.action_id),
                                    str(row["entry"].candidate.candidate_id),
                                )
                            )
                            best_seek = seek_pool[0]
                            seek_margin = float(max(self.sequence_probe_score_margin, 0.45))
                            if bool(best_seek["features"].get("reaches_target", False)) or float(
                                best_seek["score"]
                            ) <= (best_sequence_score + seek_margin):
                                selected_entry = best_seek["entry"]
                                sequence_causal_probe_applied = True
                                sequence_causal_probe_reason = "seek_target_priority"
                    if sequence_causal_probe_applied:
                        least_tried_probe_applied = True
                        if selected_entry is not entries[0]:
                            entries.remove(selected_entry)
                            entries.insert(0, selected_entry)
                sequence_causal_probe_candidates = [
                    {
                        "candidate_id": str(row["entry"].candidate.candidate_id),
                        "action_id": int(row["entry"].candidate.action_id),
                        "score": float(row["score"]),
                        "blocked_hard_skip": bool(row.get("blocked_hard_skip", False)),
                        "blocked_soft_penalty": float(row.get("blocked_soft_penalty", 0.0)),
                        "blocked_reasons": list(row.get("blocked_reasons", [])),
                        "stage": str(row["features"].get("stage", "idle")),
                        "current_region_key": str(
                            row["features"].get("current_region_key", "NA")
                        ),
                        "predicted_region_key": str(
                            row["features"].get("predicted_region_key", "NA")
                        ),
                        "predicted_distance_to_target": int(
                            row["features"].get("predicted_distance_to_target", 10**6)
                        ),
                        "advances_to_target": bool(
                            row["features"].get("advances_to_target", False)
                        ),
                        "reaches_target": bool(
                            row["features"].get("reaches_target", False)
                        ),
                        "verify_action_candidate": bool(
                            row["features"].get("verify_action_candidate", False)
                        ),
                        "bonus_hint": float(row["features"].get("bonus_hint", 0.0)),
                        "penalty_hint": float(row["features"].get("penalty_hint", 0.0)),
                        "steps_remaining": int(row["features"].get("steps_remaining", 0)),
                    }
                    for row in sequence_rows[:12]
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
                    predicted_direction = self._entry_navigation_direction(entry)
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

        selected_blocked_hard_skip_raw = False
        selected_blocked_hard_skip_recovered = False
        selected_blocked_hard_skip = False
        selected_blocked_hard_skip_replacement_applied = False
        selected_blocked_hard_skip_reasons: list[str] = []

        if entries:
            top_candidate = entries[0].candidate
            top_predicted_stats = self._candidate_predicted_region_stats(top_candidate)
            top_profile = self._candidate_coverage_block_profile(
                top_candidate,
                predicted_region_stats=top_predicted_stats,
            )
            selected_blocked_hard_skip_raw = bool(
                self._coverage_hard_skip(
                    profile=top_profile,
                    predicted_region_stats=top_predicted_stats,
                )
            )
            if selected_blocked_hard_skip_raw:
                top_focus = self._candidate_high_info_focus_features(top_candidate)
                selected_blocked_hard_skip_recovered = bool(
                    self._high_info_recoverable_hard_skip(
                        profile=top_profile,
                        focus_features=top_focus,
                    )
                )
            selected_blocked_hard_skip = bool(
                selected_blocked_hard_skip_raw and (not selected_blocked_hard_skip_recovered)
            )
            if selected_blocked_hard_skip:
                selected_blocked_hard_skip_reasons = [
                    str(reason)
                    for reason in top_profile.get("reasons", [])
                    if isinstance(reason, str)
                ]
                safe_entries: list[FreeEnergyLedgerEntryV1] = []
                for entry in entries[1:]:
                    candidate = entry.candidate
                    predicted_stats = self._candidate_predicted_region_stats(candidate)
                    profile = self._candidate_coverage_block_profile(
                        candidate,
                        predicted_region_stats=predicted_stats,
                    )
                    hard_skip_raw = bool(
                        self._coverage_hard_skip(
                            profile=profile,
                            predicted_region_stats=predicted_stats,
                        )
                    )
                    if not hard_skip_raw:
                        safe_entries.append(entry)
                        continue
                    focus = self._candidate_high_info_focus_features(candidate)
                    hard_skip_recovered = bool(
                        self._high_info_recoverable_hard_skip(
                            profile=profile,
                            focus_features=focus,
                        )
                    )
                    if not hard_skip_recovered:
                        safe_entries.append(entry)
                if safe_entries:
                    safe_entries.sort(
                        key=lambda entry: (
                            float(
                                selection_score_by_candidate.get(
                                    entry.candidate.candidate_id,
                                    entry.total_efe,
                                )
                            ),
                            int(action_count_map.get(int(entry.candidate.action_id), 0)),
                            int(entry.candidate.action_id),
                            str(entry.candidate.candidate_id),
                        )
                    )
                    replacement_entry = safe_entries[0]
                    if replacement_entry is not entries[0]:
                        entries.remove(replacement_entry)
                        entries.insert(0, replacement_entry)
                        selected_blocked_hard_skip_replacement_applied = True
                    selected_blocked_hard_skip = False

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
        if fixed_two_pass_traversal_applied:
            tie_breaker_rule_applied = (
                "fixed_two_pass_traversal_prepass(bfs_goal,min_visit<target,cross_priority)"
            )
        elif early_probe_applied:
            tie_breaker_rule_applied = "early_probe_budget_least_tried"
        elif high_info_focus_probe_applied:
            tie_breaker_rule_applied = (
                f"high_info_focus_probe_{str(high_info_focus_probe_reason)}"
                "(target_score,stage,distance,bonus,score,action_usage)"
            )
        elif sequence_causal_probe_applied:
            tie_breaker_rule_applied = (
                f"sequence_causal_probe_{str(sequence_causal_probe_reason)}"
                "(stage,bonus_hint,predicted_distance,score,action_usage)"
            )
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
            "sequence_causal_term_enabled": bool(self.sequence_causal_term_enabled),
            "high_info_focus_active_present": bool(high_info_focus_active_present),
            "high_info_focus_priority_available": bool(
                high_info_focus_priority_available
            ),
            "high_info_focus_strong_signal_available": bool(
                high_info_focus_strong_signal_available
            ),
            "high_info_focus_strong_signal_target_score": float(
                high_info_focus_strong_signal_target_score
            ),
            "high_info_focus_strong_signal_remaining_samples": int(
                high_info_focus_strong_signal_remaining_samples
            ),
            "high_info_focus_strong_signal_queue_length": int(
                high_info_focus_strong_signal_queue_length
            ),
            "high_info_after_first_pass_gate_open": bool(
                high_info_after_first_pass_gate_open
            ),
            "high_info_release_action_counter": int(high_info_release_action_counter),
            "high_info_release_min_target_score": float(
                self.high_info_release_min_target_score
            ),
            "high_info_release_min_remaining_samples": int(
                self.high_info_release_min_remaining_samples
            ),
            "high_info_release_min_queue_length": int(
                self.high_info_release_min_queue_length
            ),
            "fixed_two_pass_suppressed_by_high_info": bool(
                fixed_two_pass_suppressed_by_high_info
            ),
            "high_info_focus_probe_applied": bool(high_info_focus_probe_applied),
            "high_info_focus_probe_reason": str(high_info_focus_probe_reason),
            "sequence_causal_probe_applied": bool(sequence_causal_probe_applied),
            "sequence_causal_probe_reason": str(sequence_causal_probe_reason),
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
            "coverage_min_region_visit_count": int(coverage_min_region_visit_count),
            "coverage_regions_visited_at_least_twice": int(
                coverage_regions_visited_at_least_twice
            ),
            "coverage_target_region_count": int(coverage_target_region_count),
            "coverage_sweep_min_region_visits": int(self.coverage_sweep_min_region_visits),
            "coverage_prepass_steps": int(self.coverage_prepass_steps),
            "coverage_hard_prepass_active": bool(coverage_hard_prepass_active),
            "coverage_prepass_goal_kind": str(coverage_prepass_goal_kind),
            "coverage_prepass_goal_region": {
                "x": int(coverage_prepass_goal_region.get("x", -1)),
                "y": int(coverage_prepass_goal_region.get("y", -1)),
            },
            "coverage_prepass_bfs_applied": bool(coverage_prepass_bfs_applied),
            "coverage_prepass_bfs_next_region_key": str(
                coverage_prepass_bfs_next_region_key
            ),
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
            "fixed_two_pass_traversal_applied": bool(fixed_two_pass_traversal_applied),
            "fixed_two_pass_traversal_v1": dict(fixed_two_pass_traversal_v1),
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
            "high_info_focus_probe_candidates": list(high_info_focus_probe_candidates),
            "sequence_causal_probe_candidates": list(sequence_causal_probe_candidates),
            "selected_blocked_hard_skip_raw": bool(selected_blocked_hard_skip_raw),
            "selected_blocked_hard_skip_recovered": bool(
                selected_blocked_hard_skip_recovered
            ),
            "selected_blocked_hard_skip": bool(selected_blocked_hard_skip),
            "selected_blocked_hard_skip_replacement_applied": bool(
                selected_blocked_hard_skip_replacement_applied
            ),
            "selected_blocked_hard_skip_reasons": list(
                selected_blocked_hard_skip_reasons
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
                    "min_region_visit_count": int(
                        row.get("min_region_visit_count", 0)
                    ),
                    "regions_visited_at_least_twice": int(
                        row.get("regions_visited_at_least_twice", 0)
                    ),
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
                    "predicted_direction_model_only": str(
                        row.get("predicted_direction_model_only", "na")
                    ),
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
