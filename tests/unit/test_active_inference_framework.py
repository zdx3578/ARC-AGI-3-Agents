from types import SimpleNamespace

import pytest

from agents.templates.active_inference.diagnostics import StageDiagnosticsCollectorV1
from agents.templates.active_inference.agent import ActiveInferenceEFE
from agents.templates.active_inference.contracts import (
    ActionCandidateV1,
    FreeEnergyLedgerEntryV1,
)
from agents.templates.active_inference.hypothesis_bank import (
    ActiveInferenceHypothesisBankV1,
    build_causal_event_signature_v1,
)
from agents.templates.active_inference.policy import ActiveInferencePolicyEvaluatorV1
from agents.templates.active_inference.representation import (
    build_action_candidates_v1,
    build_observation_packet_v1,
    build_representation_state_v1,
)


def _frame_data_stub(
    *,
    frame: list[list[int]] | None = None,
    levels_completed: int = 0,
    state_name: str = "NOT_FINISHED",
    available_actions: list[object] | None = None,
) -> SimpleNamespace:
    frame_payload = frame or [[1, 1, 0], [1, 0, 0], [0, 2, 2]]
    return SimpleNamespace(
        available_actions=available_actions or [6, 6, "3", "bad"],
        frame=[frame_payload],
        state=SimpleNamespace(name=state_name),
        levels_completed=levels_completed,
        win_levels=3,
    )


@pytest.mark.unit
def test_observation_packet_normalizes_nested_frame() -> None:
    frame_chain = [
        SimpleNamespace(frame=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
        SimpleNamespace(frame=[[1, 1, 0], [1, 0, 0], [0, 2, 2]]),
    ]
    packet = build_observation_packet_v1(
        _frame_data_stub(),
        game_id="ls20",
        card_id="card-x",
        action_counter=7,
        frame_chain=frame_chain,
    )
    assert packet.available_actions == [3, 6]
    assert len(packet.frame) == 3
    assert len(packet.frame[0]) == 3
    assert packet.frame[0][0] == 1
    assert packet.num_frames_received >= 2
    assert packet.frame_chain_digests
    assert packet.frame_chain_macro_signature["micro_signature_count"] >= 1
    assert packet.frame_chain_macro_signature["dominant_micro_object_change_type"]
    assert packet.frame_chain_micro_signatures[0]["micro_pixel_change_type"]
    assert packet.frame_chain_micro_signatures[0]["micro_object_change_type"]


@pytest.mark.unit
def test_representation_and_action6_candidates_expose_coverage() -> None:
    packet = build_observation_packet_v1(
        _frame_data_stub(),
        game_id="ls20",
        card_id="card-x",
        action_counter=7,
    )
    representation = build_representation_state_v1(
        packet,
        connectivity=8,
        max_action6_points=12,
    )
    assert representation.summary["object_count"] >= 1
    assert representation.summary["action6_coordinate_proposal_count"] >= 1
    assert representation.summary["action6_coordinate_proposal_coverage"] > 0.0
    assert representation.summary["action6_candidate_diagnostics"]["proposal_count"] >= 1
    assert (
        representation.summary["action6_candidate_diagnostics"]["unique_region_count"] >= 1
    )
    assert "same_color_4" in representation.component_views
    assert "same_color_8" in representation.component_views
    assert "mixed_color_4" in representation.component_views
    assert "mixed_color_8" in representation.component_views
    assert representation.hierarchy_links

    candidates = build_action_candidates_v1(packet, representation)
    action6 = [candidate for candidate in candidates if candidate.action_id == 6]
    assert action6
    assert "proposal_coverage" in action6[0].metadata
    assert "proposal_diagnostics" in action6[0].metadata
    assert "coordinate_context_feature" in action6[0].metadata


@pytest.mark.unit
def test_policy_accepts_weight_override() -> None:
    packet = build_observation_packet_v1(
        _frame_data_stub(),
        game_id="ls20",
        card_id="card-x",
        action_counter=2,
    )
    representation = build_representation_state_v1(packet)
    candidates = build_action_candidates_v1(packet, representation)
    bank = ActiveInferenceHypothesisBankV1()
    policy = ActiveInferencePolicyEvaluatorV1(
        weight_overrides={
            "explore": {"information_gain_mechanism_dynamics": 1.75}
        }
    )
    entries = policy.evaluate_candidates(
        packet=packet,
        representation=representation,
        candidates=candidates,
        hypothesis_bank=bank,
        phase="explore",
    )
    assert entries
    assert entries[0].witness["weights"]["information_gain_mechanism_dynamics"] == pytest.approx(1.75)
    assert entries[0].information_gain_mechanism_dynamics >= 0.0


@pytest.mark.unit
def test_policy_ignores_action_cost_even_with_override() -> None:
    packet = build_observation_packet_v1(
        _frame_data_stub(),
        game_id="ls20",
        card_id="card-x",
        action_counter=2,
    )
    representation = build_representation_state_v1(packet)
    candidates = build_action_candidates_v1(packet, representation)
    bank = ActiveInferenceHypothesisBankV1()
    policy = ActiveInferencePolicyEvaluatorV1(
        weight_overrides={"explore": {"action_cost": 9.0}},
        ignore_action_cost=True,
    )
    entries = policy.evaluate_candidates(
        packet=packet,
        representation=representation,
        candidates=candidates,
        hypothesis_bank=bank,
        phase="explore",
    )
    assert entries
    assert entries[0].witness["weights"]["action_cost"] == pytest.approx(0.0)
    assert entries[0].witness["objective_policy_v1"]["ignore_action_cost"] is True
    assert entries[0].witness["objective_policy_v1"]["applied_action_cost_weight"] == pytest.approx(
        0.0
    )


@pytest.mark.unit
def test_policy_selection_reports_tie_diagnostics() -> None:
    packet = build_observation_packet_v1(
        _frame_data_stub(),
        game_id="ls20",
        card_id="card-x",
        action_counter=2,
    )
    representation = build_representation_state_v1(packet)
    candidates = build_action_candidates_v1(packet, representation)
    bank = ActiveInferenceHypothesisBankV1()
    policy = ActiveInferencePolicyEvaluatorV1(rollout_horizon=1)
    selected, entries = policy.select_action(
        packet=packet,
        representation=representation,
        candidates=candidates,
        hypothesis_bank=bank,
        phase="explore",
        remaining_budget=1,
        action_select_count={int(candidate.action_id): 0 for candidate in candidates},
        candidate_select_count={str(candidate.candidate_id): 0 for candidate in candidates},
    )
    assert selected.candidate_id
    assert entries
    diagnostics = entries[0].witness["selection_diagnostics_v1"]
    assert "best_vs_second_best_delta_total_efe" in diagnostics
    assert "tie_group_size" in diagnostics
    assert "tie_breaker_rule_applied" in diagnostics
    assert "least_tried_probe_applied" in diagnostics


@pytest.mark.unit
def test_policy_early_probe_budget_forces_under_tried_action() -> None:
    packet = build_observation_packet_v1(
        _frame_data_stub(available_actions=[1, 2]),
        game_id="ls20",
        card_id="card-x",
        action_counter=2,
    )
    representation = build_representation_state_v1(packet)
    candidate_a = ActionCandidateV1(candidate_id="a1", action_id=1, source="test")
    candidate_b = ActionCandidateV1(candidate_id="a2", action_id=2, source="test")
    entry_a = FreeEnergyLedgerEntryV1(
        schema_name="test",
        schema_version=1,
        phase="explore",
        candidate=candidate_a,
        risk=0.0,
        ambiguity=0.0,
        information_gain=0.0,
        action_cost=1.0,
        complexity_penalty=0.0,
        total_efe=1.0,
        predictive_signature_distribution={},
        witness={},
    )
    entry_b = FreeEnergyLedgerEntryV1(
        schema_name="test",
        schema_version=1,
        phase="explore",
        candidate=candidate_b,
        risk=0.0,
        ambiguity=0.0,
        information_gain=0.0,
        action_cost=1.0,
        complexity_penalty=0.0,
        total_efe=2.0,
        predictive_signature_distribution={},
        witness={},
    )

    policy = ActiveInferencePolicyEvaluatorV1(rollout_horizon=1)
    policy.evaluate_candidates = lambda **kwargs: [entry_a, entry_b]  # type: ignore[method-assign]
    selected, entries = policy.select_action(
        packet=packet,
        representation=representation,
        candidates=[candidate_a, candidate_b],
        hypothesis_bank=ActiveInferenceHypothesisBankV1(),
        phase="explore",
        remaining_budget=1,
        action_select_count={1: 5, 2: 0},
        candidate_select_count={"a1": 5, "a2": 0},
        early_probe_budget_remaining=4,
    )
    diagnostics = entries[0].witness["selection_diagnostics_v1"]
    assert selected.action_id == 2
    assert diagnostics["early_probe_applied"] is True
    assert diagnostics["tie_breaker_rule_applied"] == "early_probe_budget_least_tried"


@pytest.mark.unit
def test_agent_memory_policy_is_hard_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTIVE_INFERENCE_TRACE_ENABLED", "0")
    monkeypatch.setenv("ACTIVE_INFERENCE_ENABLE_CROSS_EPISODE_MEMORY", "1")
    agent = ActiveInferenceEFE(
        card_id="card-x",
        game_id="ls20",
        agent_name="activeinferenceefe",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=SimpleNamespace(),
    )
    memory_policy = agent._memory_policy_v1()
    assert memory_policy["cross_episode_memory"] == "off_hard"
    assert memory_policy["persistent_learning_store_used"] is False
    assert memory_policy["enable_requested"] is True
    assert memory_policy["override_blocked"] is True

    diagnostics = StageDiagnosticsCollectorV1()
    failure_reasoning = agent._reasoning_for_failure(
        packet=None,
        diagnostics=diagnostics,
        failure_code="TEST_FAILURE",
        failure_message="test",
    )
    assert failure_reasoning["memory_policy_v1"]["cross_episode_memory"] == "off_hard"
    assert "exploration_policy_v1" in failure_reasoning


@pytest.mark.unit
def test_agent_effective_explore_steps_uses_budget_bounded_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACTIVE_INFERENCE_TRACE_ENABLED", "0")
    monkeypatch.setenv("ACTIVE_INFERENCE_MAX_ACTIONS", "100")
    monkeypatch.setenv("ACTIVE_INFERENCE_EXPLORE_STEPS", "10")
    monkeypatch.setenv("ACTIVE_INFERENCE_EXPLORATION_MIN_STEPS", "12")
    monkeypatch.setenv("ACTIVE_INFERENCE_EXPLORATION_MAX_STEPS", "60")
    monkeypatch.setenv("ACTIVE_INFERENCE_EXPLORATION_FRACTION", "0.4")
    monkeypatch.setenv("ACTIVE_INFERENCE_EARLY_PROBE_BUDGET", "8")
    agent = ActiveInferenceEFE(
        card_id="card-x",
        game_id="ls20",
        agent_name="activeinferenceefe",
        ROOT_URL="http://localhost",
        record=False,
        arc_env=SimpleNamespace(),
    )
    packet = build_observation_packet_v1(
        _frame_data_stub(available_actions=[1, 2, 3, 4, 5, 6]),
        game_id="ls20",
        card_id="card-x",
        action_counter=3,
    )
    effective = agent._effective_explore_steps(packet)
    assert effective == 40

    exploration_policy = agent._exploration_policy_v1(
        packet=packet,
        effective_explore_steps=effective,
        remaining_budget=90,
        early_probe_budget_remaining=5,
    )
    assert exploration_policy["effective_explore_steps"] == 40
    assert exploration_policy["available_action_count"] == 6
    assert exploration_policy["exploration_budget_remaining"] == 40
    assert exploration_policy["action_cost_in_objective"] == "off_hard"
    assert exploration_policy["action_cost_override_blocked"] is False


@pytest.mark.unit
def test_hypothesis_bank_split_information_gain_contract() -> None:
    packet = build_observation_packet_v1(
        _frame_data_stub(),
        game_id="ls20",
        card_id="card-x",
        action_counter=2,
    )
    representation = build_representation_state_v1(packet)
    candidates = build_action_candidates_v1(packet, representation)
    bank = ActiveInferenceHypothesisBankV1()
    split = bank.split_information_gain(packet, candidates[0], representation)
    assert set(split.keys()) == {
        "action_semantics",
        "mechanism_dynamics",
        "causal_mapping",
    }
    assert all(value >= 0.0 for value in split.values())


@pytest.mark.unit
def test_hypothesis_bank_posterior_delta_report_fields() -> None:
    previous_packet = build_observation_packet_v1(
        _frame_data_stub(frame=[[1, 1, 0], [1, 0, 0], [0, 2, 2]]),
        game_id="ls20",
        card_id="card-x",
        action_counter=1,
    )
    current_packet = build_observation_packet_v1(
        _frame_data_stub(frame=[[1, 1, 0], [1, 3, 0], [0, 2, 2]]),
        game_id="ls20",
        card_id="card-x",
        action_counter=2,
    )
    previous_representation = build_representation_state_v1(previous_packet)
    current_representation = build_representation_state_v1(current_packet)
    executed_candidate = build_action_candidates_v1(previous_packet, previous_representation)[0]

    signature = build_causal_event_signature_v1(
        previous_packet,
        current_packet,
        previous_representation,
        current_representation,
        executed_candidate,
    )
    bank = ActiveInferenceHypothesisBankV1()
    bank.update_with_observation(
        previous_packet=previous_packet,
        current_packet=current_packet,
        executed_candidate=executed_candidate,
        previous_representation=previous_representation,
        observed_signature=signature,
    )
    report = bank.last_posterior_delta_report
    assert report["schema_name"] == "active_inference_posterior_delta_report_v1"
    assert "eliminated_count_by_reason" in report
    assert "survivor_family_histogram" in report
    assert "mode_transition_count" in report
    assert "mode_transition_soft_confidence" in report


@pytest.mark.unit
def test_hypothesis_bank_action_space_constraints_report() -> None:
    bank = ActiveInferenceHypothesisBankV1()
    report = bank.apply_action_space_constraints([1, 2, 3, 4])
    assert report["schema_name"] == "active_inference_action_space_constraint_report_v1"
    assert "mode_elimination_due_to_action_space_incompatibility" in report
    assert report["active_hypothesis_count_after"] <= report["active_hypothesis_count_before"]


@pytest.mark.unit
def test_stage_diagnostics_reports_first_rejected_stage() -> None:
    collector = StageDiagnosticsCollectorV1()
    collector.start("a1")
    collector.finish_ok("a1", {"ok": True})
    collector.start("a2")
    collector.finish_rejected("a2", "representation_invalid")
    collector.start("a3")
    collector.finish_ok("a3")

    bottleneck = collector.bottleneck_stage()
    assert bottleneck["stage_name"] == "a2"
    assert bottleneck["criterion"] == "first_rejected_stage"
