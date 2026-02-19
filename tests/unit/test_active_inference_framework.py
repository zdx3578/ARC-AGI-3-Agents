from types import SimpleNamespace

import pytest

from agents.templates.active_inference.diagnostics import StageDiagnosticsCollectorV1
from agents.templates.active_inference.hypothesis_bank import ActiveInferenceHypothesisBankV1
from agents.templates.active_inference.policy import ActiveInferencePolicyEvaluatorV1
from agents.templates.active_inference.representation import (
    build_action_candidates_v1,
    build_observation_packet_v1,
    build_representation_state_v1,
)


def _frame_data_stub() -> SimpleNamespace:
    return SimpleNamespace(
        available_actions=[6, 6, "3", "bad"],
        frame=[[[1, 1, 0], [1, 0, 0], [0, 2, 2]]],
        state=SimpleNamespace(name="NOT_FINISHED"),
        levels_completed=0,
        win_levels=3,
    )


@pytest.mark.unit
def test_observation_packet_normalizes_nested_frame() -> None:
    packet = build_observation_packet_v1(
        _frame_data_stub(),
        game_id="ls20",
        card_id="card-x",
        action_counter=7,
    )
    assert packet.available_actions == [3, 6]
    assert len(packet.frame) == 3
    assert len(packet.frame[0]) == 3
    assert packet.frame[0][0] == 1


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

    candidates = build_action_candidates_v1(packet, representation)
    action6 = [candidate for candidate in candidates if candidate.action_id == 6]
    assert action6
    assert "proposal_coverage" in action6[0].metadata


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
        weight_overrides={"explore": {"information_gain": 1.75}}
    )
    entries = policy.evaluate_candidates(
        packet=packet,
        representation=representation,
        candidates=candidates,
        hypothesis_bank=bank,
        phase="explore",
    )
    assert entries
    assert entries[0].witness["weights"]["information_gain"] == pytest.approx(1.75)


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
