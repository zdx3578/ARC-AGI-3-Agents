from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ObservationPacketV1:
    schema_name: str
    schema_version: int
    game_id: str
    card_id: str
    action_counter: int
    state: str
    levels_completed: int
    win_levels: int
    available_actions: list[int]
    frame: list[list[int]]
    action_cost_per_step: int = 1
    action6_coordinate_min: int = 0
    action6_coordinate_max: int = 63

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "task": {
                "game_id": self.game_id,
                "card_id": self.card_id,
                "action_counter": int(self.action_counter),
            },
            "observation": {
                "state": self.state,
                "levels_completed": int(self.levels_completed),
                "win_levels": int(self.win_levels),
                "available_actions": [int(v) for v in self.available_actions],
                "frame": self.frame,
            },
            "constraints": {
                "action_cost_per_step": int(self.action_cost_per_step),
                "action6_coordinate_min": int(self.action6_coordinate_min),
                "action6_coordinate_max": int(self.action6_coordinate_max),
            },
        }


@dataclass(slots=True)
class ComponentNodeV1:
    component_id: str
    connectivity: int
    color_connectivity: str
    area: int
    centroid_x: int
    centroid_y: int
    bbox_min_x: int
    bbox_min_y: int
    bbox_max_x: int
    bbox_max_y: int
    touches_boundary: bool
    color_signature: list[int]
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "connectivity": int(self.connectivity),
            "color_connectivity": self.color_connectivity,
            "area": int(self.area),
            "centroid_x": int(self.centroid_x),
            "centroid_y": int(self.centroid_y),
            "bbox_min_x": int(self.bbox_min_x),
            "bbox_min_y": int(self.bbox_min_y),
            "bbox_max_x": int(self.bbox_max_x),
            "bbox_max_y": int(self.bbox_max_y),
            "touches_boundary": bool(self.touches_boundary),
            "color_signature": [int(v) for v in self.color_signature],
            "digest": self.digest,
        }


@dataclass(slots=True)
class ObjectNodeV1:
    object_id: str
    color: int
    area: int
    centroid_x: int
    centroid_y: int
    bbox_min_x: int
    bbox_min_y: int
    bbox_max_x: int
    bbox_max_y: int
    touches_boundary: bool
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "color": int(self.color),
            "area": int(self.area),
            "centroid_x": int(self.centroid_x),
            "centroid_y": int(self.centroid_y),
            "bbox_min_x": int(self.bbox_min_x),
            "bbox_min_y": int(self.bbox_min_y),
            "bbox_max_x": int(self.bbox_max_x),
            "bbox_max_y": int(self.bbox_max_y),
            "touches_boundary": bool(self.touches_boundary),
            "digest": self.digest,
        }


@dataclass(slots=True)
class RepresentationStateV1:
    schema_name: str
    schema_version: int
    frame_height: int
    frame_width: int
    background_color: int
    object_nodes: list[ObjectNodeV1]
    action6_coordinate_proposals: list[tuple[int, int]]
    summary: dict[str, Any]
    component_views: dict[str, list[ComponentNodeV1]] = field(default_factory=dict)
    hierarchy_links: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "frame_height": int(self.frame_height),
            "frame_width": int(self.frame_width),
            "background_color": int(self.background_color),
            "object_nodes": [obj.to_dict() for obj in self.object_nodes],
            "action6_coordinate_proposals": [
                {"x": int(x), "y": int(y)} for (x, y) in self.action6_coordinate_proposals
            ],
            "summary": self.summary,
            "component_views": {
                str(view_name): [node.to_dict() for node in nodes]
                for (view_name, nodes) in self.component_views.items()
            },
            "hierarchy_links": [dict(link) for link in self.hierarchy_links],
        }


@dataclass(slots=True)
class ActionCandidateV1:
    candidate_id: str
    action_id: int
    x: int | None = None
    y: int | None = None
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "candidate_id": self.candidate_id,
            "action_id": int(self.action_id),
            "source": self.source,
            "metadata": self.metadata,
        }
        if self.x is not None:
            out["x"] = int(self.x)
        if self.y is not None:
            out["y"] = int(self.y)
        return out


@dataclass(slots=True)
class FreeEnergyLedgerEntryV1:
    schema_name: str
    schema_version: int
    phase: str
    candidate: ActionCandidateV1
    risk: float
    ambiguity: float
    information_gain: float
    action_cost: float
    complexity_penalty: float
    total_efe: float
    predictive_signature_distribution: dict[str, float]
    witness: dict[str, Any]
    information_gain_action_semantics: float = 0.0
    information_gain_mechanism_dynamics: float = 0.0
    information_gain_causal_mapping: float = 0.0
    vfe_current: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "phase": self.phase,
            "candidate": self.candidate.to_dict(),
            "risk": float(self.risk),
            "ambiguity": float(self.ambiguity),
            "information_gain": float(self.information_gain),
            "information_gain_action_semantics": float(
                self.information_gain_action_semantics
            ),
            "information_gain_mechanism_dynamics": float(
                self.information_gain_mechanism_dynamics
            ),
            "information_gain_causal_mapping": float(self.information_gain_causal_mapping),
            "action_cost": float(self.action_cost),
            "complexity_penalty": float(self.complexity_penalty),
            "vfe_current": float(self.vfe_current),
            "total_efe": float(self.total_efe),
            "predictive_signature_distribution": {
                str(k): float(v)
                for (k, v) in self.predictive_signature_distribution.items()
            },
            "witness": self.witness,
        }


@dataclass(slots=True)
class CausalEventSignatureV1:
    schema_name: str
    schema_version: int
    changed_pixel_count: int
    changed_bbox: tuple[int, int, int, int] | None
    level_delta: int
    state_transition: str
    changed_object_count: int
    event_tags: list[str]
    signature_digest: str
    obs_change_type: str = "OBSERVED_UNCLASSIFIED"
    changed_area_ratio: float = 0.0
    palette_delta_topk: dict[str, int] = field(default_factory=dict)
    cc_match_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "obs_change_type": self.obs_change_type,
            "changed_pixel_count": int(self.changed_pixel_count),
            "changed_area_ratio": float(self.changed_area_ratio),
            "level_delta": int(self.level_delta),
            "state_transition": self.state_transition,
            "changed_object_count": int(self.changed_object_count),
            "event_tags": list(self.event_tags),
            "palette_delta_topk": {
                str(k): int(v) for (k, v) in self.palette_delta_topk.items()
            },
            "cc_match_summary": self.cc_match_summary,
            "signature_digest": self.signature_digest,
        }
        if self.changed_bbox is not None:
            out["changed_bbox"] = {
                "min_x": int(self.changed_bbox[0]),
                "min_y": int(self.changed_bbox[1]),
                "max_x": int(self.changed_bbox[2]),
                "max_y": int(self.changed_bbox[3]),
            }
        return out


@dataclass(slots=True)
class HiddenModeStateV1:
    mode_id: str
    switch_vector: dict[str, int] = field(default_factory=dict)
    inventory_state: dict[str, int] = field(default_factory=dict)
    region_latent_label: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "switch_vector": {str(k): int(v) for (k, v) in self.switch_vector.items()},
            "inventory_state": {
                str(k): int(v) for (k, v) in self.inventory_state.items()
            },
            "region_latent_label": self.region_latent_label,
        }


@dataclass(slots=True)
class MechanismParameterV1:
    rule_family_id: str
    parameter_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_family_id": self.rule_family_id,
            "parameter_id": self.parameter_id,
        }


@dataclass(slots=True)
class StageDiagnosticV1:
    schema_name: str
    schema_version: int
    stage_name: str
    status: str
    duration_ms: float
    reject_reason_v1: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "stage_name": self.stage_name,
            "status": self.status,
            "duration_ms": float(self.duration_ms),
            "summary": self.summary,
        }
        if self.reject_reason_v1 is not None:
            out["reject_reason_v1"] = self.reject_reason_v1
        return out
