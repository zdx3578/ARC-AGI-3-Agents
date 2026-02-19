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
            "schema_version": self.schema_version,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "phase": self.phase,
            "candidate": self.candidate.to_dict(),
            "risk": float(self.risk),
            "ambiguity": float(self.ambiguity),
            "information_gain": float(self.information_gain),
            "action_cost": float(self.action_cost),
            "complexity_penalty": float(self.complexity_penalty),
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

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "schema_name": self.schema_name,
            "schema_version": int(self.schema_version),
            "changed_pixel_count": int(self.changed_pixel_count),
            "level_delta": int(self.level_delta),
            "state_transition": self.state_transition,
            "changed_object_count": int(self.changed_object_count),
            "event_tags": list(self.event_tags),
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
