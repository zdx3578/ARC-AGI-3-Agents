from __future__ import annotations

import time
from typing import Any

from .contracts import StageDiagnosticV1


class StageDiagnosticsCollectorV1:
    """Collect stage-level diagnostics for one decision step."""

    def __init__(self) -> None:
        self._records: list[StageDiagnosticV1] = []
        self._stage_start_time: dict[str, float] = {}

    def start(self, stage_name: str) -> None:
        self._stage_start_time[stage_name] = time.perf_counter()

    def finish_ok(self, stage_name: str, summary: dict[str, Any] | None = None) -> None:
        self._records.append(
            StageDiagnosticV1(
                schema_name="active_inference_stage_diagnostic_v1",
                schema_version=1,
                stage_name=stage_name,
                status="ok",
                duration_ms=self._elapsed_ms(stage_name),
                reject_reason_v1=None,
                summary=summary or {},
            )
        )

    def finish_rejected(
        self,
        stage_name: str,
        reject_reason_v1: str,
        summary: dict[str, Any] | None = None,
    ) -> None:
        self._records.append(
            StageDiagnosticV1(
                schema_name="active_inference_stage_diagnostic_v1",
                schema_version=1,
                stage_name=stage_name,
                status="rejected",
                duration_ms=self._elapsed_ms(stage_name),
                reject_reason_v1=reject_reason_v1,
                summary=summary or {},
            )
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._records]

    def bottleneck_stage(self) -> dict[str, Any]:
        if not self._records:
            return {
                "stage_name": "none",
                "criterion": "no_records",
                "duration_ms": 0.0,
            }

        for record in self._records:
            if record.status != "ok":
                return {
                    "stage_name": record.stage_name,
                    "criterion": "first_rejected_stage",
                    "duration_ms": float(record.duration_ms),
                    "reject_reason_v1": record.reject_reason_v1,
                }

        slowest = max(self._records, key=lambda record: float(record.duration_ms))
        return {
            "stage_name": slowest.stage_name,
            "criterion": "max_duration_ms",
            "duration_ms": float(slowest.duration_ms),
        }

    def _elapsed_ms(self, stage_name: str) -> float:
        started = self._stage_start_time.pop(stage_name, None)
        if started is None:
            return 0.0
        return (time.perf_counter() - started) * 1000.0
