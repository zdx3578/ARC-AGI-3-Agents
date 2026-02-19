from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any


def _jsonable(value_any: Any) -> Any:
    if value_any is None:
        return None
    if isinstance(value_any, (str, int, float, bool)):
        return value_any
    if isinstance(value_any, dict):
        return {str(k): _jsonable(v) for (k, v) in value_any.items()}
    if isinstance(value_any, (list, tuple, set)):
        return [_jsonable(v) for v in value_any]
    if is_dataclass(value_any):
        return _jsonable(asdict(value_any))
    if hasattr(value_any, "to_dict"):
        return _jsonable(value_any.to_dict())
    return str(value_any)


class ActiveInferenceTraceRecorderV1:
    def __init__(
        self,
        *,
        root_dir: str,
        game_id: str,
        agent_name: str,
        card_id: str,
    ) -> None:
        safe_game_id = game_id.replace("/", "_")
        safe_agent_name = agent_name.replace("/", "_")
        trace_dir = os.path.join(root_dir, "active_inference_traces")
        os.makedirs(trace_dir, exist_ok=True)
        filename = (
            f"{safe_game_id}.{safe_agent_name}.{int(time.time())}."
            f"{uuid.uuid4().hex[:12]}.trace.jsonl"
        )
        self.path = os.path.join(trace_dir, filename)
        self._fp = open(self.path, "w", encoding="utf-8")
        self.write(
            {
                "schema_name": "active_inference_trace_start_v1",
                "schema_version": 1,
                "started_at_unix_seconds": int(time.time()),
                "game_id": game_id,
                "card_id": card_id,
                "agent_name": agent_name,
            }
        )

    def write(self, record: dict[str, Any]) -> None:
        payload = _jsonable(record)
        self._fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
        self._fp.flush()

    def close(self) -> None:
        if self._fp.closed:
            return
        self.write(
            {
                "schema_name": "active_inference_trace_end_v1",
                "schema_version": 1,
                "ended_at_unix_seconds": int(time.time()),
            }
        )
        self._fp.close()
