import json
import logging
import os
import subprocess
from typing import Any

from arcengine import FrameData, GameAction, GameState

from ..agent import Agent

logger = logging.getLogger(__name__)

_ACTION6_MIN_COORD = 0
_ACTION6_MAX_COORD = 63


class CatCsetArcBridge(Agent):
    """Bridge agent for wiring ARC-AGI-3 loop to an external catcsetarc policy."""

    MAX_ACTIONS = 80

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.policy_command = os.getenv("CATCSETARC_POLICY_COMMAND", "").strip()
        self.policy_timeout_seconds = float(
            os.getenv("CATCSETARC_POLICY_TIMEOUT_SECONDS", "1.0")
        )

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = {
                "agent": "catcsetarc_bridge",
                "reason": "reset_required_by_state",
                "state": latest_frame.state.name,
            }
            return action

        available_action_ids = self._extract_available_action_ids(latest_frame)
        observation_packet = self._build_observation_packet(
            latest_frame, available_action_ids
        )

        if self.policy_command:
            decision = self._query_external_policy(observation_packet)
            if decision is not None:
                action = self._action_from_decision(decision, available_action_ids)
                if action is not None:
                    return action

        return self._fallback_action(available_action_ids)

    def _extract_available_action_ids(self, latest_frame: FrameData) -> list[int]:
        out: list[int] = []
        for value in latest_frame.available_actions:
            try:
                out.append(int(value))
            except Exception:
                continue
        return out

    def _build_observation_packet(
        self, latest_frame: FrameData, available_action_ids: list[int]
    ) -> dict[str, Any]:
        return {
            "schema_name": "catcsetarc_arcagi3_observation_packet_v1",
            "schema_version": 1,
            "task": {
                "game_id": self.game_id,
                "card_id": self.card_id,
                "action_counter": int(self.action_counter),
            },
            "observation": {
                "state": latest_frame.state.name,
                "levels_completed": int(latest_frame.levels_completed),
                "win_levels": int(latest_frame.win_levels),
                "available_actions": available_action_ids,
                "frame": latest_frame.frame,
            },
            "constraints": {
                "action_cost_per_step": 1,
                "action6_coordinate_min": _ACTION6_MIN_COORD,
                "action6_coordinate_max": _ACTION6_MAX_COORD,
            },
        }

    def _query_external_policy(
        self, observation_packet: dict[str, Any]
    ) -> dict[str, Any] | None:
        try:
            completed = subprocess.run(
                self.policy_command,
                shell=True,
                text=True,
                input=json.dumps(observation_packet, ensure_ascii=True),
                capture_output=True,
                timeout=self.policy_timeout_seconds,
                check=False,
            )
        except Exception as exc:
            logger.warning("catcsetarc bridge policy command failed: %s", exc)
            return None

        if completed.returncode != 0:
            logger.warning(
                "catcsetarc bridge policy returned non-zero code=%s stderr=%s",
                completed.returncode,
                completed.stderr.strip(),
            )
            return None

        output_text = completed.stdout.strip()
        if not output_text:
            logger.warning("catcsetarc bridge policy returned empty stdout")
            return None

        try:
            parsed = json.loads(output_text)
        except json.JSONDecodeError:
            lines = [line for line in output_text.splitlines() if line.strip()]
            if not lines:
                return None
            try:
                parsed = json.loads(lines[-1])
            except json.JSONDecodeError:
                logger.warning(
                    "catcsetarc bridge policy stdout is not valid JSON: %s",
                    output_text[:200],
                )
                return None

        if not isinstance(parsed, dict):
            logger.warning("catcsetarc bridge policy JSON must be object")
            return None
        return parsed

    def _action_from_decision(
        self, decision: dict[str, Any], available_action_ids: list[int]
    ) -> GameAction | None:
        action_id_any = decision.get("action_id", None)
        if action_id_any is None:
            return None

        try:
            action_id = int(action_id_any)
        except Exception:
            return None

        if action_id != GameAction.RESET.value and action_id not in available_action_ids:
            return None

        try:
            action = GameAction.from_id(action_id)
        except Exception:
            return None

        if action == GameAction.ACTION6:
            x = self._coerce_coord(decision.get("x", 0))
            y = self._coerce_coord(decision.get("y", 0))
            action.set_data({"x": x, "y": y})

        reasoning_any = decision.get("reasoning", None)
        if isinstance(reasoning_any, (dict, str)):
            action.reasoning = reasoning_any
        else:
            action.reasoning = {
                "agent": "catcsetarc_bridge",
                "source": "external_policy",
                "decision_schema": decision.get("schema_name", ""),
            }
        return action

    def _coerce_coord(self, value_any: Any) -> int:
        try:
            value_int = int(value_any)
        except Exception:
            value_int = 0
        return max(_ACTION6_MIN_COORD, min(_ACTION6_MAX_COORD, value_int))

    def _fallback_action(self, available_action_ids: list[int]) -> GameAction:
        if not available_action_ids:
            action = GameAction.RESET
            action.reasoning = {
                "agent": "catcsetarc_bridge",
                "source": "fallback",
                "reason": "no_available_actions",
            }
            return action

        # Prefer non-complex actions for a stable baseline loop.
        preferred_order = [1, 2, 3, 4, 5, 7, 6]
        selected_id = 0
        for action_id in preferred_order:
            if action_id in available_action_ids:
                selected_id = action_id
                break
        if selected_id == 0:
            selected_id = available_action_ids[0]

        action = GameAction.from_id(selected_id)
        if action == GameAction.ACTION6:
            action.set_data({"x": 31, "y": 31})

        action.reasoning = {
            "agent": "catcsetarc_bridge",
            "source": "fallback",
            "available_actions": available_action_ids,
            "selected_action_id": selected_id,
        }
        return action
