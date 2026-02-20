# ruff: noqa: E402
import os

from dotenv import load_dotenv

# Preserve explicit runtime environment overrides (e.g. OPERATION_MODE=offline uv run ...)
# before loading dotenv files.
_RUNTIME_ENV_OVERRIDES = {
    key: os.environ[key]
    for key in (
        "ARC_API_KEY",
        "ARC_BASE_URL",
        "OPERATION_MODE",
        "ENVIRONMENTS_DIR",
        "RECORDINGS_DIR",
        "ONLINE_ONLY",
        "OFFLINE_ONLY",
    )
    if key in os.environ
}

load_dotenv(dotenv_path=".env.example")
load_dotenv(dotenv_path=".env", override=True)

import argparse
import json
import logging
import signal
import sys
import threading
from functools import partial
from types import FrameType
from typing import Optional

from arc_agi import Arcade
from agents import AVAILABLE_AGENTS, Swarm
from agents.tracing import initialize as init_agentops

# Re-apply runtime overrides after imports so explicit shell vars keep highest precedence.
os.environ.update(_RUNTIME_ENV_OVERRIDES)

logger = logging.getLogger()

SCHEME = os.environ.get("SCHEME", "http")
HOST = os.environ.get("HOST", "localhost")
PORT = os.environ.get("PORT", 8001)

# Hide standard ports in URL
if (SCHEME == "http" and str(PORT) == "80") or (
    SCHEME == "https" and str(PORT) == "443"
):
    ROOT_URL = f"{SCHEME}://{HOST}"
else:
    ROOT_URL = f"{SCHEME}://{HOST}:{PORT}"


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _parse_requested_games(game_arg: Optional[str]) -> list[str]:
    if not game_arg:
        return []
    return [token.strip() for token in game_arg.split(",") if token.strip()]


def _collect_available_games() -> list[str]:
    try:
        arcade = Arcade()
        environments = arcade.get_environments()
        game_ids = _dedupe_preserve_order(
            [env.game_id for env in environments if env.game_id]
        )
        if game_ids:
            logger.info(
                "Discovered %d game(s) from Arcade environment catalog", len(game_ids)
            )
        else:
            logger.warning(
                "Arcade environment catalog is empty (operation_mode=%s, environments_dir=%s)",
                arcade.operation_mode,
                arcade.environments_dir,
            )
        return game_ids
    except Exception as e:
        logger.error("Failed to discover games from Arcade catalog: %s", e)
        return []


def _resolve_games(
    full_games: list[str], requested_games: list[str]
) -> tuple[list[str], bool]:
    if not requested_games:
        return full_games[:], False

    if full_games:
        games = [
            gid
            for gid in full_games
            if any(gid.startswith(prefix) for prefix in requested_games)
        ]
        return _dedupe_preserve_order(games), False

    # Catalog unavailable: trust user-provided game ids and let Arcade resolve at runtime.
    return _dedupe_preserve_order(requested_games), True


def run_agent(swarm: Swarm) -> None:
    swarm.main()
    os.kill(os.getpid(), signal.SIGINT)


def cleanup(
    swarm: Swarm,
    signum: Optional[int],
    frame: Optional[FrameType],
) -> None:
    logger.info("Received SIGINT, exiting...")
    card_id = swarm.card_id
    if card_id:
        scorecard = swarm.close_scorecard(card_id)
        if scorecard:
            logger.info("--- EXISTING SCORECARD REPORT ---")
            logger.info(json.dumps(scorecard.model_dump(), indent=2))
            swarm.cleanup(scorecard)

        # Provide web link to scorecard
        if card_id:
            scorecard_url = f"{ROOT_URL}/scorecards/{card_id}"
            logger.info(f"View your scorecard online: {scorecard_url}")

    sys.exit(0)


def main() -> None:
    log_level = logging.INFO
    if os.environ.get("DEBUG", "False") == "True":
        log_level = logging.DEBUG

    logger.setLevel(log_level)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("logs.log", mode="w")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # logging.getLogger("requests").setLevel(logging.CRITICAL)
    # logging.getLogger("werkzeug").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="ARC-AGI-3-Agents")
    parser.add_argument(
        "-a",
        "--agent",
        choices=AVAILABLE_AGENTS.keys(),
        help="Choose which agent to run.",
    )
    parser.add_argument(
        "-g",
        "--game",
        help="Choose a specific game_id for the agent to play. If none specified, an agent swarm will play all available games.",
    )
    parser.add_argument(
        "-t",
        "--tags",
        type=str,
        help="Comma-separated list of tags for the scorecard (e.g., 'experiment,v1.0')",
        default=None,
    )

    args = parser.parse_args()

    if not args.agent:
        logger.error("An Agent must be specified")
        return

    full_games = _collect_available_games()
    requested_games = _parse_requested_games(args.game)
    games, using_unvalidated_request = _resolve_games(full_games, requested_games)

    # For playback agents, we can derive the game from the recording filename
    if not full_games and args.agent and args.agent.endswith(".recording.jsonl"):
        from agents.recorder import Recorder

        game_prefix = Recorder.get_prefix_one(args.agent)
        full_games = [game_prefix]
        games = [game_prefix]
        logger.info(
            "Using game '%s' derived from playback recording filename", game_prefix
        )

    if using_unvalidated_request and games:
        logger.warning(
            "Environment catalog unavailable; attempting requested game id(s) directly: %s",
            games,
        )

    logger.info(f"Game list: {games}")

    if not games:
        if requested_games and full_games:
            logger.error(
                f"The specified game '{args.game}' does not exist or is not available with your API key. Please try a different game."
            )
        else:
            logger.error(
                "No games available to play. Provide --game explicitly, or run in offline mode with local environments."
            )
        return

    # Start with Empty tags, "agent" and agent name will be added by the Swarm later
    tags: list[str] = []

    # Append user-provided tags if any
    if args.tags:
        user_tags = [tag.strip() for tag in args.tags.split(",")]
        tags.extend(user_tags)

    # Initialize AgentOps client
    init_agentops(api_key=os.getenv("AGENTOPS_API_KEY"), log_level=log_level)

    swarm = Swarm(
        args.agent,
        ROOT_URL,
        games,
        tags=tags,  # Pass tags as keyword argument
    )
    agent_thread = threading.Thread(target=partial(run_agent, swarm))
    agent_thread.daemon = True  # die when the main thread dies
    agent_thread.start()

    signal.signal(signal.SIGINT, partial(cleanup, swarm))  # handler for Ctrl+C

    try:
        # Wait for the agent thread to complete
        while agent_thread.is_alive():
            agent_thread.join(timeout=5)  # Check every 5 second
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received in main thread")
        cleanup(swarm, signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Unexpected error in main thread: {e}")
        cleanup(swarm, None, None)


if __name__ == "__main__":
    os.environ["TESTING"] = "False"
    main()
