import json
import os
import shutil
from pathlib import Path

import pytest

from agents.runtime_settings import reload_runtime_config
from agents.structs import FrameData, GameState


def get_test_recordings_dir():
    conftest_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(conftest_dir, "recordings")


def _runtime_local_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "runtime_config.local.json"


def _write_runtime_local_config(payload: dict[str, object]) -> None:
    path = _runtime_local_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    reload_runtime_config()


@pytest.fixture(scope="session", autouse=True)
def clean_test_recordings():
    test_recordings_dir = get_test_recordings_dir()
    config_path = _runtime_local_config_path()
    original_config_text: str | None = None
    if config_path.is_file():
        original_config_text = config_path.read_text(encoding="utf-8")
    _write_runtime_local_config(
        {
            "runtime": {
                "RECORDINGS_DIR": test_recordings_dir,
                "ARC_API_KEY": "test-key",
                "SCHEME": "https",
                "HOST": "three.arcprize.org",
                "PORT": 443,
            },
            "agents": {
                "OPENAI_API_KEY": "test-openai-key",
            },
        }
    )

    if os.path.exists(test_recordings_dir):
        shutil.rmtree(test_recordings_dir)
    os.makedirs(test_recordings_dir, exist_ok=True)

    yield test_recordings_dir

    if original_config_text is None:
        if config_path.exists():
            config_path.unlink()
    else:
        config_path.write_text(original_config_text, encoding="utf-8")
    reload_runtime_config()


@pytest.fixture
def temp_recordings_dir(clean_test_recordings):
    test_recordings_dir = get_test_recordings_dir()

    os.makedirs(test_recordings_dir, exist_ok=True)

    yield test_recordings_dir


@pytest.fixture
def sample_frame():
    return FrameData(
        game_id="test-game",
        frame=[[[1, 2], [3, 4]]],
        state=GameState.NOT_FINISHED,
        score=5,
    )


@pytest.fixture
def use_env_vars():
    _write_runtime_local_config(
        {
            "runtime": {
                "ARC_API_KEY": "test-key",
                "SCHEME": "https",
                "HOST": "three.arcprize.org",
                "PORT": 443,
            },
            "agents": {
                "OPENAI_API_KEY": "test-openai-key",
            },
        }
    )
    yield
