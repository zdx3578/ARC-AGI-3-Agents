from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import main


@pytest.mark.unit
def test_parse_requested_games():
    assert main._parse_requested_games(None) == []
    assert main._parse_requested_games("") == []
    assert main._parse_requested_games("ls20, ft09 ,vc33") == ["ls20", "ft09", "vc33"]


@pytest.mark.unit
def test_resolve_games_with_catalog_prefix_matching():
    games, using_unvalidated_request = main._resolve_games(
        full_games=["ls20-aaaa", "ft09-bbbb", "vc33-cccc"],
        requested_games=["ls20", "vc33"],
    )

    assert using_unvalidated_request is False
    assert games == ["ls20-aaaa", "vc33-cccc"]


@pytest.mark.unit
def test_resolve_games_without_catalog_uses_requested_ids():
    games, using_unvalidated_request = main._resolve_games(
        full_games=[],
        requested_games=["ls20", "ft09"],
    )

    assert using_unvalidated_request is True
    assert games == ["ls20", "ft09"]


@pytest.mark.unit
@patch("main.Arcade")
def test_collect_available_games_from_arcade_catalog(mock_arcade_cls):
    mock_arcade = Mock()
    mock_arcade.get_environments.return_value = [
        SimpleNamespace(game_id="ls20"),
        SimpleNamespace(game_id="ls20"),
        SimpleNamespace(game_id="ft09"),
        SimpleNamespace(game_id=""),
    ]
    mock_arcade_cls.return_value = mock_arcade

    games = main._collect_available_games()

    assert games == ["ls20", "ft09"]
    mock_arcade.get_environments.assert_called_once()
