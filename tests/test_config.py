"""Tests for configuration loading (no API keys or servers needed)."""

import json

import pytest

from mcp_client_cli import config as config_module


def _write_config(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_missing_mcp_servers_raises_clear_error(tmp_path, monkeypatch):
    """A config without 'mcpServers' must raise a clear error, not KeyError."""
    cfg = tmp_path / "mcp-server-config.json"
    _write_config(cfg, {"llm": {}, "systemPrompt": "be helpful"})
    monkeypatch.setattr(config_module, "CONFIG_FILE", str(cfg))

    with pytest.raises(ValueError, match="mcpServers"):
        config_module.AppConfig.load()


def test_load_missing_system_prompt_raises_clear_error(tmp_path, monkeypatch):
    """A config without 'systemPrompt' must raise a clear error, not KeyError."""
    cfg = tmp_path / "mcp-server-config.json"
    _write_config(cfg, {"llm": {}, "mcpServers": {}})
    monkeypatch.setattr(config_module, "CONFIG_FILE", str(cfg))

    with pytest.raises(ValueError, match="systemPrompt"):
        config_module.AppConfig.load()


def test_load_valid_minimal_config(tmp_path, monkeypatch):
    """A config with both required keys loads normally."""
    cfg = tmp_path / "mcp-server-config.json"
    _write_config(cfg, {"llm": {}, "systemPrompt": "be helpful", "mcpServers": {}})
    monkeypatch.setattr(config_module, "CONFIG_FILE", str(cfg))

    app_config = config_module.AppConfig.load()

    assert app_config.system_prompt == "be helpful"
    assert app_config.mcp_servers == {}
