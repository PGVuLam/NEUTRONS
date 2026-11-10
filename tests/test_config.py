from __future__ import annotations

import pytest

from neutrons.config import Settings


def test_settings_load_environment_values() -> None:
    configuration = Settings.from_env(
        {
            "NEUTRONS_MODEL": "qwen3:4b",
            "NEUTRONS_REQUEST_TIMEOUT": "45",
            "NEUTRONS_TEMPERATURE": "0.4",
            "NEUTRONS_NUM_PREDICT": "256",
            "NEUTRONS_API_PORT": "8010",
        }
    )

    assert configuration.model_name == "qwen3:4b"
    assert configuration.request_timeout_seconds == 45
    assert configuration.temperature == 0.4
    assert configuration.num_predict == 256
    assert configuration.api_port == 8010


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("NEUTRONS_REQUEST_TIMEOUT", "NaN", "positive finite"),
        ("NEUTRONS_TEMPERATURE", "3", "between 0 and 2"),
        ("NEUTRONS_NUM_PREDICT", "0", "at least 1"),
        ("NEUTRONS_API_PORT", "70000", "between 1 and 65535"),
        ("OLLAMA_BASE_URL", "file:///local/model", r"HTTP\(S\) URL"),
        ("OLLAMA_BASE_URL", "http://user@localhost:11434", "without credentials"),
        ("OLLAMA_BASE_URL", "http://localhost:99999", "valid port number"),
    ],
)
def test_settings_reject_invalid_values(name: str, value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Settings.from_env({name: value})


def test_settings_reports_non_numeric_environment_value() -> None:
    with pytest.raises(ValueError, match="NEUTRONS_NUM_CTX must be an integer"):
        Settings.from_env({"NEUTRONS_NUM_CTX": "large"})


def test_zero_history_character_budget_is_supported() -> None:
    assert Settings.from_env({"NEUTRONS_MAX_HISTORY_CHARS": "0"}).max_history_chars == 0


def test_readiness_timeout_is_bounded() -> None:
    with pytest.raises(ValueError, match="at most 30 seconds"):
        Settings.from_env({"NEUTRONS_READINESS_TIMEOUT": "31"})
