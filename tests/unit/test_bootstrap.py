from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import minicoder.bootstrap as bootstrap_module
from minicoder.adapters.openai_compatible_chat import OpenAICompatibleChatAdapter
from minicoder.bootstrap import ApplicationFactory
from minicoder.config import AppConfig
from minicoder.platforms import OperatingSystem


def test_factory_creates_validated_bootstrap_context(tmp_path: Path) -> None:
    context = ApplicationFactory.create_bootstrap_context(
        environ={
            "MINICODER_API_KEY": "key",
            "MINICODER_BASE_URL": "https://api.deepseek.com",
            "MINICODER_MODEL": "deepseek-v4-pro",
        },
        workspace=tmp_path,
        platform_name="win32",
    )

    assert context.config.workspace == tmp_path.resolve()
    assert context.operating_system is OperatingSystem.WINDOWS


def test_factory_configures_sdk_client_without_hidden_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = AppConfig.from_environment(
        {
            "MINICODER_API_KEY": "secret-key",
            "MINICODER_BASE_URL": "https://models.example.com/v1",
            "MINICODER_MODEL": "coding-model",
            "MINICODER_MODEL_TIMEOUT_SECONDS": "12.5",
        },
        workspace=tmp_path,
    )
    captured: dict[str, Any] = {}
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=object()))

    def fake_openai(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr(bootstrap_module, "OpenAI", fake_openai)

    adapter = ApplicationFactory.create_model_adapter(config)

    assert isinstance(adapter, OpenAICompatibleChatAdapter)
    assert captured == {
        "api_key": "secret-key",
        "base_url": "https://models.example.com/v1",
        "timeout": 12.5,
        "max_retries": 0,
    }
