"""Provider-agnostic JSON chat client.

One function, :func:`chat_json`, sends a system + user message and returns the
model's raw text response. The provider is chosen by the ``LLM_PROVIDER``
environment variable:

    LLM_PROVIDER=ollama    # default; local, no API key
    LLM_PROVIDER=deepseek  # hosted, OpenAI-compatible (needs DEEPSEEK_API_KEY)
    LLM_PROVIDER=mimo      # hosted, OpenAI-compatible (needs MIMO_API_KEY)

Callers keep parsing the returned string with their own ``_parse`` helpers, so
switching providers changes only which service produced the JSON, never how it
is handled downstream. All providers are asked for JSON-only output.

Model selection is role-based. :func:`resolve_model` maps a role
("decomposer" / "reasoner" / "auditor") to the right model name for the active
provider, honouring per-provider environment overrides and falling back to a
sane default. This keeps the call sites free of provider-specific names.
"""

from __future__ import annotations

import os
from typing import Literal

Role = Literal["decomposer", "reasoner", "auditor"]

# Default model per (provider, role). Env vars override any of these.
# DeepSeek's legacy `deepseek-chat` alias was deprecated on 2026-07-24, so the
# default targets the current `deepseek-v4-flash`. Override via env if needed.
_DEFAULT_MODELS: dict[str, dict[str, str]] = {
    "ollama": {
        "decomposer": "qwen2.5:7b",
        "reasoner": "glm-4.7-flash:latest",
        "auditor": "hermes3:8b",
    },
    "deepseek": {
        "decomposer": "deepseek-v4-flash",
        "reasoner": "deepseek-v4-flash",
        "auditor": "deepseek-v4-flash",
    },
    "mimo": {
        "decomposer": "mimo-v2.5-pro",
        "reasoner": "mimo-v2.5-pro",
        "auditor": "mimo-v2.5-pro",
    },
}

# Env var that overrides the default model, per (provider, role).
_MODEL_ENV: dict[str, dict[str, str]] = {
    "ollama": {
        "decomposer": "OLLAMA_DECOMPOSER_MODEL",
        "reasoner": "OLLAMA_REASONER_MODEL",
        "auditor": "OLLAMA_AUDITOR_MODEL",
    },
    "deepseek": {
        "decomposer": "DEEPSEEK_DECOMPOSER_MODEL",
        "reasoner": "DEEPSEEK_REASONER_MODEL",
        "auditor": "DEEPSEEK_AUDITOR_MODEL",
    },
    "mimo": {
        "decomposer": "MIMO_DECOMPOSER_MODEL",
        "reasoner": "MIMO_REASONER_MODEL",
        "auditor": "MIMO_AUDITOR_MODEL",
    },
}

# OpenAI-compatible hosted providers: (base-url env var, default base url,
# api-key env var).
_OPENAI_COMPAT: dict[str, tuple[str, str, str]] = {
    "deepseek": ("DEEPSEEK_BASE_URL", "https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "mimo": ("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1", "MIMO_API_KEY"),
}


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "ollama").strip().lower() or "ollama"


def resolve_model(role: Role) -> str:
    """Return the model name for ``role`` under the active provider.

    Precedence: the provider+role env override, else the built-in default.
    """
    provider = _provider()
    env_var = _MODEL_ENV.get(provider, {}).get(role)
    if env_var:
        override = os.environ.get(env_var)
        if override and override.strip():
            return override.strip()
    defaults = _DEFAULT_MODELS.get(provider) or _DEFAULT_MODELS["ollama"]
    return defaults[role]


def chat_json(
    system: str,
    user: str,
    model: str,
    temperature: float = 0.0,
) -> str:
    """Send a system+user prompt and return the raw response text (JSON-only).

    Dispatches on ``LLM_PROVIDER``. Ollama is the default and needs no key.
    DeepSeek and MiMo are OpenAI-compatible and require their respective keys.

    Raises:
        RuntimeError: if a hosted provider is selected but its API key is unset.
        ValueError: if ``LLM_PROVIDER`` names an unknown provider.
    """
    provider = _provider()

    if provider == "ollama":
        return _chat_ollama(system, user, model, temperature)

    if provider in _OPENAI_COMPAT:
        base_env, base_default, key_env = _OPENAI_COMPAT[provider]
        api_key = os.environ.get(key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"LLM_PROVIDER={provider} but {key_env} is not set. "
                f"Add your key to .env (or export it) before running."
            )
        base_url = os.environ.get(base_env, "").strip() or base_default
        return _chat_openai_compatible(system, user, model, temperature, base_url, api_key)

    raise ValueError(
        f"Unknown LLM_PROVIDER={provider!r}. Use one of: ollama, deepseek, mimo."
    )


def _chat_ollama(system: str, user: str, model: str, temperature: float) -> str:
    import ollama

    response = ollama.chat(
        model=model,
        format="json",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"]


def _chat_openai_compatible(
    system: str,
    user: str,
    model: str,
    temperature: float,
    base_url: str,
    api_key: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return content or ""
