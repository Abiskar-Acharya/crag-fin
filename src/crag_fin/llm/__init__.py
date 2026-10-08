"""LLM client abstraction.

Exposes a single provider-agnostic entry point, :func:`chat_json`, so the
decomposer, reasoner, and auditor do not care whether the model is served
locally by Ollama or by a hosted OpenAI-compatible API (DeepSeek, Xiaomi
MiMo, or any other OpenAI-shaped endpoint).
"""

from crag_fin.llm.client import chat_json, resolve_model

__all__ = ["chat_json", "resolve_model"]
