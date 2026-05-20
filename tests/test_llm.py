"""Unit tests for llm.py and the analyse.py delegation layer.

All tests run fully offline – no network calls are made.
"""

from __future__ import annotations

import importlib
import json
import os
import types
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_JSON = json.dumps({
    "Grammar and Syntax": {"marks": 8, "suggestion": "Good overall."},
    "Vocabulary and Language Use": {"marks": 7, "suggestion": "Expand vocabulary."},
    "Comprehension and Responsiveness": {"marks": 9, "suggestion": "Very responsive."},
    "Content and Structure": {"marks": 8, "suggestion": "Well structured."},
    "Creativity and Originality": {"marks": 6, "suggestion": "Be more creative."},
})


def _make_ai_message(content: str):
    """Return a minimal AIMessage-like object with a .content attribute."""
    msg = MagicMock()
    msg.content = content
    return msg


# ---------------------------------------------------------------------------
# test_build_chain_returns_runnable
# ---------------------------------------------------------------------------

def test_build_chain_returns_runnable():
    """build_chain() with a mock LLM should return something callable."""
    import llm

    mock_llm = MagicMock()
    # The pipe operator on a PromptTemplate produces a RunnableSequence.
    chain = llm.build_chain(llm=mock_llm)
    assert callable(chain.invoke), "chain must expose an .invoke() method"


# ---------------------------------------------------------------------------
# test_get_completion_langchain
# ---------------------------------------------------------------------------

def test_get_completion_langchain(monkeypatch):
    """get_completion() should return the model's content string."""
    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    import llm
    # Reload so monkeypatched env vars are picked up by _make_llm.
    importlib.reload(llm)

    mock_llm = MagicMock()
    mock_llm.__or__ = lambda self, other: MagicMock(
        invoke=MagicMock(return_value=_make_ai_message(FAKE_JSON))
    )

    with patch.object(llm, "build_chain", return_value=MagicMock(
        invoke=MagicMock(return_value=_make_ai_message(FAKE_JSON))
    )):
        result = llm.get_completion("I went to the market yesterday")

    assert result == FAKE_JSON
    # Must be JSON-parseable.
    parsed = json.loads(result)
    assert "Grammar and Syntax" in parsed


# ---------------------------------------------------------------------------
# test_provider_env_var_langchain
# ---------------------------------------------------------------------------

def test_provider_env_var_langchain(monkeypatch):
    """When LLM_PROVIDER=langchain, ChatOpenAI should be used."""
    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    import llm
    importlib.reload(llm)

    with patch("langchain_openai.ChatOpenAI") as mock_chat_openai:
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance

        built_llm = llm._make_llm()

    mock_chat_openai.assert_called_once()
    assert built_llm is mock_instance


# ---------------------------------------------------------------------------
# test_provider_env_var_huggingface
# ---------------------------------------------------------------------------

def test_provider_env_var_huggingface(monkeypatch):
    """When LLM_PROVIDER=huggingface, HuggingFaceHub should be used."""
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setenv("HUGGINGFACEHUB_API_TOKEN", "hf-test")

    import llm
    importlib.reload(llm)

    with patch("langchain_community.llms.HuggingFaceHub") as mock_hf:
        mock_instance = MagicMock()
        mock_hf.return_value = mock_instance

        built_llm = llm._make_llm()

    mock_hf.assert_called_once()
    assert built_llm is mock_instance


# ---------------------------------------------------------------------------
# test_analyse_delegates_to_llm
# ---------------------------------------------------------------------------

def test_analyse_delegates_to_llm(monkeypatch):
    """analyse.get_completion should call llm.get_completion exactly once."""
    import llm
    import analyse as an

    mock_get_completion = MagicMock(return_value=FAKE_JSON)
    monkeypatch.setattr(llm, "get_completion", mock_get_completion)

    # Reload analyse so it picks up the patched llm module.
    importlib.reload(an)
    # Re-apply the patch after reload because reload re-binds the reference.
    monkeypatch.setattr(an._llm, "get_completion", mock_get_completion)

    result = an.get_completion("test")

    mock_get_completion.assert_called_once_with("test")
    assert result == FAKE_JSON
