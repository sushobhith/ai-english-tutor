import importlib
import sys

from langchain_core.runnables import RunnableLambda


def reload_modules():
    for module_name in ("analyse", "llm"):
        sys.modules.pop(module_name, None)

    llm = importlib.import_module("llm")
    analyse = importlib.import_module("analyse")
    return llm, analyse


def test_build_chain_returns_runnable():
    llm, _ = reload_modules()
    chain = llm.build_chain(RunnableLambda(lambda prompt: '{"ok": true}'))

    assert chain.invoke({"input_text": "I went to the market yesterday"}) == '{"ok": true}'


def test_get_completion_langchain(monkeypatch):
    llm, _ = reload_modules()
    expected = '{"Grammar and Syntax": {"marks": 8}}'

    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    monkeypatch.setattr(llm, "get_langchain_completion", lambda text: expected)

    assert llm.get_completion("I went to the market yesterday") == expected


def test_provider_env_var_langchain(monkeypatch):
    llm, _ = reload_modules()
    calls = []

    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    monkeypatch.setattr(llm, "get_langchain_completion", lambda text: calls.append(text) or "langchain")
    monkeypatch.setattr(llm, "get_huggingface_completion", lambda text: "huggingface")

    assert llm.get_completion("test") == "langchain"
    assert calls == ["test"]


def test_provider_env_var_huggingface(monkeypatch):
    llm, _ = reload_modules()
    calls = []

    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setattr(llm, "get_huggingface_completion", lambda text: calls.append(text) or "huggingface")
    monkeypatch.setattr(llm, "get_langchain_completion", lambda text: "langchain")

    assert llm.get_completion("test") == "huggingface"
    assert calls == ["test"]


def test_unknown_provider_defaults_to_langchain(monkeypatch):
    llm, _ = reload_modules()

    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.setattr(llm, "get_langchain_completion", lambda text: "langchain")
    monkeypatch.setattr(llm, "get_huggingface_completion", lambda text: "huggingface")

    assert llm.get_completion("test") == "langchain"


def test_analyse_delegates_to_llm(monkeypatch):
    llm, analyse = reload_modules()
    calls = []

    monkeypatch.setattr(llm, "get_completion", lambda text: calls.append(text) or "delegated")

    assert analyse.get_completion("test") == "delegated"
    assert calls == ["test"]
