import analyse
import llm


class DummyChain:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.response


def test_build_chain_returns_invokable_runnable():
    from langchain_core.runnables import RunnableLambda

    fake_model = RunnableLambda(lambda _prompt: '{"grammar": {"score": 10}}')
    chain = llm.build_chain(llm=fake_model)

    assert chain.invoke({"input_text": "I went to the market yesterday"})


def test_get_completion_langchain(monkeypatch):
    chain = DummyChain('{"grammar": {"score": 9}}')
    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    monkeypatch.setattr(llm, "build_chain", lambda **_kwargs: chain)

    result = llm.get_completion("I went to the market yesterday")

    assert result == '{"grammar": {"score": 9}}'
    assert chain.calls == [{"input_text": "I went to the market yesterday"}]


def test_provider_env_var_langchain(monkeypatch):
    chain = DummyChain('{"provider": "langchain"}')
    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    monkeypatch.setattr(llm, "build_chain", lambda **_kwargs: chain)
    monkeypatch.setattr(
        llm,
        "build_huggingface_chain",
        lambda: (_ for _ in ()).throw(AssertionError("wrong provider")),
    )

    assert llm.get_completion("test") == '{"provider": "langchain"}'


def test_unknown_provider_defaults_to_langchain(monkeypatch):
    chain = DummyChain('{"provider": "langchain"}')
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.setattr(llm, "build_chain", lambda **_kwargs: chain)

    assert llm.get_completion("test") == '{"provider": "langchain"}'


def test_provider_env_var_huggingface(monkeypatch):
    chain = DummyChain('{"provider": "huggingface"}')
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setattr(llm, "build_huggingface_chain", lambda: chain)
    monkeypatch.setattr(
        llm,
        "build_chain",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("wrong provider")),
    )

    assert llm.get_completion("test") == '{"provider": "huggingface"}'


def test_analyse_delegates_to_llm(monkeypatch):
    calls = []

    def fake_get_completion(text, **_kwargs):
        calls.append(text)
        return '{"ok": true}'

    monkeypatch.setattr(analyse.llm, "get_completion", fake_get_completion)

    assert analyse.get_completion("test") == '{"ok": true}'
    assert calls == ["test"]
