import sys
import types

import analyse
import llm


class FakePromptTemplate:
    def __init__(self, template):
        self.template = template

    @classmethod
    def from_template(cls, template):
        return cls(template)

    def __or__(self, fake_llm):
        return FakeChain(self, fake_llm)


class FakeChain:
    def __init__(self, prompt, fake_llm):
        self.prompt = prompt
        self.fake_llm = fake_llm

    def invoke(self, values):
        return self.fake_llm.invoke(self.prompt.template.format(**values))


class FakeLLM:
    def __init__(self, content='{"score": 10}'):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return types.SimpleNamespace(content=self.content)


def install_fake_langchain(monkeypatch):
    langchain = types.ModuleType("langchain")
    prompts = types.ModuleType("langchain.prompts")
    prompts.PromptTemplate = FakePromptTemplate
    monkeypatch.setitem(sys.modules, "langchain", langchain)
    monkeypatch.setitem(sys.modules, "langchain.prompts", prompts)


def test_build_chain_returns_callable_without_network(monkeypatch):
    install_fake_langchain(monkeypatch)
    fake_llm = FakeLLM()

    chain = llm.build_chain(llm=fake_llm)
    response = chain.invoke({"input_text": "I goed home"})

    assert response.content == '{"score": 10}'
    assert "I goed home" in fake_llm.prompts[0]


def test_get_completion_uses_langchain_provider(monkeypatch):
    fake_chain = types.SimpleNamespace(
        invoke=lambda values: types.SimpleNamespace(content='{"provider": "langchain"}')
    )
    monkeypatch.setenv("LLM_PROVIDER", "langchain")
    monkeypatch.setattr(llm, "build_chain", lambda: fake_chain)

    assert llm.get_completion("hello") == '{"provider": "langchain"}'


def test_unknown_provider_defaults_to_langchain(monkeypatch):
    fake_chain = types.SimpleNamespace(
        invoke=lambda values: types.SimpleNamespace(content='{"provider": "langchain"}')
    )
    monkeypatch.setenv("LLM_PROVIDER", "unknown")
    monkeypatch.setattr(llm, "build_chain", lambda: fake_chain)

    assert llm.get_completion("hello") == '{"provider": "langchain"}'


def test_provider_env_var_huggingface(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "huggingface")
    monkeypatch.setattr(llm, "_get_legacy_completion", lambda text: '{"provider": "huggingface"}')

    assert llm.get_completion("hello") == '{"provider": "huggingface"}'


def test_analyse_delegates_to_llm(monkeypatch):
    calls = []

    def fake_get_completion(text):
        calls.append(text)
        return '{"ok": true}'

    monkeypatch.setattr(analyse.llm, "get_completion", fake_get_completion)

    assert analyse.get_completion("test") == '{"ok": true}'
    assert calls == ["test"]

