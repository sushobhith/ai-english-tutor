import unittest
from types import SimpleNamespace

from llm import LangChainChatProvider, OpenAIChatProvider, create_llm_provider


class FakeLangChainResponse:
    content = "langchain response"


class FakeLangChainModel:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return FakeLangChainResponse()


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai response"))]
        )


class FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


class LLMTests(unittest.TestCase):
    def test_langchain_provider_invokes_chat_model(self):
        chat_model = FakeLangChainModel()
        provider = LangChainChatProvider(model="test-model", chat_model=chat_model)

        result = provider.complete("hello")

        self.assertEqual(result, "langchain response")
        self.assertEqual(chat_model.messages, [{"role": "user", "content": "hello"}])

    def test_openai_provider_keeps_legacy_api_shape(self):
        client = FakeClient()
        provider = OpenAIChatProvider(model="test-model", client=client)

        result = provider.complete("hello")

        self.assertEqual(result, "openai response")
        self.assertEqual(client.chat.completions.kwargs["model"], "test-model")
        self.assertEqual(client.chat.completions.kwargs["messages"], [{"role": "user", "content": "hello"}])

    def test_factory_defaults_to_langchain(self):
        provider = create_llm_provider(model="test-model")

        self.assertIsInstance(provider, LangChainChatProvider)

    def test_factory_can_select_openai_provider(self):
        provider = create_llm_provider(model="test-model", provider="openai")

        self.assertIsInstance(provider, OpenAIChatProvider)


if __name__ == "__main__":
    unittest.main()
