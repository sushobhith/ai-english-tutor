import importlib
import sys
import types
import unittest
from unittest import mock


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChain:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        return self.response


class FakePromptTemplate:
    template = None

    @classmethod
    def from_template(cls, template):
        cls.template = template
        return cls()

    def __or__(self, llm):
        return FakeChain(llm.response)


class FakeChatOpenAI:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
        self.response = FakeMessage('{"ok": true}')


class FakeHuggingFaceHub:
    def __init__(self, repo_id, model_kwargs):
        self.repo_id = repo_id
        self.model_kwargs = model_kwargs
        self.response = '{"ok": "hf"}'


def install_fake_langchain_modules():
    langchain = types.ModuleType("langchain")
    prompts = types.ModuleType("langchain.prompts")
    prompts.PromptTemplate = FakePromptTemplate

    langchain_openai = types.ModuleType("langchain_openai")
    langchain_openai.ChatOpenAI = FakeChatOpenAI

    langchain_community = types.ModuleType("langchain_community")
    llms = types.ModuleType("langchain_community.llms")
    llms.HuggingFaceHub = FakeHuggingFaceHub

    modules = {
        "langchain": langchain,
        "langchain.prompts": prompts,
        "langchain_openai": langchain_openai,
        "langchain_community": langchain_community,
        "langchain_community.llms": llms,
    }

    patcher = mock.patch.dict(sys.modules, modules)
    patcher.start()
    return patcher


class LlmIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.langchain_modules = install_fake_langchain_modules()
        self.env = mock.patch.dict("os.environ", {}, clear=True)
        self.env.start()
        import llm

        self.llm = importlib.reload(llm)

    def tearDown(self):
        self.env.stop()
        self.langchain_modules.stop()

    def test_build_chain_returns_runnable(self):
        chain = self.llm.build_chain(llm=FakeChatOpenAI(model="test-model", temperature=0))

        result = chain.invoke({"input_text": "I went to the market yesterday"})

        self.assertEqual(result.content, '{"ok": true}')
        self.assertIn("{input_text}", FakePromptTemplate.template)

    def test_get_completion_langchain(self):
        fake_chain = FakeChain(FakeMessage('{"score": 10}'))

        with mock.patch.object(self.llm, "build_chain", return_value=fake_chain):
            result = self.llm.get_completion("I went to the market yesterday")

        self.assertEqual(result, '{"score": 10}')
        self.assertEqual(fake_chain.calls, [{"input_text": "I went to the market yesterday"}])

    def test_provider_env_var_langchain(self):
        with mock.patch.object(self.llm, "get_langchain_completion", return_value="langchain") as langchain_completion:
            with mock.patch.object(self.llm, "get_huggingface_completion", return_value="huggingface") as hf_completion:
                result = self.llm.get_completion("text")

        self.assertEqual(result, "langchain")
        langchain_completion.assert_called_once_with("text")
        hf_completion.assert_not_called()

    def test_provider_env_var_huggingface(self):
        with mock.patch.dict("os.environ", {"LLM_PROVIDER": "huggingface"}, clear=True):
            with mock.patch.object(self.llm, "get_langchain_completion", return_value="langchain") as langchain_completion:
                with mock.patch.object(self.llm, "get_huggingface_completion", return_value="huggingface") as hf_completion:
                    result = self.llm.get_completion("text")

        self.assertEqual(result, "huggingface")
        hf_completion.assert_called_once_with("text")
        langchain_completion.assert_not_called()

    def test_unknown_provider_defaults_to_langchain(self):
        with mock.patch.dict("os.environ", {"LLM_PROVIDER": "unknown"}, clear=True):
            with mock.patch.object(self.llm, "get_langchain_completion", return_value="langchain") as langchain_completion:
                result = self.llm.get_completion("text")

        self.assertEqual(result, "langchain")
        langchain_completion.assert_called_once_with("text")

    def test_analyse_delegates_to_llm(self):
        import analyse

        analyse = importlib.reload(analyse)
        with mock.patch.object(analyse.llm, "get_completion", return_value='{"delegated": true}') as get_completion:
            result = analyse.get_completion("test")

        self.assertEqual(result, '{"delegated": true}')
        get_completion.assert_called_once_with("test")


if __name__ == "__main__":
    unittest.main()
