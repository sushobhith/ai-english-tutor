import importlib
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAKE_JSON = json.dumps(
    {
        "Grammar and Syntax": {"marks": 8, "suggestion": "Good overall."},
        "Vocabulary and Language Use": {"marks": 7, "suggestion": "Use more precise words."},
        "Comprehension and Responsiveness": {"marks": 8, "suggestion": "Answer directly."},
        "Content and Structure": {"marks": 7, "suggestion": "Add a clearer order."},
        "Creativity and Originality": {"marks": 6, "suggestion": "Use more original examples."},
    }
)


class FakeRunnable:
    def __init__(self, response):
        self.response = response

    def invoke(self, _payload):
        return self.response


class FakePromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        return self.template.format(**kwargs)

    def __or__(self, llm):
        return llm


class FakeMessage:
    content = FAKE_JSON


class LlmTests(unittest.TestCase):
    def setUp(self):
        self._old_modules = dict(sys.modules)
        self._old_env = os.environ.copy()
        self._install_fake_langchain()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._old_env)
        sys.modules.clear()
        sys.modules.update(self._old_modules)

    def _install_fake_langchain(self):
        dotenv_module = types.ModuleType("dotenv")
        dotenv_module.find_dotenv = lambda: ""
        dotenv_module.load_dotenv = lambda _path="": False

        langchain_module = types.ModuleType("langchain")
        prompts_module = types.ModuleType("langchain.prompts")
        prompts_module.PromptTemplate = FakePromptTemplate

        community_module = types.ModuleType("langchain_community")
        community_llms_module = types.ModuleType("langchain_community.llms")
        community_llms_module.HuggingFaceHub = MagicMock(return_value=FakeRunnable(FAKE_JSON))
        community_module.llms = community_llms_module

        openai_module = types.ModuleType("langchain_openai")
        openai_module.ChatOpenAI = MagicMock(return_value=FakeRunnable(FakeMessage()))

        sys.modules["dotenv"] = dotenv_module
        sys.modules["langchain"] = langchain_module
        sys.modules["langchain.prompts"] = prompts_module
        sys.modules["langchain_community"] = community_module
        sys.modules["langchain_community.llms"] = community_llms_module
        sys.modules["langchain_openai"] = openai_module

    def _reload_llm(self):
        sys.modules.pop("llm", None)
        import llm

        return importlib.reload(llm)

    def test_build_chain_returns_runnable(self):
        llm = self._reload_llm()
        fake_llm = FakeRunnable(FAKE_JSON)

        chain = llm.build_chain(llm=fake_llm)

        self.assertTrue(callable(chain.invoke))
        self.assertEqual(chain.invoke({"input_text": "test"}), FAKE_JSON)

    def test_get_completion_langchain_returns_json_string(self):
        os.environ["LLM_PROVIDER"] = "langchain"
        os.environ["OPENAI_API_KEY"] = "sk-test"
        llm = self._reload_llm()

        result = llm.get_completion("I went to the market yesterday")

        self.assertEqual(result, FAKE_JSON)
        self.assertIn("Grammar and Syntax", json.loads(result))

    def test_provider_env_var_langchain_uses_chat_openai(self):
        os.environ["LLM_PROVIDER"] = "langchain"
        llm = self._reload_llm()

        result = llm._build_llm()

        import langchain_openai

        langchain_openai.ChatOpenAI.assert_called_once()
        self.assertIsInstance(result, FakeRunnable)

    def test_unknown_provider_defaults_to_langchain(self):
        os.environ["LLM_PROVIDER"] = "not-real"
        llm = self._reload_llm()

        self.assertEqual(llm.get_provider(), "langchain")

    def test_provider_env_var_huggingface_uses_huggingfacehub(self):
        os.environ["LLM_PROVIDER"] = "huggingface"
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf-test"
        llm = self._reload_llm()

        result = llm._build_llm()

        import langchain_community.llms

        langchain_community.llms.HuggingFaceHub.assert_called_once()
        self.assertIsInstance(result, FakeRunnable)

    def test_analyse_delegates_to_llm(self):
        fake_llm = types.ModuleType("llm")
        fake_llm.get_completion = MagicMock(return_value=FAKE_JSON)
        fake_llm.get_prompt = MagicMock(return_value="prompt")
        sys.modules["llm"] = fake_llm
        sys.modules.pop("analyse", None)

        import analyse

        result = analyse.get_completion("test")

        fake_llm.get_completion.assert_called_once_with("test", model="gpt-3.5-turbo")
        self.assertEqual(result, FAKE_JSON)


if __name__ == "__main__":
    unittest.main()
