import importlib
import json
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage


FAKE_JSON = json.dumps(
    {
        "Grammar and Syntax": {"marks": 8, "suggestion": "Good overall."},
        "Vocabulary and Language Use": {"marks": 7, "suggestion": "Use more precise words."},
        "Comprehension and Responsiveness": {"marks": 9, "suggestion": "Keep this clarity."},
        "Content and Structure": {"marks": 8, "suggestion": "Add a clearer conclusion."},
        "Creativity and Originality": {"marks": 6, "suggestion": "Use more original examples."},
    }
)


class LangChainIntegrationTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("HUGGINGFACEHUB_API_TOKEN", None)

    def reload_llm(self):
        sys.modules.pop("llm", None)
        import llm

        return importlib.reload(llm)

    def test_build_chain_returns_runnable_without_network_calls(self):
        llm = self.reload_llm()

        chain = llm.build_chain(llm=lambda _: AIMessage(content=FAKE_JSON))

        self.assertTrue(callable(chain.invoke))
        self.assertEqual(chain.invoke({"input_text": "I spoke clearly."}), FAKE_JSON)

    def test_get_completion_returns_json_string_from_chain(self):
        llm = self.reload_llm()
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = FAKE_JSON

        with patch.object(llm, "build_chain", return_value=fake_chain):
            result = llm.get_completion("I went to the market yesterday")

        fake_chain.invoke.assert_called_once_with({"input_text": "I went to the market yesterday"})
        self.assertIn("Grammar and Syntax", json.loads(result))

    def test_langchain_provider_uses_chat_openai(self):
        os.environ["LLM_PROVIDER"] = "langchain"
        os.environ["OPENAI_API_KEY"] = "sk-test"
        llm = self.reload_llm()

        with patch("langchain_openai.ChatOpenAI") as chat_openai:
            expected_model = MagicMock()
            chat_openai.return_value = expected_model

            actual_model = llm.make_llm()

        chat_openai.assert_called_once()
        self.assertIs(actual_model, expected_model)

    def test_huggingface_provider_uses_huggingface_hub(self):
        os.environ["LLM_PROVIDER"] = "huggingface"
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf-test"
        llm = self.reload_llm()

        with patch("langchain_community.llms.HuggingFaceHub") as huggingface_hub:
            expected_model = MagicMock()
            huggingface_hub.return_value = expected_model

            actual_model = llm.make_llm()

        huggingface_hub.assert_called_once()
        self.assertIs(actual_model, expected_model)

    def test_unknown_provider_defaults_to_langchain(self):
        os.environ["LLM_PROVIDER"] = "not-real"
        llm = self.reload_llm()

        self.assertEqual(llm.get_provider(), "langchain")

    def test_analyse_delegates_to_llm(self):
        import analyse

        with patch.object(analyse.llm, "get_completion", return_value=FAKE_JSON) as get_completion:
            result = analyse.get_completion("test")

        get_completion.assert_called_once_with("test", model=None)
        self.assertEqual(result, FAKE_JSON)

    def test_analyse_keeps_prompt_helper_available(self):
        import analyse

        prompt = analyse.get_prompt("sample")

        self.assertIn("sample", prompt)
        self.assertIn("JSON structure", prompt)


if __name__ == "__main__":
    unittest.main()
