import os
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import analyse
import llm


class ContentResult:
    def __init__(self, content):
        self.content = content


class LLMTests(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("LLM_PROVIDER", None)

    def test_build_chain_returns_runnable(self):
        chain = llm.build_chain(RunnableLambda(lambda prompt: "ok"))

        self.assertEqual(chain.invoke({"analysis_prompt": "hello"}), "ok")

    def test_get_completion_langchain(self):
        fake_model = RunnableLambda(lambda prompt: ContentResult('{"score": 10}'))

        with patch.object(llm, "_create_chat_openai", return_value=fake_model):
            self.assertEqual(llm.get_completion("I went to the market yesterday"), '{"score": 10}')

    def test_provider_env_var_langchain(self):
        os.environ["LLM_PROVIDER"] = "langchain"

        with patch.object(llm, "_get_completion_langchain", return_value="langchain") as langchain_mock:
            with patch.object(llm, "_get_completion_huggingface") as huggingface_mock:
                self.assertEqual(llm.get_completion("test"), "langchain")

        langchain_mock.assert_called_once_with("test")
        huggingface_mock.assert_not_called()

    def test_provider_env_var_huggingface(self):
        os.environ["LLM_PROVIDER"] = "huggingface"

        with patch.object(llm, "_get_completion_huggingface", return_value="huggingface") as huggingface_mock:
            with patch.object(llm, "_get_completion_langchain") as langchain_mock:
                self.assertEqual(llm.get_completion("test"), "huggingface")

        huggingface_mock.assert_called_once_with("test")
        langchain_mock.assert_not_called()

    def test_unknown_provider_defaults_to_langchain(self):
        os.environ["LLM_PROVIDER"] = "unknown"

        with patch.object(llm, "_get_completion_langchain", return_value="langchain") as langchain_mock:
            with patch.object(llm, "_get_completion_huggingface") as huggingface_mock:
                self.assertEqual(llm.get_completion("test"), "langchain")

        langchain_mock.assert_called_once_with("test")
        huggingface_mock.assert_not_called()

    def test_analyse_delegates_to_llm(self):
        with patch.object(llm, "get_completion", return_value="delegated") as get_completion_mock:
            self.assertEqual(analyse.get_completion("test"), "delegated")

        get_completion_mock.assert_called_once_with("test")


if __name__ == "__main__":
    unittest.main()
