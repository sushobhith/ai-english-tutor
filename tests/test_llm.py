import os
import unittest
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

import analyse
import llm


MOCKED_JSON = '{"Grammar and Syntax": {"marks": 8, "suggestion": "Use clearer tense."}}'


def fake_model(response=MOCKED_JSON):
    return RunnableLambda(lambda _: response)


class LangChainIntegrationTests(unittest.TestCase):
    def test_build_chain_returns_runnable(self):
        chain = llm.build_chain(llm=fake_model())

        self.assertTrue(hasattr(chain, "invoke"))
        self.assertEqual(chain.invoke({"input_text": "I went to the market yesterday"}), MOCKED_JSON)

    def test_get_completion_langchain(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "langchain"}, clear=False):
            with patch("llm.ChatOpenAI", return_value=fake_model()) as chat_openai:
                result = llm.get_completion("I went to the market yesterday")

        self.assertEqual(result, MOCKED_JSON)
        chat_openai.assert_called_once()

    def test_provider_env_var_langchain(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "langchain"}, clear=False):
            with patch("llm.ChatOpenAI", return_value=fake_model()) as chat_openai:
                with patch("llm.HuggingFaceHub") as huggingface_hub:
                    result = llm.get_completion("test")

        self.assertEqual(result, MOCKED_JSON)
        chat_openai.assert_called_once()
        huggingface_hub.assert_not_called()

    def test_provider_env_var_huggingface(self):
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "huggingface",
                "HUGGINGFACE_API_KEY": "test-token",
            },
            clear=False,
        ):
            with patch("llm.HuggingFaceHub", return_value=fake_model()) as huggingface_hub:
                result = llm.get_completion("test")

        self.assertEqual(result, MOCKED_JSON)
        huggingface_hub.assert_called_once()

    def test_unknown_provider_defaults_to_langchain(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown"}, clear=False):
            with patch("llm.ChatOpenAI", return_value=fake_model()) as chat_openai:
                result = llm.get_completion("test")

        self.assertEqual(result, MOCKED_JSON)
        chat_openai.assert_called_once()

    def test_analyse_delegates_to_llm(self):
        with patch("analyse.llm.get_completion", return_value=MOCKED_JSON) as get_completion:
            result = analyse.get_completion("test")

        self.assertEqual(result, MOCKED_JSON)
        get_completion.assert_called_once_with("test", model=analyse.repo_id)


if __name__ == "__main__":
    unittest.main()
