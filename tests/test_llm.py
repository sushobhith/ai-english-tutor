import importlib
import json
from unittest import TestCase
from unittest.mock import Mock, patch

from langchain_core.runnables import RunnableLambda

import analyse
import llm


FAKE_ANALYSIS = json.dumps({
    'Grammar and Syntax': {'marks': 8, 'suggestion': 'Good overall.'},
    'Vocabulary and Language Use': {'marks': 7, 'suggestion': 'Use more varied words.'},
})


class LlmTest(TestCase):
    def test_build_chain_returns_runnable(self):
        fake_llm = RunnableLambda(lambda prompt: FAKE_ANALYSIS)

        chain = llm.build_chain(llm_instance=fake_llm)

        self.assertTrue(callable(chain.invoke))
        self.assertEqual(chain.invoke({'input_text': 'I went to the market'}), FAKE_ANALYSIS)

    def test_get_completion_langchain(self):
        message = Mock(content=FAKE_ANALYSIS)
        chain = Mock()
        chain.invoke.return_value = message

        with patch.object(llm, 'build_chain', return_value=chain):
            self.assertEqual(llm.get_completion('hello'), FAKE_ANALYSIS)

        chain.invoke.assert_called_once_with({'input_text': 'hello'})

    def test_provider_env_var_langchain(self):
        with patch.dict('os.environ', {'LLM_PROVIDER': 'langchain', 'OPENAI_API_KEY': 'sk-test'}):
            importlib.reload(llm)
            with patch('langchain_openai.ChatOpenAI') as chat_openai:
                chat_openai.return_value = 'openai-llm'

                self.assertEqual(llm.build_llm(), 'openai-llm')

        chat_openai.assert_called_once()

    def test_provider_env_var_huggingface(self):
        env = {'LLM_PROVIDER': 'huggingface', 'HUGGINGFACEHUB_API_TOKEN': 'hf-test'}
        with patch.dict('os.environ', env):
            importlib.reload(llm)
            with patch('langchain_community.llms.HuggingFaceHub') as huggingface_hub:
                huggingface_hub.return_value = 'huggingface-llm'

                self.assertEqual(llm.build_llm(), 'huggingface-llm')

        huggingface_hub.assert_called_once()

    def test_analyse_delegates_to_llm(self):
        with patch.object(llm, 'get_completion', return_value=FAKE_ANALYSIS) as get_completion:
            self.assertEqual(analyse.get_completion('test'), FAKE_ANALYSIS)

        get_completion.assert_called_once_with('test', model='gpt-3.5-turbo')
