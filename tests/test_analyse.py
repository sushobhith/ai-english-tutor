import unittest
from unittest.mock import Mock, patch

import analyse


class AnalyseTests(unittest.TestCase):
    def test_get_completion_uses_selected_provider(self):
        provider = Mock()
        provider.complete.return_value = '{"Grammar and Syntax": {"marks": 8}}'

        with patch("analyse.create_llm_provider", return_value=provider) as create_provider:
            result = analyse.get_completion("I has a apple.", model="test-model", provider="langchain")

        create_provider.assert_called_once_with(model="test-model", provider="langchain")
        provider.complete.assert_called_once()
        self.assertIn("proof read this text", provider.complete.call_args.args[0])
        self.assertEqual(result, '{"Grammar and Syntax": {"marks": 8}}')

    def test_prompt_contains_expected_rubric(self):
        prompt = analyse.get_prompt("Hello world")

        self.assertIn("Grammar and Syntax", prompt)
        self.assertIn("Vocabulary and Language Use", prompt)
        self.assertIn("Comprehension and Responsiveness", prompt)
        self.assertIn("Content and Structure", prompt)
        self.assertIn("Creativity and Originality", prompt)


if __name__ == "__main__":
    unittest.main()
