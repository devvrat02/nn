import os
import unittest
from unittest.mock import Mock, patch

from utils.generation_recovery import generate_structured, parse_choice, StructuredGenerationError
from utils.local_llm import LocalGenerationLimitError
from method.hidden_info_mining import IntentArgumentation


class RecoveryTests(unittest.TestCase):
    def test_single_literal_print_choice_is_parsed_and_audited(self):
        text = '```python\n# This is a sample solution.\nprint("A")\n```'
        audit = []
        self.assertEqual(parse_choice(text, "ABC", audit, "counterfactual"), "A")
        self.assertEqual(audit[0]["policy"], "literal-print-choice-v1")
        self.assertEqual(audit[0]["stage"], "counterfactual")

    def test_choice_parser_rejects_ambiguity_and_executable_expressions(self):
        for text in ('', 'AB', 'ABC', 'A or B',
                     '```python\nprint("A")\nprint("B")\n```',
                     '```python\nprint(open("secret").read())\n```',
                     '```python\nimport os\nprint("A")\n```',
                     '```python\nprint("D")\n```'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_choice(text, "ABC")

    def test_limit_recovery_changes_prompt_and_is_audited(self):
        generate = Mock(side_effect=[LocalGenerationLimitError("limit"), "<first||second>"])
        assessor = IntentArgumentation.__new__(IntentArgumentation)
        audit = []
        with patch.dict(os.environ, {"TRACER_BACKEND": "local"}):
            result = generate_structured(generate, "claim and evidence", assessor._extract_answer,
                                         "assumptions", "Be concise", audit)
        self.assertEqual(result, ["first", "second"])
        self.assertEqual(audit[0]["stage"], "assumptions")
        self.assertNotEqual(generate.call_args_list[0], generate.call_args_list[1])
        self.assertTrue(generate.call_args_list[1].args[0].startswith("claim and evidence"))

    def test_success_does_not_change_prompt_or_add_audit(self):
        generate = Mock(return_value="C")
        audit = []
        self.assertEqual(generate_structured(generate, "prompt", lambda s: parse_choice(s, "ABC"), "choice", "short", audit), "C")
        generate.assert_called_once_with("prompt")
        self.assertEqual(audit, [])

    def test_bad_format_gets_one_recovery(self):
        generate = Mock(side_effect=["long explanation", "B"])
        with patch.dict(os.environ, {"TRACER_BACKEND": "local"}):
            self.assertEqual(generate_structured(generate, "prompt", lambda s: parse_choice(s, "ABC"), "choice", "letter"), "B")

    def test_double_failure_is_not_a_prediction(self):
        generate = Mock(side_effect=LocalGenerationLimitError("limit"))
        with patch.dict(os.environ, {"TRACER_BACKEND": "local"}), self.assertRaises(StructuredGenerationError):
            generate_structured(generate, "prompt", str, "stage", "short")
        self.assertEqual(generate.call_count, 2)

    def test_transport_error_is_not_misidentified_as_format_error(self):
        generate = Mock(side_effect=RuntimeError("out of memory"))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            generate_structured(generate, "prompt", str, "stage", "short")
        generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
