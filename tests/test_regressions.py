import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from method import claim_verification_hiss as verification
from method import hidden_info_mining as mining
from method import reassessment
from method.eval import evaluate
from method.prompts import GPT4oMiniWrapper
from utils import utils


class RegressionTests(unittest.TestCase):
    def test_hiss_plain_verdict_and_evidence_url_are_audited(self):
        audit = {}
        response = "False\n\n### Evidence\nhttps://www.politifact.com/factchecks/example/"
        self.assertEqual(verification.extract_hiss_answer(response, audit), "false")
        self.assertEqual(audit["generation_recovery"], "explicit-hiss-verdict-v1")
        for bad in ("False or true", "False\n### Evidence\nActually true", "False\n<unknown>"):
            with self.assertRaises(ValueError):
                verification.extract_hiss_answer(bad)

    def test_local_hiss_format_retry_changes_prompt(self):
        with patch.dict("os.environ", {"TRACER_BACKEND": "local"}), \
             patch.object(verification, "call_gpt", side_effect=["no answer", "<false>"]) as call:
            self.assertEqual(verification.promptf("claim", verification.prompt, [], audit={})[1], "false")
        self.assertNotEqual(call.call_args_list[0].args[0], call.call_args_list[1].args[0])

    def test_output_limit_uses_marked_bounded_recovery(self):
        from utils.local_llm import LocalGenerationLimitError
        audit = {}
        with patch.object(verification, "call_gpt", side_effect=[LocalGenerationLimitError("limit"), "Verdict: <false>"]) as call:
            self.assertEqual(verification.promptf("claim", verification.prompt, [], audit=audit)[1], "false")
        self.assertNotEqual(call.call_args_list[0].args[0], call.call_args_list[1].args[0])
        self.assertEqual(audit["generation_recovery"], "bounded-hiss-v1")

    def test_failed_recovery_does_not_invent_a_label(self):
        from utils.local_llm import LocalGenerationLimitError
        with patch.object(verification, "call_gpt", side_effect=LocalGenerationLimitError("limit")) as call:
            with self.assertRaises(LocalGenerationLimitError):
                verification.promptf("claim", verification.prompt, [])
        self.assertEqual(call.call_count, 2)

    def test_verdict_with_trailing_empty_tag(self):
        response = "The claim can be classified as <false>.\n\n<>false"
        self.assertEqual(verification.extract_answer(response), "false")
        with patch.object(verification, "call_gpt", return_value=response) as call:
            self.assertEqual(verification.promptf("claim", verification.prompt, []), (response, "false"))
        call.assert_called_once()

    def test_verdict_ignores_multiple_blank_tags_and_normalizes_case(self):
        self.assertEqual(verification.extract_answer("Verdict: < Half-True >. <> <  >"), "half-true")

    def test_latest_nonempty_verdict_takes_precedence(self):
        self.assertEqual(verification.extract_answer("Initially <true>, finally <false>. <>"), "false")
        with self.assertRaisesRegex(ValueError, "Unsupported final verdict"):
            verification.extract_answer("Initially <true>, finally <unverifiable>. <>")

    def test_empty_or_absent_verdict_is_not_guessed(self):
        for response in ("<>false", "< >", "The evidence mentions true and false."):
            with self.subTest(response=response), self.assertRaisesRegex(ValueError, "No nonempty"):
                verification.extract_answer(response)

    def test_wrapper_uses_configured_client(self):
        fake = Mock()
        fake.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=" answer "))])
        with patch.object(utils, "client", fake):
            self.assertEqual(GPT4oMiniWrapper()("question"), "answer")
        self.assertEqual(fake.chat.completions.create.call_args.kwargs["model"], "gpt-4o-mini")

    def test_invalid_verification_is_not_a_blank_prediction(self):
        with patch.object(verification, "call_gpt", return_value="no valid label"):
            with self.assertRaises(RuntimeError):
                verification.promptf("claim", verification.prompt, [])

    def test_intent_uses_predicted_hidden_evidence_without_gold_fields(self):
        assessor = mining.IntentArgumentation.__new__(mining.IntentArgumentation)
        assessor.intent_model = "test-model"
        event = {"claim": "claim", "evidence": ["PRESENTED_SENTENCE", "HIDDEN_SENTENCE"], "prediction": [0, 1]}
        with patch.object(mining, "completion_finetune", return_value="<intent>") as call:
            self.assertEqual(assessor._generate_intent(event), "intent")
        content = call.call_args.args[1][1]["content"]
        self.assertIn("HIDDEN_SENTENCE", content)
        self.assertNotIn("PRESENTED_SENTENCE", content)

    def test_empty_evidence_never_calls_neural_ranker(self):
        assessor = mining.IntentArgumentation.__new__(mining.IntentArgumentation)
        self.assertEqual(assessor.ranking_top_k_evidence("assumption", [], {}, "entailment"), [])

    def test_invalid_reassessment_is_an_error(self):
        with patch.object(reassessment, "call_gpt", return_value="Bad response"):
            with self.assertRaises(ValueError):
                reassessment.post_fix("true", "reason", {}, {"assumption": []})

    def test_unverifiable_preserves_label(self):
        with patch.object(reassessment, "call_gpt", return_value="D."):
            self.assertEqual(reassessment.post_fix("true", "reason", {}, {"assumption": []}), "true")

    def test_jsonl_is_one_record_per_line(self):
        with tempfile.TemporaryDirectory() as directory:
            handler = utils.DataHandler("test", directory, exact_folder=True)
            handler.save_file("hidden_info.jsonl", [{"value": 1}, {"value": 2}])
            handler.close()
            rows = [json.loads(line) for line in (Path(directory) / "hidden_info.jsonl").read_text().splitlines()]
            self.assertEqual(rows, [{"value": 1}, {"value": 2}])

    def test_empty_evaluation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.jsonl"
            path.write_text("")
            with self.assertRaises(ValueError):
                evaluate(path)


if __name__ == "__main__":
    unittest.main()
