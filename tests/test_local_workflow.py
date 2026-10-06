import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import local_workflow as workflow
from utils import utils


class LocalWorkflowTests(unittest.TestCase):
    def test_larger_output_budget_preserves_log_and_records_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "input.json"
            source.write_text("[]")
            args = SimpleNamespace(output=base / "run", llm=base / "model", command="verify", context=100, max_new_tokens=10)
            workflow.stage_manifest(args, [source])
            log = args.output / "log.jsonl"
            original = b'{"example_id": 42, "pred": "true"}\n'
            log.write_bytes(original)
            args.max_new_tokens = 20
            workflow.stage_manifest(args, [source])
            self.assertEqual(log.read_bytes(), original)
            history = json.loads((args.output / "budget_changes.jsonl").read_text())
            self.assertEqual(history["completed_records_preserved"], 1)
            self.assertEqual(history["previous_config"]["max_new_tokens"], 10)
            args.max_new_tokens = 10
            with self.assertRaises(ValueError):
                workflow.stage_manifest(args, [source])

    def test_every_shared_chat_helper_routes_locally_without_api_client(self):
        with patch.dict(os.environ, {"TRACER_BACKEND": "local"}), \
             patch("utils.local_llm.complete_local", return_value="local answer") as local, \
             patch.object(utils, "get_client", side_effect=AssertionError("Hosted API forbidden")):
            self.assertEqual(utils.call_gpt("question", model="gpt-3.5-turbo"), "local answer")
            self.assertEqual(utils.completion_finetune("ft:anything", [{"role": "user", "content": "intent"}]), "local answer")
        self.assertEqual(local.call_count, 2)

    def test_prompt_imports_cannot_bypass_local_dispatch(self):
        from method import prompts, hidden_info_mining, reassessment
        with patch.dict(os.environ, {"TRACER_BACKEND": "local"}), \
             patch("utils.local_llm.complete_local", return_value="C"), \
             patch.object(utils, "get_client", side_effect=AssertionError("Hosted API forbidden")):
            for module in (prompts, hidden_info_mining, reassessment):
                self.assertEqual(module.call_gpt("counterfactual"), "C")
            self.assertEqual(prompts.GPT4oMiniWrapper()("assumption"), "C")

    def test_resume_preserves_completed_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = utils.DataHandler("test", tmp, exact_folder=True)
            first.log_iteration({"example_id": 42, "pred": "true"})
            first.close()
            second = utils.DataHandler("test", tmp, exact_folder=True, resume=True)
            self.assertEqual(second.completed_ids, {42})
            second.log_iteration({"example_id": 43, "pred": "false"})
            second.close()
            self.assertEqual(len(workflow.read_predictions(Path(tmp) / "log.jsonl")), 2)

    def test_changed_input_cannot_resume_old_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "input.json"
            source.write_text("[]")
            args = SimpleNamespace(output=base / "run", llm=base / "model", command="verify", context=100, max_new_tokens=10)
            workflow.stage_manifest(args, [source])
            workflow.stage_manifest(args, [source])
            source.write_text("[1]")
            with self.assertRaises(ValueError):
                workflow.stage_manifest(args, [source])

    def test_duplicate_prediction_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            path.write_text('{"example_id": 1}\n' * 2)
            with self.assertRaises(ValueError):
                workflow.read_predictions(path)

    def test_comparison_rejects_incomplete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "input.json"
            workflow.write_json(data, [{"example_id": 1, "claim": "test", "evidence": [], "annotation": [], "veracity": "true"}])
            log = base / "log.jsonl"
            log.write_text("")
            args = SimpleNamespace(data=data, literal=log, reassessed=log, output=base / "compare")
            with self.assertRaises(ValueError):
                workflow.compare(args)

    def test_comparison_marks_subset_and_computes_measured_gain(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "input.json"
            workflow.write_json(data, [{"example_id": "fixture", "claim": "test", "evidence": [], "annotation": [], "veracity": "half-true"}])
            for stage, prediction in (("literal", "true"), ("reassessed", "half-true")):
                folder = base / stage
                folder.mkdir()
                (folder / "log.jsonl").write_text(json.dumps({"example_id": "fixture", "veracity": "half-true", "pred": prediction}) + "\n")
                workflow.write_json(folder / "local_run.json", {"status": "complete", "config": {"backend": "local-transformers"}})
            args = SimpleNamespace(data=data, literal=base / "literal/log.jsonl", reassessed=base / "reassessed/log.jsonl", output=base / "compare")
            workflow.compare(args)
            report = json.loads((args.output / "comparison.json").read_text())
            self.assertFalse(report["full_official_test"])
            self.assertEqual(report["local_gain_percentage_points"]["accuracy"], 100)


if __name__ == "__main__":
    unittest.main()
