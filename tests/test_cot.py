import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import local_workflow as workflow
from method import claim_verification_cot as cot


class CotTests(unittest.TestCase):
    def test_cot_uses_only_claim_and_evidence_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            source.write_text(json.dumps([{"example_id": 1, "claim": "CLAIM", "evidence": ["EVIDENCE"],
                                          "veracity": "false", "ruling": "SECRET_RULING", "intent_gold": "SECRET_INTENT"}]))
            args = SimpleNamespace(datapath=source, output_dir=root / "result", model="fixture", resume=True)
            with patch.object(cot, "call_gpt", return_value="Justification. <true>") as generate:
                cot.main(args)
                cot.main(args)
            generate.assert_called_once()
            sent = generate.call_args.args[0]
            self.assertIn("Think step by step", sent)
            self.assertIn("CLAIM", sent)
            self.assertNotIn("SECRET", sent)
            self.assertNotIn("Examples:", sent)
            row = json.loads((args.output_dir / "log.jsonl").read_text())
            self.assertEqual(row["pred"], "true")
            self.assertEqual(row["verifier"], "cot")

    def test_cot_cannot_resume_hiss_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input.json"
            source.write_text("[]")
            args = SimpleNamespace(output=root / "result", llm=root / "model", command="verify", context=100, max_new_tokens=10)
            workflow.stage_manifest(args, [source])
            args.verifier = "cot"
            with self.assertRaises(ValueError):
                workflow.stage_manifest(args, [source])

    def test_four_method_report_and_cot_pairing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "input.json"
            workflow.write_json(data, [{"example_id": 1, "claim": "c", "evidence": [], "annotation": [], "veracity": "half-true"}])
            for name, pred in [("hiss", "false"), ("hiss_ra", "false"), ("cot", "true"), ("cot_ra", "half-true")]:
                folder = root / name
                folder.mkdir()
                (folder / "log.jsonl").write_text(json.dumps({"example_id": 1, "veracity": "half-true", "pred": pred}) + "\n")
                config = {"verifier": "cot" if name.startswith("cot") else "hiss"}
                if name == "cot_ra":
                    config["input_hashes"] = {str((root / "cot/log.jsonl").resolve()): workflow.digest(root / "cot/log.jsonl")}
                workflow.write_json(folder / "local_run.json", {"status": "complete", "config": config})
            args = SimpleNamespace(data=data, literal=root / "hiss/log.jsonl", reassessed=root / "hiss_ra/log.jsonl",
                                   cot_literal=root / "cot/log.jsonl", cot_reassessed=root / "cot_ra/log.jsonl", output=root / "compare")
            workflow.compare(args)
            report = json.loads((args.output / "comparison.json").read_text())
            self.assertEqual(len(report["local_metrics"]), 4)
            self.assertEqual(report["gains_by_method_percentage_points"]["CoT"]["accuracy"], 100)
            self.assertEqual(report["gains_by_method_percentage_points"]["HiSS"]["accuracy"], 0)
            args.cot_reassessed = args.reassessed
            with self.assertRaises(ValueError):
                workflow.compare(args)
