import unittest
from types import SimpleNamespace
from unittest.mock import patch

from utils.local_llm import GenerationProgress


class GenerationProgressTests(unittest.TestCase):
    def test_reports_first_token_and_interval_without_stopping(self):
        with patch("utils.local_llm.time.monotonic", side_effect=[0, 2, 3, 18, 19]), \
             patch("builtins.print") as output:
            progress = GenerationProgress(100, 50)
            for count in [1, 2, 10]:
                self.assertIs(progress(SimpleNamespace(shape=(1, 100 + count)), None), False)
            progress.finish(11)
        lines = [call.args[0] for call in output.call_args_list]
        self.assertEqual(len(lines), 4)
        self.assertIn("1/50 tokens", lines[1])
        self.assertIn("10/50 tokens", lines[2])
        self.assertIn("11 tokens", lines[3])

    def test_transformers_stopping_list_remains_false(self):
        import torch
        from transformers import StoppingCriteriaList
        with patch("builtins.print"):
            progress = GenerationProgress(2, 10)
            result = StoppingCriteriaList([progress])(torch.tensor([[1, 2, 3]]), None)
        self.assertFalse(result.any().item())


if __name__ == "__main__":
    unittest.main()
