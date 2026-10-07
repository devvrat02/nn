import unittest
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import local_workflow as workflow

from local_workflow import apply_model_profile
from utils.local_llm import effective_context, render_chat, stop_token_ids


class LlamaProfileTests(unittest.TestCase):
    def test_existing_login_is_referenced_without_copying_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root = Path(tmp)
            cache = root / "user-cache"
            token = cache / "huggingface/token"
            token.parent.mkdir(parents=True)
            token.write_text("test-placeholder")
            os.environ["XDG_CACHE_HOME"] = str(cache)
            args = SimpleNamespace(model_profile="llama3-8b", llm=root / "model", context=8192,
                                   max_new_tokens=1536, device_map="auto", command="download")
            with patch.object(workflow, "ROOT", root / "project"):
                workflow.configure(args)
            self.assertEqual(os.environ["HF_TOKEN_PATH"], str(token))
            self.assertFalse((Path(os.environ["HF_HOME"]) / "token").exists())
            self.assertNotIn("HF_TOKEN", os.environ)

    def test_profiles_keep_qwen_defaults_and_bound_llama_context(self):
        for profile, folder, context, tokens, device in (
            ("qwen", "qwen", 16384, 1536, "cuda"),
            ("llama3-8b", "llama3-8b", 8192, 1536, "cuda"),
        ):
            args = SimpleNamespace(model_profile=profile, llm=None, context=None,
                                   max_new_tokens=None, device_map=None)
            apply_model_profile(args)
            self.assertEqual((args.llm.name, args.context, args.max_new_tokens, args.device_map),
                             (folder, context, tokens, device))

    def test_user_overrides_are_preserved(self):
        args = SimpleNamespace(model_profile="llama3-8b", llm="custom", context=3000,
                               max_new_tokens=256, device_map="cuda")
        apply_model_profile(args)
        self.assertEqual((args.llm, args.context, args.max_new_tokens, args.device_map),
                         ("custom", 3000, 256, "cuda"))

    def test_cannot_fake_longer_context(self):
        self.assertEqual(effective_context(8192, 8192), 8192)
        with self.assertRaisesRegex(ValueError, "native limit"):
            effective_context(16384, 8192)

    def test_default_cli_selects_exact_requested_base_model(self):
        with patch("sys.argv", ["local_workflow.py", "doctor"]), \
             patch.object(workflow, "configure"), patch.object(workflow, "doctor") as doctor:
            workflow.main()
        args = doctor.call_args.args[0]
        self.assertEqual(args.model_profile, "llama3-8b")
        self.assertEqual(args.context, 8192)
        self.assertEqual(workflow.MODEL_PROFILES[args.model_profile][1], "meta-llama/Meta-Llama-3-8B")

    def test_base_model_is_not_given_llama2_chat_tokens(self):
        tokenizer = SimpleNamespace(chat_template=None, bos_token="<|begin_of_text|>")
        messages = [{"role": "system", "content": "Rule"}, {"role": "user", "content": "All evidence"}]
        text = render_chat(tokenizer, messages, "llama", "base-v1")
        self.assertIn("All evidence", text)
        self.assertTrue(text.endswith("### Response\n"))
        self.assertNotIn("[INST]", text)
        with self.assertRaises(ValueError):
            render_chat(tokenizer, messages, "llama")

    def test_all_model_end_tokens_are_accepted(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[128001, 128009]))
        tokenizer = SimpleNamespace(eos_token_id=128001)
        self.assertIn(128009, stop_token_ids(model, tokenizer))
        model.generation_config.eos_token_id = None
        self.assertEqual(stop_token_ids(model, tokenizer), [128001])


if __name__ == "__main__":
    unittest.main()
