import unittest
from types import SimpleNamespace

from local_workflow import apply_model_profile
from utils.local_llm import effective_context, render_chat, stop_token_ids


class LlamaProfileTests(unittest.TestCase):
    def test_profiles_keep_qwen_defaults_and_bound_llama_context(self):
        for profile, folder, context, tokens, device in (
            ("qwen", "qwen", 16384, 1536, "cuda"),
            ("llama2-7b", "llama2-7b-chat", 4096, 512, "cuda"),
            ("llama31-8b", "llama31-8b", 16384, 1536, "cuda"),
            ("llama31-8b-instruct", "llama31-8b-instruct", 16384, 1536, "cuda"),
        ):
            args = SimpleNamespace(model_profile=profile, llm=None, context=None,
                                   max_new_tokens=None, device_map=None)
            apply_model_profile(args)
            self.assertEqual((args.llm.name, args.context, args.max_new_tokens, args.device_map),
                             (folder, context, tokens, device))

    def test_user_overrides_are_preserved(self):
        args = SimpleNamespace(model_profile="llama2-7b", llm="custom", context=3000,
                               max_new_tokens=256, device_map="cuda")
        apply_model_profile(args)
        self.assertEqual((args.llm, args.context, args.max_new_tokens, args.device_map),
                         ("custom", 3000, 256, "cuda"))

    def test_cannot_fake_longer_context(self):
        self.assertEqual(effective_context(4096, 4096), 4096)
        with self.assertRaisesRegex(ValueError, "native limit"):
            effective_context(16384, 4096)

    def test_llama_fallback_includes_entire_evidence_and_system(self):
        tokenizer = SimpleNamespace(chat_template=None, bos_token="<s>")
        messages = [{"role": "system", "content": "Format rule"},
                    {"role": "user", "content": "Claim and all evidence"}]
        self.assertEqual(render_chat(tokenizer, messages, "llama"),
                         "<s>[INST] <<SYS>>\nFormat rule\n<</SYS>>\n\nClaim and all evidence [/INST]")
        with self.assertRaises(ValueError):
            render_chat(tokenizer, messages, "unknown")

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
