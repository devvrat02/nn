import unittest
from types import SimpleNamespace

from local_workflow import apply_model_profile
from utils.local_llm import effective_context, render_chat


class LlamaProfileTests(unittest.TestCase):
    def test_profiles_keep_qwen_defaults_and_bound_llama_context(self):
        for profile, folder, context, tokens, device in (
            ("qwen", "qwen", 16384, 1536, "cuda"),
            ("llama2-7b", "llama2-7b-chat", 4096, 512, "cuda"),
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


if __name__ == "__main__":
    unittest.main()
