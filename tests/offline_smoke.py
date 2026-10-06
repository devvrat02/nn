"""Real tiny alignment training + scripted LLM/NLI integration, not a benchmark."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def run(output):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaModel
    from sentence_alignment.train_model_script import main as train, AlignmentDataset
    from sentence_alignment.predict_model_script import predict, predict_single_claim
    from sentence_alignment.model import Aligner
    from method import claim_verification_hiss as literal, reassessment as ra, hidden_info_mining as mining
    from method.eval import evaluate

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    encoder = output / "tiny_encoder"
    encoder.mkdir(exist_ok=True)
    vocab = {word: i for i, word in enumerate(["<pad>", "<unk>", "<s>", "</s>", "[SPLIT]",
                                              "Jobs", "rose", "Policy", "worked", "Temporary", "jobs", "."])}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", unk_token="<unk>",
                                        bos_token="<s>", eos_token="</s>", additional_special_tokens=["[SPLIT]"])
    tokenizer.save_pretrained(encoder)
    RobertaModel(RobertaConfig(vocab_size=len(tokenizer), hidden_size=24, num_hidden_layers=1,
                              num_attention_heads=2, intermediate_size=32, max_position_embeddings=514,
                              pad_token_id=0)).save_pretrained(encoder)
    records = [{"example_id": i, "claim": "Jobs rose .", "evidence": ["Jobs rose .", "Temporary jobs ."],
                "annotation": [0, 1], "veracity": label, "ruling": "TEST REFERENCE ONLY"}
               for i, label in enumerate(["half-true", "false", "half-true", "true"])]
    data_path = output / "fixture_input.json"
    data_path.write_text(json.dumps(records), encoding="utf-8")
    checkpoint = output / "alignment_model"
    train(SimpleNamespace(train_dataset=str(data_path), encoder_name=str(encoder), max_evi=2,
                          output_dir=str(checkpoint), train_epoch=1, batch_size=2,
                          gradient_accumulation_steps=1, gradient_checkpointing=False,
                          learning_rate=1e-5, max_steps=2))
    aligned = output / "alignment.json"
    predict(str(checkpoint), str(data_path), str(aligned), 2, True)
    predicted = json.loads(aligned.read_text(encoding="utf-8"))
    assert all(len(x["prediction"]) == len(x["evidence"]) for x in predicted)
    # Force truncation and verify that masked-out separators never become training targets.
    dataset = AlignmentDataset(str(data_path), tokenizer, max_length=4, max_evidence_count=2)
    sample = dataset[0]
    assert int(sample["labels_mask"].sum()) <= int((sample["input_ids"] == vocab["[SPLIT]"]).sum())
    loaded = Aligner.from_pretrained(checkpoint)
    loaded.eval()
    long_predictions = predict_single_claim(loaded, tokenizer, "Jobs " * 100,
                                             ["Jobs " * 100, "Temporary jobs ."], 2, 32)
    assert len(long_predictions) == 2

    # Deterministic branch fixtures are separate from learned alignment outputs.
    # The tiny randomly initialized model is not scientifically meaningful.
    for record in predicted:
        record["prediction"] = [0, 1] if record["example_id"] != 3 else [0, 0]
    branch_path = output / "scripted_alignment.json"
    branch_path.write_text(json.dumps(predicted), encoding="utf-8")
    labels = iter(["true", "false", "half-true", "true"])

    def verify_response(*args, **kwargs):
        assert "TEST REFERENCE ONLY" not in args[0]
        return "Scripted test justification. <" + next(labels) + ">"

    class Ranker:
        device = "cpu"
        def __init__(self, *args, **kwargs):
            pass
        def encode(self, text):
            return np.array([1.0, 0.0])

    class NLI:
        model = SimpleNamespace(device="cpu")
        def __init__(self, *args, **kwargs):
            pass
        def predict(self, pairs, **kwargs):
            return np.array([[5.0, 0.0, 0.0] for _ in pairs])

    def intent_response(model, messages):
        assert "annotation" not in str(messages)
        assert "TEST REFERENCE ONLY" not in str(messages)
        return "<Policy worked.>"

    def scripted_generation(self, prompt, **kwargs):
        assert "TEST REFERENCE ONLY" not in prompt
        return "<Were the jobs permanent?>" if "yes-no" in prompt else "<The new jobs were permanent.>"

    # A network call from any overlooked code path fails the test immediately.
    with patch("socket.socket.connect", side_effect=AssertionError("Network forbidden in offline test")), \
         patch.object(literal, "call_gpt", side_effect=verify_response), \
         patch.object(mining, "SentenceTransformer", Ranker), patch.object(mining, "CrossEncoder", NLI), \
         patch.object(mining, "completion_finetune", side_effect=intent_response), \
         patch("method.prompts.GPT4oMiniWrapper.__call__", scripted_generation), \
         patch.object(mining, "call_gpt", return_value=" C. "), \
         patch.object(ra, "call_gpt", return_value="B"):
        literal.main(SimpleNamespace(datapath=str(data_path), output_dir=str(output / "literal"), model="fixture"))
        ra.main(SimpleNamespace(datafile=str(branch_path), literal=str(output / "literal/log.jsonl"),
                                intent_model="fixture", output_dir=str(output / "reassessment")))
    results = [json.loads(line) for line in (output / "reassessment/log.jsonl").read_text().splitlines()]
    assert [x["pred"] for x in results] == ["half-true", "false", "half-true", "true"]
    metrics = {"WARNING": "SCRIPTED INTEGRATION FIXTURES ONLY. Not research results or model accuracy.",
               "literal": evaluate(output / "literal/log.jsonl"),
               "reassessment": evaluate(output / "reassessment/log.jsonl"),
               "checks": ["tiny model training", "checkpoint reload", "alignment inference", "truncation",
                          "literal verification", "intent/questions/assumptions", "counterfactual filtering",
                          "CHE ranking", "true to half-true", "no CHE preserves true", "other labels preserved",
                          "JSONL artifacts", "evaluation", "no API calls"]}
    (output / "smoke_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
