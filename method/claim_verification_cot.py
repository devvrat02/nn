"""Local zero-shot CoT baseline; this prompt is not supplied by the authors."""
import json
from tqdm import tqdm
from utils.utils import DataHandler, call_gpt
from utils.generation_recovery import generate_structured
from method.claim_verification_hiss import extract_answer

PROMPT = """Determine the veracity of the claim using the supplied evidence.
Think step by step about whether the evidence supports, contradicts, or qualifies
the claim. Give a brief evidence-based justification, then a final verdict.
Choose exactly one label: true, half-true, or false.
Write the final verdict inside angle brackets: <true>, <half-true>, or <false>.

Evidence:
{evidence}

Claim: {claim}
"""


def build_prompt(item):
    return PROMPT.format(claim=item["claim"], evidence="\n".join(item["evidence"]))


def main(args):
    with open(args.datapath, encoding="utf-8") as stream:
        data = json.load(stream)
    logger = DataHandler("Local zero-shot CoT verification (cot-v1)", args.output_dir,
                         exact_folder=True, resume=getattr(args, "resume", False))
    try:
        for item in tqdm(data):
            if item["example_id"] in logger.completed_ids:
                continue
            audit = []
            rationale, pred = generate_structured(
                lambda text: call_gpt(text, model=args.model), build_prompt(item),
                lambda text: (text, extract_answer(text)), "CoT verification",
                "Give at most 150 words of evidence-based justification, followed by exactly one "
                "final <true>, <half-true>, or <false>. Do not repeat or invent quotations.",
                audit, policy="bounded-cot-v1")
            logger.log_iteration({"example_id": item["example_id"], "veracity": item["veracity"],
                                  "claim": item["claim"], "ruling": item.get("ruling", ""),
                                  "pred": pred, "rationale": rationale, "verifier": "cot", "prompt_version": "cot-v1",
                                  **({"generation_recovery": "bounded-cot-v1", "generation_recoveries": audit} if audit else {})})
    finally:
        logger.close()
