"""Local zero-shot CoT baseline; this prompt is not supplied by the authors."""
import json
import re
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


def extract_cot_answer(text, audit=None):
    try:
        return extract_answer(text)
    except ValueError:
        # Never override a malformed/unsupported tagged verdict. Only accept an
        # explicit opening assertion from a normally completed generation.
        if "<" in text or ">" in text:
            raise
        match = re.match(r"\A\s*The claim is (half-true|true|false)[.!](?:\s|$)", text, re.I)
        labels = re.findall(r"\b(?:half[- ]true|true|false)\b", text.lower())
        if (not match or set(labels) != {match.group(1).lower()}
                or re.search(r"\b(?:not|uncertain|unclear|cannot|however|but|although)\b", text, re.I)):
            raise
        if audit is not None:
            audit.append({"stage": "CoT verdict parsing", "policy": "explicit-opening-verdict-v1",
                          "matched_verdict": match.group(0).strip()})
        return match.group(1).lower()


def main(args):
    with open(args.datapath, encoding="utf-8") as stream:
        data = json.load(stream)
    logger = DataHandler("Local zero-shot CoT verification (cot-v1)", args.output_dir,
                         exact_folder=True, resume=getattr(args, "resume", False))
    try:
        for item in tqdm(data):
            if item["example_id"] in logger.completed_ids:
                continue
            tqdm.write(f"CoT verification: example {item['example_id']}")
            audit = []
            rationale, pred = generate_structured(
                lambda text: call_gpt(text, model=args.model), build_prompt(item),
                lambda text: (text, extract_cot_answer(text, audit)), "CoT verification",
                "Give at most 150 words of evidence-based justification, followed by exactly one "
                "final <true>, <half-true>, or <false>. Do not repeat or invent quotations.",
                audit, policy="bounded-cot-v1")
            logger.log_iteration({"example_id": item["example_id"], "veracity": item["veracity"],
                                  "claim": item["claim"], "ruling": item.get("ruling", ""),
                                  "pred": pred, "rationale": rationale, "verifier": "cot", "prompt_version": "cot-v1",
                                  "verdict_parser": "cot-explicit-opening-v1",
                                  **({"generation_recovery": ("bounded-cot-v1" if any(
                                      entry["policy"] == "bounded-cot-v1" for entry in audit)
                                      else "explicit-opening-verdict-v1"),
                                      "generation_recoveries": audit} if audit else {})})
    finally:
        logger.close()
