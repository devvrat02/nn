# This script is adapted from https://github.com/jadeCurl/HiSS
# HiSS: Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method (AACL 2023)

import os

import json
import re
import argparse
from tqdm import tqdm
from utils.utils import *
from utils.local_llm import LocalGenerationLimitError


prompt = ['''You are required to determine the veracity of a claim.

Requirements:
1. Follow the format as the examples.
2. Give the veracity at the end of your answer inside a <>.
3. Answer must be one from {false, half-true, true}.

Examples: 
Claim: "Emerson Moser, who was Crayola’s top crayon molder for almost 40 years, was colorblind."
A fact checker will decompose the claim into 4 subclaims that are easier to verify:
1.Emerson Moser was a crayon molder at Crayola.
2.Moser worked at Crayola for almost 40 years.
3.Moser was Crayola's top crayon molder.
4.Moser was colorblind.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is there any official record or documentation indicating that Emerson Moser worked as a crayon molder at Crayola?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there any official records or documentation confirming Emerson Moser's length of employment at Crayola?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Emerson Moser, who is retiring next week after 35 years, isn't colorblind in the sense that he can't see color at all. It's just that some ...
To verify subclaim 3, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there credible sources or publications that mention Emerson Moser as Crayola's top crayon molder?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
To verify subclaim 4, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Are there any credible sources or records indicating that Emerson Moser was colorblind?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Yes.
Question: Was Emerson Moser's colorblindness only confusing for certain colors?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Moser has had tritanomaly, a type of colorblindness that makes it difficult to distinguish between blue and green and between yellow and red.
Based on the answers to these questions, it is clear that among false, half-true, true, the claim can be classified as <half-true>.

Claim: "Bernie Sanders said 85 million Americans have no health insurance."
A fact checker will not split the claim since the original claim is easier to verify.
To verify the claim, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: How many Americans did Bernie Sanders claim had no health insurance?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: "We have 85 million Americans who have no health insurance," Sanders said Dec. 11 on CNN's State of the Union.
Question: How did Bernie Sanders define "no health insurance"?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Sanders spokesperson Mike Casca said the senator was referring to the number of uninsured and under-insured Americans and cited a report about those numbers for adults.
Question: How many Americans were uninsured or under-insured according to the Commonwealth Fund survey?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: The Commonwealth Fund survey found that 43% of working-age adults 19 to 64, or about 85 million Americans, were uninsured or inadequately insured.
Question: Is the statement "we have 85 million Americans who have no health insurance" partially accurate according to the information in the passage?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Bernie Sanders omitted that his figure included people who either have no health insurance or are under-insured.
Based on the answers to these questions, it is clear that among false, half-true, true, the claim can be classified as <half-true>.

Claim: "JAG charges Nancy Pelosi with treason and seditious conspiracy."
A fact checker will decompose the claim into 2 subclaims that are easier to verify:
1. JAG has made a claim or accusation against Nancy Pelosi.
2. The specific charges or allegations made against Nancy Pelosi are treason and seditious conspiracy.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is it true that JAG has made a claim or accusation against Nancy Pelosi?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: There is no evidence to support this claim and a spokesperson for the U.S. Navy Judge Advocate General's Corps has stated that it is not true.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she raise each question and look for an answer:
Question: Is it true that the specific charges or allegations made against Nancy Pelosi are treason and seditious conspiracy?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: There is no evidence to support this claim.
Question: Where is the source of the claim?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: Real Raw News, a disclaimer stating that it contains "humor, parody and satire" and has a history of publishing fictitious stories.
Based on the answers to these questions, it is clear that among false, half-true, true, the claim can be classified as <false>.

Claim: "Cheri Beasley “backs tax hikes — even on families making under $75,000."
A fact checker will decompose the claim into 2 subclaims that are easier to verify:
1.Cheri Beasley supports tax increases.
2.Cheri Beasley supports tax increases for families with an income under $75,000.
To verify subclaim 1, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she might raise each question and look for an answer:
Question: Does Cheri Beasley support tax increases?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: Yes.
Answer: Beasley supports student loan bailouts for the wealthy.
To verify subclaim 2, a fact-checker will go through a step-by-step process to ask and answer a series of questions relevant to its factuality. Here are the specific steps he/she might raise each question and look for an answer:
Question: Does the ad accurately link Beasley's position on student loan debt forgiveness with her stance on tax hikes for families making under $75,000 per year?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: The ad makes a misleading connection between the two issues and does not accurately represent Beasley's position on tax hikes for families making under $75,000 per year.
Question: Has Cheri Beasley ever advocated for tax hikes specifically on families making under $75,000?
Tell me if you are confident to answer the question or not. Answer me ‘yes’ or ‘no’: No.
Answer: No evidence found that Cheri Beasley has explicitly advocated for such a tax hike.
Based on the answers to these questions, it is clear that among false, half-true, true, the claim can be classified as <false>.


Please follow the format above.
Your answer should be based on evidence below:
          
Evidence:
[EVIDENCE]


Claim: ''', '''A fact checker will''',
 ]



def extract_answer(text):
    # Models occasionally append an empty <> after the actual verdict. It is
    # formatting noise, not a replacement verdict. Still reject an explicit
    # unsupported final value rather than using an earlier, possibly revised one.
    matches = [value.strip().lower() for value in re.findall(r'<([^<>]*)>', text)
               if value.strip()]
    if not matches:
        raise ValueError("No nonempty <true>, <half-true>, or <false> verdict found")
    answer = matches[-1]
    if answer not in {"true", "half-true", "false"}:
        raise ValueError(f"Unsupported final verdict {answer!r}; expected true, half-true, or false")
    return answer

def extract_hiss_answer(text, audit=None):
    try:
        return extract_answer(text)
    except ValueError:
        match = re.fullmatch(
            r"\s*(true|half-true|false)\s*(?:\n\s*### Evidence\s*\n\s*(https?://[^\s<>]+(?:\s+https?://[^\s<>]+)*))?\s*",
            text, re.I)
        if not match:
            raise
        if audit is not None:
            audit.setdefault("generation_recovery", "explicit-hiss-verdict-v1")
            audit.setdefault("generation_recoveries", []).append(
                {"stage": "HiSS verdict parsing", "policy": "explicit-hiss-verdict-v1"})
        return match.group(1).lower()


def promptf(question, prompt, evidence, model="gpt-3.5-turbo", audit=None,
            intermediate="\nAnswer:",
            followup="Intermediate Question",
            finalans='\nBased on the answers to these questions, it is clear that among among false, half-true, true, the claim '):
    claim = question
    cur_prompt = prompt[0] + question
    
    cur_prompt = cur_prompt.replace('[EVIDENCE]', '\n'.join(evidence))
    
    max_retries = 3
    attempts = 0
    rationale, pred = None, None

    last_error = None
    recovered = False
    
    while attempts < max_retries:
        try:
            ret_text = call_gpt(cur_prompt, model=model)
            rationale = ret_text
            pred = extract_hiss_answer(ret_text, audit)
            break
        except LocalGenerationLimitError:
            if recovered:
                raise
            # Keep the original claim/evidence. Do not recycle an unfinished,
            # possibly repetitive rationale or invent a label on failure.
            recovered = True
            if audit is not None:
                audit["generation_recovery"] = "bounded-hiss-v1"
            print("Output limit reached; retrying once with bounded HiSS instructions.", flush=True)
            cur_prompt += ("\n\nFor this answer, use at most three verification questions and at most 200 words total. "
                           "Use only the supplied evidence. Do not repeat questions or lists. "
                           "Do not invent quotations. Finish now with one final verdict: <true>, <half-true>, or <false>.")
        except Exception as e:
            if isinstance(e, ValueError) and os.environ.get("TRACER_BACKEND") == "local":
                if recovered:
                    raise RuntimeError(f"HiSS format recovery failed: {e}") from e
                recovered = True
                if audit is not None:
                    audit["generation_recovery"] = "bounded-hiss-format-v1"
                print("Invalid HiSS answer format; retrying once with an explicit verdict requirement.", flush=True)
                cur_prompt += ("\n\nOUTPUT REQUIREMENT: Give at most 200 words of evidence-based explanation. "
                               "End with exactly one verdict: <true>, <half-true>, or <false>. "
                               "Do not provide only a URL or repeat the prompt.")
                attempts += 1
                continue
            last_error = e
            print(f"Attempt {attempts + 1} failed: {type(e).__name__}: {e}")
            rationale = ""
            pred = ""
        
        attempts += 1
    
    if not pred:
        raise RuntimeError(f"Claim verification failed after three attempts: {last_error}") from last_error
    return rationale, pred.lower()


def extract_question(generated):
    generated = generated.split('Question: ')[-1].split('Answer')[0]
    return generated

def main(args):
    dataset_path = args.datapath
    with open(dataset_path, 'r', encoding='utf-8') as json_file:
        json_list = json.load(json_file)

    logger = DataHandler(task_description="Claim Verification HiSS", result_folder=args.output_dir, exact_folder=True, resume=getattr(args, "resume", False))

    for json_str in tqdm(json_list):
        if json_str["example_id"] in logger.completed_ids:
            continue
        result = json_str
        label = result["veracity"]
        claim = result["claim"]
        idx = result["example_id"]
        evidence = result["evidence"]
        ruling = result["ruling"]

        question = claim 
        audit = {}
        rationale, pred = promptf(question, 
                                  prompt, 
                                  evidence=evidence, model=args.model, audit=audit)
        logger.log_iteration({
            **audit,
            "example_id": idx,
            "veracity": label,
            "pred": pred,
            "claim": claim,
            "ruling": ruling,
            "rationale": rationale,
        })
    
    logger.close()


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--datapath", type=str, default="dataset/test.json")
   parser.add_argument("--output_dir", default="method/results/literal")
   parser.add_argument("--model", default="gpt-3.5-turbo")
   args = parser.parse_args()
   main(args)
