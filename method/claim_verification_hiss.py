# This script is adapted from https://github.com/jadeCurl/HiSS
# HiSS: Towards LLM-based Fact Verification on News Claims with a Hierarchical Step-by-Step Prompting Method (AACL 2023)

import os

import json
import re
import argparse
from tqdm import tqdm
from utils.utils import *


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

cot_prompt = ['''You are a careful fact checker. Determine whether the claim is true,
half-true, or false based only on the evidence below.

Think through the key facts and any important context step by step. Then give
your final verdict. Your response must end with exactly one of these labels:
<true>, <half-true>, or <false>.

Claim: {claim}

Evidence:
{evidence}

Reasoning and final verdict:
''']


def extract_answer(text):
    labels = r"(?:half[\s-]+true|true|false)"
    matches = re.findall(rf"<\s*({labels})\s*>", text, flags=re.IGNORECASE)
    if matches:
        return re.sub(r"\s+", "-", matches[-1].lower())

    verdict_prefix = (
        r"(?:we\s+(?:rate|classify)\s+(?:(?:(?:the|this)\s+)?"
        r"(?:claim|statement)|it)\s+"
        r"(?:as\s+)?|(?:the\s+)?(?:claim|statement|verdict|classification|rating)\s+"
        r"(?:(?:is|was|should be|is classified as|is rated as)\s*|:\s*))"
    )
    match = re.search(
        rf"(?:^|[\n.!?]\s*){verdict_prefix}({labels})[.!]?\s*$",
        text.rstrip(),
        flags=re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", "-", match.group(1).lower())

    final_line = text.rstrip().splitlines()[-1].strip() if text.strip() else ""
    match = re.fullmatch(
        rf"({labels})[.!]?",
        final_line,
        flags=re.IGNORECASE,
    )
    if match:
        return re.sub(r"\s+", "-", match.group(1).lower())

    raise ValueError(
        "Model response did not include a valid <true>, <half-true>, or <false> "
        f"label. Response ended with: {text[-300:]!r}"
    )

def promptf(question, prompt, evidence,
            intermediate="\nAnswer:",
            followup="Intermediate Question",
            finalans='\nBased on the answers to these questions, it is clear that among among false, half-true, true, the claim ',
            prompt_template=None):
    claim = question
    template = prompt_template or prompt[0]
    if "{claim}" in template:
        cur_prompt = template.format(claim=question, evidence="\n".join(evidence))
    else:
        cur_prompt = template + question
        cur_prompt = cur_prompt.replace('[EVIDENCE]', '\n'.join(evidence))
    cur_prompt += (
        "\n\nOutput-format requirement: End your response with exactly one of "
        "<true>, <half-true>, or <false>."
    )

    rationale = call_gpt(
        cur_prompt, stop='Answer me ‘yes’ or ‘no’: No.', model="gpt-3.5-turbo"
    )
    try:
        return rationale, extract_answer(rationale)
    except ValueError:
        recovery_prompt = (
            "Classify the veracity of the claim using only the supplied evidence. "
            "Return exactly one label and nothing else: <true>, <half-true>, "
            "or <false>.\n\n"
            f"Claim: {question}\n\nEvidence:\n" + "\n".join(evidence)
        )
        recovery = call_gpt(recovery_prompt, model="gpt-3.5-turbo")
        try:
            pred = extract_answer(recovery)
        except ValueError as recovery_error:
            raise RuntimeError(
                "Claim verification failed: the model omitted a valid label in "
                f"both its answer and the format-recovery response. "
                f"Recovery response ended with: {recovery[-300:]!r}"
            ) from recovery_error

        return f"{rationale}\n\nFormat-recovery response: {recovery}", pred


def extract_question(generated):
    generated = generated.split('Question: ')[-1].split('Answer')[0]
    return generated

def main(args):
    dataset_path = args.datapath
    with open(dataset_path, 'r') as json_file:
        json_list = json.load(json_file)

    result_dir = os.environ.get("FACT_CHECK_RESULT_DIR", "method/result/")
    method_name = args.method
    logger = DataHandler(
        task_description=f"Claim Verification {method_name.upper()}", result_folder=result_dir
    )

    start_index = int(os.environ.get("FACT_CHECK_START", "0"))
    end_value = os.environ.get("FACT_CHECK_END")
    limit_value = os.environ.get("FACT_CHECK_LIMIT")
    if start_index < 0:
        raise ValueError("FACT_CHECK_START must be zero or a positive integer.")
    if end_value and limit_value:
        raise ValueError("Set only one of FACT_CHECK_END or FACT_CHECK_LIMIT.")
    if end_value:
        end_index = int(end_value)
    elif limit_value:
        limit = int(limit_value)
        if limit < 1:
            raise ValueError("FACT_CHECK_LIMIT must be a positive integer.")
        end_index = start_index + limit
    else:
        end_index = len(json_list)
    if end_index < start_index:
        raise ValueError("FACT_CHECK_END must be greater than or equal to FACT_CHECK_START.")
    json_list = json_list[start_index:end_index]
    if not json_list:
        raise ValueError(
            f"No records in dataset range [{start_index}, {end_index})."
        )

    selected_prompt = prompt if method_name == "hiss" else cot_prompt
    for record in tqdm(json_list, desc=f"{method_name.upper()} fact checks"):
        rationale, pred = promptf(
            record["claim"],
            selected_prompt,
            evidence=record["evidence"],
            prompt_template=selected_prompt[0] if method_name == "cot" else None,
        )
        logger.log_iteration({
            "example_id": record["example_id"],
            "veracity": record["veracity"],
            "pred": pred,
            "claim": record["claim"],
            "ruling": record["ruling"],
            "rationale": rationale,
        })
    
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="dataset/test.json")
    parser.add_argument(
        "--method",
        choices=("hiss", "cot"),
        default=os.environ.get("FACT_CHECK_METHOD", "hiss"),
    )
    args = parser.parse_args()
    main(args)
