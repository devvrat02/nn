import json
import os
import re
from utils import *
from tqdm import tqdm
import argparse
from utils.utils import *
from method.hidden_info_mining import *
from copy import deepcopy


PROMPT_POST_PROCESS = """A claim can be literally correct but still misleading in an implicit way. You are required to review the fact-checking of a claim and refine its veracity.

We have fact-checked a claim and generated a justification. Now, we have discovered hidden information related to the intended conclusion of the claim.

Please review the justification and determine whether the fact-checking has taken the hidden information into account. Then, refine the veracity of the claim accordingly.

You will be provided with the argument supporting the intended conclusion and relevant evidence.

Evidence: [EVIDENCE]
Argument: [ARGUMENT]
Justification: [JUSTIFICATION]

Please refine the veracity of the claim based on the above information.

Original Veracity: [VERACITY]

Choose one of the following options:
A. True
B. Half-true
C. False
D. Unverifiable (e.g., the assumption does not strongly support the intended conclusion. not enough information)

Only output the veracity option. Do not include additional text.
From {A, B, C, D}, your answer is: 
"""

PROMPT_STRUCTURED_CLASSIFICATION = """Reassess the claim using the original fact-checking justification, the supplied evidence, and the argument about hidden information.

Claim classification must follow this dataset's rating mapping:
- TRUE: the original rating is True.
- HALF-TRUE: the original rating is Mostly True or Half True.
- FALSE: the original rating is Mostly False, False, or Pants on Fire.

First list the claim's individual factual components. For each component, assess the available evidence as SUPPORTED, CONTRADICTED, or INSUFFICIENT EVIDENCE. Then choose one overall label using the dataset mapping and the materiality of the evidence:
- Choose TRUE when the evidence supports the material claim and does not establish a misleading omission.
- Choose FALSE when the material claim is substantially contradicted.
- Choose HALF-TRUE when it is partly accurate or important context materially changes the impression.

Insufficient evidence is not itself proof of falsity or a reason by itself to choose HALF-TRUE. Do not let one supporting or contradictory item decide the whole claim. If the evidence does not justify changing the original veracity, retain it. Give a concise explanation for the final label.

Claim: [CLAIM]
Evidence: [EVIDENCE]
Argument: [ARGUMENT]
Original justification: [JUSTIFICATION]
Original veracity: [VERACITY]

The first line must be exactly one of `Label: TRUE`, `Label: HALF-TRUE`, or `Label: FALSE`.
The second line must start with `Explanation:` and give one concise explanation.
"""

PROMPT_EVIDENCE_CONSISTENCY = """Check the consistency between the claim's factual components and the supplied evidence before assigning a final label.

Use this dataset's rating mapping:
- TRUE maps only to the original True rating.
- HALF-TRUE maps to Mostly True or Half True.
- FALSE maps to Mostly False, False, or Pants on Fire.

Analyze factual components of the CLAIM only; do not turn evidence details into claim components. For each material component, indicate whether it is CENTRAL or SECONDARY, assess it as SUPPORTED, CONTRADICTED, or INSUFFICIENT EVIDENCE, and cite relevant evidence IDs. Keep each component to one short line and preserve both sides when evidence conflicts.

Then classify the claim as a whole:
- FALSE when evidence contradicts the central factual assertion. Incidental accurate details do not make a false central assertion half-true.
- HALF-TRUE only when evidence establishes a meaningful mixture: at least one important component is supported and another important component is contradicted or materially qualified. Missing or incomplete evidence alone is not a mixture and is not sufficient for HALF-TRUE.
- TRUE when the material components are supported and no important contradiction or misleading qualification is established.

Do not treat insufficient evidence as proof of FALSE or HALF-TRUE. If the evidence does not establish a meaningful basis to change the starting prediction, retain that prediction. Do not use any information beyond the claim, evidence, justification, starting prediction, and supplied TRACER argument.

Claim: [CLAIM]
Starting prediction: [VERACITY]
HiSS justification: [JUSTIFICATION]
Evidence:
[EVIDENCE]
TRACER argument, if available:
[ARGUMENT]

Return only the compact component lines followed by these two lines. Do not summarize or quote the evidence:
`C1 | CENTRAL or SECONDARY | SUPPORTED, CONTRADICTED, or INSUFFICIENT | Evidence IDs`
...
Label: TRUE, HALF-TRUE, or FALSE
Explanation: one concise explanation
"""

DECISION_CONSISTENCY_GUARD = False
DECISION_GUARD_PROMPT_ID = "decision-consistency-guard-v3"
PROMPT_DECISION_GUARD_ANALYSIS = """Analyze how the supplied evidence relates to factual components asserted by the claim. Do not classify the claim and do not treat evidence details as claim components.

Use the project's existing label mapping if a classification is discussed later: TRUE is the original True rating; HALF-TRUE is Mostly True or Half True; FALSE is Mostly False, False, or Pants on Fire.

Identify the claim's central factual proposition and any material secondary components. For every component, mark every applicable status:
- SUPPORTED: cited evidence directly supports this component.
- CONTRADICTED: cited evidence directly contradicts this component.
- INSUFFICIENT: evidence does not establish support or contradiction for this component.
Preserve conflicting support and contradiction for the same component. Do not treat INSUFFICIENT as CONTRADICTED. Mark components only SECONDARY when they are not material to the main claim.

Return only component lines in this exact pipe-delimited format, one component per line:
CENTRAL | short factual proposition | SUPPORTED=Evidence_1,Evidence_2 | CONTRADICTED=none | INSUFFICIENT=none

The first field must be CENTRAL, MATERIAL_SECONDARY, or SECONDARY. Use exact field order and include all five fields. List every applicable status separately with only evidence IDs that directly support it; use `none` for an absent status. If a component is INSUFFICIENT but has no evidence ID, write `INSUFFICIENT=no evidence IDs`. Preserve both support and contradiction when both apply. Every cited ID must exactly match an ID supplied below. Mark at least one component CENTRAL. Do not add a heading, JSON, explanation, or trailing prose. Central status is derived from the CENTRAL component lines.

Example of conflicting evidence:
CENTRAL | short factual proposition | SUPPORTED=Evidence_1 | CONTRADICTED=Evidence_2 | INSUFFICIENT=none

Claim: [CLAIM]
Current proposed prediction: [PREDICTION]
HiSS justification: [JUSTIFICATION]
Evidence:
[EVIDENCE]
"""

PROMPT_DECISION_GUARD_RECONSIDER = """Reconsider the proposed claim classification because it conflicts with the evidence-status analysis. Choose a label using the project's existing label definitions and the same evidence; do not force a particular replacement label.

Label mapping:
- TRUE: original rating True.
- HALF-TRUE: original rating Mostly True or Half True.
- FALSE: original rating Mostly False, False, or Pants on Fire.

Distinguish central assertions from material secondary details. Require direct evidence contradicting a supported central proposition before choosing FALSE. Do not infer contradiction from INSUFFICIENT evidence. Use HALF-TRUE when the dataset's definition applies to a materially partly accurate or contextually misleading claim; do not use it just because an irrelevant detail is uncertain. Use TRUE only when the material components are adequately supported under the dataset's definitions. Preserve conflicting evidence.

Claim: [CLAIM]
Proposed prediction: [PREDICTION]
HiSS justification: [JUSTIFICATION]
Evidence:
[EVIDENCE]
Evidence-status analysis:
[ANALYSIS]
Reason for reconsideration:
[REASON]

Return exactly two lines:
Label: TRUE, HALF-TRUE, or FALSE
Explanation: one concise reason grounded in the supplied evidence
"""


_DECISION_GUARD_STATUS_ORDER = (
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT",
)
_DECISION_GUARD_STATUSES = set(_DECISION_GUARD_STATUS_ORDER)
_DECISION_GUARD_IMPORTANCE = {"CENTRAL", "MATERIAL_SECONDARY", "SECONDARY"}


def _extract_json_object(response):
    response = (response or "").strip()
    if response.startswith("```"):
        response = re.sub(
            r"^```(?:json)?\s*|\s*```$", "", response, flags=re.IGNORECASE
        )
    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("The evidence analysis did not contain a JSON object.")
    return json.loads(response[start : end + 1])


def _parse_decision_guard_analysis(response, evidence):
    response = (response or "").strip()
    if "{" in response:
        parsed = _extract_json_object(response)
        if not isinstance(parsed, dict) or not isinstance(parsed.get("components"), list):
            raise ValueError("The evidence analysis has an invalid JSON schema.")
        raw_components = parsed["components"]
    else:
        raw_components = []
        for line in response.splitlines():
            fields = [field.strip() for field in line.split("|")]
            if len(fields) != 5:
                raise ValueError("A component line must contain exactly five fields.")
            importance, name, *status_fields = fields
            evidence_ids = {}
            statuses = []
            for expected_status, status_field in zip(
                _DECISION_GUARD_STATUS_ORDER, status_fields
            ):
                if "=" not in status_field:
                    raise ValueError("A component status field is malformed.")
                label, values = (part.strip() for part in status_field.split("=", 1))
                if label.upper() != expected_status:
                    raise ValueError("Component status fields are out of order.")
                if not values:
                    raise ValueError("A component status field must not be empty.")
                if values.lower() == "none":
                    ids = []
                elif expected_status == "INSUFFICIENT" and values.lower() == "no evidence ids":
                    ids = []
                    statuses.append(expected_status)
                else:
                    ids = [item.strip() for item in values.split(",") if item.strip()]
                    if ids:
                        statuses.append(expected_status)
                evidence_ids[expected_status] = ids
            raw_components.append({
                "component": name,
                "importance": importance.upper(),
                "status": statuses,
                "evidence_ids": evidence_ids,
            })
        if not raw_components:
            raise ValueError("The evidence analysis did not contain component lines.")

    supplied_ids = set(evidence)
    normalized = []
    for component in raw_components:
        if not isinstance(component, dict):
            raise ValueError("Evidence-analysis components must be objects.")
        name = component.get("component")
        importance = component.get("importance")
        statuses = component.get("status")
        evidence_ids = component.get("evidence_ids")
        if (
            not isinstance(name, str)
            or not name.strip()
            or importance not in _DECISION_GUARD_IMPORTANCE
            or not isinstance(statuses, list)
            or not statuses
            or not isinstance(evidence_ids, dict)
        ):
            raise ValueError("An evidence-analysis component is malformed.")
        statuses = {str(status).upper() for status in statuses}
        if not statuses.issubset(_DECISION_GUARD_STATUSES):
            raise ValueError("An evidence-analysis component has an unknown status.")
        if not set(evidence_ids).issubset(_DECISION_GUARD_STATUSES):
            raise ValueError("An evidence-analysis component has an unknown status key.")
        ids_by_status = {}
        for status in _DECISION_GUARD_STATUS_ORDER:
            ids = evidence_ids.get(status, [])
            if not isinstance(ids, list) or not all(
                isinstance(item, str) for item in ids
            ):
                raise ValueError("Evidence citations must be lists of evidence IDs.")
            if not set(ids).issubset(supplied_ids):
                raise ValueError("The evidence analysis cited an ID not supplied to it.")
            if status in {"SUPPORTED", "CONTRADICTED"} and status in statuses and not ids:
                raise ValueError(f"{status} requires at least one cited evidence ID.")
            if status not in statuses and ids:
                raise ValueError(f"Evidence IDs were supplied for unmarked status {status}.")
            ids_by_status[status] = list(dict.fromkeys(ids))
        normalized.append({
            "component": name.strip(),
            "importance": importance,
            "status": [
                status
                for status in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT")
                if status in statuses
            ],
            "evidence_ids": ids_by_status,
        })
    central_components = [
        item for item in normalized if item["importance"] == "CENTRAL"
    ]
    if not central_components:
        raise ValueError("The evidence analysis did not identify a central component.")
    central_statuses = {
        status
        for item in central_components
        for status in item["status"]
    }
    if {"SUPPORTED", "CONTRADICTED"}.issubset(central_statuses):
        central_status = "MIXED"
    elif "CONTRADICTED" in central_statuses:
        central_status = "CONTRADICTED"
    elif "SUPPORTED" in central_statuses:
        central_status = "SUPPORTED"
    else:
        central_status = "INSUFFICIENT"
    return {"central_claim_status": central_status, "components": normalized}


def _decision_guard_check(prediction, analysis):
    central_status = analysis["central_claim_status"]
    material = [
        item for item in analysis["components"]
        if item["importance"] in {"CENTRAL", "MATERIAL_SECONDARY"}
    ]
    supports_material = any("SUPPORTED" in item["status"] for item in material)
    contradicts_material = any("CONTRADICTED" in item["status"] for item in material)
    insufficient_material = any("INSUFFICIENT" in item["status"] for item in material)
    meaningful_mixture = supports_material and contradicts_material

    if prediction == "false":
        if central_status == "SUPPORTED":
            return "RECONSIDER", "The central proposition is marked SUPPORTED without a cited central contradiction."
        if central_status == "INSUFFICIENT":
            return "RECONSIDER", "The central proposition is INSUFFICIENT; insufficient evidence alone does not establish FALSE."
        if central_status == "MIXED":
            return "ESCALATE", "The central proposition has conflicting support and contradiction."
        if meaningful_mixture:
            return "RECONSIDER", "Material components include both supported and contradicted content."
        return "KEEP", "The central proposition is directly contradicted by cited evidence."

    if prediction == "half-true":
        if meaningful_mixture:
            return "KEEP", "Material components include both supported and contradicted content, consistent with a partial-accuracy label."
        if insufficient_material:
            return "ESCALATE", "The material evidence is insufficient to confirm or refute the partial-accuracy interpretation."
        if central_status == "CONTRADICTED" and not supports_material:
            return "RECONSIDER", "The central proposition is contradicted and no material supported component establishes a meaningful mixture."
        if central_status == "SUPPORTED" and not contradicts_material:
            return "RECONSIDER", "The central and material components are supported without a material contradiction; reconsider whether HALF-TRUE is justified."
        return "ESCALATE", "The evidence statuses do not establish a clear basis for retaining or changing HALF-TRUE."

    if prediction == "true":
        if contradicts_material:
            return "RECONSIDER", "At least one material component is marked CONTRADICTED."
        if central_status != "SUPPORTED" or insufficient_material:
            return "ESCALATE", "The central proposition or another material component is not adequately established as supported."
        return "KEEP", "The central proposition and all material components are supported without a material contradiction."

    raise ValueError(f"Unsupported prediction for decision guard: {prediction!r}")


def decision_consistency_guard(
    prediction, rationale, claim, evidence, argument=None, return_details=False
):
    evidence_text = "\n".join(
        f"{key}: {value}" for key, value in evidence.items()
    ) or "(No evidence available.)"
    analysis_prompt = (
        PROMPT_DECISION_GUARD_ANALYSIS
        .replace("[CLAIM]", claim)
        .replace("[PREDICTION]", prediction)
        .replace("[JUSTIFICATION]", rationale or "(No justification available.)")
        .replace("[EVIDENCE]", evidence_text)
    )
    if argument:
        analysis_prompt += "\nTRACER argument, for context only:\n" + json.dumps(
            argument, ensure_ascii=False
        )
    analysis_response = None
    correction_response = None
    details = {
        "original_prediction": prediction,
        "central_claim_status": "INSUFFICIENT",
        "supported_components": [],
        "contradicted_components": [],
        "insufficient_components": [],
        "evidence_statuses": [],
        "consistency_check": None,
        "consistency_reason": "",
        "guard_action": "ESCALATE",
        "final_prediction": prediction,
        "analysis_parse_valid": False,
        "analysis_response": None,
        "correction_response": None,
        "prompt_configuration": DECISION_GUARD_PROMPT_ID,
    }
    try:
        analysis_response = call_gpt(analysis_prompt)
        analysis = _parse_decision_guard_analysis(analysis_response, evidence)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        details["consistency_reason"] = (
            f"Evidence analysis was malformed or unverifiable: {error}"
        )
        details["analysis_response"] = analysis_response
        if return_details:
            return prediction, details
        return prediction

    details["analysis_parse_valid"] = True
    details["central_claim_status"] = analysis["central_claim_status"]
    details["evidence_statuses"] = analysis["components"]
    details["supported_components"] = [
        item for item in analysis["components"] if "SUPPORTED" in item["status"]
    ]
    details["contradicted_components"] = [
        item for item in analysis["components"] if "CONTRADICTED" in item["status"]
    ]
    details["insufficient_components"] = [
        item for item in analysis["components"] if "INSUFFICIENT" in item["status"]
    ]
    action, reason = _decision_guard_check(prediction, analysis)
    details["guard_action"] = action
    details["consistency_reason"] = reason
    details["consistency_check"] = action == "KEEP"
    details["analysis_response"] = analysis_response

    if action == "RECONSIDER":
        correction_prompt = (
            PROMPT_DECISION_GUARD_RECONSIDER
            .replace("[CLAIM]", claim)
            .replace("[PREDICTION]", prediction)
            .replace("[JUSTIFICATION]", rationale or "(No justification available.)")
            .replace("[EVIDENCE]", evidence_text)
            .replace("[ANALYSIS]", json.dumps(analysis, ensure_ascii=False))
            .replace("[REASON]", reason)
        )
        correction_response = call_gpt(correction_prompt)
        match = re.search(
            r"(?im)^\s*Label:\s*(true|half[- ]true|false)\s*$",
            correction_response or "",
        )
        explanation_match = re.search(
            r"(?im)^\s*Explanation:\s*\S.*$",
            correction_response or "",
        )
        if match and explanation_match:
            details["final_prediction"] = match.group(1).lower().replace(" ", "-")
            details["correction_response"] = correction_response
        else:
            details["guard_action"] = "ESCALATE"
            details["consistency_check"] = None
            details["consistency_reason"] += (
                " Reconsideration response was malformed; retained the original prediction."
            )
            details["correction_response"] = correction_response
    if return_details:
        return details["final_prediction"], details
    return details["final_prediction"]


def post_fix(
    pred,
    rationale,
    relevant_evidence,
    argument,
    evidence_filter="original",
    classifier_prompt="original",
    return_response=False,
):
    if classifier_prompt not in ("original", "structured"):
        raise ValueError(f"Unknown classifier prompt: {classifier_prompt}")

    all_hidden_evidence_key = []
    for assumption in argument["assumption"]:
        if evidence_filter == "original":
            assumption["backing"] = (
                assumption["supported_by"] + assumption["refuted_by"]
            )
            assumption.pop("supported_by")
            assumption.pop("refuted_by")
        else:
            supported_by = assumption.get("supported_by", [])
            refuted_by = assumption.get("refuted_by", [])
            neutral = assumption.get("neutral", [])
            assumption["backing"] = list(dict.fromkeys(
                supported_by + refuted_by + neutral
            ))
        all_hidden_evidence_key += assumption["backing"]
    all_hidden_evidence_key = list(set(all_hidden_evidence_key))
    hidden_evidence = {key: relevant_evidence[key] for key in all_hidden_evidence_key}
    if classifier_prompt == "structured":
        prompt = (
            PROMPT_STRUCTURED_CLASSIFICATION
            .replace("[CLAIM]", argument["claim"])
            .replace("[EVIDENCE]", json.dumps(hidden_evidence, indent=4))
            .replace("[ARGUMENT]", json.dumps(argument, indent=4))
            .replace("[JUSTIFICATION]", rationale)
            .replace("[VERACITY]", pred)
        )
    else:
        prompt = PROMPT_POST_PROCESS.replace(
            "[EVIDENCE]", json.dumps(hidden_evidence, indent=4)
        ).replace("[ARGUMENT]", json.dumps(argument, indent=4)).replace(
            "[JUSTIFICATION]", rationale
        ).replace("[VERACITY]", pred)
    if evidence_filter == "assumption-aware" and classifier_prompt == "original":
        prompt += (
            "\n\nInterpret each assumption separately using its evidence "
            "relationships. Preserve both supporting and contradictory evidence; "
            "neither one supporting item nor one contradiction decides the whole "
            "claim. Treat neutral/insufficient evidence as missing support, not as "
            "a contradiction. If only some parts of the intended conclusion are "
            "supported while others are contradicted or insufficient, consider "
            "Half-true rather than forcing an all-true or all-false verdict."
        )
    response = call_gpt(prompt)
    if classifier_prompt == "structured":
        match = re.search(
            r"(?im)^\s*Label:\s*(true|half[- ]true|false)\s*$",
            response or "",
        )
        if not match:
            print(
                "[structured-classifier] malformed classification response; "
                f"preserving original verdict {pred!r}. Response: {response!r}",
                flush=True,
            )
            result = pred
        else:
            result = match.group(1).lower().replace(" ", "-")
    elif evidence_filter == "assumption-aware":
        match = re.match(r"^\s*([ABCD])(?:\b|[.)])", response or "", re.IGNORECASE)
        if not match:
            print(
                "[assumption-aware] malformed final classification response; "
                f"preserving original verdict {pred!r}. Response: {response!r}",
                flush=True,
            )
            result = pred
        else:
            label = match.group(1).upper()
            result = {"A": "true", "B": "half-true", "C": "false", "D": pred}[label]
    else:
        if "A" in response[:2]:
            result = "true"
        elif "B" in response[:2]:
            result = "half-true"
        elif "C" in response[:2]:
            result = "false"
        else:
            result = pred
    return (result, response) if return_response else result


def evidence_consistency_check(
    pred, rationale, claim, evidence, argument=None, return_response=False
):
    evidence_text = "\n".join(
        f"{key}: {value}" for key, value in evidence.items()
    ) or "(No evidence available.)"
    argument_text = (
        json.dumps(argument, ensure_ascii=False, indent=2)
        if argument
        else "(No TRACER argument available.)"
    )
    prompt = (
        PROMPT_EVIDENCE_CONSISTENCY
        .replace("[CLAIM]", claim)
        .replace("[VERACITY]", pred)
        .replace("[JUSTIFICATION]", rationale or "(No justification available.)")
        .replace("[EVIDENCE]", evidence_text)
        .replace("[ARGUMENT]", argument_text)
    )
    response = call_gpt(prompt)
    match = re.search(
        r"(?im)^\s*Label:\s*(true|half[- ]true|false)\s*$",
        response or "",
    )
    response_valid = match is not None
    if not response_valid:
        print(
            "[evidence-consistency] malformed response; preserving starting "
            f"prediction {pred!r}. Response: {response!r}",
            flush=True,
        )
        result = pred
    else:
        result = match.group(1).lower().replace(" ", "-")
    return (result, response, response_valid) if return_response else result



def main(args):
    pred_dict = {}
    if (
        args.decision_consistency_guard
        and args.classifier_prompt == "evidence-consistency"
    ):
        raise ValueError(
            "Decision Consistency Guard cannot be combined with the evidence-consistency extension."
        )
    result_dir = os.environ.get("REASSESS_RESULT_DIR", "method/result/")
    logger = DataHandler(
        task_description=(
            f"post-process ({args.evidence_filter} evidence filter, "
            f"{args.classifier_prompt} classifier, "
            f"decision guard={args.decision_consistency_guard})\n{args.datafile}"
        ),
        result_folder=result_dir,
    )
    literal_path = args.literal

    intent_assessor = IntentArgumentation(
        intent_model_id=args.intent_model or os.environ.get("LLM_MODEL_PATH"),
        evidence_filter=args.evidence_filter,
    )
    hidden_info_collections = []

    with open(literal_path, "r") as f:
        for line in f:
            record = json.loads(line)
            pred_dict[record['example_id']] = record
    with open(args.datafile, 'r') as f:
        data = json.load(f)
    for event in tqdm(data):
        pred = pred_dict[event['example_id']]['pred']
        pred_new = pred
        rationale = pred_dict[event['example_id']]['rationale']
        classifier_response = None
        classifier_parse_valid = None
        evidence = event["evidence"]
        evidence_dict = {
            f"Evidence_{i+1}": evidence[i] for i in range(len(evidence))
        }
        argument = None
        if pred == "true":
            event["relevant_evidence"] = evidence_dict
            event = intent_assessor.mining_one_event(event)
            argument = event["argument"]
            hidden_info_collections.append(event)
            if (
                argument["assumption"]
                and args.classifier_prompt != "evidence-consistency"
            ):
                ans, classifier_response = post_fix(
                    pred,
                    rationale,
                    evidence_dict,
                    deepcopy(argument),
                    evidence_filter=args.evidence_filter,
                    classifier_prompt=args.classifier_prompt,
                    return_response=True,
                )
                pred_new = ans
        if args.classifier_prompt == "evidence-consistency":
            pred_new, classifier_response, classifier_parse_valid = evidence_consistency_check(
                pred,
                rationale,
                event["claim"],
                evidence_dict,
                argument=argument,
                return_response=True,
            )
        prediction_before_guard = pred_new
        guard_details = None
        if args.decision_consistency_guard:
            pred_new, guard_details = decision_consistency_guard(
                prediction_before_guard,
                rationale,
                event["claim"],
                evidence_dict,
                argument=argument,
                return_details=True,
            )
        log_record = {
            "example_id": event['example_id'],
            "veracity": event['veracity'],
            "pred": pred_new,
            "old_pred": pred,
            "ruling": event["ruling"],
            "rationale": rationale,
            "relevant_evidence": evidence_dict,
            "classifier_response": classifier_response,
            "classifier_parse_valid": classifier_parse_valid,
        }
        if guard_details is not None:
            log_record.update({
                "original_prediction": guard_details["original_prediction"],
                "central_claim_status": guard_details["central_claim_status"],
                "supported_components": guard_details["supported_components"],
                "contradicted_components": guard_details["contradicted_components"],
                "insufficient_components": guard_details["insufficient_components"],
                "evidence_statuses": guard_details["evidence_statuses"],
                "consistency_check": guard_details["consistency_check"],
                "consistency_reason": guard_details["consistency_reason"],
                "guard_action": guard_details["guard_action"],
                "final_prediction": guard_details["final_prediction"],
                "prediction_changed_by_guard": (
                    guard_details["final_prediction"]
                    != guard_details["original_prediction"]
                ),
                "guard_analysis_parse_valid": guard_details["analysis_parse_valid"],
                "guard_analysis_response": guard_details["analysis_response"],
                "guard_correction_response": guard_details["correction_response"],
                "model_name": args.intent_model or os.environ.get("LLM_MODEL_PATH"),
                "ground_truth": event["veracity"],
                "prompt_configuration": guard_details["prompt_configuration"],
            })
        logger.log_iteration(log_record)

    logger.save_file("hidden_info.jsonl", hidden_info_collections)
    logger.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, help="The data file to be processed.")
    parser.add_argument("--literal", type=str, help="The data file to of literal checking.")
    parser.add_argument(
        "--intent_model",
        type=str,
        default=os.environ.get("LLM_MODEL_PATH"),
        help="Local model path (or OpenAI fine-tuned model ID when not using LLM_MODEL_PATH).",
    )
    parser.add_argument(
        "--evidence-filter",
        choices=("original", "assumption-aware"),
        default="original",
        help="Evidence selection mode; original preserves the existing TRACER behavior.",
    )
    parser.add_argument(
        "--classifier-prompt",
        choices=("original", "structured", "evidence-consistency"),
        default="original",
        help=(
            "Final classification mode; evidence-consistency checks every example, "
            "while the default preserves the original True-only TRACER path."
        ),
    )
    parser.add_argument(
        "--decision-consistency-guard",
        action="store_true",
        default=DECISION_CONSISTENCY_GUARD,
        help=(
            "Validate the original TRACER prediction against component-level "
            "evidence statuses; cannot be combined with evidence-consistency."
        ),
    )
    args = parser.parse_args()
    main(args)