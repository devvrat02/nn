import os
import json
import re
from utils.utils import *
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder
from scipy.special import softmax
import random
import time
from method.prompts import *
from intent_generation.prompts import *


question_num = 4
assumption_num = 5
evidence_num = 3


class IntentArgumentation:
    def __init__(self, 
                 intent_model_id,
                 question_model="gpt-4o-mini", 
                 assumption_model="gpt-4o-mini",
                 classifier_model="gpt-4o-mini",
                 evidence_filter="original"):
        
        model_mapping = {
            "gpt-4o-mini": GPT4oMiniWrapper,
            "Qwen/Qwen2.5-3B-Instruct": QwenModelWrapper
        }
        
        self.intent_model = intent_model_id
        if evidence_filter not in ("original", "assumption-aware"):
            raise ValueError(f"Unknown evidence filter: {evidence_filter}")
        self.evidence_filter = evidence_filter
        self.question_generator = model_mapping.get(question_model, GPT4oMiniWrapper)()
        self.assumption_generator = model_mapping.get(assumption_model, GPT4oMiniWrapper)()
        auxiliary_device = os.environ.get("AUXILIARY_MODEL_DEVICE")
        if auxiliary_device is None and os.environ.get("LLM_MODEL_PATH"):
            auxiliary_device = "cpu"
        self.relevance_ranker = SentenceTransformer(
            'all-MiniLM-L6-v2',
            device=auxiliary_device,
        )
        self.nli_reranker = CrossEncoder(
            'cross-encoder/nli-deberta-v3-large',
            device=auxiliary_device,
        )
        self.classifier = model_mapping.get(classifier_model, GPT4oMiniWrapper)()
        print("Ranker device:", self.relevance_ranker.device)
        print("Reranker device:", self.nli_reranker.model.device)
    

    def _generate_intent(self, event):
        claim = event['claim']
        evidence = event['evidence']
        evidence = [evi for evi, label in zip(event["evidence"], event["annotation"]) if label != 0]
        evidence = '\n'.join(event['evidence'])

        inputs = FINETUNE_TEMPLATE_TASK.replace("{claim}", claim).replace("{evidence}", evidence)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": inputs
            },
        ]

        max_retries = 3
        retries = 0
        last_error = None
        while retries < max_retries:
            try:
                response = completion_finetune(self.intent_model, messages)
                intent = self._extract_content(response)
                break
            except Exception as e:
                print(e)
                last_error = e
                retries += 1
        else:
            raise RuntimeError("Intent generation failed after three attempts.") from last_error

        return intent

    def _extract_content(self, input_string):
        pattern = re.compile(r"<([^>]+)>")
        matches = pattern.findall(input_string)
        if matches:
            return matches[0].strip()

        labeled_intent = re.search(
            r"(?im)^(?:the intended conclusion(?: of the claim)?|intent)\s*:\s*(.+)$",
            input_string,
        )
        if labeled_intent:
            return labeled_intent.group(1).strip()

        for prefix in (
            r"(?i)^intent:",
            r"(?i)^intentded understanding:",
            r"(?i)^intentded conclusion:",
        ):
            input_string = re.sub(prefix, "", input_string.strip(), count=1)

        lines = [line.strip() for line in input_string.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Intent response was empty.")
        return lines[-1]
    def _extract_answer(self, input_string):
        # Use regex to find all matches between angle brackets and trim extra whitespace
        matches = re.findall(r'<\s*(.*?)\s*>', input_string)
        text = matches[0].split("||")
        return text

    def _extract_question(self, input_string):
        # Use regex to find all matches between angle brackets and trim extra whitespace
        matches = re.findall(r'<\s*(.*?)\s*>', input_string)
        return matches

    def implicit_assumption(self, event):
        claim = event['claim']
        intent = event['intent']
        implicit_question = event['question']
        implicit_question_string = '\n'.join(
            [
                f"{i+1}. {implicit_question[i]}" for i in range(len(implicit_question))
            ]
        )
        global assumption_num
        prompt = PROMPT_IMPLICIT_ASSUMPTION.format(claim=claim, intention=intent, questions=implicit_question_string,
                                                   assumption_max_number=assumption_num)
        response = self.assumption_generator(prompt)
        assumptions = self._extract_answer(response)
        return assumptions
    
    def implicit_question(self, event):
        global question_num
        evidence = '\n'.join(event['evidence'])
        claim = event['claim']
        intent = event['intent']

        prompt = PROMPT_IMPLICIT_QUESTION_HYPER.format(claim=claim, intent=intent, evidence=evidence, question_num=question_num)
        response = self.question_generator(prompt)
        questions = self._extract_question(response)
        return questions
    
    def _cross_encoder_pipeline(self, premises, hypotheses):
        inputs = [(premise, hypotheses) for premise in premises]
        label_mapping = ['contradiction', 'entailment', 'neutral']
        scores = self.nli_reranker.predict(inputs)
        labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
        softmax_scores = softmax(scores, axis=1)

        label_indices = scores.argmax(axis=1)
        final_scores = [softmax_scores[i][label_indices[i]] for i in range(len(label_indices))]

        return labels, final_scores
    
    def ranking_top_k_evidence(self, assumption, sentences_key, relevant_evidence_dict, type, k=3):
        assumption_embedding = self.relevance_ranker.encode(assumption)
        evidence_similarities = []
        for key in sentences_key:
            evidence_text = relevant_evidence_dict[key]
            evidence_embedding = self.relevance_ranker.encode(evidence_text)
            similarity = cosine_similarity([assumption_embedding], [evidence_embedding])[0][0]
            evidence_similarities.append((key, evidence_text, similarity))
        premises = [text for _, text, _ in evidence_similarities]
        labels, _ = self._cross_encoder_pipeline(premises, assumption)
        filtered_evidence = [(key, text, sim) for (key, text, sim), label in zip(evidence_similarities, labels) if label == type]
        filtered_evidence = [(key, sim) for key, _, sim in filtered_evidence if sim >= 0.45]
        filtered_evidence.sort(key=lambda x: x[1], reverse=True)
        top_k_keys = [key for key, _ in filtered_evidence[:k]]
        return top_k_keys
    

    def _exist_hidden(self, input_string):
        input_string = input_string[:2]
        if "A" in input_string or "B" in input_string or "C" in input_string:
            return False
        elif "D" in input_string or "E" in input_string:
            return True
        else:
            return random.choice([True, False])

    def hidden_info_mining(self, event):
        if self.evidence_filter == "assumption-aware":
            return self.assumption_aware_evidence_filtering(event)
        
        relevant_evidence_dict = event["relevant_evidence"]
        hidden_evidence_key = []
        for evi, pred in zip(event['relevant_evidence'], event['prediction']):
            if pred == 1:
                hidden_evidence_key.append(evi)

        argument = {
            "intent": event['intent'],
            "claim": event['claim'],
            "assumption":[]
        }

        if len(hidden_evidence_key) == 0:
            return argument
        
        global evidence_num
        
        for i, assumption in enumerate(event['assumption']):
            refuted_by = self.ranking_top_k_evidence(assumption, hidden_evidence_key, relevant_evidence_dict, "contradiction", evidence_num)
            supported_by = self.ranking_top_k_evidence(assumption, hidden_evidence_key, relevant_evidence_dict, "entailment", evidence_num)
            backing = supported_by + refuted_by
            if not backing:
                continue
            argument["assumption"].append(
                {
                    "content": assumption,
                    "supported_by": supported_by,
                    "refuted_by": refuted_by,
                }
            )

        return argument

    def assumption_aware_evidence_filtering(self, event):
        relevant_evidence = event.get("relevant_evidence") or {}
        predictions = event.get("prediction") or []
        assumptions = event.get("assumption") or []
        argument = {
            "intent": event.get("intent", ""),
            "claim": event.get("claim", ""),
            "assumption": [],
        }
        filter_log = []
        hidden_candidates = []
        seen_texts = set()

        for key, prediction in zip(relevant_evidence, predictions):
            evidence_text = relevant_evidence[key]
            if prediction != 1:
                filter_log.append({
                    "evidence_key": key,
                    "decision": "excluded",
                    "reason": "alignment model marked evidence as presented",
                })
                continue
            if not isinstance(evidence_text, str) or not evidence_text.strip():
                filter_log.append({
                    "evidence_key": key,
                    "decision": "excluded",
                    "reason": "evidence text is empty or malformed",
                })
                continue
            normalized_text = " ".join(evidence_text.casefold().split())
            if normalized_text in seen_texts:
                filter_log.append({
                    "evidence_key": key,
                    "decision": "excluded",
                    "reason": "duplicate evidence text",
                })
                print(
                    f"[assumption-aware] {key}: excluded duplicate evidence text.",
                    flush=True,
                )
                continue
            seen_texts.add(normalized_text)
            hidden_candidates.append((key, evidence_text))

        if not isinstance(assumptions, list):
            print(
                "[assumption-aware] malformed assumptions; no evidence was "
                "attached to assumptions.",
                flush=True,
            )
            assumptions = []

        for assumption_index, assumption in enumerate(assumptions):
            if not isinstance(assumption, str) or not assumption.strip():
                print(
                    f"[assumption-aware] assumption {assumption_index + 1}: "
                    "skipped empty or malformed assumption.",
                    flush=True,
                )
                continue
            if not hidden_candidates:
                print(
                    f"[assumption-aware] assumption {assumption_index + 1}: "
                    "no hidden evidence available.",
                    flush=True,
                )
                continue

            keys = [key for key, _ in hidden_candidates]
            texts = [text for _, text in hidden_candidates]
            embeddings = self.relevance_ranker.encode([assumption, *texts])
            similarities = cosine_similarity(
                [embeddings[0]], embeddings[1:]
            )[0]
            labels, confidences = self._cross_encoder_pipeline(texts, assumption)
            if len(labels) != len(texts) or len(confidences) != len(texts):
                raise RuntimeError(
                    "NLI model returned a different number of relationships than "
                    "the evidence candidates."
                )

            supported_by = []
            refuted_by = []
            neutral = []
            relationships = []
            for key, label, similarity, confidence in zip(
                keys, labels, similarities, confidences
            ):
                if label == "entailment":
                    relationship = "supports"
                    supported_by.append(key)
                    retain = True
                    reason = "NLI supports assumption"
                elif label == "contradiction":
                    relationship = "contradicts"
                    refuted_by.append(key)
                    retain = True
                    reason = "NLI contradicts assumption; retained despite conflict"
                elif label == "neutral":
                    relationship = "neutral/insufficient"
                    retain = float(similarity) >= 0.45
                    reason = (
                        "NLI neutral and similarity meets relevance threshold"
                        if retain
                        else "NLI neutral and below relevance threshold"
                    )
                    if retain:
                        neutral.append(key)
                else:
                    raise ValueError(f"Unexpected NLI label: {label!r}")

                relationships.append({
                    "evidence_key": key,
                    "relationship": relationship,
                    "similarity": float(similarity),
                    "confidence": float(confidence),
                    "retained": retain,
                    "reason": reason,
                })

            retained = supported_by + refuted_by + neutral
            print(
                f"[assumption-aware] assumption {assumption_index + 1}: "
                f"{len(supported_by)} support, {len(refuted_by)} contradict, "
                f"{len(neutral)} neutral retained; "
                f"{len(hidden_candidates) - len(retained)} irrelevant removed.",
                flush=True,
            )
            if retained:
                argument["assumption"].append({
                    "content": assumption,
                    "supported_by": supported_by,
                    "refuted_by": refuted_by,
                    "neutral": neutral,
                    "evidence_relationships": relationships,
                })
            filter_log.append({
                "assumption": assumption,
                "relationships": relationships,
            })

        event["evidence_filter_log"] = filter_log
        return argument
    
    def counterfactual_evaluation(self, event):
        claim = event['claim']
        assumption = event['assumption']
        if isinstance(event["intent"], list):
            intent = event["intent"]
        else:
            intent = event['intent']

        argument = {
            "Z": intent,
            "linked_by":{
                "X": claim
            }
        }
        for i, assume in enumerate(assumption):
            if self.evidence_filter == "assumption-aware":
                assumption_evidence = next(
                    (
                        item
                        for item in (event.get("argument") or {}).get("assumption", [])
                        if item.get("content") == assume
                    ),
                    {},
                )
                argument["linked_by"][f"Y{i+1}"] = {
                    "assumption": assume,
                    "evidence_relationships": assumption_evidence.get(
                        "evidence_relationships", []
                    ),
                }
            else:
                argument["linked_by"][f"Y{i+1}"] = assume
        valid_assumption = []
        for i, assume in enumerate(assumption):
            letter = f"Y{i+1}"
            query = f"Evaluate △P(Z|do({letter}=¬{letter})). More specifically, how does the probability " \
            f"of Z change when we set {letter} from {letter} to ¬{letter}?"
            prompt = PROMPT_CAUSAL_EVAL.format(argument=json.dumps(argument, indent=4), query=query)
            ans = call_gpt(prompt)
            if ans == "B" or ans == "C":
                valid_assumption.append(assume)
        event["assumption"] = valid_assumption
    

    def mining_one_event(self, event):
        max_retries = 3
        retries = 0
        last_error = None
        while retries < max_retries:
            try:
                intent = self._generate_intent(event)
                event["intent"] = intent
                questions = self.implicit_question(event)
                event["question"] = questions
                assumptions = self.implicit_assumption(event)
                event["assumption"] = assumptions
                if self.evidence_filter == "original":
                    self.counterfactual_evaluation(event)
                    argument = self.hidden_info_mining(event)
                else:
                    argument = self.hidden_info_mining(event)
                    event["argument"] = argument
                    self.counterfactual_evaluation(event)
                    valid_assumptions = set(event["assumption"])
                    argument["assumption"] = [
                        item
                        for item in argument["assumption"]
                        if item["content"] in valid_assumptions
                    ]
                event["argument"] = argument
                break
            except Exception as e:
                print(e)
                last_error = e
                retries += 1
                time.sleep(1)
        else:
            raise RuntimeError("Hidden-information mining failed after three attempts.") from last_error

        return event
