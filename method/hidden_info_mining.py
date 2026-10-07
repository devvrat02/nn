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
from utils.generation_recovery import generate_structured, parse_choice, StructuredGenerationError
from utils.local_llm import LocalGenerationLimitError


question_num = 4
assumption_num = 5
evidence_num = 3


class IntentArgumentation:
    def __init__(self, 
                 intent_model_id,
                 question_model="gpt-4o-mini", 
                 assumption_model="gpt-4o-mini",
                 classifier_model="gpt-4o-mini"):
        
        model_mapping = {
            "gpt-4o-mini": GPT4oMiniWrapper,
            "Qwen/Qwen2.5-3B-Instruct": QwenModelWrapper
        }
        
        self.intent_model = intent_model_id
        self.question_generator = model_mapping.get(question_model, GPT4oMiniWrapper)()
        self.assumption_generator = model_mapping.get(assumption_model, GPT4oMiniWrapper)()
        ranker_device = os.environ.get("TRACER_RANKER_DEVICE")
        self.relevance_ranker = SentenceTransformer(os.environ.get("TRACER_RANKER_MODEL", 'all-MiniLM-L6-v2'), device=ranker_device)
        self.nli_reranker = CrossEncoder(os.environ.get("TRACER_NLI_MODEL", 'cross-encoder/nli-deberta-v3-large'), device=ranker_device, max_length=512)
        self.classifier = model_mapping.get(classifier_model, GPT4oMiniWrapper)()
        print("Ranker device:", self.relevance_ranker.device)
        print("Reranker device:", self.nli_reranker.model.device)
    

    def _generate_intent(self, event):
        claim = event['claim']
        evidence = '\n'.join(evi for evi, label in zip(event["evidence"], event["prediction"]) if label == 1)

        inputs = FINETUNE_TEMPLATE_TASK.replace("{claim}", claim).replace("{evidence}", evidence)
        if os.environ.get("TRACER_BACKEND") == "local":
            inputs += "\nOutput one implicit intended conclusion inside angle brackets: <intended conclusion>."
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

        def generate(text):
            return completion_finetune(self.intent_model, [messages[0], {"role": "user", "content": text}])
        return generate_structured(generate, inputs, self._extract_content, "intent",
                                   "Output just one short sentence inside <...>, under 60 words. No explanation.",
                                   event.setdefault("generation_recoveries", []))

    def _extract_content(self, input_string):
        patterns = [
            r"(?i)^intent:",
            r"(?i)^intentded understanding:",
            r"(?i)^intentded conclusion:"
        ]
        
        for pattern in patterns:
            input_string = re.sub(pattern, "", input_string, count=1).strip()

        pattern = re.compile(r"<([^>]+)>")
        matches = pattern.findall(input_string)
        if not matches or not matches[0].strip():
            raise ValueError("Missing nonempty <intent>")
        return matches[0].strip()
    def _extract_answer(self, input_string):
        # Use regex to find all matches between angle brackets and trim extra whitespace
        matches = re.findall(r'<\s*(.*?)\s*>', input_string)
        if not matches or not matches[0].strip():
            raise ValueError("Missing nonempty <assumption||assumption> block")
        text = [value.strip() for value in matches[0].split("||")]
        if any(not value for value in text) or len(text) > assumption_num:
            raise ValueError("Expected one to five nonempty assumptions")
        return text

    def _extract_question(self, input_string):
        # Use regex to find all matches between angle brackets and trim extra whitespace
        matches = re.findall(r'<\s*(.*?)\s*>', input_string)
        if not matches or any(not value.strip() for value in matches) or len(matches) > question_num:
            raise ValueError("Expected one to four nonempty bracketed questions")
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
        return generate_structured(self.assumption_generator, prompt, self._extract_answer, "assumptions",
                                   "Output one to five distinct short assumptions in a SINGLE bracketed block: "
                                   "<first assumption||second assumption>. At most 120 words total. "
                                   "No rationale, empty brackets, repeated assumptions, or text outside the block.",
                                   event.setdefault("generation_recoveries", []))
    
    def implicit_question(self, event):
        global question_num
        evidence = '\n'.join(event['evidence'])
        claim = event['claim']
        intent = event['intent']

        prompt = PROMPT_IMPLICIT_QUESTION_HYPER.format(claim=claim, intent=intent, evidence=evidence, question_num=question_num)
        return generate_structured(self.question_generator, prompt, self._extract_question, "questions",
                                   "Output one to four short yes/no questions, each inside its own <...>. "
                                   "At most 100 words total. No explanation or repetition.",
                                   event.setdefault("generation_recoveries", []))
    
    def _cross_encoder_pipeline(self, premises, hypotheses):
        inputs = [(premise, hypotheses) for premise in premises]
        label_mapping = ['contradiction', 'entailment', 'neutral']
        scores = self.nli_reranker.predict(inputs, batch_size=4, show_progress_bar=False)
        labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
        softmax_scores = softmax(scores, axis=1)

        label_indices = scores.argmax(axis=1)
        final_scores = [softmax_scores[i][label_indices[i]] for i in range(len(label_indices))]

        return labels, final_scores
    
    def ranking_top_k_evidence(self, assumption, sentences_key, relevant_evidence_dict, type, k=3):
        if not sentences_key:
            return []
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
            argument["linked_by"][f"Y{i+1}"] = assume
        valid_assumption = []
        for i, assume in enumerate(assumption):
            letter = f"Y{i+1}"
            query = f"Evaluate △P(Z|do({letter}=¬{letter})). More specifically, how does the probability " \
            f"of Z change when we set {letter} from {letter} to ¬{letter}?"
            prompt = PROMPT_CAUSAL_EVAL.format(argument=json.dumps(argument, indent=4), query=query)
            ans = generate_structured(call_gpt, prompt, lambda text: parse_choice(
                text, "ABC", event.setdefault("generation_recoveries", []), "counterfactual"), "counterfactual",
                                      "Output exactly one letter: A, B, or C. No explanation.",
                                      event.setdefault("generation_recoveries", []))
            if ans == "B" or ans == "C":
                valid_assumption.append(assume)
        event["assumption"] = valid_assumption
    

    def mining_one_event(self, event):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                intent = self._generate_intent(event)
                event["intent"] = intent
                questions = self.implicit_question(event)
                event["question"] = questions
                assumptions = self.implicit_assumption(event)
                event["assumption"] = assumptions
                self.counterfactual_evaluation(event)
                argument = self.hidden_info_mining(event)
                event["argument"] = argument
                break
            except (StructuredGenerationError, LocalGenerationLimitError):
                raise
            except Exception as e:
                print(e)
                retries += 1
                time.sleep(1)
        if "argument" not in event:
            raise RuntimeError(f"Hidden evidence mining failed for {event.get('example_id')} after three attempts.")
        return event
