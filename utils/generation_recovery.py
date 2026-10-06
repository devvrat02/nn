"""Bounded, audited recovery for local structured generations."""
import hashlib
import os
from utils.local_llm import LocalGenerationLimitError


class StructuredGenerationError(ValueError):
    pass


def generate_structured(generate, prompt, parse, stage, instruction, audit=None, policy="bounded-reassessment-v1"):
    current = prompt
    for attempt in range(2):
        error = None
        try:
            response = generate(current)
        except LocalGenerationLimitError as exc:
            error = exc
        else:
            try:
                return parse(response)
            except (ValueError, IndexError) as exc:
                error = exc
        if os.environ.get("TRACER_BACKEND") != "local":
            raise error
        if attempt:
            raise StructuredGenerationError(f"{stage} failed after bounded recovery: {error}") from error
        entry = {"stage": stage, "policy": policy,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
        if audit is not None and entry not in audit:
            audit.append(entry)
        print(f"{stage}: {type(error).__name__}; retrying with a concise, explicit output format.", flush=True)
        current = prompt + "\n\nOUTPUT REQUIREMENT (finish once, do not repeat): " + instruction


def parse_choice(text, choices):
    answer = text.strip().strip(".()").upper()
    if answer not in choices:
        raise ValueError(f"Expected one of {choices}, received {answer!r}")
    return answer
