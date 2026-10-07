"""Bounded, audited recovery for local structured generations."""
import hashlib
import os
import ast
import re
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


def parse_choice(text, choices, audit=None, stage="choice"):
    answer = text.strip().strip(".()").upper()
    if len(answer) == 1 and answer in choices:
        return answer
    # A base completion model may wrap a literal option in a Python print.
    # Inspect syntax only: no eval/exec, calls, or expressions are evaluated.
    fenced = re.fullmatch(r"\s*```python\s*\n(.*?)\n```\s*", text, re.S | re.I)
    if fenced:
        try:
            body = ast.parse(fenced.group(1)).body
        except (SyntaxError, ValueError):
            body = []
        if len(body) == 1 and isinstance(body[0], ast.Expr):
            call = body[0].value
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "print" and len(call.args) == 1 and not call.keywords
                    and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str)):
                option = call.args[0].value
                if len(option) == 1 and option in choices:
                    if audit is not None:
                        audit.append({"stage": stage, "policy": "literal-print-choice-v1",
                                      "choice": option,
                                      "response_sha256": hashlib.sha256(text.encode()).hexdigest()})
                    return option
    raise ValueError(f"Expected one of {choices}, received {answer!r}")
