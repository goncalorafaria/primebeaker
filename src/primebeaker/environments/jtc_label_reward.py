"""Utilities for verifiable JTC label rewards."""

import re

from primebeaker.runtime.code_utils import extract_python_code


LABEL_RE = re.compile(r"<label>\s*(.*?)\s*</label>", re.IGNORECASE | re.DOTALL)
FEEDBACK_RE = re.compile(
    r"<feedback>\s*(.*?)\s*</feedback>",
    re.IGNORECASE | re.DOTALL,
)
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*</output>", re.IGNORECASE | re.DOTALL)
CODE_RE = re.compile(r"<code>\s*(.*?)\s*</code>", re.IGNORECASE | re.DOTALL)
PYTHON_FENCE_RE = re.compile(
    r"<code>\s*```(?:python|py)\s*\n.+?\n```\s*</code>",
    re.IGNORECASE | re.DOTALL,
)


def normalize_label(label):
    """Normalize labels for exact ground-truth comparison."""
    if label is None:
        return None
    return " ".join(str(label).strip().lower().split())


def extract_label(text):
    """Return normalized label text from <label>...</label>, or None."""
    if not text:
        return None
    match = LABEL_RE.search(text)
    if not match:
        return None
    label = normalize_label(match.group(1))
    return label or None


def extract_feedback(text):
    """Return feedback text from <feedback>...</feedback>, or None."""
    if not text:
        return None
    match = FEEDBACK_RE.search(text)
    if not match:
        return None
    feedback = match.group(1).strip()
    return feedback or None


def has_exactly_one_label(text):
    """Return True iff text has exactly one non-empty <label>...</label>."""
    if not text:
        return False
    matches = LABEL_RE.findall(text)
    return len(matches) == 1 and bool(normalize_label(matches[0]))


def has_exactly_one_feedback(text):
    """Return True iff text has exactly one non-empty <feedback>...</feedback>."""
    if not text:
        return False
    matches = FEEDBACK_RE.findall(text)
    return len(matches) == 1 and bool(matches[0].strip())


def has_exactly_one_code(text):
    """Return True iff text has exactly one valid Python <code> block."""
    if not text:
        return False
    code_tags = CODE_RE.findall(text)
    if len(code_tags) != 1:
        return False
    return bool(PYTHON_FENCE_RE.fullmatch(text.strip()) and extract_python_code(text))


def extract_task_output(messages):
    """Extract the judged model output from the user prompt, if present."""
    for message in messages or []:
        if message.get("role") != "user":
            continue
        match = OUTPUT_RE.search(message.get("content") or "")
        if match:
            output = match.group(1).strip()
            return output or None
    return None


def format_reward(completion_text):
    """Reward 1.0 iff exactly one non-empty feedback and label tag appear."""
    if not has_exactly_one_feedback(completion_text):
        return 0.0
    if not has_exactly_one_label(completion_text):
        return 0.0
    return 1.0


def code_format_reward(completion_text):
    """Reward 1.0 iff completion has exactly one Python <code> block."""
    return 1.0 if has_exactly_one_code(completion_text) else 0.0


def exact_label_reward(completion_text, answer):
    """Reward 1.0 iff completion label exactly agrees with ground-truth answer."""
    if not format_reward(completion_text):
        return 0.0
    predicted = extract_label(completion_text)
    expected = normalize_label(answer)
    if expected is None:
        return 0.0
    return 1.0 if predicted == expected else 0.0


def split_sft_row_for_label_rl(row):
    """Convert one Prime-RL SFT row to a verifiable RL row.

    Returns {"prompt": first_two_messages, "answer": label} or None.
    """
    messages = row.get("messages") or []
    if len(messages) < 2:
        return None

    answer = None
    for message in messages:
        if message.get("role") == "assistant":
            answer = extract_label(message.get("content", "")) or answer

    if answer is None:
        return None

    return {
        "prompt": messages[:2],
        "answer": answer,
        "output": extract_task_output(messages),
    }
