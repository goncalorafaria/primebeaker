"""Core logic for a two-step JTC code-tool RL environment.

The rollout policy is:

1. The model may answer directly:
       <feedback>...</feedback>
       <label>...</label>

2. Or it may request one Python execution:
       <code>```python
       ...
       ```</code>

   The environment executes that code on the HTTP code server, appends a tool
   response, then lets the model produce the final feedback + label.

The exact Verifiers harness wrapper can stay thin around these functions.
"""


from .jtc_label_reward import (
    exact_label_reward,
    format_reward,
    code_format_reward,
)


DEFAULT_CODE_SERVER_URL = "http://127.0.0.1:1212/python"


def normalize_truncation_limit(limit):
    """Coerce env/TOML values for optional code output truncation."""
    if limit is None or limit == "None":
        return None
    return int(limit)


def classify_response(text):
    """Classify a model response as 'code', 'final', or 'invalid'."""
    if code_format_reward(text):
        return "code"
    if format_reward(text):
        return "final"
    return "invalid"


def first_turn_format_reward(text):
    """Reward direct final format or valid code-request format."""
    return 1.0 if classify_response(text) in ("code", "final") else 0.0


def final_turn_format_reward(text):
    """Reward final feedback + label format."""
    return format_reward(text)


def final_label_reward(text, answer):
    """Reward exact final-label agreement with ground truth."""
    return exact_label_reward(text, answer)
