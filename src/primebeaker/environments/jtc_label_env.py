"""One-step Verifiers environment for JTC label RL.

Install/copy this as a Prime-RL `verifiers` environment and point the RL config
at its environment id.

Reward:
    format: model emits non-empty <feedback>...</feedback> and <label>...</label>
    correctness: model <label>...</label> exactly matches ground-truth answer
"""

import json

from datasets import Dataset
import verifiers as vf

from .jtc_label_reward import exact_label_reward, format_reward


def _load_jsonl(path, split):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row.setdefault("split", split)
            rows.append(row)
    return Dataset.from_list(rows)


def load_environment(dataset, split="train"):
    rows = _load_jsonl(dataset, split)

    async def valid_format(completion, answer):
        completion_text = completion[-1].get("content", "") if completion else ""
        return format_reward(completion_text)

    async def correct_label(completion, answer):
        completion_text = completion[-1].get("content", "") if completion else ""
        return exact_label_reward(completion_text, answer)

    rubric = vf.Rubric(funcs=[valid_format, correct_label])
    return vf.SingleTurnEnv(dataset=rows, rubric=rubric)
