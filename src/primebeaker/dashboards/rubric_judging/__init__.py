"""Rubric-judging view: the first consumer of the shared dashboard API."""

CURVES = {
    "time_per_step": {"metric": "time/step", "unit": "seconds", "requires": ["time/forward_backward"]},
    "reward_mean": {"metric": "train/agg/all/agent/reward/mean", "unit": "reward"},
    "total_tokens_mean": {"metric": "train/agg/all/num_total_tokens/mean", "unit": "tokens"},
}
