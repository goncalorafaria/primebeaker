"""Signed criterion scoring: pass means the described behavior is present."""
import math


def nonzero_weight(value, *, field='weight'):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{field} must be numeric')
    weight = float(value)
    if not math.isfinite(weight) or weight == 0:
        raise ValueError(f'{field} must be finite and nonzero')
    return weight


def scoring_weight(value, *, mode="weighted"):
    """Scoring only: retain original magnitude or use its sign; sampling is independent."""
    weight = nonzero_weight(value)
    if mode == "weighted":
        return weight
    if mode == "sign_only":
        return 1.0 if weight > 0 else -1.0
    raise ValueError("rubric_scoring must be 'weighted' or 'sign_only'")


def weight_normalizer(rubrics):
    total = sum(max(0.0, nonzero_weight(r['weight'])) for r in rubrics)
    if total == 0:
        raise ValueError('Reward normalization requires at least one positive-weight rubric')
    return total


def judge_criterion(rubric):
    text = str(rubric['text'])
    if rubric['weight'] < 0:
        return ('This is a penalty condition. Return pass ONLY if the described error or bad behavior '
                'is present in the response; return fail if it is absent. Pass here means the '
                'penalty applies, not that the response is good. Condition: ' + text)
    return text


def signed_sample_indexes(rubrics, limit, rng, *, sampling="uniform"):
    """Sample without replacement; weighted draws use absolute signed weights."""
    if sampling not in {"uniform", "weighted"}:
        raise ValueError("rubric_sampling must be 'uniform' or 'weighted'")
    if sampling == "uniform":
        return sorted(rng.sample(range(len(rubrics)), limit))
    if not 0 <= limit <= len(rubrics):
        raise ValueError("sample size must be between zero and the rubric count")
    remaining = list(range(len(rubrics)))
    weights = [abs(nonzero_weight(r['weight'])) for r in rubrics]
    selected = []
    for _ in range(limit):
        # Scaling avoids overflowing the sum for large finite weights.
        scale = max(weights)
        position = rng.choices(range(len(remaining)), weights=[w / scale for w in weights], k=1)[0]
        selected.append(remaining.pop(position))
        weights.pop(position)
    return sorted(selected)
