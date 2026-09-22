"""Signed criterion scoring: pass means the described behavior is present."""
import math


def nonzero_weight(value, *, field='weight'):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f'{field} must be numeric')
    weight = float(value)
    if not math.isfinite(weight) or weight == 0:
        raise ValueError(f'{field} must be finite and nonzero')
    return weight


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


def signed_sample_indexes(rubrics, limit, rng):
    """Sample without replacement, independent of rubric sign or weight."""
    return sorted(rng.sample(range(len(rubrics)), limit))
