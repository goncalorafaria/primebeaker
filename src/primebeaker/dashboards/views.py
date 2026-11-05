"""Explicit view registration; shared sources do not depend on view internals."""
from importlib.resources import files


def get_view(kind: str):
    if kind == "rubric-judging":
        from .rubric_judging import CURVES
        return files("primebeaker.dashboards.rubric_judging").joinpath("assets"), CURVES
    raise ValueError(f"Unknown dashboard kind {kind!r}; supported: rubric-judging")
