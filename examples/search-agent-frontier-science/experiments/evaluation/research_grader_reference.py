# Copied verbatim in semantics from Steven's frontierscience_evaluator.py using AST.
import dataclasses
import re

RESEARCH_JUDGE_PROMPT = '\nYou are grading a science exam.\n\nYou will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.\n\nEvaluate the attempted answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric).\n\n***\nThe problem: {problem}\n***\nThe rubric: {rubric}\n***\nThe attempted answer: {answer}\n***\n\nFirst, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.\n'.strip()

@dataclasses.dataclass(frozen=True)
class FrontierScienceJudgeResult:
    points: float | None
    correct: bool
    verdict: str | float | None
    parse_error: bool

def _extract_verdict_line(text: str) -> str:
    for line in reversed(str(text).splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return ''

def parse_research_judge_output(judge_output: str) -> FrontierScienceJudgeResult:
    verdict_line = _extract_verdict_line(judge_output)
    match = re.search('VERDICT:\\s*(-?\\d+(?:\\.\\d+)?)\\s*$', verdict_line, re.I)
    if not match:
        return FrontierScienceJudgeResult(points=0.0, correct=False, verdict=None, parse_error=True)
    raw_points = float(match.group(1))
    points = min(10.0, max(0.0, raw_points))
    return FrontierScienceJudgeResult(points=points, correct=points >= 7.0, verdict=points, parse_error=False)
