import ast
import re
from typing import Any

BINARY_RUBRIC_SCORES: dict[str, str] = {
    "pass": "Passes all of the requirements.",
    "fail": "Does not pass all of the requirements.",
}


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None

# Cap ``safe_literal`` evaluation of ``str * int`` folding (assert criteria parsing).
_MAX_STR_REPEAT_OPS = 10_000
_MAX_STR_REPEAT_OUTPUT_CHARS = 200_000
_MAX_SAFE_LITERAL_DEPTH = 64


def _safe_unparse(node: ast.AST, *, fallback: str = "<expr>") -> str:
    try:
        return ast.unparse(node)
    except RecursionError:
        return fallback


def inject_assert_failure_messages(code: str, message: str) -> str:
    """Rewrite top-level asserts to ``assert expr, explanation`` when msg is omitted.

    If parse fails or *message* is empty, returns *code* unchanged.
    Asserts that already specify a comma message are left as-is.
    """
    stripped = code.strip()
    if not stripped or not (message or "").strip():
        return code
    msg = message.strip()
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return code

    touched = False
    for stmt in tree.body:
        if isinstance(stmt, ast.Assert) and stmt.msg is None:
            stmt.msg = ast.Constant(value=msg)
            touched = True
    if not touched:
        return code
    return _safe_unparse(tree, fallback=code)


def add_prints_after_asserts(code: str) -> str:
    lines = []
    for line in code.splitlines():
        lines.append(line)
        stripped = line.lstrip()
        if not stripped.startswith("assert "):
            continue

        indent = line[: len(line) - len(stripped)]
        lines.append(f"{indent}print({f'passed: {stripped}'!r})")
    return "\n".join(lines)


def safe_literal(node: ast.AST, *, _depth: int = 0) -> Any:
    if _depth > _MAX_SAFE_LITERAL_DEPTH:
        return _safe_unparse(node)

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [safe_literal(item, _depth=_depth + 1) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(safe_literal(item, _depth=_depth + 1) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            safe_literal(key, _depth=_depth + 1): safe_literal(value, _depth=_depth + 1)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left = safe_literal(node.left, _depth=_depth + 1)
        right = safe_literal(node.right, _depth=_depth + 1)
        if isinstance(left, str) and isinstance(right, int):
            if (
                0 <= right <= _MAX_STR_REPEAT_OPS
                and len(left) * right <= _MAX_STR_REPEAT_OUTPUT_CHARS
            ):
                return left * right
            return _safe_unparse(node)
        if isinstance(left, int) and isinstance(right, str):
            if (
                0 <= left <= _MAX_STR_REPEAT_OPS
                and len(right) * left <= _MAX_STR_REPEAT_OUTPUT_CHARS
            ):
                return left * right
            return _safe_unparse(node)
    return _safe_unparse(node)


def parse_assertion_snippet(snippet: Any) -> dict[str, Any]:
    if not isinstance(snippet, str):
        raise TypeError(
            f"parse_assertion_snippet expected str, got {type(snippet).__name__}; "
            "normalize ground_truth snippets to strings before calling."
        )
    text = snippet.strip()
    try:
        return _parse_assertion_snippet_text(text)
    except (RecursionError, SyntaxError):
        msg = text or "Assertion failed."
        return binary_code_rubric(
            add_prints_after_asserts(inject_assert_failure_messages(text, msg)),
            criteria=text,
        )


def _parse_assertion_snippet_text(text: str) -> dict[str, Any]:
    tree = ast.parse(text)
    stmt = tree.body[0]

    if not isinstance(stmt, ast.Assert):
        msg = text or "Assertion failed."
        return binary_code_rubric(
            add_prints_after_asserts(inject_assert_failure_messages(text, msg)),
            criteria=text,
        )

    test = stmt.test
    if not (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Call)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
    ):
        rubric = binary_code_rubric(
            add_prints_after_asserts(inject_assert_failure_messages(text, text)),
            criteria=text,
        )
        rubric["name"] = "check"
        return rubric

    call = test.left
    function_name = _safe_unparse(call.func, fallback="check")
    args = [safe_literal(arg) for arg in call.args]
    kwargs = {kw.arg: safe_literal(kw.value) for kw in call.keywords if kw.arg is not None}
    expected = safe_literal(test.comparators[0])

    criteria = (
        f"Run {function_name} with arguments "
        f"{tuple(args)!r}"
        f"{f' and keyword arguments {kwargs!r}' if kwargs else ''}: "
        f"should return {expected!r}."
    )

    annotated = inject_assert_failure_messages(text, criteria)

    return {
        "criteria": criteria,
        "code": add_prints_after_asserts(annotated),
        "kwargs": kwargs,
        "feedback": None,
        "scores": BINARY_RUBRIC_SCORES,
        "label": None,
    }


def binary_code_rubric(code: str, *, criteria: str | None = None) -> dict[str, Any]:
    return {
        "criteria": criteria or code,
        "code": code,
        "feedback": None,
        "scores": BINARY_RUBRIC_SCORES,
        "label": None,
    }


def prepend_output_code(output_code: str | None, check_code: str) -> str:
    if output_code:
        return f"{output_code}\n\n{check_code}"
    return check_code


def _strip_code_fence(text: str) -> str:
    # Find the opening fence (```python, ```py, or plain ```) anywhere in the
    # text, then locate the closing fence using str.rfind("```") rather than a
    # non-greedy regex.
    #
    # WHY search anywhere, not just at the start:
    # When the model wraps code inside <code>...</code>, it sometimes includes
    # a brief preamble before the fence, e.g.:
    #
    #   "Here is the solution:\n```python\ndef check(...)..."
    #
    # Using re.match (anchored to the start) misses those cases and returns the
    # raw content — which still contains ``` — causing HasCleanDefinitionOutput
    # to reject the extraction as "wrapper syntax present".
    #
    # WHY rfind FOR THE CLOSING FENCE:
    # Generated code frequently contains triple-backtick literals inside Python
    # string arguments — e.g. a regex pattern like
    #
    #   code_blocks = re.findall(r'```(?:cpp)?\n(.*?)```', response, ...)
    #
    # A non-greedy pattern (.*?)``` would stop at the ``` inside the string,
    # returning a truncated snippet.  rfind anchors to the LAST ```, which is
    # always the real closing fence.
    opening = re.search(r"```(?:python|py)?\s*\n?", text, flags=re.IGNORECASE)
    if opening:
        inner = text[opening.end():]
        last_fence = inner.rfind("```")
        if last_fence != -1:
            return inner[:last_fence].strip()

    # Fallback: no opening fence found — return the text as-is (already
    # stripped of the outer wrapper by the caller).
    return text.strip()


def extract_python_code(text: Any) -> str | None:
    text = clean_text(text)
    if text is None:
        return None

    # Use rfind to locate the LAST <code> tag.  The model's thinking section
    # often contains literal "<code>" in phrases like "produce the answer with
    # <code> tags" — a re.findall would start matching from that first stray
    # tag and return the entire thinking block as the "code", not the real
    # code block near the end.  rfind skips all such earlier occurrences.
    text_lower = text.lower()
    last_open = text_lower.rfind("<code>")
    if last_open != -1:
        # Find the </code> that closes this last <code>
        close_pos = text_lower.find("</code>", last_open)
        if close_pos != -1:
            content = text[last_open + len("<code>") : close_pos].strip()
            return _strip_code_fence(content)

    # No <code> block found — look for the last ```python / ```py / ``` fence.
    # Again use rfind so we start from the rightmost opening fence, avoiding
    # fence markers that appear inside the model's reasoning text.
    for opener in ("```python\n", "```py\n", "```\n"):
        pos = text.rfind(opener)
        if pos == -1:
            continue
        inner_start = pos + len(opener)
        # The closing fence is the last ``` that comes after the opening.
        close_pos = text.rfind("```", inner_start)
        if close_pos > inner_start:
            return text[inner_start:close_pos].strip()

    return None
