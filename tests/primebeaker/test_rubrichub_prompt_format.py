import asyncio
import json
from types import SimpleNamespace

import pytest

from primebeaker.environments.jtc_rubrichub_judge_env import JTCJudgeRubric, _prompt_text


@pytest.mark.parametrize('wrap', [lambda q: q, lambda q: {'role': 'user', 'content': q},
                                 lambda q: [{'role': 'user', 'content': q}],
                                 lambda q: [SimpleNamespace(role='user', content=q)]])
def test_single_user_prompt_preserves_content(wrap):
    question = "Bachelor's degree in cognitive behavioral neuroscience\nα & β"
    assert _prompt_text(wrap(question)) == question


def test_multiturn_prompt_preserves_all_roles_and_content():
    messages = [{'role': 'system', 'content': 'Be concise.'},
                {'role': 'user', 'content': 'Tell me about neuroscience.'},
                {'role': 'assistant', 'content': 'Which degree?'},
                {'role': 'user', 'content': "Bachelor's degree."}]
    assert _prompt_text(messages) == (
        "system:\nBe concise.\n\nuser:\nTell me about neuroscience.\n\n"
        "assistant:\nWhich degree?\n\nuser:\nBachelor's degree.")


def test_judge_request_receives_plain_question():
    captured = {}

    class Judge:
        async def verify_output(self, **request):
            captured.update(request)
            return {'judgments': [{'rubric_index': 0, 'label': 'pass'}]}

    question = "Bachelor's degree in cognitive behavioral neuroscience"
    rubric = JTCJudgeRubric(judge_client=Judge(), judge_model_path='test')
    score = asyncio.run(rubric.jtc_weighted_score(
        completion=[{'role': 'assistant', 'content': 'Which aspect interests you?'}],
        prompt=[{'role': 'user', 'content': question}],
        answer=json.dumps({'record_id': 'test', 'rubrics': [{'text': 'Asks for clarification', 'weight': 1}]}),
        state={},
    ))
    assert captured['input'] == question
    assert captured['output'] == 'Which aspect interests you?'
    assert score == 1.0
