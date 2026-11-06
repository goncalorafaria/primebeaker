import copy
import json
from pathlib import Path
from web_quality import reject_challenge_page

def test_saved_long_challenge():
    audit = Path('/weka/gfaria/search-agent-repr/analysis/step400-web-v2-review/fresh29.json')
    entry = json.loads(audit.read_text())[15]
    response = json.loads(Path(entry['cache_path']).read_text())['response']
    original = copy.deepcopy(response)
    assert len(response['data']['content']) == 64000
    rejected = reject_challenge_page(response)
    assert rejected['success'] is False and rejected['data'] is None
    assert response == original  # cache contents must not be mutated
    # Still reject if extra scripts make the page even longer.
    padded = copy.deepcopy(response)
    padded['data']['content'] += 'var script = 1; ' * 10000
    assert not reject_challenge_page(padded)['success']
    # An article quoting these strings should not be rejected by the long rule.
    article = copy.deepcopy(response)
    article['data']['title'] = 'Analysis of CAPTCHA interfaces'
    assert reject_challenge_page(article)['success']
    # Neither a title nor an isolated challenge phrase is sufficient.
    article['data']['title'] = 'Just a moment...'
    article['data']['content'] = 'An article discusses: are you a robot?\n' + 'Scientific content. ' * 4000
    assert reject_challenge_page(article)['success']

if __name__ == '__main__':
    test_saved_long_challenge()
    print('PASS: saved 64k CAPTCHA, padded CAPTCHA, benign article controls, immutable input')
