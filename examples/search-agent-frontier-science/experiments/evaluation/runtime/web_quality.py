import re
from urllib.parse import urlsplit
from serper_resolve import is_google_redirect

def reject_challenge_page(response, *, reject_google_redirects=False):
    """Reject unreadable HTTP-200 fetches and, when enabled, opaque destinations."""
    data = response.get("data")
    if response.get("mode") != "url" or not response.get("success") or not isinstance(data, dict):
        return response
    title = str(data.get("title") or "").strip().casefold()
    content = str(data.get("content") or "").casefold()
    if not content.strip():
        return {**response, "success": False, "data": None, "error": "Remote website returned no readable content."}
    destination = str(data.get('final_url') or data.get('finalUrl') or data.get('reported_url') or '')
    if not destination and title.startswith(('https://', 'http://')):
        destination = title  # Older cached Jina responses retained asset URL only in title.
    if urlsplit(destination).path.lower().endswith(('.svg', '.png', '.jpg', '.jpeg')) and not urlsplit(str(data.get('url') or '')).path.lower().endswith(('.svg', '.png', '.jpg', '.jpeg')):
        return {**response, 'success': False, 'data': None, 'error': 'Article request resolved to an image asset instead of document text.'}
    if 'page not found' in title and ('# page not found' in content[:2500] or 'page you were looking for does not exist' in content[:2500]):
        return {**response, 'success': False, 'data': None, 'error': 'Remote website returned a page-not-found document.'}
    # Require paired interstitial markers near the start, not a lone mention
    # of cookies, CAPTCHA, or Cloudflare in an otherwise useful article.
    lead = content[:2500]
    pairs = (
        ('performing security verification', 'verifies you are not a bot'),
        ('security check required', "unusual activity from your network"),
        ('checking your browser before accessing', 'recaptcha/challengepage'),
        ("let's confirm you are human", 'complete the security check'),
        ('are you a robot?', 'confirm you are a human'),
    )
    interstitial = len(content) < 6000 and any(a in lead and b in lead for a, b in pairs)
    # ScienceDirect can append tens of thousands of characters of scripts to
    # its CAPTCHA. Identify the leading challenge structure, not body length.
    long_captcha = (
        title in {'just a moment...', 'just a moment', 'are you a robot?'}
        and re.search(r'^#{1,3}\s+are you a robot\?\s*$', lead, re.MULTILINE) is not None
        and 'please confirm you are a human by completing the captcha challenge below' in lead
        and re.search(r'verification successful\. waiting for [^\n]+ to respond', lead) is not None
    )
    if interstitial or long_captcha:
        return {**response, 'success': False, 'data': None,
                'error': 'Remote website returned a security-check interstitial, not source content. Try a direct PDF, an alternate source, or search again.'}
    host = (urlsplit(str(data.get('url') or '')).hostname or '').lower()
    ads_shell = (host == 'adsabs.harvard.edu' or host.endswith('.adsabs.harvard.edu')) and len(content) < 2000 and 'mirador viewer' in lead and 'scan explorer' in lead
    if ads_shell:
        return {**response, 'success': False, 'data': None,
                'error': 'Remote website returned only an ADS scan-viewer shell; no document text was extracted. Try a direct PDF or another source.'}
    # Short content alone is not a failure. Flag likely captions without
    # discarding potentially useful scientific evidence.
    if len(content.strip()) < 500 and re.match(r'^(?:a |an )?(?:scatter plot|plot|figure|diagram|image)\b', content.strip()):
        data = {**data, 'quality_warnings': list(data.get('quality_warnings') or []) + ['possible_figure_only_content']}
        response = {**response, 'data': data}
    google_challenge = (
        any(is_google_redirect(data.get(key)) for key in ("url", "resolved_url"))
        and (title == "verifying your request" or "verifying your request" in content[:500])
    )
    if google_challenge or (title == "robot challenge screen" and "checking the site connection security" in content):
        return {
            **response,
            "success": False,
            "data": None,
            "error": "Remote website returned a bot-challenge page; its content is unavailable.",
        }
    if reject_google_redirects and any(is_google_redirect(data.get(key)) for key in ("url", "resolved_url")):
        return {
            **response, "success": False, "data": None,
            "error": "Page destination is an unresolved Google link; its content is unavailable.",
        }
    return response

