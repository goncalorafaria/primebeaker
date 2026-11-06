from eval_web_access import blocked_url, blocked_response, filter_web_response
DOMAINS = ("huggingface.co", "hf.co")
def filter_response(response):
    response = filter_web_response(response, domains=DOMAINS)
    data = response.get("data")
    if response.get("mode") == "url" and isinstance(data, dict):
        for key in ("reported_url", "final_url"):
            if blocked_url(data.get(key) or "", domains=DOMAINS):
                return {**response, **blocked_response()}
        if (response.get("success") and str(data.get("title") or "").strip().casefold() == "robot challenge screen"
            and "checking the site connection security" in str(data.get("content") or "").casefold()):
            return {**response, "success": False, "data": None,
                    "error": "Remote website returned a bot-challenge page; its content is unavailable."}
    return response
