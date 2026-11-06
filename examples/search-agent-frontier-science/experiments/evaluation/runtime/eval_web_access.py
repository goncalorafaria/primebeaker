"""URL-based benchmark-source policy, applied to fresh and cached web responses."""

import json
import posixpath
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

_SOURCE_URL_FIELDS = {"link", "url", "website", "resolved_url"}
BLOCKLIST_KEYS = ("domains", "github_repos", "url_prefixes", "hf_datasets", "arxiv_ids")


def _url_parts(url):
    parsed = urlsplit(url if "://" in url or url.startswith("//") else "//" + url)
    host = unquote(parsed.hostname or "").encode("idna").decode("ascii").lower().rstrip(".")
    path = posixpath.normpath("/" + unquote(parsed.path).lstrip("/"))
    return parsed, host, path


def _string_list(values, field):
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{field} must be a list of strings")
    return list(dict.fromkeys(value.strip() for value in values))


def normalize_blocked_domains(domains: list[str]) -> list[str]:
    normalized = []
    for domain in _string_list(domains, "web_blocked_domains"):
        domain = domain.encode("idna").decode("ascii").lower().rstrip(".")
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", domain) or ".." in domain:
            raise ValueError("web_blocked_domains entries must be hostnames, without schemes, paths, or wildcards")
        if domain not in normalized:
            normalized.append(domain)
    return normalized


def normalize_blocked_github_repos(repos: list[str]) -> list[str]:
    normalized = []
    for repo in _string_list(repos, "web_blocked_github_repos"):
        repo = repo.lower().removesuffix(".git")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*/[a-z0-9_.-]+", repo) or repo.split("/")[1] in {".", ".."}:
            raise ValueError("web_blocked_github_repos entries must be owner/repo, without URLs or paths")
        if repo not in normalized:
            normalized.append(repo)
    return normalized


def normalize_blocked_hf_datasets(datasets: list[str]) -> list[str]:
    normalized = []
    for dataset in _string_list(datasets, "web_blocked_hf_datasets"):
        dataset = dataset.lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*/[a-z0-9][a-z0-9_.-]*", dataset):
            raise ValueError("web_blocked_hf_datasets entries must be owner/dataset, without URLs or paths")
        if dataset not in normalized:
            normalized.append(dataset)
    return normalized


def normalize_blocked_url_prefixes(prefixes: list[str]) -> list[str]:
    normalized = []
    for prefix in _string_list(prefixes, "web_blocked_url_prefixes"):
        parsed, host, path = _url_parts(prefix)
        if (not host or parsed.scheme not in {"", "http", "https"} or parsed.query or parsed.fragment
                or parsed.username or parsed.password or parsed.port or "*" in prefix):
            raise ValueError("web_blocked_url_prefixes entries must be a hostname/path without query, fragment or wildcard")
        normalize_blocked_domains([host])
        prefix = host + path.rstrip("/")
        if prefix not in normalized:
            normalized.append(prefix)
    return normalized


def normalize_blocked_arxiv_ids(ids: list[str]) -> list[str]:
    normalized = []
    for paper in _string_list(ids, "web_blocked_arxiv_ids"):
        if not re.fullmatch(r"\d{4}\.\d{4,5}", paper):
            raise ValueError("web_blocked_arxiv_ids entries must be unversioned paper IDs, e.g. 2504.12516")
        if paper not in normalized:
            normalized.append(paper)
    return normalized


BLOCKLIST_VALIDATORS = {
    "domains": normalize_blocked_domains,
    "github_repos": normalize_blocked_github_repos,
    "url_prefixes": normalize_blocked_url_prefixes,
    "hf_datasets": normalize_blocked_hf_datasets,
    "arxiv_ids": normalize_blocked_arxiv_ids,
}
_DEFAULT_BLOCKLIST = json.loads(Path(__file__).with_name("web_blocklist.json").read_text())
DEFAULT_BLOCKLIST = {key: BLOCKLIST_VALIDATORS[key](_DEFAULT_BLOCKLIST[key]) for key in BLOCKLIST_KEYS}
DEFAULT_BLOCKED_DOMAINS = tuple(DEFAULT_BLOCKLIST["domains"])


def blocked_domain(url: str, domains: list[str] | tuple[str, ...]) -> str | None:
    if not url or not domains:
        return None
    _, host, _ = _url_parts(url)
    return next((domain for domain in domains if host == domain or host.endswith("." + domain)), None)


def _github_repo(host, path):
    parts = path.strip("/").lower().split("/")
    if host == "media.githubusercontent.com" and parts[:1] == ["media"]:
        parts = parts[1:]
    elif host == "api.github.com" and parts[:1] == ["repos"]:
        parts = parts[1:]
    elif host.endswith(".github.io"):
        parts.insert(0, host.removesuffix(".github.io"))
    elif host not in {"github.com", "www.github.com", "raw.githubusercontent.com", "raw.github.com",
                      "media.githubusercontent.com", "codeload.github.com"}:
        return None
    return "/".join((parts[0], parts[1].removesuffix(".git"))) if len(parts) >= 2 else None


def _hf_datasets(parsed, host, path):
    parts = path.strip("/").lower().split("/")
    if host in {"huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co"}:
        if parts[:1] == ["api"]:
            parts = parts[1:]
        if len(parts) >= 3 and parts[0] == "datasets":
            return ["/".join(parts[1:3])]
    if host == "datasets-server.huggingface.co":
        datasets = [value.lower() for value in parse_qs(parsed.query).get("dataset", [])]
        if len(parts) >= 3 and parts[0] == "cached-assets":
            datasets.append("/".join(parts[1:3]))
        return datasets
    return []


def _arxiv_id(host, path):
    host = host.removeprefix("www.")
    patterns = {
        "arxiv.org": r"/(?:abs|pdf|html|src|e-print)/",
        "export.arxiv.org": r"/(?:abs|pdf|html|src|e-print)/",
        "ar5iv.labs.arxiv.org": r"/html/",
        "ar5iv.org": r"/html/",
        "alphaxiv.org": r"/(?:abs|pdf|html|overview)/",
        "huggingface.co": r"/papers/",
        "hf.co": r"/papers/",
        "doi.org": r"/10\.48550/arxiv\.",
    }
    if host not in patterns:
        return None
    match = re.fullmatch(patterns[host] + r"(\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/?", path, re.IGNORECASE)
    return match[1] if match else None


def _wrapped_url(parsed, host):
    # Only unwrap known proxy/redirect formats; arbitrary query text is not a URL source.
    if host == "r.jina.ai":
        return unquote(parsed.path).lstrip("/") + ("?" + parsed.query if parsed.query else "")
    if host == "web.archive.org":
        match = re.match(r"/web/[^/]+/(https?://.+)", unquote(parsed.path))
        return (match[1] + ("?" + parsed.query if parsed.query else "")) if match else None
    keys = {
        ("www.google.com", "/url"): ("url", "q"),
        ("google.com", "/url"): ("url", "q"),
        ("translate.google.com", "/translate"): ("u",),
    }.get((host, parsed.path), ())
    query = parse_qs(parsed.query)
    return next((value for key in keys for value in query.get(key, []) if value.startswith(("https://", "http://"))), None)


def blocked_url(url: str, domains=(), github_repos=(), *, url_prefixes=(), hf_datasets=(), arxiv_ids=()) -> str | None:
    for _ in range(5):
        if not url:
            return None
        parsed, host, path = _url_parts(url)
        domain = next((d for d in domains if host == d or host.endswith("." + d)), None)
        if domain:
            return domain
        repo = _github_repo(host, path)
        if repo in github_repos:
            return "github:" + repo
        for dataset in _hf_datasets(parsed, host, path):
            if dataset in hf_datasets:
                return "hf_dataset:" + dataset
        paper = _arxiv_id(host, path)
        if paper in arxiv_ids:
            return "arxiv:" + paper
        for prefix in url_prefixes:
            _, prefix_host, prefix_path = _url_parts(prefix)
            if host.removeprefix("www.") == prefix_host.removeprefix("www.") and (
                path == prefix_path or path.startswith(prefix_path.rstrip("/") + "/")
            ):
                return "url_prefix:" + prefix
        url = _wrapped_url(parsed, host)
    return None


def blocked_response() -> dict:
    return {
        "success": False,
        "mode": "url",
        "data": None,
        "upstream_status": None,
        "error": "This domain cannot be accessed.",
    }


def filter_web_response(response: dict, domains=(), github_repos=(), *, url_prefixes=(), hf_datasets=(), arxiv_ids=(),
                        on_block: Callable[[str, str], None] | None = None) -> dict:
    """Return a sanitized copy; never modify the shared raw cache entry."""
    data = response.get("data")
    if not (domains or github_repos or url_prefixes or hf_datasets or arxiv_ids) or not isinstance(data, dict):
        return response

    def matches(url):
        rule = blocked_url(url, domains, github_repos, url_prefixes=url_prefixes,
                           hf_datasets=hf_datasets, arxiv_ids=arxiv_ids)
        if rule and on_block:
            on_block(url, rule)
        return bool(rule)

    def has_blocked_source(value):
        if isinstance(value, dict):
            return any(
                (key in _SOURCE_URL_FIELDS and isinstance(child, str) and matches(child))
                or has_blocked_source(child)
                for key, child in value.items()
            )
        if isinstance(value, list):
            return any(has_blocked_source(child) for child in value)
        return False

    if response.get("mode") == "url":
        for key in ("url", "resolved_url"):
            if matches(data.get(key) or ""):
                return {**response, **blocked_response()}
        return response
    # Remove each matching result with its snippet, including auxiliary answer sections.
    filtered = {}
    for key, value in data.items():
        if isinstance(value, list):
            filtered[key] = [item for item in value if not has_blocked_source(item)]
        elif not has_blocked_source(value):
            filtered[key] = value
    return {**response, "data": filtered}
