"""Explicit legacy-ID alias for the judge-scored search + visit harness.

Use legacy environment ID ``jtc-search-agent-visit-judge-env``. The existing
``jtc-search-agent-visit-env`` continues to use exact-match scoring.
"""

from .jtc_search_visit_agent_judge_env import load_environment

__all__ = ["load_environment"]
