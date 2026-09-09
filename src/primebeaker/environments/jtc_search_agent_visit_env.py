"""Explicit legacy-ID alias for the search + direct-visit harness.

Use legacy environment ID ``jtc-search-agent-visit-env``.
"""

from .jtc_search_visit_agent_env import load_environment

__all__ = ["load_environment"]
