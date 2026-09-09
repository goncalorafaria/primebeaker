"""Explicit legacy-ID alias for the search + webterminal harness.

Use legacy environment ID ``jtc-search-agent-webterminal-env``.  The older
``jtc-search-agent-env`` remains supported by ``jtc_search_agent_env``.
"""

from .jtc_search_agent_env import load_environment

__all__ = ["load_environment"]
