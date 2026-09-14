"""Print the search-agent tool schemas for preparing a trace replay manifest."""
import json
from primebeaker.environments.search_agent_env import SearchAgentEnv
from verifiers.envs.stateful_tool_env import filter_signature
from verifiers.utils.tool_utils import convert_func_to_tool_def

# Inspect the bound signatures without connecting to live tool services.
env = SearchAgentEnv.__new__(SearchAgentEnv)
definitions = [
    convert_func_to_tool_def(env.search),
    convert_func_to_tool_def(filter_signature(env.webterminal, ['asset_store'])),
]
tools = []
for definition in definitions:
    raw = definition.model_dump(exclude_none=True)
    tools.append({'type': 'function', 'function': {
        key: value for key, value in raw.items()
        if key in {'name', 'description', 'parameters'}
    }})
print(json.dumps(tools, indent=2))
