# PrimeBeaker clients

The canonical client API is `primebeaker.client`. Each service client has its
own module and all clients share the same asynchronous `ToolClient` transport.

| Client | Dedicated module | Purpose |
| --- | --- | --- |
| `RemoteCodeExecutionClient` | `primebeaker.client.code` | Remote Python execution |
| `TerminalExecutionClient` | `primebeaker.client.terminal` | Restricted terminal pipelines |
| `PodmanExecutionClient` | `primebeaker.client.podman` | Stateful container sessions |
| `SearchClient` | `primebeaker.client.search` | Search queries |
| `FetchClient` | `primebeaker.client.fetch` | URL extraction via Jina or LiteRegistry |
| `WebTerminalExecutionClient` | `primebeaker.client.webterminal` | Browse/fetch plus terminal pipelines |
| `JudgeClient` | `primebeaker.client.judge` | Rubric judging |
| `RewardModelClient` | `primebeaker.client.reward_model` | Sequence classification |
| `SubmitToolClient` | `primebeaker.client.submission` | Rollout-local rubric submissions |

Import from the package for normal use:

```python
from primebeaker.client import (
    FetchClient,
    SearchClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
)

terminal = TerminalExecutionClient()
fetch = FetchClient(
    local_search_server_url="http://127.0.0.1:1212/search",
    local_search_model_path="fetch",
)
webterminal = WebTerminalExecutionClient(terminal, fetch)
search = SearchClient(model_path="search")
```

`primebeaker.clients` remains a compatibility facade for existing callers.
