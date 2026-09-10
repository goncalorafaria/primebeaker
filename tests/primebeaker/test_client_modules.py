from literegistry_tool_client import (
    FetchClient,
    JudgeClient,
    PodmanExecutionClient,
    RemoteCodeExecutionClient,
    RewardModelClient,
    SearchClient,
    SubmitToolClient,
    TerminalExecutionClient,
    ToolClient,
    WebTerminalExecutionClient,
)
from literegistry_tool_client._transport import ToolClient as ModuleToolClient
from literegistry_tool_client.code import (
    RemoteCodeExecutionClient as ModuleCodeClient,
)
from literegistry_tool_client.fetch import FetchClient as ModuleFetchClient
from literegistry_tool_client.judge import JudgeClient as ModuleJudgeClient
from literegistry_tool_client.podman import PodmanExecutionClient as ModulePodmanClient
from literegistry_tool_client.reward_model import (
    RewardModelClient as ModuleRewardClient,
)
from literegistry_tool_client.search import SearchClient as ModuleSearchClient
from literegistry_tool_client.submission import SubmitToolClient as ModuleSubmitClient
from literegistry_tool_client.terminal import (
    TerminalExecutionClient as ModuleTerminalClient,
)
from literegistry_tool_client.webterminal import (
    WebTerminalExecutionClient as ModuleWebTerminalClient,
)


def test_every_public_client_uses_its_dedicated_implementation() -> None:
    assert ToolClient is ModuleToolClient
    assert RemoteCodeExecutionClient is ModuleCodeClient
    assert TerminalExecutionClient is ModuleTerminalClient
    assert PodmanExecutionClient is ModulePodmanClient
    assert SearchClient is ModuleSearchClient
    assert FetchClient is ModuleFetchClient
    assert WebTerminalExecutionClient is ModuleWebTerminalClient
    assert JudgeClient is ModuleJudgeClient
    assert RewardModelClient is ModuleRewardClient
    assert SubmitToolClient is ModuleSubmitClient


def test_every_concrete_client_inherits_the_shared_transport() -> None:
    for client_type in (
        RemoteCodeExecutionClient,
        TerminalExecutionClient,
        PodmanExecutionClient,
        SearchClient,
        FetchClient,
        WebTerminalExecutionClient,
        JudgeClient,
        RewardModelClient,
        SubmitToolClient,
    ):
        assert issubclass(client_type, ToolClient)


def test_literegistry_owns_clients_and_shared_asset_types() -> None:
    import literegistry_tool_client as canonical
    from primebeaker.runtime.asset_store import AssetStore, WebAssetStore

    assert AssetStore is canonical.AssetStore
    assert WebAssetStore is canonical.WebAssetStore
