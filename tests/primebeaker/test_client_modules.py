from primebeaker.client import (
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
from primebeaker.client._transport import ToolClient as ModuleToolClient
from primebeaker.client.code import RemoteCodeExecutionClient as ModuleCodeClient
from primebeaker.client.fetch import FetchClient as ModuleFetchClient
from primebeaker.client.judge import JudgeClient as ModuleJudgeClient
from primebeaker.client.podman import PodmanExecutionClient as ModulePodmanClient
from primebeaker.client.reward_model import RewardModelClient as ModuleRewardClient
from primebeaker.client.search import SearchClient as ModuleSearchClient
from primebeaker.client.submission import SubmitToolClient as ModuleSubmitClient
from primebeaker.client.terminal import (
    TerminalExecutionClient as ModuleTerminalClient,
)
from primebeaker.client.webterminal import (
    WebTerminalExecutionClient as ModuleWebTerminalClient,
)
from primebeaker.clients import (
    FetchClient as CompatibilityFetchClient,
    JudgeClient as CompatibilityJudgeClient,
    PodmanExecutionClient as CompatibilityPodmanClient,
    RemoteCodeExecutionClient as CompatibilityCodeClient,
    RewardModelClient as CompatibilityRewardClient,
    SearchClient as CompatibilitySearchClient,
    SubmitToolClient as CompatibilitySubmitClient,
    TerminalExecutionClient as CompatibilityTerminalClient,
    ToolClient as CompatibilityToolClient,
    WebTerminalExecutionClient as CompatibilityWebTerminalClient,
)


def test_every_public_client_uses_its_dedicated_implementation() -> None:
    assert ToolClient is ModuleToolClient is CompatibilityToolClient
    assert RemoteCodeExecutionClient is ModuleCodeClient is CompatibilityCodeClient
    assert TerminalExecutionClient is ModuleTerminalClient is CompatibilityTerminalClient
    assert PodmanExecutionClient is ModulePodmanClient is CompatibilityPodmanClient
    assert SearchClient is ModuleSearchClient is CompatibilitySearchClient
    assert FetchClient is ModuleFetchClient is CompatibilityFetchClient
    assert (
        WebTerminalExecutionClient
        is ModuleWebTerminalClient
        is CompatibilityWebTerminalClient
    )
    assert JudgeClient is ModuleJudgeClient is CompatibilityJudgeClient
    assert RewardModelClient is ModuleRewardClient is CompatibilityRewardClient
    assert SubmitToolClient is ModuleSubmitClient is CompatibilitySubmitClient


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
