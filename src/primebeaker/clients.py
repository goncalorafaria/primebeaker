"""Compatibility facade for the public :mod:`primebeaker.client` API."""

from primebeaker.client import (
    FetchClient,
    FencedToolOutputDisplay,
    JudgeClient,
    PlainJsonToolOutputDisplay,
    PodmanExecutionClient,
    RemoteCodeExecutionClient,
    RewardModelClient,
    SearchClient,
    SubmissionStore,
    SubmitToolClient,
    TerminalExecutionClient,
    ToolClient,
    ToolOutputDisplay,
    WebTerminalExecutionClient,
    code_output_to_tool_content,
)
from primebeaker.runtime.asset_store import AssetStore, WebAssetStore

__all__ = [
    "AssetStore",
    "FetchClient",
    "FencedToolOutputDisplay",
    "JudgeClient",
    "PlainJsonToolOutputDisplay",
    "PodmanExecutionClient",
    "RemoteCodeExecutionClient",
    "RewardModelClient",
    "SearchClient",
    "SubmissionStore",
    "SubmitToolClient",
    "TerminalExecutionClient",
    "ToolClient",
    "ToolOutputDisplay",
    "WebAssetStore",
    "WebTerminalExecutionClient",
    "code_output_to_tool_content",
]
