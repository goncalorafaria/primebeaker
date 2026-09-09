"""Compatibility imports; implementations live in literegistry_tool_client."""

from literegistry_tool_client._transport import (
    ToolClient as ToolClient,
    RETRYABLE_HTTP_STATUSES as RETRYABLE_HTTP_STATUSES,
    HTTP_RETRY_BASE_DELAY_SECONDS as HTTP_RETRY_BASE_DELAY_SECONDS,
    HTTP_RETRY_MAX_DELAY_SECONDS as HTTP_RETRY_MAX_DELAY_SECONDS,
)
