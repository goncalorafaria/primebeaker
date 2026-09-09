"""Compatibility imports; implementations live in literegistry_tool_client."""

from literegistry_tool_client.output import (
    FencedToolOutputDisplay as FencedToolOutputDisplay,
    PlainJsonToolOutputDisplay as PlainJsonToolOutputDisplay,
    ToolOutputDisplay as ToolOutputDisplay,
    code_output_to_tool_content as code_output_to_tool_content,
    sanitize_tool_output as sanitize_tool_output,
    terminal_output_payload as terminal_output_payload,
    terminal_output_to_tool_content as terminal_output_to_tool_content,
    truncate_output_text as truncate_output_text,
    truncate_tool_output as truncate_tool_output,
)
