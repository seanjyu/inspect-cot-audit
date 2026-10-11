"""Unit tests for parsing utilities."""

from inspect_ai.model import (
    ChatMessageAssistant,
    ContentReasoning,
    ContentText,
    ToolCall,
)

from inspect_cot_audit.core.parsers import extract_thinking_trace, extract_tool_calls_with_context


def test_extract_thinking_trace_content_reasoning() -> None:
    """Tests extraction from structured ContentReasoning blocks."""
    msg = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning="Structured reasoning trace."),
            ContentText(text="Final output."),
        ]
    )
    assert extract_thinking_trace([msg]) == "Structured reasoning trace."


def test_extract_thinking_trace_think_tag_fallback() -> None:
    """Tests extraction from open-weight <think> tags."""
    msg = ChatMessageAssistant(content="<think>DeepSeek style thinking.</think> Final answer.")
    assert extract_thinking_trace([msg]) == "DeepSeek style thinking."


def test_extract_thinking_trace_legacy_tag_fallback() -> None:
    """Tests extraction from <thinking> tags."""
    msg = ChatMessageAssistant(content="<thinking>Legacy thinking trace.</thinking> Final answer.")
    assert extract_thinking_trace([msg]) == "Legacy thinking trace."


def test_extract_tool_calls_with_context_structured() -> None:
    """Tests tool call context extraction with ContentReasoning."""
    tool_call = ToolCall(id="call_1", function="read_file", arguments={"path": "test.txt"})
    msg = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning="Checking file before executing call."),
        ],
        tool_calls=[tool_call],
    )

    results = extract_tool_calls_with_context([msg])
    assert len(results) == 1
    context, tool_calls = results[0]
    assert context == "Checking file before executing call."
    assert tool_calls[0].id == "call_1"
