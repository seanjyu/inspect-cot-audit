# """Parsing utilities for extracting thinking blocks and tool call payloads from TaskState."""

# import re
# from typing import Any

# from inspect_ai.model import ChatMessage, ToolCall

# THINKING_TAG_PATTERNS: list[re.Pattern[str]] = [
#     re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE),
#     re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE),
#     re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE),
# ]


# def extract_thinking_trace(messages: list[ChatMessage]) -> str:
#     """Extracts intermediate reasoning blocks from assistant messages.

#     Searches for explicitly tagged XML blocks (<thinking>, <thought>, <reasoning>)
#     or inspect reasoning attributes across assistant turns.
#     """
#     extracted_blocks: list[str] = []

#     for message in messages:
#         if message.role != "assistant":
#             continue

#         text_content = ""
#         if isinstance(message.content, str):
#             text_content = message.content
#         elif isinstance(message.content, list):
#             for part in message.content:
#                 if getattr(part, "type", None) == "text":
#                     text_content += getattr(part, "text", "")

#         # Check for explicit reasoning fields (e.g. Inspect or Claude native reasoning)
#         if hasattr(message, "reasoning") and message.reasoning:
#             extracted_blocks.append(str(message.reasoning).strip())

#         # Extract tagged blocks via regex
#         for pattern in THINKING_TAG_PATTERNS:
#             matches = pattern.findall(text_content)
#             for match in matches:
#                 cleaned = match.strip()
#                 if cleaned and cleaned not in extracted_blocks:
#                     extracted_blocks.append(cleaned)

#     return "\n\n---\n\n".join(extracted_blocks)


# def extract_tool_calls_with_context(
#     messages: list[ChatMessage],
# ) -> list[tuple[str, list[ToolCall]]]:
#     """Extracts executed tool calls paired with the preceding assistant text message.

#     Returns:
#         List of tuples: (preceding_assistant_text, list_of_tool_calls)
#     """
#     results: list[tuple[str, list[ToolCall]]] = []

#     for message in messages:
#         if message.role == "assistant" and message.tool_calls:
#             text_content = ""
#             if isinstance(message.content, str):
#                 text_content = message.content
#             elif isinstance(message.content, list):
#                 for part in message.content:
#                     if getattr(part, "type", None) == "text":
#                         text_content += getattr(part, "text", "")

#             results.append((text_content.strip(), message.tool_calls))

#     return results


# def serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
#     """Converts an Inspect ToolCall instance into a clean dictionary payload."""
#     return {
#         "id": tool_call.id,
#         "function": tool_call.function,
#         "arguments": tool_call.arguments,
#     }

"""Parsing utilities for extracting thinking blocks and tool call payloads from TaskState."""

import re
from typing import Any

from inspect_ai.model import ChatMessage, ContentReasoning
from inspect_ai.tool import ToolCall

THINKING_TAG_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<thought>(.*?)</thought>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE),
]


def extract_thinking_trace(messages: list[ChatMessage]) -> str:
    """Extracts intermediate reasoning blocks from assistant messages.

    Priority:
    1. Structured `ContentReasoning` blocks within `ChatMessage.content`.
    2. Native `message.reasoning` attribute.
    3. Regex tag matching (<think>, <thinking>, <thought>, <reasoning>) on text content.
    """
    extracted_blocks: list[str] = []

    for message in messages:
        if message.role != "assistant":
            continue

        message_reasoning_found = False

        # 1. Structured ContentReasoning blocks inside message.content
        if isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, ContentReasoning) or getattr(part, "type", None) == "reasoning":
                    reasoning_text = getattr(part, "reasoning", "") or str(part)
                    if reasoning_text.strip():
                        extracted_blocks.append(reasoning_text.strip())
                        message_reasoning_found = True

        # 2. Native message.reasoning attribute
        if not message_reasoning_found and hasattr(message, "reasoning") and message.reasoning:
            reasoning_text = str(message.reasoning).strip()
            if reasoning_text:
                extracted_blocks.append(reasoning_text)
                message_reasoning_found = True

        # 3. Fallback: Parse XML tags from raw text content if no structured reasoning was found
        if not message_reasoning_found:
            text_content = ""
            if isinstance(message.content, str):
                text_content = message.content
            elif isinstance(message.content, list):
                for part in message.content:
                    if getattr(part, "type", None) == "text":
                        text_content += getattr(part, "text", "")

            if text_content:
                for pattern in THINKING_TAG_PATTERNS:
                    matches = pattern.findall(text_content)
                    for match in matches:
                        cleaned = match.strip()
                        if cleaned and cleaned not in extracted_blocks:
                            extracted_blocks.append(cleaned)

    return "\n\n---\n\n".join(extracted_blocks)


def extract_tool_calls_with_context(
    messages: list[ChatMessage],
) -> list[tuple[str, list[ToolCall]]]:
    """Extracts executed tool calls paired with the preceding assistant thinking or text context.

    Returns:
        List of tuples: (preceding_thinking_or_text, list_of_tool_calls)
    """
    results: list[tuple[str, list[ToolCall]]] = []

    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            # First attempt to retrieve structured reasoning for the turn
            context = extract_thinking_trace([message])

            # Fallback to standard text if no reasoning was emitted
            if not context:
                if isinstance(message.content, str):
                    context = message.content.strip()
                elif isinstance(message.content, list):
                    text_parts = [
                        getattr(p, "text", "")
                        for p in message.content
                        if getattr(p, "type", None) == "text"
                    ]
                    context = "".join(text_parts).strip()

            results.append((context, message.tool_calls))

    return results


def serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Converts an Inspect ToolCall instance into a clean dictionary payload."""
    return {
        "id": tool_call.id,
        "function": tool_call.function,
        "arguments": tool_call.arguments,
    }
