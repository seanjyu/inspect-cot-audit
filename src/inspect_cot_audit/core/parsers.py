"""Parsing utilities for extracting thinking blocks and tool call payloads from TaskState."""

import json
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


def extract_multi_agent_traces(
    messages: list[ChatMessage],
) -> dict[str, list[dict[str, Any]]]:
    """Extracts reasoning traces and tool calls or structured choices grouped by agent identity."""
    agent_traces: dict[str, list[dict[str, Any]]] = {}

    for message in messages:
        if message.role != "assistant":
            continue

        # Safely resolve agent identity
        agent_id = "primary_agent"
        source = getattr(message, "source", None)
        if source:
            agent_id = str(source)
        else:
            metadata = getattr(message, "metadata", None)
            if isinstance(metadata, dict):
                for key in ("agent", "agent_id", "sender", "name"):
                    if metadata.get(key):
                        agent_id = str(metadata[key])
                        break

        # Extract thinking/CoT trace
        context = extract_thinking_trace([message])
        if not context and hasattr(message, "reasoning") and message.reasoning:
            context = str(message.reasoning)

        # Fallback to text content if no reasoning tag was found
        if not context:
            content = getattr(message, "content", "")
            if isinstance(content, str):
                context = content.strip()
            elif isinstance(content, list):
                text_parts = [
                    getattr(part, "text", "")
                    for part in content
                    if getattr(part, "type", None) == "text"
                ]
                context = "".join(text_parts).strip()
            elif hasattr(message, "text"):
                context = str(message.text)

        # Collect tool calls or structured payloads safely
        tool_calls = getattr(message, "tool_calls", None) or []
        structured_val = getattr(message, "structured_value", {})

        # Fallback to parsing text content if structured_value isn't directly attached to message
        if not structured_val and context:
            try:
                structured_val = json.loads(context)
            except Exception:
                structured_val = {}

        if agent_id not in agent_traces:
            agent_traces[agent_id] = []

        agent_traces[agent_id].append(
            {
                "reasoning": context,
                "tool_calls": tool_calls,
                "structured_value": structured_val,
                "completion": context,
            }
        )

    return agent_traces
