from typing import Any

from inspect_ai.model import ChatMessage
from inspect_ai.tool import ToolCall

from inspect_cot_audit.core.parsers import extract_thinking_trace


def serialize_tool_call(tool_call: ToolCall) -> dict[str, Any]:
    """Converts an Inspect ToolCall instance into a clean dictionary payload."""
    return {
        "id": tool_call.id,
        "function": tool_call.function,
        "arguments": tool_call.arguments,
    }


def serialize_full_single_agent_trajectory(messages: list[ChatMessage]) -> str:
    """Serializes the entire multi-round execution history of a single agent
    into a chronological narrative transcript for long-horizon auditing.
    """
    transcript_lines: list[str] = []
    turn_counter = 1

    for message in messages:
        if message.role == "user":
            transcript_lines.append(f"\n[Round {turn_counter} - User Input]:\n{message.content}")

        elif message.role == "assistant":
            # Extract reasoning/thinking for this turn
            thinking = extract_thinking_trace([message])
            thought_text = thinking if thinking else "[No explicit reasoning emitted]"

            transcript_lines.append(
                f"\n[Round {turn_counter} - Agent Reasoning (CoT)]:\n{thought_text}"
            )

            # If the assistant executed tool calls, serialize them
            if message.tool_calls:
                serialized = [serialize_tool_call(tc) for tc in message.tool_calls]
                transcript_lines.append(
                    f"[Round {turn_counter} - Executed Tool Calls]:\n{serialized}"
                )

            turn_counter += 1

        elif message.role == "tool":
            # Include tool outputs so the auditor sees what the agent observed
            transcript_lines.append(f"\n[Tool Output Response]:\n{message.content}")

    return "\n".join(transcript_lines)


def serialize_multi_agent_traces(traces: dict[str, list[tuple[str, list[ToolCall]]]]) -> str:
    """Serializes grouped agent traces into a structured text transcript for LLM judges."""
    if not traces:
        return "[No agent interaction recorded]"

    sections: list[str] = []

    for agent_id, turns in traces.items():
        agent_block = [f"=== AGENT IDENTITY: {agent_id} ==="]

        for turn_idx, (thinking, tool_calls) in enumerate(turns, 1):
            thought_text = thinking if thinking else "[No explicit reasoning emitted]"
            agent_block.append(f"  [Turn {turn_idx} Reasoning]:\n  {thought_text}")

            if tool_calls:
                serialized_calls = [serialize_tool_call(tc) for tc in tool_calls]
                agent_block.append(
                    f"  [Turn {turn_idx} Executed Tool Calls]:\n  {serialized_calls}"
                )
            else:
                agent_block.append(f"  [Turn {turn_idx} Executed Tool Calls]: None")

        sections.append("\n".join(agent_block))

    return "\n\n".join(sections)
