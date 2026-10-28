"""Core data structures and parsers."""

# from scratch.types import ConsistencyAuditResult, DeceptionAuditResult

from inspect_cot_audit.core.parsers import extract_thinking_trace, extract_tool_calls_with_context
from inspect_cot_audit.core.runner import CoTCheck, run_cot_audit
from inspect_cot_audit.core.serializers import (
    serialize_full_single_agent_trajectory,
    serialize_multi_agent_traces,
    serialize_tool_call,
)

__all__ = [
    "CoTCheck",
    # "ConsistencyAuditResult",
    # "DeceptionAuditResult",
    "extract_thinking_trace",
    "extract_tool_calls_with_context",
    "run_cot_audit",
    "serialize_full_single_agent_trajectory",
    "serialize_multi_agent_traces",
    "serialize_tool_call",
]
