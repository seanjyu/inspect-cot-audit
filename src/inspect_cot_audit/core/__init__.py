"""Core data structures and parsers."""

# from scratch.types import ConsistencyAuditResult, DeceptionAuditResult

from inspect_cot_audit.core.parsers import extract_thinking_trace, extract_tool_calls_with_context
from inspect_cot_audit.core.runner import CoTCheck, run_cot_audit

__all__ = [
    "CoTCheck",
    # "ConsistencyAuditResult",
    # "DeceptionAuditResult",
    "extract_thinking_trace",
    "extract_tool_calls_with_context",
    "run_cot_audit",
]
