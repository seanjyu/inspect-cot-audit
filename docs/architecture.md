# Architecture

This document outlines the overarching architecture of this framework as well as various high level decisions and tradeoffs.

# Lifecycle
The following describes the general lifecycle of a CoT auditor scorer. Note each scorer performs its own test so the individual mechanics for each scorer will not be explained here.
1. A scorer can parse the COT through the `core.parser.extract_thinking_trace` function. Specifically this function looks for the COT using three methods sequentially, trying the next method only when the previous one fails:
    <ol type="I">
    <li>Structured `ContentReasoning` blocks within `ChatMessage.content`.</li>
    <li>Native `message.reasoning` attribute.</li>
    <li>Regex tag matching (<think>, <thinking>, <thought>, <reasoning>) on text content.</li>
    </ol>

     Tool calls can also be used in the auditing process by using the `core.parser.extract_tool_calls_with_context`. This function uses `extract_thinking_trace` to extract the CoT and additionally extracts the tool calls for each round of messages.
2. Once extracted, the CoT trace undergoes a task-specific evaluation that conforms to the `core.runner.CotCheck` protocol." This check instance contains several important things:
    - The method to create the a method which is used to create the prompt for the auditor.
    - The method to parse the CoT output is implemented here. Note this method returns a typed `AuditOutcome` containing results and operational status (e.g. `ok`, `repaired`, `parse_failed`, etc). 
    - A repair/re-attempt prompt is also implemented here
    - There also is a default method which is a fallback for when the LLM judge is unable to generate a response. 
3. With the CoT check instantiated the shared runner can be used to obtain the CoT from the model. Note this uses Inspect AI's `get_model`. The runner has an input for max retries for an auditor to 
4. The runner returns with the task specific metrics, the score is calculated in the scorer method.
5. Basic audit metrics were also implemented and can be found in `evaluation.metrics`


# DIY 
The following is a basic guide of the minimum methods that have to implemented to make a new CoT-audit experiment
1. Implement Dataclass
2. Write Prompt
3. Write check class
    1. Implement init method
    2. Implement build_prompt method 
    3. Implement parse method
    4. Implement build_repair_prompt
    5. Implement default
4. Write solver
    1. Extract CoT (using either extract_thinking_trace or extract_tool_calls_with_context)
    2. Call run_cot_audit
    3. Calculate and return score

# Various Software Architectural Decisions
The following section outlines the core design choices made during the development of the framework.
- 