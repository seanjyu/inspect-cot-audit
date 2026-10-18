import asyncio

import pytest
from inspect_ai.solver import TaskState

from inspect_cot_audit.core.runner import run_cot_audit


# --- Mock Check Implementation for Testing ---
class DummyCheck:
    name = "dummy_check"

    def __init__(self, parse_responses: list[str | None], fail_with_exception: bool = False):
        # A sequence of responses to return across retry attempts
        self.parse_responses = parse_responses
        self.fail_with_exception = fail_with_exception
        self.calls = 0

    def build_prompt(self, state: TaskState) -> str:
        return "Test base prompt"

    def parse(self, completion: str):
        if self.fail_with_exception:
            raise RuntimeError("Intentional parse crash")

        # Pop the next expected parse response
        if self.calls < len(self.parse_responses):
            res = self.parse_responses[self.calls]
        else:
            res = None
        self.calls += 1
        return res

    def default(self) -> str:
        return "DEFAULT_FALLBACK"

    def build_repair_prompt(self, base_prompt: str, last_completion: str) -> str:
        return f"{base_prompt}\nREPAIR: fix your format."


class MockModelResponse:
    def __init__(self, completion: str | None):
        self.completion = completion


class MockModel:
    def __init__(self, completions: list[str | Exception]):
        self.completions = completions
        self.call_count = 0

    async def generate(self, prompt: str):
        if self.call_count >= len(self.completions):
            res = "VERDICT: OK"
        else:
            res = self.completions[self.call_count]

        self.call_count += 1

        if isinstance(res, Exception):
            raise res
        return MockModelResponse(completion=res)


# --- Pytest Test Cases ---


@pytest.mark.asyncio
async def test_run_cot_audit_success_on_first_attempt(monkeypatch):
    """Test that a valid parse on attempt 1 returns status 'ok'."""
    mock_model = MockModel(completions=["VERDICT: CONSISTENT"])
    monkeypatch.setattr("inspect_cot_audit.core.runner.get_model", lambda _: mock_model)

    check = DummyCheck(parse_responses=["SUCCESS_VALUE"])
    state = TaskState(sample_id="1", model="test", epoch=1, input="test", messages=[])

    outcome = await run_cot_audit("mock_model", check, state, max_attempts=2)

    assert outcome.value == "SUCCESS_VALUE"
    assert outcome.status == "ok"
    assert outcome.attempts == 1
    assert mock_model.call_count == 1


@pytest.mark.asyncio
async def test_run_cot_audit_repair_and_success(monkeypatch):
    """Test that a failed parse on attempt 1 triggers repair prompt and succeeds on attempt 2."""
    mock_model = MockModel(completions=["BAD_OUTPUT", "GOOD_OUTPUT"])
    monkeypatch.setattr("inspect_cot_audit.core.runner.get_model", lambda _: mock_model)

    # First parse returns None (fail), second returns value (success)
    check = DummyCheck(parse_responses=[None, "REPAIRED_SUCCESS"])
    state = TaskState(sample_id="1", model="test", epoch=1, input="test", messages=[])

    outcome = await run_cot_audit("mock_model", check, state, max_attempts=2)

    assert outcome.value == "REPAIRED_SUCCESS"
    assert outcome.status == "repaired"
    assert outcome.attempts == 2
    assert mock_model.call_count == 2


@pytest.mark.asyncio
async def test_run_cot_audit_parse_failure_exhausts_attempts(monkeypatch):
    """Test that persistent parse failures return 'parse_failed' and default value."""
    mock_model = MockModel(completions=["BAD_1", "BAD_2"])
    monkeypatch.setattr("inspect_cot_audit.core.runner.get_model", lambda _: mock_model)

    check = DummyCheck(parse_responses=[None, None])
    state = TaskState(sample_id="1", model="test", epoch=1, input="test", messages=[])

    outcome = await run_cot_audit("mock_model", check, state, max_attempts=2)

    assert outcome.value == "DEFAULT_FALLBACK"
    assert outcome.status == "parse_failed"
    assert outcome.attempts == 2
    assert outcome.raw == "BAD_2"


@pytest.mark.asyncio
async def test_run_cot_audit_generation_timeout(monkeypatch):
    """Test that an asyncio.TimeoutError catches correctly and returns 'gen_failed'."""
    mock_model = MockModel(completions=[asyncio.TimeoutError()])
    monkeypatch.setattr("inspect_cot_audit.core.runner.get_model", lambda _: mock_model)

    check = DummyCheck(parse_responses=[])
    state = TaskState(sample_id="1", model="test", epoch=1, input="test", messages=[])

    outcome = await run_cot_audit("mock_model", check, state, max_attempts=1, timeout=1)

    assert outcome.value == "DEFAULT_FALLBACK"
    assert outcome.status == "gen_failed"
    assert "TimeoutError" in outcome.raw
    assert outcome.attempts == 1


@pytest.mark.asyncio
async def test_run_cot_audit_general_exception_handling(monkeypatch):
    """Test that unexpected generation exceptions trigger retry/failure accounting."""
    mock_model = MockModel(completions=[ConnectionError("API down")])
    monkeypatch.setattr("inspect_cot_audit.core.runner.get_model", lambda _: mock_model)

    check = DummyCheck(parse_responses=[])
    state = TaskState(sample_id="1", model="test", epoch=1, input="test", messages=[])

    outcome = await run_cot_audit("mock_model", check, state, max_attempts=1)

    assert outcome.value == "DEFAULT_FALLBACK"
    assert outcome.status == "gen_failed"
    assert outcome.raw == "API down"
    assert outcome.attempts == 1
