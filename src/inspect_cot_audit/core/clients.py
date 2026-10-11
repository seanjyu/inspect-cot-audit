import instructor


def get_auditor_client(model_str: str):
    """
    Initializes and returns an Instructor-patched async client
    based on the provider-prefixed model string (e.g., 'ollama/...', 'openai/...').
    """
    return instructor.from_provider(model_str, async_client=True, mode=instructor.Mode.MD_JSON)
