"""Utilities to extract audit metrics from Inspect EvalLogs into Pandas DataFrames."""

from typing import Any

import pandas as pd
from inspect_ai.log import EvalLog


def eval_logs_to_dataframe(logs: EvalLog | list[EvalLog]) -> pd.DataFrame:
    """Converts one or more Inspect AI EvalLogs into a tabular Pandas DataFrame for analysis.

    Args:
        logs: Single EvalLog or list of EvalLogs returned by inspect_ai.read_eval_log().

    Returns:
        pd.DataFrame containing sample-level audit metrics and metadata.
    """
    if isinstance(logs, EvalLog):
        log_list = [logs]
    else:
        log_list = logs

    records: list[dict[str, Any]] = []

    for log in log_list:
        eval_id = getattr(log.eval, "run_id", "unknown")
        model_name = getattr(log.eval, "model", "unknown")
        task_name = getattr(log.eval, "task", "unknown")

        if not log.samples:
            continue

        for sample in log.samples:
            row: dict[str, Any] = {
                "eval_id": eval_id,
                "task": task_name,
                "model": model_name,
                "sample_id": sample.id,
                "target": str(sample.target),
            }

            if sample.scores:
                for score_name, score_obj in sample.scores.items():
                    row[f"score_{score_name}"] = score_obj.value
                    if score_obj.metadata:
                        for meta_k, meta_v in score_obj.metadata.items():
                            row[f"{score_name}_{meta_k}"] = meta_v

            records.append(row)

    return pd.DataFrame(records)
