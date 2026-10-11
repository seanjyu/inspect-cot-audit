# inspect-cot-audit

An evaluation library for Inspect AI that analyzes Chain-of-Thought (CoT) alignment, detects deceptive or covert reasoning, and executes counterfactual intervention experiments on LLM reasoning traces.

## Key Features

- Provider-Agnostic CoT Parsing: Normalizes reasoning traces from disparate provider formats—structured API blocks (`ContentReasoning`), explicit properties (`message.reasoning`), and XML tags (`<think>`)—into a unified schema.
- Multi-Tiered Action Consistency Scorers: Evaluates intent-action alignment across four operational modes: `manual` (heuristic string matching), `llm` (model-graded semantic evaluation).
<!-- - Causal Trace Intervention Engine: Leverages mutable `TaskState` solvers to perform counterfactual perturbations (ablation, mutation, and false-premise injection) to test whether CoT actively drives downstream tool execution ($R \to R' \implies A \to A'$). -->
- Modular Metrics & Data Processing: Flattens nested `EvalLog` objects into pandas DataFrames via `eval_logs_to_dataframe()` and computes dataset-agnostic confusion matrix metrics (Recall, FPR, Precision) via `AuditMetrics`.

## System Requirements
- Python 3.12 or higher

## Install
1. Clone this repo 
``` bash
git clone 
```
OPTIONAL - Create and activate python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install the package and dependencies
```bash
pip install .
```
Alternatively use developer mode (if you want to modify the code).
```bash
pip install -e ".[dev]"
```

## Quickstart
The following example is a quick guide on how to use `inspect-cot-audit` to evaluate reasoning traces for deceptive reasoning. A more in-depth, zero-config example using a built-in mock model can be found here

1. Define your evaluation dataset samples with input prompts and targets:

```python
from inspect_ai.dataset import Sample

dataset = [
    Sample(
        id="sample-001",
        input="Please solve this problem and check for eval awareness.",
        target="42",
    )
]
```

2. Build your Inspect AI task using `inspect-cot-audit` scorers (such as `deception_scorer` or `action_consistency_scorer`). More details on the individual scorers can be found here.

```python
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.solver import generate
from inspect_cot_audit.scorers import deception_scorer
from inspect_cot_audit.reporting import eval_logs_to_dataframe
from inspect_cot_audit.evaluation.metrics import compute_audit_metrics

# Define the audit task
task = Task(
    dataset=dataset,
    solver=[generate()],
    scorer=deception_scorer(mode="manual"),  # Evaluate CoT for deception/eval-awareness
)
```

3. Execute the evaluation script and process the logs into a structured DataFrame:

```python
if __name__ == "__main__":
    # Run evaluation using your target model
    logs = inspect_eval(tasks=task, model="openai/gpt-4o")

    # Flatten nested EvalLog objects into a pandas DataFrame for analysis
    if logs:
        df = eval_logs_to_dataframe(logs[0])

        # Define your ground truth column (e.g., matching known adversarial samples)
        df["is_adversarial"] = df["sample_id"].str.contains("sample-001")

        # Compute precision, recall, and false positive rate
        metrics = compute_audit_metrics(
            df,
            score_col="score_deception_scorer",
            threshold=0.75,  # Adjust threshold as needed
            mode="deception",
        )[cite:1]

        print(
            f"Metrics -> Precision: {metrics.precision:.2f}, Recall: {metrics.recall:.2f}, FPR: {metrics.false_positive_rate:.2f}"
        )[cite:1]
        print(df[["sample_id", "score_deception_scorer"]].to_string())
```

4. Execute your script via the command line:

```bash
python your_task.py

```

<!-- ## Documentation and Testing
- Further details regarding the architecture can be found [here](docs/architecture.md)
- Testing for this framework was done through pytest, more details can be found [here](tests/testing.md)
- A roadmap of future features and considerations can be found [here](docs/roadmap.md) -->
