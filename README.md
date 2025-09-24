# personal-ai-voice-assistant
A private, locally hosted AI-powered voice assistant for your mobile device.

## Setup

- Python 3.11 recommended
- Install via uv (or pip):
```bash
uv sync
```
- Create a `.env` in project root (see `.env.example`):
```
OPENAI_API_KEY=sk-...
HUGGINGFACE_TOKEN=hf_...
HF_DATASET_REPO_ID=your-org/function-calling-dataset
```

## Settings

All configuration lives in `settings.py` (Pydantic v2). Notable fields:
- `settings.dataset.LLM_MODEL` (default `gpt-4o-mini`)
- `settings.dataset.FEW_SHOT_EXAMPLES_DATASET` (HF dataset for few-shots)
- `settings.auth.HF_DATASET_REPO_ID` (fallback repo id for Hub push)

## CLI: Generate Dataset

Entrypoint: `src/dataset/create_dataset.py`

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 50 \
  --multi-tool-examples 50 \
  --unknown-intent-examples 30 \
  --paraphrase-count 100 \
  --dataset-name dataset.json
```

Options for few-shots (to keep prompts small/fast):
- `--hf-examples <int>`: number of HF examples in multi-tool prompts (default 3)
- `--no-hf-examples`: disable HF examples entirely

Format checking (LLM-based) and reporting:
- `--validate-format`: validate each entry with an LLM
- `--fail-on-invalid`: abort save if any invalid entries
- `--format-report <path>`: write JSON report of invalid entries

Execution-gated Hub push (versioned):
- `--push-to-hub`: push final dataset to the Hugging Face Hub
- `--hub-repo-id <repo>`: overrides `HF_DATASET_REPO_ID` in `.env`
- `--hub-config-name <name>`: version/config label (e.g., `v0.3`)
- `--min-success-rate <0.0-1.0>`: minimum execution success rate to allow push
- `--skip-execution-checker`: skip execution gate
- `--exec-sample-size <int>`: sample size for the execution checker (default 50)

Example (small, fast, validated, gated, and versioned push):
```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 2 \
  --multi-tool-examples 2 \
  --unknown-intent-examples 2 \
  --paraphrase-count 0 \
  --hf-examples 3 \
  --validate-format \
  --format-report data\format_report.json \
  --min-success-rate 0.9 \
  --exec-sample-size 20 \
  --push-to-hub \
  --hub-config-name v0.3
```

## CLI: Execution Checker (standalone)

Runs generated function calls and reports pass/fail.

```bash
uv run -m src.dataset.execution_checker
```

Outputs `execution_report.json` and prints a summary.

## CLI recipe examples

Click the dropdown button on each example for the CLI commands.

<details>
  <summary><strong>(small, fast, validated, gated, and versioned push)</strong></summary>

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 2 \
  --multi-tool-examples 2 \
  --unknown-intent-examples 2 \
  --paraphrase-count 0 \
  --hf-examples 3 \
  --validate-format \
  --format-report data\format_report.json \
  --min-success-rate 0.9 \
  --exec-sample-size 20 \
  --push-to-hub \
  --hub-config-name v0.3
```
</details>

<details>
  <summary><strong>(generate locally with minimal HF context; no validation, no push)</strong></summary>

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 10 \
  --multi-tool-examples 10 \
  --unknown-intent-examples 5 \
  --paraphrase-count 0 \
  --hf-examples 3 \
  --dataset-name dataset.json
```
</details>

<details>
  <summary><strong>(disable HF few-shots entirely for quickest generation)</strong></summary>

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 5 \
  --multi-tool-examples 5 \
  --unknown-intent-examples 5 \
  --paraphrase-count 0 \
  --no-hf-examples \
  --dataset-name dataset.json
```
</details>

<details>
  <summary><strong>(validate format and fail if any invalid entries)</strong></summary>

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 10 \
  --multi-tool-examples 10 \
  --unknown-intent-examples 5 \
  --paraphrase-count 0 \
  --hf-examples 3 \
  --validate-format \
  --fail-on-invalid \
  --format-report data\format_report.json
```
</details>

<details>
  <summary><strong>(gated push with explicit repo id and semantic version)</strong></summary>

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 20 \
  --multi-tool-examples 20 \
  --unknown-intent-examples 10 \
  --paraphrase-count 0 \
  --hf-examples 5 \
  --validate-format \
  --format-report data\format_report.json \
  --min-success-rate 0.95 \
  --exec-sample-size 50 \
  --push-to-hub \
  --hub-repo-id your-username/function-calling-dataset \
  --hub-config-name v1.0.0
```
</details>

<details>
  <summary><strong>(push without gating: skip execution checker)</strong></summary>

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 10 \
  --multi-tool-examples 10 \
  --unknown-intent-examples 10 \
  --paraphrase-count 0 \
  --hf-examples 3 \
  --validate-format \
  --format-report data\format_report.json \
  --skip-execution-checker \
  --push-to-hub \
  --hub-config-name v0.4
```
</details>

<details>
  <summary><strong>(heavier dataset, light paraphrasing, and gated push)</strong></summary>

```bash
uv run -m src.dataset.create_dataset \
  --single-tool-examples 50 \
  --multi-tool-examples 50 \
  --unknown-intent-examples 30 \
  --paraphrase-count 20 \
  --hf-examples 5 \
  --validate-format \
  --format-report data\format_report.json \
  --min-success-rate 0.9 \
  --exec-sample-size 100 \
  --push-to-hub \
  --hub-config-name v1.1.0
```
</details>

<details>
  <summary><strong>(standalone execution checker on existing dataset)</strong></summary>

```bash
uv run -m src.dataset.execution_checker
```
</details>

## CLI: Format Checker (standalone)

Use the format checker utilities to validate a dataset you already have:

```python
from src.dataset.check_format import run_format_checker
import json

with open("data/dataset.json", encoding="utf-8") as f:
    ds = json.load(f)

valid, invalid = run_format_checker(ds)
print(len(valid), "valid", len(invalid), "invalid")
```

## Hugging Face Dataset Versioning

- Adopt semantic versions for `--hub-config-name` (e.g., `v1.0.0`).
- Ensure you’re logged in:
```bash
huggingface-cli login
```
- Consumers can load a specific version:
```python
from datasets import load_dataset
ds = load_dataset("your-org/function-calling-dataset", revision="v1.0.0")
```

## Troubleshooting

- Slow multi-tool generation: reduce `--hf-examples` or add `--no-hf-examples`.
- HF dataset 404/timeouts: code falls back and continues without few-shots.
- Windows console emoji errors: logs may include emoji; output still proceeds.