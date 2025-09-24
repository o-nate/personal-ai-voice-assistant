"""
Module for creating and managing the dataset generation pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

from datasets import Dataset
from huggingface_hub import list_repo_files

from settings import settings
from src.utils.logging_config import get_logger
from src.functions.functions import available_function_calls

from .generation.generate_multi_tool_points import (
    generate_multi_tool_examples,
)
from .generation.generate_negative_points import (
    generate_unknown_intent_examples,
)
from .generation.generate_single_tool_points import (
    generate_single_tool_examples,
)
from .generation.paraphrase import paraphrase_dataset
from .tools_description import TOOLS
from .check_format import run_format_checker
from .execution_checker import run_execution_checker_parallel

logger = get_logger(__name__)


def build_dataset(
    single_tool_examples_per_tool=50,
    multi_tool_examples=50,
    unknown_intent_examples=30,
    paraphrase_count=100,
    dataset_name: str = "dataset.json",
    use_hf_examples: bool = True,
    hf_num_examples: int = 5,
    validate_format: bool = False,
    fail_on_invalid: bool = False,
    format_report: Optional[str] = None,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None,
    hub_config_name: Optional[str] = None,
    min_success_rate: float = 0.0,
    skip_execution_checker: bool = False,
    exec_sample_size: int = 50,
):
    """
    Build a dataset by generating single-tool, multi-tool, and unknown intent examples,
    optionally adding paraphrased variants, and saving the result to a file.

    Args:
        single_tool_examples_per_tool (int, optional): Number of examples to generate for each tool.
        Defaults to 50.
        multi_tool_examples (int, optional): Number of multi-tool examples to generate. Defaults to
        50.
        unknown_intent_examples (int, optional): Number of unknown intent examples to generate.
        Defaults to 30.
        paraphrase_count (int, optional): Number of paraphrased examples to generate from the base
        dataset.
        Defaults to 100.
        dataset_name (str, optional): Name of the output dataset file. Defaults to "dataset.json".

    Returns:
        None. The generated dataset is saved to the specified file.
    """
    logger.info("Starting dataset generation pipeline...")
    dataset = []
    idx = 1

    # Generate single tool examples to teach the model how to handle simple, direct tasks
    logger.info("Generating single-tool examples...")
    total_single_tools = 0
    for tool_name, tool_data in TOOLS.items():
        single_tool, idx = generate_single_tool_examples(
            tool_name, tool_data, single_tool_examples_per_tool, idx
        )
        dataset.extend(single_tool)
        total_single_tools += len(single_tool)
    logger.info("Generated %d single-tool examples.", total_single_tools)

    # Generate multi-tool examples, using few-shot examples from HuggingFace model
    logger.info("Generating multi-tool examples...")
    multi_tool, idx = generate_multi_tool_examples(
        multi_tool_examples,
        idx,
        use_hf_examples=use_hf_examples,
        num_examples=hf_num_examples,
    )
    dataset.extend(multi_tool)
    logger.info("Generated %d multi-tool examples.", len(multi_tool))

    # Generate unknown intent examples
    logger.info("Generating unknown intent examples...")
    unknowns, idx = generate_unknown_intent_examples(unknown_intent_examples, idx)
    dataset.extend(unknowns)
    logger.info("Generated %d unknown intent examples.", len(unknowns))

    logger.info("Generated base dataset with %d examples.", len(dataset))

    # Paraphrase
    if paraphrase_count > 0:
        paraphrased = paraphrase_dataset(dataset, 1000, paraphrase_count)
        full_dataset = dataset + paraphrased
        logger.info("Added %d paraphrased examples.", len(paraphrased))
    else:
        full_dataset = dataset
        logger.info("Skipping paraphrasing as paraphrase_count is 0.")

    # Optionally validate format with LLM
    final_dataset = full_dataset
    if validate_format:
        logger.info("Validating dataset format with LLM...")
        valid, invalid = run_format_checker(full_dataset)
        logger.info("Format check: %d valid, %d invalid", len(valid), len(invalid))

        # Optionally write a report of invalid entries
        if format_report:
            try:
                report_payload = {
                    "valid_count": len(valid),
                    "invalid_count": len(invalid),
                    "invalid": [
                        {"id": entry_id, "reason": reason}
                        for entry_id, reason in invalid
                    ],
                }
                with open(format_report, "w", encoding="utf-8") as rf:
                    json.dump(report_payload, rf, indent=2)
                logger.info("Wrote format report to %s", format_report)
            except IOError as e:
                logger.error(
                    "Failed to write format report to %s: %s", format_report, e
                )

        if fail_on_invalid and invalid:
            logger.error(
                "Invalid entries found and fail_on_invalid is set. Aborting save."
            )
            return

        # Use only valid entries if validation was requested
        final_dataset = valid

    # Save
    try:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        out_path = data_dir / dataset_name
        logger.info("Saving final dataset to %s...", out_path)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(final_dataset, f, indent=2)
        logger.info(
            "Saved final dataset with %d examples to %s", len(final_dataset), out_path
        )
    except IOError as e:
        logger.error("Failed to save dataset to %s: %s", out_path, e, exc_info=True)

    logger.info("Dataset generation pipeline finished.")

    # Optionally run execution checker and gate on success rate
    if not skip_execution_checker:
        logger.info("Running execution checker on a sample of the dataset...")
        try:
            sample = final_dataset[:exec_sample_size]

            report = run_execution_checker_parallel(sample, available_function_calls)
            logger.info(
                "Execution checker: %d calls | success rate %.2f%%",
                report["total_function_calls"],
                report["success_rate"],
            )
            if report["success_rate"] < min_success_rate * 100.0:
                logger.error(
                    "Success rate %.2f%% is below threshold %.2f%%; skipping Hub push.",
                    report["success_rate"],
                    min_success_rate * 100.0,
                )
                push_to_hub = False
        except Exception as e:
            logger.error("Execution checker failed: %s", e, exc_info=True)
            push_to_hub = False

    # Optionally push to Hugging Face Hub
    if push_to_hub:
        try:
            repo_id = hub_repo_id or settings.auth.HF_DATASET_REPO_ID
            if not repo_id:
                logger.error(
                    "No Hub repo id provided. Set --hub-repo-id or HF_DATASET_REPO_ID."
                )
                return
            ds = Dataset.from_list(final_dataset)

            # Dynamically assign version if not provided, to avoid overwriting
            config_name = hub_config_name
            if not config_name:

                # Try to find the next available version tag (v0, v1, ...)
                base_version = 0
                existing_versions = set()
                try:
                    files = list_repo_files(repo_id)
                    for f in files:
                        # Look for config_name in filenames like 'dataset/config_name/xxx'
                        parts = f.split("/")
                        if len(parts) > 1:
                            existing_versions.add(parts[1])
                    while f"v{base_version}" in existing_versions:
                        base_version += 1
                    config_name = f"v{base_version}"
                    logger.info("Auto-selected config_name/version: %s", config_name)
                except Exception as e:
                    logger.warning(
                        "Could not check existing versions on Hub, defaulting to v0: %s",
                        e,
                    )
                    config_name = "v0"

            logger.info(
                "Pushing dataset to Hub repo '%s' with config '%s'...",
                repo_id,
                config_name,
            )
            push_kwargs = {"config_name": config_name}
            ds.push_to_hub(repo_id, **push_kwargs)
            logger.info(
                "Successfully pushed dataset to Hugging Face Hub: %s (config: %s)",
                repo_id,
                config_name,
            )
        except Exception as e:
            logger.error(
                "Failed to push dataset to Hugging Face Hub: %s", e, exc_info=True
            )


def main() -> None:
    "Run script"
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument(
        "--single-tool-examples",
        type=int,
        default=50,
        help="Number of examples per single tool",
    )
    parser.add_argument(
        "--multi-tool-examples",
        type=int,
        default=50,
        help="Number of multi-tool examples",
    )
    parser.add_argument(
        "--unknown-intent-examples",
        type=int,
        default=30,
        help="Number of unknown intent examples",
    )
    parser.add_argument(
        "--paraphrase-count",
        type=int,
        default=10,
        help="Number of paraphrased examples",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="dataset.json",
        help="Output dataset filename",
    )
    parser.add_argument(
        "--hf-examples",
        type=int,
        default=3,
        help="Number of HF few-shot examples to include in multi-tool prompts",
    )
    parser.add_argument(
        "--no-hf-examples",
        action="store_true",
        help="Disable using HF few-shot examples in multi-tool prompts",
    )
    parser.add_argument(
        "--validate-format",
        action="store_true",
        help="Validate dataset entries with an LLM format checker before saving",
    )
    parser.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="If validation finds invalid entries, abort saving",
    )
    parser.add_argument(
        "--format-report",
        type=str,
        default=None,
        help="Optional path to write a JSON report of invalid entries",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the final dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="Target HF repo id (fallbacks to HF_DATASET_REPO_ID in .env)",
    )
    parser.add_argument(
        "--hub-config-name",
        type=str,
        default=None,
        help="Optional Hub config/version name, e.g., v0.3",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.0,
        help="Minimum execution success rate (0.0-1.0) required to push",
    )
    parser.add_argument(
        "--skip-execution-checker",
        action="store_true",
        help="Skip execution checker gating before pushing",
    )
    parser.add_argument(
        "--exec-sample-size",
        type=int,
        default=50,
        help="Number of examples to sample for execution checker",
    )

    args = parser.parse_args()

    build_dataset(
        single_tool_examples_per_tool=args.single_tool_examples,
        multi_tool_examples=args.multi_tool_examples,
        unknown_intent_examples=args.unknown_intent_examples,
        paraphrase_count=args.paraphrase_count,
        dataset_name=args.dataset_name,
        use_hf_examples=not args.no_hf_examples,
        hf_num_examples=args.hf_examples,
        validate_format=args.validate_format,
        fail_on_invalid=args.fail_on_invalid,
        format_report=args.format_report,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id,
        hub_config_name=args.hub_config_name,
        min_success_rate=args.min_success_rate,
        skip_execution_checker=args.skip_execution_checker,
        exec_sample_size=args.exec_sample_size,
    )


if __name__ == "__main__":
    main()
