"""
Module for validating the format of function-calling dataset entries using LLMs.
"""

import json

from litellm import completion

from settings import settings
from src.utils.logging_config import get_logger

from .prompts import FORMAT_CHECK_PROMPT

logger = get_logger(__name__)


def check_entry_format(entry: dict, model=settings.dataset.LLM_MODEL):
    """
    Validate the format of a dataset entry using an LLM.

    Args:
        entry (dict): The dataset entry to validate.
        model (str, optional): The LLM model to use for validation. Defaults to
        settings.dataset.LLM_MODEL.

    Returns:
        str: The LLM's response indicating whether the entry is valid or describing any issues.
    """
    logger.debug("Entry format checking model: %d", model)
    message = (
        f"Validate the following entry:\n\n```json\n{json.dumps(entry, indent=2)}\n```"
    )

    response = completion(
        model=model,
        messages=[
            {"role": "system", "content": FORMAT_CHECK_PROMPT},
            {"role": "user", "content": message},
        ],
    )

    result = response["choices"][0]["message"]["content"].strip()
    return result


def run_format_checker(dataset: list):
    """
    Check the format of each entry in the dataset using an LLM.

    Args:
        dataset (list): List of dataset entries (dicts) to validate.

    Returns:
        tuple: (valid, invalid)
            valid (list): Entries deemed valid by the LLM.
            invalid (list): Tuples of (entry id, LLM response) for invalid entries.
    """
    valid, invalid = [], []

    for entry in dataset:
        result = check_entry_format(entry)
        if result.startswith("VALID"):
            valid.append(entry)
        else:
            invalid.append((entry["id"], result))

    return valid, invalid
