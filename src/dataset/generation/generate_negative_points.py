"""
Module for generating negative (unknown intent) examples for the dataset.

We want to show the model what kind of requests it is not capable of comleting.
"""

import random

from src.dataset.tools_description import UNKNOWN_INTENTS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def generate_unknown_intent_examples(count, start_id):
    """
    Generate a list of negative (unknown intent) examples for the dataset.

    Args:
        count (int): Number of unknown intent examples to generate.
        start_id (int): Starting ID for the generated examples.

    Returns:
        tuple: (examples, next_start_id)
            examples (list): List of generated example dicts.
            next_start_id (int): The next available ID after the last generated example.
    """
    examples = []
    logger.info(
        "Generating %d unknown intent examples starting from id %d", count, start_id
    )
    for _ in range(count):
        query = random.choice(UNKNOWN_INTENTS)
        logger.debug("Selected query: %s for id %d", query, start_id)
        examples.append(
            {
                "id": start_id,
                "query": query,
                "answers": [],
                "tools": [],
            }
        )
        start_id += 1
    logger.info("Generated %d examples. Next start_id: %d", len(examples), start_id)
    return examples, start_id
