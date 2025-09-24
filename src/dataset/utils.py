import json

from datasets import load_dataset
from huggingface_hub.errors import EntryNotFoundError
from requests.exceptions import RequestException
from settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def render_template(template, args):
    return template.format(**args)


def extract_datapoints_hf_dataset(
    dataset=settings.dataset.FEW_SHOT_EXAMPLES_DATASET, num_datapoints=1000
) -> list[dict]:
    """
    Extracts a specified number of datapoints from a Hugging Face dataset.

    Args:
        dataset (str): The name of the Hugging Face dataset to load.
        num_datapoints (int): The number of datapoints to extract.
    """

    logger.debug("Using %s for few-shot examples", dataset)

    str_data = []

    try:
        datasets = load_dataset(dataset)

        shuffled = datasets["train"].shuffle(seed=42)

        first_1000 = shuffled.select(range(num_datapoints))

        for datapoint in first_1000:
            str_data.append(json.dumps(datapoint, indent=2))
    except (
        EntryNotFoundError,
        FileNotFoundError,
        RequestException,
        ValueError,
        RuntimeError,
    ) as e:
        # Non-fatal: proceed without few-shot examples
        logger.warning(
            "Failed to load HF dataset '%s': %s. Continuing without examples.",
            dataset,
            e,
        )
        return []

    return str_data
