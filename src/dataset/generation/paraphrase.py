"""
Module for generating paraphrased queries using an LLM.

We want to expand the dataset with diverse single-tool, multi-tool, and negative examples.
This shows the model the latent space of meaning that each request reflects.
"""

from litellm import completion

from settings import settings
from src.dataset.prompts import create_paraphrase_prompt
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def paraphrase_query(query):
    """
    Generate a paraphrased version of the given query using an LLM.

    Args:
        query (str): The input query string to be paraphrased.

    Returns:
        str or None: The paraphrased query if successful, otherwise None.
    """
    logger.info("Sending paraphrasing request for query: %r", query)
    prompt = create_paraphrase_prompt(query)
    response = completion(
        model=settings.dataset.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    logger.debug("Received response: %s", response)
    try:
        paraphrased_text = (
            response.get("choices", [{}])[0].get("message", {}).get("content")
        )
    except (KeyError, IndexError, TypeError, AttributeError) as parse_error:
        logger.warning(
            "Malformed paraphrase response for query %r: %s", query, parse_error
        )
        return None

    if paraphrased_text:
        logger.info("Paraphrased query: %r", paraphrased_text.strip())
        return paraphrased_text.strip()
    else:
        logger.warning(
            "Could not extract paraphrase from response for query: %r", query
        )
        return None


def paraphrase_dataset(dataset, start_id, count):
    """
    Generate paraphrased variants of queries from a dataset.

    Args:
        dataset (list): List of example dicts, each containing 'id', 'query', 'answers', and 'tools'.
        start_id (int): Starting ID to assign to the new paraphrased examples.
        count (int): Number of examples to paraphrase (at most).

    Returns:
        list: List of new example dicts with paraphrased queries and updated IDs.
    """
    paraphrased = []
    num_to_paraphrase = min(count, len(dataset))
    if num_to_paraphrase == 0:
        logger.info("No examples selected for paraphrasing.")
        return []

    logger.info("Starting paraphrasing for %d examples.", num_to_paraphrase)
    for i in range(num_to_paraphrase):
        ex = dataset[i]
        logger.info(
            "Processing example %d/%d (id=%s)",
            i + 1,
            num_to_paraphrase,
            ex.get("id", "N/A"),
        )
        new_query = paraphrase_query(ex["query"])
        if new_query and new_query != ex["query"]:
            logger.info("Original: %r | Paraphrased: %r", ex["query"], new_query)
            new_ex = {
                "id": start_id,
                "query": new_query,
                "answers": ex["answers"],
                "tools": ex["tools"],
            }
            paraphrased.append(new_ex)
            start_id += 1
            # Log progress periodically
            if (i + 1) % 20 == 0 or (i + 1) == num_to_paraphrase:
                logger.info("Paraphrased %d/%d examples...", i + 1, num_to_paraphrase)
        elif not new_query:
            logger.warning(
                "Skipping example %s due to paraphrasing error or empty result.",
                ex.get("id", "N/A"),
            )
        else:
            logger.info(
                "Paraphrased query is identical to original for example %s, skipping.",
                ex.get("id", "N/A"),
            )

    logger.info("Finished paraphrasing. Generated %d new examples.", len(paraphrased))
    return paraphrased
