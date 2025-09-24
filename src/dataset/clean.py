"""
Module for cleaning and deduplicating dataset entries using semantic similarity.
"""

import torch
from sentence_transformers import SentenceTransformer, util

from settings import settings
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def deduplicate_and_decontaminate_dataset(
    dataset, reference_queries=None, similarity_threshold=0.85
):
    """
    Remove duplicate and contaminated entries from a dataset using semantic similarity.

    Args:
        dataset (list): List of dataset entries (dicts), each containing a 'query' field.
        reference_queries (list, optional): List of queries to check for contamination.
        Defaults to None.
        similarity_threshold (float, optional): Cosine similarity threshold above which
        entries are considered duplicates or contaminated. Defaults to 0.85.

    Returns:
        list: Cleaned dataset with duplicates and contaminated entries removed.
    """
    if reference_queries is None:
        reference_queries = []
    logger.debug(
        "Sentence transformers model: %s", settings.dataset.SENTENCE_TRANSFORMER_MODEL
    )
    model = SentenceTransformer(settings.dataset.SENTENCE_TRANSFORMER_MODEL)
    clean_dataset = []
    seen_embeddings = []

    ref_embeddings = (
        model.encode(
            reference_queries, convert_to_tensor=True, normalize_embeddings=True
        )
        if reference_queries
        else None
    )

    for entry in dataset:
        query = entry.get("query", "").strip()
        if len(query) < 5:
            continue

        query_embedding = model.encode(
            query, convert_to_tensor=True, normalize_embeddings=True
        )

        if seen_embeddings:
            similarities = util.pytorch_cos_sim(
                query_embedding, torch.stack(seen_embeddings)
            )
            if torch.any(similarities > similarity_threshold):
                continue

        if ref_embeddings is not None:
            ref_similarities = util.pytorch_cos_sim(query_embedding, ref_embeddings)
            if torch.any(ref_similarities > similarity_threshold):
                continue

        seen_embeddings.append(query_embedding)
        clean_dataset.append(entry)

    return clean_dataset
