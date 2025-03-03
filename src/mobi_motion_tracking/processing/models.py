"""Dataclass storing all similarity metrics."""

from typing import Any

import numpy as np


class SimilarityMetrics:
    """Stores similarity metrics between two time-series sequences."""

    def __init__(self, method: str, metrics: dict[str, Any]) -> None:
        """Initialize a SimilarityMetrics instance.

        Args:
            method: The name of the similarity computation method.
            metrics: A dictionary storing computed values.
        """
        self.method = method
        self.metrics = metrics


def create_similarity_metrics_from_dtw(
    distance: float, path: list[tuple[int, int]]
) -> SimilarityMetrics:
    """Creates a SimilarityMetrics instance from DTW output.

    Args:
        distance: The cumulative distance between experimental and target sequences.
        path: A list of tuples representing the warping path.

    Returns:
        A SimilarityMetrics instance storing DTW-specific metrics.
    """
    return SimilarityMetrics(
        method="DTW",
        metrics={
            "distance": distance,
            "target_path": np.array([p[0] for p in path]),
            "experimental_path": np.array([p[1] for p in path]),
        },
    )
