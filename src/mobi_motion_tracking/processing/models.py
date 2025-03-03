"""Dataclass storing all similarity metrics."""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SimilarityMetrics:
    """Stores similarity metrics between two time-series sequences."""

    method: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dtw(
        cls, distance: float, path: list[tuple[int, int]]
    ) -> "SimilarityMetrics":
        """Creates a SimilarityMetrics instance from DTW output.

        Args:
            distance: The cumulative distance between experimental and target sequences.
            path: A list of tuples representing the warping path.

        Returns:
            A SimilarityMetrics instance storing DTW-specific metrics.
        """
        return cls(
            method="DTW",
            metrics={
                "distance": distance,
                "target_path": np.array([p[0] for p in path]),
                "experimental_path": np.array([p[1] for p in path]),
            },
        )
