"""Dataclass storing all similarity metrics."""

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np


@dataclass
class SimilarityMetrics:
    """Stores similarity metrics between two time-series sequences.

    Attributes:
        method: A string containing the name of the similarity function used.
        metrics A list containing speciifc outputs for each
            similarity function.
    """

    method: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dtw(
        cls, distance: float, warping_path: list[tuple[int, int]]
    ) -> "SimilarityMetrics":
        """Creates a SimilarityMetrics instance from DTW output.

        Args:
            distance: The cumulative distance between experimental and target sequences.
            warping_path: A list of tuples representing the warping path.

        Returns:
            A SimilarityMetrics instance storing DTW-specific metrics.
        """
        return cls(
            method="DTW",
            metrics={
                "distance": distance,
                "target_path": ([p[0] for p in warping_path]),
                "experimental_path": ([p[1] for p in warping_path]),
            },
        )


@dataclass
class ParticipantData:
    """Stores relevant participant information.

    Attributes:
        participant_ID: Integer value unique to every participant.
        sequence: Integer value associated with each iteration of a test.
        data: np.ndarray containing subject data for specified sequence.
    """

    participant_ID: str
    sequence_sheetname: str
    data: np.ndarray
