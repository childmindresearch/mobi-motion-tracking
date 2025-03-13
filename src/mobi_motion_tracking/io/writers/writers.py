"""Functions to write calculated outputs to a file."""

import json
import pathlib

from mobi_motion_tracking.core import models


def save_results_to_json(
    metadata: models.Metadata,
    similarity_metrics: models.SimilarityMetrics,
    output_path: pathlib.Path,
) -> None:
    """Writes output to json file."""
    results = [
        {
            "participant_ID": metadata.participant_ID,
            "sheetname": metadata.sequence_sheetname,
            "method": similarity_metrics.method,
            "metric": similarity_metrics.metrics,
        }
    ]

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
