"""Functions to write calculated outputs to a file."""

import json
import pathlib

from mobi_motion_tracking.core import models


def save_results_to_ndjson(
    gold_metadata: models.Metadata,
    subject_metadata: models.Metadata,
    similarity_metrics: models.SimilarityMetrics,
    output_path: pathlib.Path,
) -> None:
    """Appends results to a newline-delimited JSON file (NDJSON)."""
    new_entry = {
        "gold_filename": gold_metadata.participant_ID,
        "participant_ID": subject_metadata.participant_ID,
        "sheetname": subject_metadata.sequence_sheetname,
        "method": similarity_metrics.method,
        "metric": similarity_metrics.metrics,
    }

    with open(output_path, "a") as f:
        json.dump(new_entry, f)
        f.write("\n")
