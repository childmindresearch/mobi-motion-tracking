"""Functions to write calculated outputs to a file."""

import datetime
import json
import pathlib

from mobi_motion_tracking.core import models


def generate_output_filename(gold_participant_ID: str, output_dir: pathlib.Path) -> str:
    """Generates a unique filename based on gold participant ID and date."""
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    base_filename = f"results_{gold_participant_ID}_{date_str}"

    counter = 1
    while (output_dir / f"{base_filename}_{counter:03}.ndjson").exists():
        counter += 1

    return f"{base_filename}_{counter:03}.ndjson"


def save_results_to_ndjson(
    gold_metadata: models.Metadata,
    subject_metadata: models.Metadata,
    similarity_metrics: models.SimilarityMetrics,
    output_dir: pathlib.Path,
) -> None:
    """Appends results to a newline-delimited JSON file (NDJSON)."""
    new_entry = {
        "gold_filename": gold_metadata.participant_ID,
        "participant_ID": subject_metadata.participant_ID,
        "sheetname": subject_metadata.sequence_sheetname,
        "method": similarity_metrics.method,
        "metric": similarity_metrics.metrics,
    }

    output_path = generate_output_filename(gold_metadata.participant_ID, output_dir)

    with open(output_path, "a") as f:
        json.dump(new_entry, f)
        f.write("\n")
