"""smoke tests for orchestrator.py."""

import datetime
import pathlib

from mobi_motion_tracking.core import orchestrator


def test_orchestrator_good() -> None:
    """Smoke test for the orchestrator run function."""
    experimental_path = pathlib.Path("tests/sample_data/100.xlsx")
    gold_path = pathlib.Path("tests/sample_data/Gold.xlsx")
    sequence = [1]
    algorithm = "dtw"
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    expected_output_file = pathlib.Path(
        f"tests/sample_data/results_Gold_{date_str}.ndjson"
    )

    orchestrator.run(experimental_path, gold_path, sequence, algorithm)

    assert expected_output_file.exists(), "Expected file was not created"
