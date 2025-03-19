"""Test writers.py functions."""

import datetime
import pathlib
import tempfile
from typing import Generator

import pytest

from mobi_motion_tracking.io.writers import writers


@pytest.fixture
def temp_output_dir() -> Generator[pathlib.Path]:
    """Fixture to create and clean up a temporary directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield pathlib.Path(temp_dir)


def test_correct_filename_generation(temp_output_dir: pathlib.Path) -> None:
    """Test that the generated filename follows the expected format."""
    participant_id = "Gold"
    date_str = datetime.datetime.now().strftime("%m%d%Y")
    expected_filename = f"results_{participant_id}_{date_str}.ndjson"

    result = writers.generate_output_filename(participant_id, temp_output_dir)

    assert result.endswith(expected_filename), (
        "Resulting filename is assigned \
        incorrectly."
    )
    assert result.exists(), "Output file was not generated."


def test_existing_file_handling(temp_output_dir: pathlib.Path) -> None:
    """Test that the function does not overwrite an existing file."""
    participant_id = "Gold"

    result_path = pathlib.Path(
        writers.generate_output_filename(participant_id, temp_output_dir)
    )
    result_path.write_text("Test data")
    result_path = writers.generate_output_filename(participant_id, temp_output_dir)

    assert result_path.read_text() == "Test data", (
        "Contents of output file has been \
        overwritten."
    )


def test_different_participant_ids(temp_output_dir: pathlib.Path) -> None:
    """Test that different participant IDs generate different filenames."""
    id1 = "GoldA"
    id2 = "GoldB"

    file1 = writers.generate_output_filename(id1, temp_output_dir)
    file2 = writers.generate_output_filename(id2, temp_output_dir)

    assert file1 != file2, (
        "Output files are the same. A different fi;e should be \
        generated for each new Gold ID."
    )
