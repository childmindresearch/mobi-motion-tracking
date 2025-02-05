"""Fixtures used by pytest."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_excel_path() -> Path:
    """Test data for sample .xlsx file."""
    return Path(__file__).parent / "sample_data" / "valid_file.xlsx"
