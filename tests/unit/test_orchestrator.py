"""test orchestrator.py functions."""

import pytest

from mobi_motion_tracking.core import orchestrator


def test_run_file_good() -> None:
    """Tests the run_file function works properly."""
