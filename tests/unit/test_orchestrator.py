"""Tests orchestrator.py."""

import numpy as np
import pytest

from mobi_motion_tracking.core import models, orchestrator
from mobi_motion_tracking.io.readers import readers


def test_invalid_algorithm() -> None:
    """Test that an invalid algorithm raises a ValueError."""
    gold_data = np.array([[1, 2], [3, 4]])
    subject_data = np.array([[1, 2], [3, 4]])
    algorithm = "INVALID"

    with pytest.raises(ValueError, match=f"Unsupported algorithm '{algorithm}'."):
        orchestrator.run_algorithm(algorithm, gold_data, subject_data)
