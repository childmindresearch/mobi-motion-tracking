"""CLI for mobi-motion-tracking."""

import argparse
import pathlib
from typing import List, Optional

from mobi_motion_tracking.core import orchestrator


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Argument parser for mobi-motion-tracking cli.

    Args:
        args: A list of command line arguments given as strings. If None, the parser
            will take the args from `sys.argv`.

    Returns:
        Namespace object with all the input arguments and default values.
    """
