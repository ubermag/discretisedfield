"""Finite-difference fields."""
import importlib.metadata
import os
import pathlib

import matplotlib.pyplot as plt
import pytest

from . import tools
from .field import Field
from .field_rotator import FieldRotator
from .interact import interact
from .line import Line
from .mesh import Mesh
from .operators import integrate
from .region import Region

# Enable default plotting style.
plt.style.use(pathlib.Path(__file__).parent / "plotting" / "plotting-style.mplstyle")

__version__ = importlib.metadata.version(__package__)


def test():
    """Run all package tests.

    Examples
    --------
    1. Run all tests.

    >>> import discretisedfield as df
    ...
    >>> # df.test()

    """
    return pytest.main(["-v", "--pyargs", "discretisedfield", "-l"])  # pragma: no cover
