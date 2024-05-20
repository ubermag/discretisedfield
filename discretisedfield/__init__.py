"""Finite-difference fields."""

import importlib.metadata
import pathlib

import matplotlib.pyplot as plt
import pytest

from . import tools as tools
from .field import Field as Field
from .field_rotator import FieldRotator as FieldRotator
from .interact import interact as interact
from .line import Line as Line
from .mesh import Mesh as Mesh
from .operators import integrate as integrate
from .region import Region as Region

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
