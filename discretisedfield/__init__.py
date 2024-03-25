"""Finite-difference fields."""

import importlib.metadata
import pathlib

import matplotlib.pyplot as plt
import pytest

from . import tools  # noqa: F401
from .field import Field  # noqa: F401
from .field_rotator import FieldRotator  # noqa: F401
from .interact import interact  # noqa: F401
from .line import Line  # noqa: F401
from .mesh import Mesh  # noqa: F401
from .operators import integrate  # noqa: F401
from .region import Region  # noqa: F401

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
