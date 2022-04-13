"""Finite-difference fields."""
import os

import matplotlib.pyplot as plt
import pkg_resources
import pytest

from .field import Field
from .field_rotator import FieldRotator
from .interact import interact
from .line import Line
from .mesh import Mesh
from .operators import DValue, dS, dV, dx, dy, dz, integral
from .region import Region

# Enable default plotting style.
dirname = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(dirname, "./util/plotting-style.mplstyle")
plt.style.use(path)

__version__ = pkg_resources.get_distribution(__name__).version


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
