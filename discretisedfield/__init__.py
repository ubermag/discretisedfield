"""Main package."""
import os
import pytest
import pkg_resources
from .region import Region
from .mesh import Mesh
from .field import Field
from .line import Line
from .operators import DValue, dx, dy, dz, dV, dS, integral
from .interact import interact
import matplotlib.pyplot as plt

# Enable default plotting style.
dirname = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(dirname, './util/plotting-style.mplstyle')
plt.style.use(path)

__version__ = pkg_resources.get_distribution(__name__).version
__dependencies__ = pkg_resources.require(__name__)


def test():
    """Run all package tests.

    Examples
    --------
    1. Run all tests.

    >>> import discretisedfield as df
    ...
    >>> # df.test()

    """
    return pytest.main(['-v', '--pyargs',
                        'discretisedfield', '-l'])  # pragma: no cover
