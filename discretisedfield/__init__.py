import os
import pytest
import pkg_resources
from .region import Region
from .mesh import Mesh
from .field import Field
from .line import Line
from .interact import interact
import matplotlib.pyplot as plt

my_path = os.path.abspath(os.path.dirname(__file__))
path = os.path.join(my_path, 'util/discretisedfield-style.mplstyle')
plt.style.use(path)

__version__ = pkg_resources.get_distribution(__name__).version
__dependencies__ = pkg_resources.require(__name__)


def test():
    return pytest.main(['-v', '--pyargs',
                        'discretisedfield', '-l'])  # pragma: no cover
