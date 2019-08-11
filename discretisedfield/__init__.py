import pytest
import pkg_resources
import matplotlib
matplotlib.use('Agg')
from .mesh import Mesh
from .field import Field
from .region import Region

__version__ = pkg_resources.get_distribution(__name__).version
__dependencies__ = pkg_resources.require(__name__)


def test():
    return pytest.main(['-v', '--pyargs',
                        'discretisedfield', '-l'])  # pragma: no cover
