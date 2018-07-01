import pkg_resources
import matplotlib as mpl
mpl.use("agg")
from .mesh import Mesh
from .field import Field
from .read import read


def test():
    import pytest  # pragma: no cover
    return pytest.main(["-v", "--pyargs", "discretisedfield"])  # pragma: no cover

__version__ = pkg_resources.get_distribution(__name__).version
__dependencies__ = pkg_resources.require(__name__)
