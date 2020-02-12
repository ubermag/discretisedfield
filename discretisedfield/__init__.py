import pytest
import pkg_resources
import seaborn as sns
from .region import Region
from .mesh import Mesh
from .field import Field
from .line import Line
from .operators import cross, stack
from .interact import interact

sns.set(style='whitegrid')

__version__ = pkg_resources.get_distribution(__name__).version
__dependencies__ = pkg_resources.require(__name__)


def test():
    return pytest.main(['-v', '--pyargs',
                        'discretisedfield', '-l'])  # pragma: no cover
