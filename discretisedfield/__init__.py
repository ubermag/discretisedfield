import matplotlib
matplotlib.use("nbagg")

from .mesh import Mesh
from .field import Field, read_oommf_file


def test():
    import pytest  # pragma: no cover
    pytest.main()  # pragma: no cover
