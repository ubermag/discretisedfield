import matplotlib
matplotlib.use("nbagg")

from .mesh import Mesh
from .field import Field, read_oommf_file


def test():
    import pytest
    pytest.main()
