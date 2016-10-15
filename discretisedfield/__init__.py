# use most conservative backend 
import matplotlib
matplotlib.use("agg")

from .mesh import Mesh
from .field import Field, read_oommf_file


def test():
    import pytest  # pragma: no cover
    pytest.main(["-v", "--pyargs", "discretisedfield"])  # pragma: no cover
