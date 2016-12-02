import matplotlib as mpl
mpl.use("agg")  # use most conservative backend
from .mesh import Mesh
from .field import Field, read_oommf_file


def test():
    import pytest  # pragma: no cover
    pytest.main(["-v", "--pyargs", "discretisedfield"])  # pragma: no cover
