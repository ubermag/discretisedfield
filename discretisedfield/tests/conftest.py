import pytest

import discretisedfield as df

valid_mesh_args = [
    # pmin, pmax, n, cell
    [0, 20e-9, None, 1e-9],
    [(0, 0), (20e-9, 10e-9), None, (1e-9, 1e-9)],
    [(0, 0, 0), (5, 5, 5), [1, 1, 1], None],
    [(-1, 0, -3), (5, 7, 9), None, (1, 1, 1)],
    [(0, 0, 0), (10e-9, 10e-9, 10e-9), None, (1e-9, 1e-9, 1e-9)],
    [(0, 0, 0), (20e-9, 10e-9, 5e-9), None, (1e-9, 1e-9, 1e-9)],
    [(0, 0, 0), (20e-9, 10e-9, 5e-9), (1, 1, 1), None],
    [(0, 0, 0, 0), (20e-9, 10e-9, 5e-9, 2e-9), None, (1e-9, 1e-9, 1e-9, 1e-9)],
]


@pytest.fixture(params=valid_mesh_args)
def valid_mesh(request):
    p1, p2, n, cell = request.param
    return df.Mesh(p1=p1, p2=p2, n=n, cell=cell)


@pytest.fixture
def mesh_3d():
    return df.Mesh(
        p1=(0, 0, -10e-9), p2=(50e-9, 30e-9, 20e-9), cell=(10e-9, 10e-9, 5e-9)
    )
