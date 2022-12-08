import pytest

import discretisedfield as df

valid_mesh_args = [
    # pmin, pmax, n, cell
    [(0, 0, 0), (5, 5, 5), [1, 1, 1], None],
    [(-1, 0, -3), (5, 7, 9), None, (1, 1, 1)],
    [(0, 0, 0), (10e-9, 10e-9, 10e-9), None, (1e-9, 1e-9, 1e-9)],
    [(0, 0, 0), (20e-9, 10e-9, 5e-9), None, (1e-9, 1e-9, 1e-9)],
    [(0, 0, 0), (20e-9, 10e-9, 5e-9), (1, 1, 1), None],
]


@pytest.fixture(params=valid_mesh_args)
def valid_mesh(request):
    p1, p2, n, cell = request.param
    return df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
