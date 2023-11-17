import numbers
import re
import types

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pytest

import discretisedfield as df

from .test_region import html_re as region_html_re

html_re = (
    r"<strong>Mesh</strong>\s*<ul>\s*"
    rf"<li>{region_html_re}</li>\s*"
    r"<li>n = .*</li>\s*"
    r"(<li>bc = ([xyz]{1,3}|neumann|dirichlet)<li>)?\s*"
    rf"(<li>subregions:\s*<ul>\s*(<li>{region_html_re}</li>\s*)+</ul></li>)?"
    r"</ul>"
)


@pytest.mark.parametrize(
    "p1, p2, n, cell",
    [
        # 1d
        [0, 2e-10, 1, None],
        [0, 2e-10, None, 1e-10],
        [(5e-9,), (2e-10,), 1, None],
        [(5e-9,), -2e-9, None, 1e-9],
        # 2d
        [(0, 0), (5, 7), (5, 7), None],
        [(0, 0), (5, 7), None, (0.5, 0.5)],
        [(3, -1), [0, 5], (1, 1), None],
        [(3, -1), [0, 5], None, (1, 1)],
        [(0, 0), np.array([20e-9, 10e-9]), (10, 20), None],
        [(0, 0), np.array([20e-9, 10e-9]), None, (2e-9, 1e-9)],
        [[1.5e-9, -2e-9], np.array((7.5e-9, 2e-9)), (7, 11), None],
        [[1.5e-9, -2e-9], np.array((7.5e-9, 2e-9)), None, (0.5e-9, 4e-9)],
        # 3d
        [(0, 0, 0), (5, 5, 5), [1, 1, 1], None],
        [(-1, 0, -3), (5, 7, 5), None, (1, 1, 1)],
        [(0, 0, 0), (5e-9, 5e-9, 5e-9), None, (1e-9, 1e-9, 1e-9)],
        [(0, 0, 0), (5e-9, 5e-9, 5e-9), (5, 5, 5), None],
        [
            (-1.5e-9, -5e-9, 0),
            (1.5e-9, -15e-9, -10e-9),
            None,
            (1.5e-9, 0.5e-9, 10e-9),
        ],
        [(-1.5e-9, -5e-9, 0), (1.5e-9, -15e-9, -10e-9), (3, 10, 2), None],
        [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0)), None, (0.5e-9, 1e-9, 5e-9)],
        [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0)), (5, 5, 7), None],
        [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6), None, (0.5e-6, 2e-6, 2.5e-6)],
        [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6), (1, 10, 20), None],
        [(0, 125e-9, 0), (500e-9, 0, -3e-9), None, (25e-9, 25e-9, 3e-9)],
        # > 3d
        [(0, 1, 2, 3), (10, 9, 8, 7), (2, 4, 6, 8), None],
        [(0, 1, 2, 3), (10, 9, 8, 7), None, (1, 1, 1, 1)],
        [[0, 1, 2, 3, 4], [10, 9, 8, 7, 6], np.ones(5, dtype=int) * 10, None],
        [[0, 1, 2, 3, 4], [10, 9, 8, 7, 6], None, np.ones(5) * 0.5],
        [np.arange(10.0), np.arange(10.0, 20.0), np.ones(10, dtype=int) * 4, None],
        [np.arange(10.0), np.arange(10.0, 20.0), None, np.ones(10)],
    ],
)
def test_init_valid_args(p1, p2, n, cell):
    mesh1 = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)
    assert isinstance(mesh1, df.Mesh)

    assert isinstance(mesh1.region, df.Region)
    if n is not None:
        assert np.all(mesh1.n == n)
    if cell is not None:
        assert np.allclose(mesh1.cell, cell, atol=0)

    mesh2 = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
    assert isinstance(mesh2, df.Mesh)

    assert isinstance(mesh2.region, df.Region)
    if n is not None:
        assert np.all(mesh2.n == n)
    if cell is not None:
        assert np.allclose(mesh2.cell, cell, atol=0)

    assert mesh1 == mesh2


@pytest.mark.parametrize(
    "p1, p2, n, cell, error",
    [
        # 1d
        [0, 2e-10, (1, 1), None, ValueError],
        [0, 2e-10, None, (1e-10, 1e-10), ValueError],
        [0, 1e-3, None, 1, ValueError],
        [(5e-9,), (2e-10,), -1, None, ValueError],
        [(5e-9,), -2e-10, None, -1e-9, ValueError],
        [(5e-9,), -2e-10, "seven cells", None, TypeError],
        ["zero", -2e-10, None, 1e-9, TypeError],
        # 2d
        [(0, 0), (5, 7), (5, -7), None, ValueError],
        [(0, 0), (5, 7), None, (-0.5, 0.5), ValueError],
        [(3, -1), [0, 5], (1, 1j), None, TypeError],
        [(3, -1), [0, 5], None, (1, 1 + 2j), TypeError],
        [(0, 0), np.array([20e-9, 10e-9]), (10, 20.5), None, TypeError],
        [(0, 0), np.array([20e-9, 10e-9]), None, (4e-9, 2e-9, 1e-9), ValueError],
        [[1.5e-9, -2e-9], np.array((7.5e-9, 2e-9)), (7, 11, 15), None, ValueError],
        [[1.5e-9, -2e-9], np.array((7.5e-9, 2e-9)), None, (0.5e-9, "one"), TypeError],
        # 3d
        [(0, 0, 0), (5, 5, 5), None, (-1, 1, 1), ValueError],
        [(0, 0, 0), (5, 5, 5), (-1, 1, 1), None, ValueError],
        [(0, 0, 0), (5, 5, 5), "n", None, TypeError],
        [(0, 0, 0), (5, 5, 5), (1, 2, 2 + 1j), None, TypeError],
        [(0, 0, 0), (5, 5, 5), (1, 2, "2"), None, TypeError],
        [("1", 0, 0), (1, 1, 1), None, (0, 0, 1e-9), TypeError],
        [
            (-1.5e-9, -5e-9, "a"),
            (1.5e-9, 15e-9, 16e-9),
            None,
            (5, 1, -1e-9),
            TypeError,
        ],
        [
            (-1.5e-9, -5e-9, "a"),
            (1.5e-9, 15e-9, 16e-9),
            (5, 1, -1),
            None,
            TypeError,
        ],
        [
            (-1.5e-9, -5e-9, 0),
            (1.5e-9, 16e-9),
            None,
            (0.1e-9, 0.1e-9, 1e-9),
            ValueError,
        ],
        [
            (-1.5e-9, -5e-9, 0),
            (1.5e-9, 15e-9, 1 + 2j),
            None,
            (5, 1, 1e-9),
            TypeError,
        ],
        ["string", (5, 1, 1e-9), None, "string", TypeError],
        [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 16e-9), None, 2 + 2j, TypeError],
        # > 3d
        [(0, 1, 2, 3), ("infinity", 9, 8, 7), (2, 4, 6, 8), None, TypeError],
        ["origin", (10, 9, 8, 7), None, (1, 1, 1, 1), TypeError],
        [
            [0, 1, 2, 3, 4],
            [10, 9, 8, 7, 6],
            np.ones(5, dtype=int) * -10,
            None,
            ValueError,
        ],
        [[0, 1, 2, 3, 4], [10, 9, 8, 7, 6], None, np.ones(6) * 0.5, ValueError],
        [np.arange(10.0), np.arange(10.0, 20.0), np.ones(10) * 4.5, None, TypeError],
    ],
)
def test_init_invalid_args(p1, p2, n, cell, error):
    with pytest.raises(error):
        df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)

    with pytest.raises(error):
        df.Mesh(p1=p1, p2=p2, n=n, cell=cell)


@pytest.mark.parametrize(
    "p1, p2, cell, sr1_p1, sr1_p2, sr2_p1, sr2_p2",
    [
        [0, 50, 10, 0, 40, 20, 50],
        [(0, 0), (50, 10), (10, 10), (0, 0), (40, 10), (20, 0), (50, 10)],
        [
            (0, 0, 0),
            (100, 50, 10),
            (10, 10, 10),
            (0, 0, 0),
            (50, 40, 10),
            (10, 20, 0),
            (100, 50, 10),
        ],
    ],
)
def test_init_subregions(p1, p2, cell, sr1_p1, sr1_p2, sr2_p1, sr2_p2):
    subregions = {
        "r1": df.Region(p1=sr1_p1, p2=sr1_p2),
        "default": df.Region(p1=sr2_p1, p2=sr2_p2),
    }
    with pytest.warns(UserWarning, match="Subregion name ``default``"):
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
    assert isinstance(mesh, df.Mesh)
    assert mesh.subregions == subregions


@pytest.mark.parametrize(
    "p1, p2, cell, sr1_p1, sr1_p2, sr2_p1, sr2_p2",
    [
        [(0,), 50, 10, 0, 40, 20, 50],
        [(0, 0), (50, 10), (10, 10), (0, 0), (40, 10), (20, 0), (50, 10)],
        [
            (0, 0, 0),
            (100, 50, 10),
            (10, 10, 10),
            (0, 0, 0),
            (50, 40, 10),
            (10, 20, 0),
            (100, 50, 10),
        ],
    ],
)
def test_subregions_custom_parameters(p1, p2, cell, sr1_p1, sr1_p2, sr2_p1, sr2_p2):
    dims = list("abc")[: len(p1)]
    units = ["d", "ef", "ghi"][: len(p1)]
    region = df.Region(p1=p1, p2=p2, dims=dims, units=units, tolerance_factor=1e-6)
    subregions = {
        "r1": df.Region(p1=sr1_p1, p2=sr1_p2),
        "r2": df.Region(
            p1=sr2_p1,
            p2=sr2_p2,
            dims=list("rst")[: len(p1)],
            units=list("aei")[: len(p1)],
            tolerance_factor=10,
        ),
    }
    mesh = df.Mesh(region=region, cell=cell, subregions=subregions)
    assert isinstance(mesh, df.Mesh)
    assert len(mesh.subregions) == len(subregions)
    for sr_name in mesh.subregions:
        assert np.array_equal(mesh.subregions[sr_name].pmin, subregions[sr_name].pmin)
        assert np.array_equal(mesh.subregions[sr_name].pmax, subregions[sr_name].pmax)
        assert mesh.subregions[sr_name].units == mesh.region.units
        assert mesh.subregions[sr_name].dims == mesh.region.dims
        assert mesh.subregions[sr_name].tolerance_factor == mesh.region.tolerance_factor


@pytest.mark.parametrize(
    "p1, p2, cell, subregions, error",
    [
        ((0,), (100e-9,), (10e-9,), {"r1": df.Region(p1=0, p2=45e-9)}, ValueError),
        (
            (0, 0),
            (100e-9, 50e-9),
            (10e-9, 10e-9),
            {"r1": df.Region(p1=0, p2=40e-9)},
            ValueError,
        ),
        (
            (0, 0),
            (100e-9, 50e-9),
            (10e-9, 10e-9),
            {"r1": df.Region(p1=(0, 0, 0), p2=(40e-9, 50e-9, 10e-9))},
            ValueError,
        ),
        (
            (0, 0, 0),
            (100e-9, 50e-9, 10e-9),
            (10e-9, 10e-9, 10e-9),
            {"r1": df.Region(p1=(0, 0, 0), p2=(45e-9, 50e-9, 10e-9))},
            ValueError,
        ),
        (
            (0, 0, 0),
            (100e-9, 50e-9, 10e-9),
            (10e-9, 10e-9, 10e-9),
            {"r1": df.Region(p1=(5e-9, 0, 0), p2=(45e-9, 50e-9, 10e-9))},
            ValueError,
        ),
        (
            (0, 0, 0),
            (100e-9, 50e-9, 10e-9),
            (10e-9, 10e-9, 10e-9),
            {"r1": df.Region(p1=(0, 0, 0), p2=(40e-9, 50e-9, 200e-9))},
            ValueError,
        ),
        (
            (0, 0, 0),
            (100e-9, 50e-9, 10e-9),
            (10e-9, 10e-9, 10e-9),
            {1: df.Region(p1=(0, 0, 0), p2=(45e-9, 50e-9, 200e-9))},
            TypeError,
        ),
        (
            (0, 0, 0),
            (100e-9, 50e-9, 10e-9),
            (10e-9, 10e-9, 10e-9),
            {"r1": "top half of the region"},
            TypeError,
        ),
        (
            (0, 0, 0, 0),
            (100e-9, 50e-9, 10e-9, 10e-9),
            (10e-9, 10e-9, 10e-9, 10e-9),
            {"r1": df.Region(p1=(0, 0, 0), p2=(40e-9, 50e-9, 200e-9))},
            ValueError,
        ),
    ],
)
def test_invalid_subregions(p1, p2, cell, subregions, error):
    with pytest.raises(error):
        df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)


def test_init_with_region_and_points():
    p1 = (0, -4, 16.5)
    p2 = (15, 10.1, 11)
    region = df.Region(p1=p1, p2=p2)
    n = (10, 10, 10)

    with pytest.raises(ValueError):
        df.Mesh(region=region, p1=p1, p2=p2, n=n)


def test_init_with_n_and_cell():
    p1 = (0, -4, 16.5)
    p2 = (15, 10.1, 11)
    n = (15, 141, 11)
    cell = (1, 0.1, 0.5)

    with pytest.raises(ValueError):
        df.Mesh(p1=p1, p2=p2, n=n, cell=cell)


@pytest.mark.parametrize(
    "p1, p2, cell",
    [
        [(0), 150e-9, 4e-9],
        [(0, 100e-9), (150e-9, 120e-9), (4e-9, 1e-9)],
        [(0, 100e-9, 1e-9), (150e-9, 120e-9, 6e-9), (4e-9, 1e-9, 1e-9)],
        [(0, 100e-9, 0), (150e-9, 104e-9, 1e-9), (2e-9, 1.5e-9, 0.1e-9)],
        [(10e9, 10e3, 0), (11e9, 11e3, 5), (1e9, 1e3, 1.5)],
        [(0, 100e-9, 1e-9, 0), (150e-9, 120e-9, 6e-9, 70e-9), (4e-9, 1e-9, 1e-9, 3e-9)],
    ],
)
def test_region_not_aggregate_of_cell(p1, p2, cell):
    with pytest.raises(ValueError):
        df.Mesh(p1=p1, p2=p2, cell=cell)


@pytest.mark.parametrize(
    "p1, p2, cell",
    [
        [0, 1e-9, 2e-9],
        [(0, 0), (1e-9, 1e-9), (1e-9, 2e-9)],
        [(0, 0, 0), (1e-9, 1e-9, 1e-9), (2e-9, 1e-9, 1e-9)],
        [(0, 0, 0), (1e-9, 1e-9, 1e-9), (1e-9, 2e-9, 1e-9)],
        [(0, 0, 0), (1e-9, 1e-9, 1e-9), (1e-9, 1e-9, 2e-9)],
        [(0, 0, 0), (1e-9, 1e-9, 1e-9), (1e-9, 5e-9, 0.1e-9)],
    ],
)
def test_cell_greater_than_domain(p1, p2, cell):
    p1 = (0, 0, 0)
    p2 = (1e-9, 1e-9, 1e-9)

    with pytest.raises(ValueError):
        df.Mesh(p1=p1, p2=p2, cell=cell)


def test_cell_n():
    p1 = (0, 0, 0)
    p2 = (20e-9, 20e-9, 20e-9)
    cell = (2e-9, 4e-9, 1e-9)
    n = (10, 5, 20)

    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    mesh = df.Mesh(p1=p1, p2=p2, cell=np.array(cell))
    mesh = df.Mesh(p1=p1, p2=p2, cell=list(cell))
    assert np.all(mesh.n == n)

    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    mesh = df.Mesh(p1=p1, p2=p2, n=np.array(n, dtype=int))
    mesh = df.Mesh(p1=p1, p2=p2, n=list(n))
    assert np.allclose(mesh.cell, cell, atol=0)

    with pytest.raises(AttributeError):
        mesh.cell = (2e-9, 2e-9, 2e-9)
    with pytest.raises(AttributeError):
        mesh.n = (10, 10, 10)


@pytest.mark.parametrize(
    "cell, n, error",
    [
        (2e-9, None, ValueError),
        (None, 10, ValueError),
        ({"x": 2e-9, "y": 4e-9, "z": 1e-9}, None, TypeError),
        (None, {"x": 10, "y": 5, "z": 20}, TypeError),
        ((2e-9, 4e-9), None, ValueError),
        (None, (10, 5), ValueError),
        ((2e-9, 4e-9, 2e-9, 4e-9), None, ValueError),
        (None, (10, 5, 20, 10), ValueError),
        (None, (10.0, 5.0, 20.0), TypeError),
        (None, None, ValueError),
        ((2e-9, 4e-9, 1e-9), (10, 5, 20), ValueError),
    ],
)
def test_cell_n_invalid(cell, n, error):
    p1 = (0, 0, 0)
    p2 = (20e-9, 20e-9, 20e-9)
    with pytest.raises(error):
        df.Mesh(p1=p1, p2=p2, cell=cell, n=n)


def test_bc():  # TODO later
    p1 = (0, 0, 0)
    p2 = (20e-9, 20e-9, 20e-9)
    region = df.Region(p1=p1, p2=p2, dims=["x", "y", "z"])
    cell = (2e-9, 4e-9, 1e-9)

    allowed_bc = ["x", "y", "z", "xy", "yz", "zx", "Neumann", "dirichlet"]

    for bc in allowed_bc:
        df.Mesh(region=region, cell=cell, bc=bc)

    with pytest.raises(TypeError):
        df.Mesh(region=region, cell=cell, bc=2)
    with pytest.raises(ValueError):
        df.Mesh(region=region, cell=cell, bc="user")
    with pytest.raises(ValueError):
        df.Mesh(region=region, cell=cell, bc="xxz")


@pytest.mark.parametrize(
    "p1, p2, cell, length",
    [
        [0, 5, 1, 5],
        [(1, 1), (5, 7), (2, 3), 2 * 2],
        [(0, 0, 0), (5, 4, 3), (1, 1, 1), 5 * 4 * 3],
    ],
)
def test_len(p1, p2, cell, length):
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    assert isinstance(mesh, df.Mesh)
    assert len(mesh) == length


@pytest.mark.parametrize(
    "p1, p2, n, length",
    [
        [(0,), (10,), (5,), 5],
        [(0.0, 0.0), (10.0, 5.0), (5, 3), 5 * 3],
        [(0, 0, 0), (1e-5, 1e-5, 2e-5), (2, 5, 3), 2 * 5 * 3],
        [(0, 0, 0, 0), (10.0, 5.0, 2.0, 10), (5, 4, 3, 2), 5 * 4 * 3 * 2],
    ],
)
def test_indices_coordinates_iter(p1, p2, n, length):
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    assert isinstance(mesh, df.Mesh)

    assert isinstance(mesh.indices, types.GeneratorType)
    assert len(list(mesh.indices)) == length
    for index in mesh.indices:
        assert isinstance(index, tuple)
        assert len(index) == mesh.region.ndim
        assert all(isinstance(i, numbers.Integral) for i in index)
        assert all([0 <= i <= j for i, j in zip(index, n)])

    assert isinstance(mesh.__iter__(), types.GeneratorType)
    assert len(list(mesh)) == length
    for point in mesh:
        assert isinstance(point, np.ndarray)
        assert len(point) == mesh.region.ndim
        assert all(isinstance(i, numbers.Real) for i in point)
        assert all([0 <= i <= j for i, j in zip(point, p2)])


@pytest.mark.parametrize(
    "p1_1, p1_2, p2, n1, n2",
    [
        [5e-9, 6e-9, 10e-9, 5, 3],
        [(-100e-9, -10e-9), (-99e-9, -10e-9), (100e-9, 10e-9), (5, 5), (5, 3)],
        [(0, 0, 0), (3, 3, 3), (10, 10, 10), (5, 5, 5), (5, 5, 3)],
        [(0, 0, 0, 0), (3, 3, 3, 3), (10, 10, 10, 10), (5, 5, 5, 5), (5, 5, 5, 3)],
    ],
)
def test_eq(p1_1, p1_2, p2, n1, n2):
    mesh1 = df.Mesh(p1=p1_1, p2=p2, n=n1)
    mesh2 = df.Mesh(p1=p1_1, p2=p2, n=n1)
    mesh3 = df.Mesh(p1=p1_2, p2=p2, n=n1)
    mesh4 = df.Mesh(p1=p1_1, p2=p2, n=n2)

    assert isinstance(mesh1, df.Mesh)
    assert isinstance(mesh2, df.Mesh)
    assert isinstance(mesh3, df.Mesh)
    assert mesh1 == mesh2
    assert not mesh1 != mesh2
    assert mesh1 != mesh3
    assert not mesh1 == mesh3
    assert mesh1 != mesh4
    assert not mesh1 == mesh4
    assert mesh3 != mesh4
    assert not mesh3 == mesh4

    assert mesh1 != 1
    assert not mesh2 == "mesh2"


@pytest.mark.parametrize(
    "p1_1, p1_2, p2, n1, n2",
    [
        [(5e-9,), (6e-9,), (10e-9,), 5, 3],  # 1d
        # TODO the next test currently fails because of problems with atol=0 and
        # zero values in the test data; this has to be fixed in Mesh
        [(-100e-9, -10e-9), (-99e-9, -10e-9), (100e-9, 10e-9), (5, 5), (5, 3)],  # 2d
        [  # 3d
            (0, 0, 0),
            (3e-9, 3e-9, 3e-9),
            (10e-9, 10e-9, 10e-9),
            (5, 5, 5),
            (5, 5, 3),
        ],
        [  # 4d
            (0, 0, 0, 0),
            (3e-9, 3e-9, 3e-9, 3e-9),
            (10e-9, 10e-9, 10e-9, 10e-9),
            (5, 5, 5, 5),
            (5, 5, 5, 3),
        ],
    ],
)
def test_allclose(p1_1, p1_2, p2, n1, n2):
    mesh1 = df.Mesh(p1=p1_1, p2=p2, n=n1)
    mesh2 = df.Mesh(p1=np.array(p1_1), p2=np.array(p2) * (1 + 1e-13), n=n1)
    mesh3 = df.Mesh(p1=p1_2, p2=p2, n=n1)
    mesh4 = df.Mesh(p1=p1_1, p2=p2, n=n2)

    assert mesh1.allclose(mesh2)

    # deviation ~1e-9 is considered too large with the default tolerance_factor
    assert not mesh1.allclose(mesh3)
    assert mesh1.allclose(mesh3, atol=1e-8)
    mesh1.region.tolerance_factor = 1
    assert mesh1.allclose(mesh3)

    assert not mesh3.allclose(mesh4)  # different n

    with pytest.raises(TypeError):
        mesh1.allclose(mesh1.region)

    with pytest.raises(TypeError):
        mesh1.allclose(mesh3, atol="20")

    with pytest.raises(TypeError):
        mesh1.allclose(mesh3, rtol="1")


def test_allclose_cell_accuracy():
    eps = 1e-6
    n = int(1e7)
    region1 = df.Region(p1=0, p2=1, tolerance_factor=eps)
    mesh1 = df.Mesh(region=region1, n=n)
    region2 = df.Region(p1=eps, p2=1 + eps, tolerance_factor=eps)
    mesh2 = df.Mesh(region=region2, n=n)
    # NOTE the two meshes are shifted by 10 cells but this cannot be dectected with
    # the default tolerance settings
    assert mesh1.allclose(mesh2)
    # Setting atol=0 solves the problem
    assert not mesh1.allclose(mesh2, atol=0)
    # changing region.tolerance_factor so that the final tolerance is cell * 1e-3
    # "properly" fixes the problem
    mesh1.region.tolerance_factor = 1e-3 * min(mesh1.cell) / min(mesh1.region.edges)
    assert not mesh1.allclose(mesh2)

    mesh3 = df.Mesh(p1=eps / 10, p2=1, n=n)
    print(
        mesh1.region.tolerance_factor,
        mesh1.region.tolerance_factor * np.min(mesh1.cell),
    )
    assert not mesh1.allclose(mesh3)  # computed atol = 1e-29
    assert mesh1.allclose(mesh3, atol=eps)


def test_repr():
    mesh = df.Mesh(p1=0, p2=10, n=5)

    rstr = "Mesh(Region(pmin=[0], pmax=[10], dims=['x'], units=['m']), n=[5])"
    assert repr(mesh) == rstr
    assert re.match(html_re, mesh._repr_html_(), re.DOTALL)

    p1 = (-1, -4, 11)
    p2 = (15, 10.1, 12.5)
    cell = (1, 0.1, 0.5)

    mesh = df.Mesh(p1=p1, p2=p2, cell=cell, bc="x")
    assert isinstance(mesh, df.Mesh)

    rstr = (
        "Mesh(Region(pmin=[-1.0, -4.0, 11.0], pmax=[15.0, 10.1, 12.5], "
        "dims=['x', 'y', 'z'], units=['m', 'm', 'm']), "
        "n=[16, 141, 3], bc=x)"
    )
    assert repr(mesh) == rstr
    assert re.match(html_re, mesh._repr_html_(), re.DOTALL)


@pytest.mark.parametrize(
    "p1, p2, n, index, expected_1, expected_min, expected_max",
    [
        [(0.0,), (-30e-9,), np.array([3]), (2), (-5e-9,), (-25e-9,), (-5e-9,)],
        [
            (0.0, 0.0),
            (30.0, 40.0),
            np.array([3, 4]),
            (2, 2),
            (25.0, 25.0),
            (5.0, 5.0),
            (25.0, 35.0),
        ],
        [
            (0, 0, 0),
            (30, 40, 50),
            np.array([3, 4, 5]),
            (2, 2, 3),
            (25, 25, 35),
            (5, 5, 5),
            (25, 35, 45),
        ],
        [
            (0, 0, 0, -50.0),
            (30, 40, 50, 150.0),
            np.array([3, 4, 5, 5]),
            (2, 2, 3, 1),
            (25, 25, 35, 10.0),
            (5, 5, 5, -30),
            (25, 35, 45, 130),
        ],
    ],
)
def test_index2point_valid(p1, p2, n, index, expected_1, expected_min, expected_max):
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    assert isinstance(mesh, df.Mesh)

    assert np.allclose(mesh.index2point(index), expected_1, atol=0)

    # Correct minimum index
    assert isinstance(mesh.index2point((0,) * mesh.region.ndim), np.ndarray)
    assert np.allclose(mesh.index2point((0,) * mesh.region.ndim), expected_min, atol=0)

    # Correct maximum index
    assert isinstance(mesh.index2point(n - 1), np.ndarray)
    assert np.allclose(mesh.index2point(n - 1), expected_max, atol=0)


@pytest.mark.parametrize(
    "p1, p2, n, index, error",
    [
        [0, 1, 3, 3, IndexError],
        [0, 1, 3, -1, IndexError],
        [0, 1, 3, "string", TypeError],
        [0, 1, 3, 0.0, TypeError],
        [0, 5, 5, slice(0, 3), TypeError],
        [(0, 0), (1, 1), (3, 3), (2, 3), IndexError],
        [(0, 0), (1, 1), (3, 3), (-1, 2), IndexError],
        [(0, 0), (1, 1), (3, 3), (2, -1), IndexError],
        [(0, 0), (1, 1), (3, 3), (2, "string"), TypeError],
        [(0, 0), (1, 1), (3, 3), (2, 2.0), TypeError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, 3), IndexError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, -1, 2), IndexError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (1, 2, -1), IndexError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, "string"), TypeError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, 2.0), TypeError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, 2, 3), IndexError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2.0, 2.0, 2.0), TypeError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2, 1, 3), IndexError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2, 2, -2), IndexError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2, 2, "string"), TypeError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2, 2, 2.0), TypeError],
        [0, 1, 3, (2, 2), IndexError],
        [(0, 0), (1, 1), (3, 3), (2, 2, 2), IndexError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, 2, 2), IndexError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2), IndexError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), 2, IndexError],
    ],
)
def test_index2point_invalid(p1, p2, n, index, error):
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    with pytest.raises(error):
        mesh.index2point(index)


@pytest.mark.parametrize(
    "p1, p2, n, point, expected",
    [
        [(0.0,), (-30e-9,), np.array([3]), (2,), (-5e-9,)],
        [
            (0.0, 0.0),
            (30.0, 40.0),
            np.array([3, 4]),
            (2, 2),
            (25.0, 25.0),
        ],
        [
            (0, 0, 0),
            (30, 40, 50),
            np.array([3, 4, 5]),
            (2, 2, 3),
            (25, 25, 35),
        ],
        [
            (0, 0, 0, -50.0),
            (30, 40, 50, 150.0),
            np.array([3, 4, 5, 5]),
            (2, 2, 3, 1),
            (25, 25, 35, 10.0),
        ],
    ],
)
def test_point2index_valid(p1, p2, n, point, expected):
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    assert isinstance(mesh, df.Mesh)

    assert np.array_equal(mesh.point2index(mesh.region.pmin), (0,) * mesh.region.ndim)
    assert np.array_equal(
        mesh.point2index(mesh.region.pmin + mesh.cell / 2), (0,) * mesh.region.ndim
    )
    assert np.array_equal(
        mesh.point2index(mesh.region.pmin + 3 * mesh.cell / 4), (0,) * mesh.region.ndim
    )
    assert np.array_equal(mesh.point2index(mesh.region.pmax), n - 1)
    assert np.array_equal(mesh.point2index(mesh.region.pmax - mesh.cell / 2), n - 1)
    assert np.array_equal(mesh.point2index(mesh.region.pmax - 3 * mesh.cell / 4), n - 1)
    assert np.array_equal(mesh.point2index(expected), point)


def test_point2index_boundaries():
    mesh = df.Mesh(p1=0, p2=10, cell=1)

    assert mesh.point2index(0.9) == (0,)
    assert mesh.point2index(1.0) == (1,)
    assert mesh.point2index(1.1) == (1,)

    # Check with even values as well
    assert mesh.point2index(1.9) == (1,)
    assert mesh.point2index(2.0) == (2,)
    assert mesh.point2index(2.1) == (2,)

    # Check inclusive boundaries
    assert mesh.point2index(0) == (0,)
    assert mesh.point2index(10) == (9,)

    # Check with floating-point accuracy
    point = -mesh.region.tolerance_factor
    assert mesh.point2index(point) == (0,)

    # tolerance_factor is internally multiplied by min(edges)=1(0,)
    point = -20 * mesh.region.tolerance_factor
    with pytest.raises(ValueError):
        mesh.point2index(point)

    point = 10 + mesh.region.tolerance_factor
    assert mesh.point2index(point) == (9,)

    point = 10 + 20 * mesh.region.tolerance_factor
    print(point, point in mesh.region)
    with pytest.raises(ValueError):
        mesh.point2index(point)

    # Check out of bounds
    with pytest.raises(ValueError):
        mesh.point2index(-10)

    with pytest.raises(ValueError):
        mesh.point2index(20)


@pytest.mark.parametrize(
    "p1, p2, n, point, error",
    [
        [0, 1, 3, 5, ValueError],
        [0, 1, 3, -1, ValueError],
        [0, 1, 3, "string", TypeError],
        [0, 50e-9, 5, slice(0, 20e-9), TypeError],
        [(0, 0), (1, 1), (3, 3), (2, 5), ValueError],
        [(0, 0), (1, 1), (3, 3), (-1, 2), ValueError],
        [(0, 0), (1, 1), (3, 3), (2, -1), ValueError],
        [(0, 0), (1, 1), (3, 3), (2, "string"), TypeError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2.0, 2.0, 5.0), ValueError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, -1, 2), ValueError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (1.0, 2.0, -1.0), ValueError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, "string"), TypeError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, 2, 3), ValueError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2, 1, 5), ValueError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2, 2, -2), ValueError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2, 2, "string"), TypeError],
        [0, 1, 3, (2, 2), ValueError],
        [(0, 0), (1, 1), (3, 3), (2, 2, 2), ValueError],
        [(0, 0, 0), (1, 1, 1), (3, 3, 3), (2, 2, 2, 2), ValueError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), (2, 2), ValueError],
        [(0, 0, 0, 0), (1, 1, 1, 1), (3, 3, 3, 3), 2, ValueError],
    ],
)
def test_point2index_invalid(p1, p2, n, point, error):
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    with pytest.raises(error):
        mesh.point2index(point)


def test_index2point_point2index_mutually_inverse():
    # 1d
    p1 = 15
    p2 = -1
    cell = 1
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
    assert isinstance(mesh, df.Mesh)

    for p in [-0.5, 14.5]:
        assert np.allclose(mesh.index2point(mesh.point2index(p)), p, atol=0)

    for i in [0, 1]:
        assert mesh.point2index(mesh.index2point(i)) == (i,)

    # 3d
    p1 = (15, -4, 12.5)
    p2 = (-1, 10.1, 11)
    cell = (1, 0.1, 0.5)
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
    assert isinstance(mesh, df.Mesh)

    for p in [(-0.5, -3.95, 11.25), (14.5, 10.05, 12.25)]:
        assert np.allclose(mesh.index2point(mesh.point2index(p)), p, atol=0)

    for i in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
        assert mesh.point2index(mesh.index2point(i)) == i


def test_region2slice():
    p1 = 0
    p2 = 4
    cell = 1
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    assert isinstance(mesh, df.Mesh)
    assert mesh.region2slices(df.Region(p1=p1, p2=p2)) == (slice(0, 4, None),)
    assert mesh.region2slices(df.Region(p1=0, p2=1)) == (slice(0, 1, None),)
    assert mesh.region2slices(df.Region(p1=2, p2=3)) == (slice(2, 3, None),)
    # check error is raised if region is greater than the mesh's region
    with pytest.raises(ValueError):
        mesh.region2slices(df.Region(p1=(-1), p2=(3)))

    p1 = (0, 0, -2)
    p2 = (4, 5, 4)
    cell = (1, 1, 1)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    assert isinstance(mesh, df.Mesh)
    assert mesh.region2slices(df.Region(p1=p1, p2=p2)) == (
        slice(0, 4, None),
        slice(0, 5, None),
        slice(0, 6, None),
    )
    assert mesh.region2slices(df.Region(p1=(0, 0, 0), p2=(1, 1, 1))) == (
        slice(0, 1, None),
        slice(0, 1, None),
        slice(2, 3, None),
    )
    assert mesh.region2slices(df.Region(p1=(2, 3, -1), p2=(3, 5, 0))) == (
        slice(2, 3, None),
        slice(3, 5, None),
        slice(1, 2, None),
    )
    with pytest.raises(ValueError):
        mesh.region2slices(df.Region(p1=(-1, 3, -1), p2=(3, 5, 0)))

    with pytest.raises(ValueError):
        mesh.region2slices(df.Region(p1=(-1, 3), p2=(3, 5)))


def test_cells():
    # 1d example (ndim=1)
    p1 = 0
    p2 = 10
    cell = 2
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    assert np.allclose(mesh.cells.x, [1.0, 3.0, 5.0, 7.0, 9.0], atol=0)

    # 3d example (ndim=3)
    p1 = (0, 0, 4)
    p2 = (10, 6, 0)
    cell = (2, 2, 1)
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    assert np.allclose(mesh.cells.x, [1.0, 3.0, 5.0, 7.0, 9.0], atol=0)
    assert np.allclose(mesh.cells.y, [1.0, 3.0, 5.0], atol=0)
    assert np.allclose(mesh.cells.z, [0.5, 1.5, 2.5, 3.5], atol=0)

    # 4d example (ndim=4)
    p1 = (0, 0, 4, 4)
    p2 = (10, 6, 0, 0)
    cell = (2, 2, 1, 1)
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    assert np.allclose(mesh.cells.x0, [1.0, 3.0, 5.0, 7.0, 9.0], atol=0)
    assert np.allclose(mesh.cells.x1, [1.0, 3.0, 5.0], atol=0)
    assert np.allclose(mesh.cells.x2, [0.5, 1.5, 2.5, 3.5], atol=0)
    assert np.allclose(mesh.cells.x3, [0.5, 1.5, 2.5, 3.5], atol=0)


def test_vertices():
    p1 = 0
    p2 = 5
    cell = 1
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    assert np.allclose(mesh.vertices.x, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], atol=0)

    p1 = (0, 1, 0)
    p2 = (5, 0, 6)
    cell = (1, 1, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    assert np.allclose(mesh.vertices.x, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], atol=0)
    assert np.allclose(mesh.vertices.y, [0.0, 1.0], atol=0)
    assert np.allclose(mesh.vertices.z, [0.0, 2.0, 4.0, 6.0], atol=0)

    p1 = (0, 1, 0, 0)
    p2 = (5, 0, 6, 6)
    cell = (1, 1, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    assert np.allclose(mesh.vertices.x0, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], atol=0)
    assert np.allclose(mesh.vertices.x1, [0.0, 1.0], atol=0)
    assert np.allclose(mesh.vertices.x2, [0.0, 2.0, 4.0, 6.0], atol=0)
    assert np.allclose(mesh.vertices.x3, [0.0, 2.0, 4.0, 6.0], atol=0)


def test_line():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (1, 1, 1)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    assert isinstance(mesh, df.Mesh)

    line = mesh.line(p1=(0, 0, 0), p2=(10, 10, 10), n=10)
    assert isinstance(line, types.GeneratorType)
    assert len(list(line)) == 10
    for point in line:
        assert isinstance(point, tuple)
        assert len(point) == 3
        assert all([0 <= i <= 10 for i in point])

    line = list(mesh.line(p1=(0, 0, 0), p2=(10, 0, 0), n=11))
    assert len(line) == 11
    assert line[0] == (0, 0, 0)
    assert line[-1] == (10, 0, 0)
    assert line[5] == (5, 0, 0)

    with pytest.raises(ValueError):
        line = list(mesh.line(p1=(-1e-9, 0, 0), p2=(10, 0, 0), n=100))

    with pytest.raises(ValueError):
        line = list(mesh.line(p1=(0, 0, 0), p2=(11, 0, 0), n=100))


def test_or():
    p1 = (-50e-9, -25e-9, 0)
    p2 = (50e-9, 25e-9, 5e-9)
    cell = (5e-9, 5e-9, 5e-9)
    mesh1 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = (-45e-9, -20e-9, 0)
    p2 = (10e-9, 20e-9, 5e-9)
    cell = (5e-9, 5e-9, 5e-9)
    mesh2 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = (-42e-9, -20e-9, 0)
    p2 = (13e-9, 20e-9, 5e-9)
    cell = (5e-9, 5e-9, 5e-9)
    mesh3 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = (-50e-9, -25e-9, 0)
    p2 = (50e-9, 25e-9, 5e-9)
    cell = (2.5e-9, 2.5e-9, 2.5e-9)
    mesh4 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    with pytest.deprecated_call():  # ensures DeprecationWarning
        assert mesh1 | mesh2 is True
        assert mesh1 | mesh3 is False
        assert mesh1 | mesh4 is False
        assert mesh1 | mesh1 is True


def test_is_aligned():
    p1 = (-50e-9, -25e-9, 0)
    p2 = (50e-9, 25e-9, 5e-9)
    cell = (5e-9, 5e-9, 5e-9)
    mesh1 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = (-45e-9, -20e-9, 0)
    p2 = (10e-9, 20e-9, 5e-9)
    cell = (5e-9, 5e-9, 5e-9)
    mesh2 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = (-42e-9, -20e-9, 0)
    p2 = (13e-9, 20e-9, 5e-9)
    cell = (5e-9, 5e-9, 5e-9)
    mesh3 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = (-50e-9, -25e-9, 0)
    p2 = (50e-9, 25e-9, 5e-9)
    cell = (2.5e-9, 2.5e-9, 2.5e-9)
    mesh4 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    assert mesh1.is_aligned(mesh2)
    assert not mesh1.is_aligned(mesh3)
    assert not mesh1.is_aligned(mesh4)
    assert mesh1.is_aligned(mesh1)

    # Test tolerance
    tol = 1e-12
    mesh5 = df.Mesh(p1=(0, 0, 0), p2=(20e-9, 20e-9, 20e-9), cell=(5e-9, 5e-9, 5e-9))
    mesh6 = df.Mesh(
        p1=(0 + 1e-13, 0, 0),
        p2=(20e-9 + 1e-13, 20e-9, 20e-9),
        cell=(5e-9, 5e-9, 5e-9),
    )
    mesh7 = df.Mesh(
        p1=(0, 0, 0 + 1e-10),
        p2=(20e-9, 20e-9, 20e-9 + 1e-10),
        cell=(5e-9, 5e-9, 5e-9),
    )

    assert mesh5.is_aligned(mesh6, tol)
    assert not mesh5.is_aligned(mesh7, tol)

    p1 = -50e-9
    p2 = 50e-9
    cell = 5e-9
    mesh1 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = (-45e-9,)
    p2 = 10e-9
    cell = 5e-9
    mesh2 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = -42e-9
    p2 = 13e-9
    cell = 5e-9
    mesh3 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    p1 = -50e-9
    p2 = 50e-9
    cell = 2.5e-9
    mesh4 = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    assert mesh1.is_aligned(mesh2)
    assert not mesh1.is_aligned(mesh3)
    assert not mesh1.is_aligned(mesh4)
    assert mesh1.is_aligned(mesh1)

    # Test tolerance
    tol = 1e-12
    mesh5 = df.Mesh(p1=0, p2=20e-9, cell=5e-9)
    mesh6 = df.Mesh(
        p1=1e-13,
        p2=20e-9 + 1e-13,
        cell=5e-9,
    )
    mesh7 = df.Mesh(
        p1=1e-10,
        p2=20e-9 + 1e-10,
        cell=5e-9,
    )
    assert mesh5.is_aligned(mesh6, tol)
    assert not mesh5.is_aligned(mesh7, tol)

    # Test exceptions
    with pytest.raises(TypeError):
        mesh5.is_aligned(mesh6.region, tol)
    with pytest.raises(TypeError):
        mesh5.is_aligned(mesh6, "1e-12")


def test_getitem():
    # Subregions dictionary
    p1 = (0, 0, 0)
    p2 = (100, 50, 10)
    cell = (5, 5, 5)
    subregions = {
        "r1": df.Region(p1=(0, 0, 0), p2=(50, 50, 10)),
        "r2": df.Region(p1=(50, 0, 0), p2=(100, 50, 10)),
    }
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

    submesh1 = mesh["r1"]
    assert isinstance(submesh1, df.Mesh)
    assert np.allclose(submesh1.region.pmin, (0, 0, 0), atol=0)
    assert np.allclose(submesh1.region.pmax, (50, 50, 10), atol=0)
    assert np.allclose(submesh1.cell, (5, 5, 5), atol=0)

    submesh2 = mesh["r2"]
    assert isinstance(submesh2, df.Mesh)
    assert np.allclose(submesh2.region.pmin, (50, 0, 0), atol=0)
    assert np.allclose(submesh2.region.pmax, (100, 50, 10), atol=0)
    assert np.allclose(submesh2.cell, (5, 5, 5), atol=0)

    assert len(submesh1) + len(submesh2) == len(mesh)

    # "newly-defined" region
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (1, 1, 1)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    submesh = mesh[df.Region(p1=(0.1, 2.2, 4.01), p2=(4.9, 3.8, 5.7))]
    assert isinstance(submesh, df.Mesh)
    assert np.allclose(submesh.region.pmin, (0, 2, 4), atol=0)
    assert np.allclose(submesh.region.pmax, (5, 4, 6), atol=0)
    assert np.allclose(submesh.cell, cell, atol=0)
    assert np.all(submesh.n == (5, 2, 2))
    assert mesh[mesh.region].allclose(mesh, atol=0)

    with pytest.raises(ValueError):
        # subregion extending beyond mesh
        # (p1 is in mesh region, but p2 is outside)
        submesh = mesh[df.Region(p1=(1, 2, 1), p2=(200, 79, 14))]

    p1 = 20e-9
    p2 = -25e-9
    cell = 5e-9
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    submesh = mesh[df.Region(p1=(11e-9,), p2=(-9e-9,))]
    assert isinstance(submesh, df.Mesh)
    assert np.allclose(submesh.region.pmin, -10e-9, atol=0)
    assert np.allclose(submesh.region.pmax, 15e-9, atol=0)
    assert np.allclose(submesh.cell, cell, atol=0)
    assert np.array_equal(submesh.n, [5])
    assert mesh[mesh.region].allclose(mesh, atol=0)

    with pytest.raises(ValueError):
        submesh = mesh[df.Region(p1=(11e-9, 22e-9, 1e-9), p2=(200e-9, 79e-9, 14e-9))]


def test_getitem_boundaries():
    subregion = df.Region(p1=2, p2=6)
    mesh = df.Mesh(p1=0, p2=10, cell=1, subregions={"a": subregion})

    assert mesh[subregion] == mesh["a"]


@pytest.mark.parametrize(
    "p1, p2, cell",
    [
        [0, 10, 1],
        [(0, 0), (5.0, 7.0), (1, 1)],
        [(-1, 2, 7), (5, 9, 4), (1, 1, 1)],
        [(-1.0, 2.0, 7.0, 10.0), (5.0, 9.0, 4.0, 3.0), (1, 1, 0.5, 1)],
    ],
)
def test_pad(p1, p2, cell):
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, cell=cell)

    for dim in mesh.region.dims:
        # adding one padding on either side, and for each dimension
        padded_mesh = mesh.pad({dim: (1, 1)})
        # Add cell to pmin and pmax to get the correct padded region
        idx = mesh.region._dim2index(dim)
        temp = mesh.region.pmin.copy()
        if mesh.region.ndim == 1:
            temp = temp - cell
        else:
            temp[idx] = temp[idx] - cell[idx]
        assert np.allclose(padded_mesh.region.pmin, temp, atol=0)
        temp = mesh.region.pmax.copy()
        if mesh.region.ndim == 1:
            temp = temp + cell
        else:
            temp[idx] = temp[idx] + cell[idx]
        assert np.allclose(padded_mesh.region.pmax, temp, atol=0)
        temp = mesh.n.copy()
        if mesh.region.ndim == 1:
            temp = temp + 2
        else:
            temp[idx] = temp[idx] + 2
        assert np.all(padded_mesh.n == temp)

    for dim in mesh.region.dims:
        # adding padding only on pmax side, and for each dimension
        padded_mesh = mesh.pad({dim: (0, 1)})
        # Add cell to pmin and pmax to get the correct padded region
        idx = mesh.region._dim2index(dim)
        assert np.allclose(padded_mesh.region.pmin, mesh.region.pmin, atol=0)
        temp = mesh.region.pmax.copy()
        if mesh.region.ndim == 1:
            temp = temp + cell
        else:
            temp[idx] = temp[idx] + cell[idx]
        assert np.allclose(padded_mesh.region.pmax, temp, atol=0)
        temp = mesh.n.copy()
        if mesh.region.ndim == 1:
            temp = temp + 1
        else:
            temp[idx] = temp[idx] + 1
        assert np.all(padded_mesh.n == temp)

    for dim in mesh.region.dims:
        # adding two layers on pmin side and 3 on pmax, for each dimension
        padded_mesh = mesh.pad({dim: (2, 3)})
        # Add cell to pmin and pmax to get the correct padded region
        idx = mesh.region._dim2index(dim)
        temp = mesh.region.pmin.copy()
        if mesh.region.ndim == 1:
            temp = temp - 2 * cell
        else:
            temp[idx] = temp[idx] - 2 * cell[idx]
        assert np.allclose(padded_mesh.region.pmin, temp, atol=0)
        temp = mesh.region.pmax.copy()
        if mesh.region.ndim == 1:
            temp = temp + 3 * cell
        else:
            temp[idx] = temp[idx] + 3 * cell[idx]
        assert np.allclose(padded_mesh.region.pmax, temp, atol=0)
        temp = mesh.n.copy()
        if mesh.region.ndim == 1:
            temp = temp + 5
        else:
            temp[idx] = temp[idx] + 5
        assert np.all(padded_mesh.n == temp)

    pad_directions = {dim: (1, 1) for dim in mesh.region.dims}
    padded_mesh = mesh.pad(pad_directions)
    assert np.allclose(padded_mesh.region.pmin, mesh.region.pmin - cell, atol=0)
    assert np.allclose(padded_mesh.region.pmax, mesh.region.pmax + cell, atol=0)
    assert np.all(padded_mesh.n == mesh.n + 2 * np.ones_like(mesh.n))


@pytest.mark.parametrize(
    "p1, p2, cell, checks",
    [
        (0, 100, 10, {"dx": 10}),
        ((0, 0), (100e-9, 80e-6), (1e-9, 5e-6), {"dx": 1e-9, "dy": 5e-6}),
        (
            (0, 0, 0),
            (100e-9, 80e-9, 10e-9),
            (1e-9, 5e-9, 10e-9),
            {"dx": 1e-9, "dy": 5e-9, "dz": 10e-9},
        ),
        (
            (-5e-9, -5e-9, -5e-9, -5e-9),
            (5e-9, 5e-9, 5e-9, 5e-9),
            (0.5e-9, 1e-9, 2e-9, 5e-9),
            {"dx0": 0.5e-9, "dx1": 1e-9, "dx2": 2e-9, "dx3": 5e-9},
        ),
    ],
)
def test_getattr(p1, p2, cell, checks):
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    for key, val in checks.items():
        assert np.isclose(getattr(mesh, key), val, atol=0)

    with pytest.raises(AttributeError):
        mesh.dk

    # single-character attributes are handled differently
    with pytest.raises(AttributeError):
        mesh.a


@pytest.mark.parametrize(
    "p1, p2, n, in_dir, not_in_dir",
    [
        ((0,), (100,), (5,), ["dx"], ["dy", "dz"]),
        ((0, 0), (100, 80), (5, 5), ["dx", "dy"], ["dz"]),
        ((0, 0, 0), (100, 80, 10), (5, 5, 10), ["dx", "dy", "dz"], []),
        ((0, 0, 0, 0), (10, 10, 10, 5), (5, 5, 5, 5), ["dx0", "dx1", "dx2", "dx3"], []),
    ],
)
def test_dir(p1, p2, n, in_dir, not_in_dir):
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)

    assert all(i in dir(mesh) for i in in_dir)
    assert all(i not in dir(mesh) for i in not_in_dir)


@pytest.mark.parametrize(
    "p1, p2, cell, dV",
    [
        [1, 11, 2, 2],
        [(0, 0), (20e-9, 10e-9), (2.5e-9, 2.5e-9), 6.25e-18],
        [(0, 0, 0), (100, 80, 10), (1, 2, 2.5), 5],
        [(0, 0, 0, 0), (5e-9, 6e-9, 3e-9, 2e-9), (5e-9, 3e-9, 1.5e-9, 2e-9), 4.5e-35],
    ],
)
def test_dV(p1, p2, cell, dV):
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    assert np.isclose(mesh.dV, dV, atol=0)


def test_mpl(valid_mesh, tmp_path):
    if valid_mesh.region.ndim != 3:
        with pytest.raises(RuntimeError):
            valid_mesh.mpl()
    else:
        valid_mesh.mpl()
        valid_mesh.mpl(box_aspect=[1, 2, 3])
        valid_mesh.mpl(filename=tmp_path / "figure.pdf")

    plt.close("all")


def test_k3d(valid_mesh):
    if valid_mesh.region.ndim != 3:
        with pytest.raises(RuntimeError):
            valid_mesh.k3d()
    else:
        valid_mesh.k3d()
        valid_mesh.sel(x=(0, 1e-9)).k3d()


def test_k3d_mpl_subregions(tmp_path):
    p1 = (0, 0, 0)
    p2 = (100, 80, 10)
    cell = (100, 5, 10)
    subregions = {
        "r1": df.Region(p1=(0, 0, 0), p2=(100, 10, 10)),
        "r2": df.Region(p1=(0, 10, 0), p2=(100, 20, 10)),
        "r3": df.Region(p1=(0, 20, 0), p2=(100, 30, 10)),
        "r4": df.Region(p1=(0, 30, 0), p2=(100, 40, 10)),
        "r5": df.Region(p1=(0, 40, 0), p2=(100, 50, 10)),
        "r6": df.Region(p1=(0, 50, 0), p2=(100, 60, 10)),
        "r7": df.Region(p1=(0, 60, 0), p2=(100, 70, 10)),
        "r8": df.Region(p1=(0, 70, 0), p2=(100, 80, 10)),
    }
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

    # matplotlib tests
    mesh.mpl.subregions(box_aspect=(1, 1, 1), show_region=True)

    mesh.mpl.subregions(filename=tmp_path / "figure.pdf")
    plt.close("all")

    # k3d tests
    mesh.k3d.subregions()


def test_scale():
    # example 1: 1d mesh, 1 subregion
    mesh = df.Mesh(p1=0, p2=10, cell=1, subregions={"sr": df.Region(p1=1, p2=5)})
    n = 10
    assert np.all(mesh.n == n)
    res = mesh.scale(2.5)
    assert isinstance(res, df.Mesh)
    assert np.allclose(res.region.pmin, -7.5, atol=0)
    assert np.allclose(res.region.pmax, 17.5, atol=0)
    assert np.allclose(res.region.edges, 25, atol=0)
    assert np.all(res.n == n)
    assert np.allclose(res.cell, 2.5, atol=0)
    assert len(res.subregions) == 1
    assert np.allclose(res.subregions["sr"].pmin, -5, atol=0)
    assert np.allclose(res.subregions["sr"].pmax, 5, atol=0)

    # example 2: 1d mesh, 1 subregion, with given reference_point
    mesh = df.Mesh(p1=0, p2=10, cell=1, subregions={"sr": df.Region(p1=1, p2=5)})
    n = 10
    assert np.all(mesh.n == n)
    mesh.scale(2.5, reference_point=-4.2, inplace=True)
    assert isinstance(mesh, df.Mesh)
    assert np.allclose(mesh.region.pmin, 6.3, atol=0)
    assert np.allclose(mesh.region.pmax, 31.3, atol=0)
    assert np.allclose(mesh.region.edges, 25, atol=0)
    assert np.all(mesh.n == n)
    assert np.allclose(mesh.cell, 2.5, atol=0)
    assert len(mesh.subregions) == 1
    assert np.allclose(mesh.subregions["sr"].pmin, 8.8, atol=0)
    assert np.allclose(mesh.subregions["sr"].pmax, 18.8, atol=0)

    # example 3: 2d mesh, no subregions
    p1 = (-50e-9, 0)
    p2 = (50e-9, 20e-9)
    cell = (1e-9, 2e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    n = (100, 10)  # for tests
    assert np.all(mesh.n == n)

    res = mesh.scale(2)
    assert isinstance(res, df.Mesh)
    assert np.allclose(res.region.pmin, (-100e-9, -10e-9), atol=0)
    assert np.allclose(res.region.pmax, (100e-9, 30e-9), atol=0)
    assert np.allclose(res.region.edges, (200e-9, 40e-9), atol=0)
    assert np.all(res.n == n)
    assert np.allclose(res.cell, (2e-9, 4e-9), atol=0)
    assert res.subregions == {}

    mesh.scale((2, 0.5), inplace=True)
    assert np.allclose(mesh.region.pmin, (-100e-9, 5e-9), atol=0)
    assert np.allclose(mesh.region.pmax, (100e-9, 15e-9), atol=0)
    assert np.allclose(mesh.region.edges, (200e-9, 10e-9), atol=0)
    assert np.all(mesh.n == n)
    assert np.allclose(mesh.cell, (2e-9, 1e-9), atol=0)
    assert mesh.subregions == {}

    # example 4: 3d mesh, 2 subregions (partial overlap), inplace=True
    p1 = (-50e-9, -50e-9, 0)
    p2 = (50e-9, 50e-9, 20e-9)
    cell = (1e-9, 1e-9, 2e-9)
    subregions = {
        "sr1": df.Region(p1=(0, 0, 0), p2=(10e-9, 10e-9, 10e-9)),
        "sr2": df.Region(p1=p1, p2=p2),
    }
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
    n = (100, 100, 10)  # for tests
    assert np.all(mesh.n == n)

    mesh.scale((2, 4, 0.5), inplace=True)
    assert np.allclose(mesh.region.pmin, (-100e-9, -200e-9, 5e-9), atol=0)
    assert np.allclose(mesh.region.pmax, (100e-9, 200e-9, 15e-9), atol=0)
    assert np.allclose(mesh.region.edges, (200e-9, 400e-9, 10e-9), atol=0)
    assert np.all(mesh.n == n)
    assert np.allclose(mesh.cell, (2e-9, 4e-9, 1e-9), atol=0)
    assert len(mesh.subregions) == 2
    assert np.allclose(mesh.subregions["sr1"].pmin, (0, 0, 5e-9), atol=0)
    assert np.allclose(mesh.subregions["sr1"].pmax, (20e-9, 40e-9, 10e-9), atol=0)
    assert np.allclose(mesh.subregions["sr2"].pmin, (-100e-9, -200e-9, 5e-9), atol=0)
    assert np.allclose(mesh.subregions["sr2"].pmax, (100e-9, 200e-9, 15e-9), atol=0)


def test_translate():
    p1 = -50
    p2 = 50
    cell = 1
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    n = 100
    assert np.all(mesh.n == n)

    res = mesh.translate(3.2)
    assert isinstance(res, df.Mesh)
    assert np.allclose(res.region.pmin, -46.8, atol=0)
    assert np.allclose(res.region.pmax, 53.2, atol=0)
    assert np.allclose(res.region.edges, 100, atol=0)
    assert np.all(mesh.n == n)
    assert np.allclose(mesh.cell, cell, atol=0)

    mesh.translate(-5, inplace=True)
    assert np.allclose(mesh.region.pmin, -55, atol=0)
    assert np.allclose(mesh.region.pmax, 45, atol=0)
    assert np.allclose(mesh.region.edges, 100, atol=0)
    assert np.all(mesh.n == n)
    assert np.allclose(mesh.cell, cell, atol=0)

    p1 = (-50e-9, -50e-9, 0)
    p2 = (50e-9, 50e-9, 20e-9)
    cell = (1e-9, 1e-9, 2e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    n = (100, 100, 10)  # for tests
    assert np.all(mesh.n == n)
    subregions = {
        "sr1": df.Region(p1=(0, 0, 0), p2=(10e-9, 10e-9, 20e-9)),
        "sr2": df.Region(p1=p1, p2=p2),
    }
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
    assert np.all(mesh.n == n)

    res = mesh.translate((50e-9, 0, -10e-9))
    assert isinstance(res, df.Mesh)
    assert np.allclose(res.region.pmin, (0, -50e-9, -10e-9), atol=0)
    assert np.allclose(res.region.pmax, (100e-9, 50e-9, 10e-9), atol=0)
    assert np.allclose(res.region.edges, (100e-9, 100e-9, 20e-9), atol=0)
    assert np.all(mesh.n == n)
    assert np.allclose(mesh.cell, cell, atol=0)
    assert len(res.subregions) == 2
    assert np.allclose(res.subregions["sr1"].pmin, (50e-9, 0, -10e-9), atol=0)
    assert np.allclose(res.subregions["sr1"].pmax, (60e-9, 10e-9, 10e-9), atol=0)
    assert np.allclose(res.subregions["sr2"].pmin, (0, -50e-9, -10e-9), atol=0)
    assert np.allclose(res.subregions["sr2"].pmax, (100e-9, 50e-9, 10e-9), atol=0)

    mesh.translate((50e-9, 0, -10e-9), inplace=True)
    assert np.allclose(mesh.region.pmin, (0, -50e-9, -10e-9), atol=0)
    assert np.allclose(mesh.region.pmax, (100e-9, 50e-9, 10e-9), atol=0)
    assert np.allclose(mesh.region.edges, (100e-9, 100e-9, 20e-9), atol=0)
    assert np.all(mesh.n == n)
    assert np.allclose(mesh.cell, cell, atol=0)
    assert len(mesh.subregions) == 2
    assert np.allclose(res.subregions["sr1"].pmin, (50e-9, 0, -10e-9), atol=0)
    assert np.allclose(res.subregions["sr1"].pmax, (60e-9, 10e-9, 10e-9), atol=0)
    assert np.allclose(res.subregions["sr2"].pmin, (0, -50e-9, -10e-9), atol=0)
    assert np.allclose(res.subregions["sr2"].pmax, (100e-9, 50e-9, 10e-9), atol=0)


def test_rotate90():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(40e-9, 20e-9, 10e-9), n=(40, 10, 5))

    rotated = mesh.rotate90("x", "y")
    assert rotated.region.allclose(mesh.region.rotate90("x", "y"))
    assert np.allclose(rotated.n, (10, 40, 5))
    assert np.allclose(mesh.n, (40, 10, 5))

    rotated = mesh.rotate90("x", "z")
    assert np.allclose(rotated.n, (5, 10, 40))

    rotated = mesh.rotate90("y", "z", k=2, reference_point=(0, 0, 0))
    # ensure that the region is rotated
    assert np.allclose(rotated.region.pmin, (0, -20e-9, -10e-9))
    assert np.allclose(rotated.region.pmax, (40e-9, 0, 0))
    # the number of cells stays constant for 180° rotations
    assert np.allclose(rotated.n, (40, 10, 5))

    # subregion rotation
    mesh.subregions = {
        "sr1": df.Region(p1=(0, 0, 0), p2=(10e-9, 2e-9, 4e-9)),
        "sr2": df.Region(p1=(30e-9, 10e-9, 8e-9), p2=(33e-9, 16e-9, 6e-9)),
    }

    rotated = mesh.rotate90("x", "y")
    assert np.allclose(rotated.n, (10, 40, 5))
    assert np.allclose(rotated.region.pmin, (10e-9, -10e-9, 0))
    assert np.allclose(rotated.region.pmax, (30e-9, 30e-9, 10e-9))
    assert rotated.subregions["sr1"].allclose(
        df.Region(p1=(28e-9, -10e-9, 0), p2=(30e-9, 0, 4e-9))
    )
    assert rotated.subregions["sr2"].allclose(
        df.Region(p1=(14e-9, 20e-9, 6e-9), p2=(20e-9, 23e-9, 8e-9))
    )
    # check that the original object has not changed
    assert mesh.subregions == {
        "sr1": df.Region(p1=(0, 0, 0), p2=(10e-9, 2e-9, 4e-9)),
        "sr2": df.Region(p1=(30e-9, 10e-9, 8e-9), p2=(33e-9, 16e-9, 6e-9)),
    }
    # inplace rotation
    mesh.rotate90("z", "y", reference_point=(1e-9, 2e-9, 1e-9), inplace=True)
    assert np.allclose(mesh.n, (40, 5, 10))
    assert np.allclose(mesh.region.pmin, (0, 1e-9, -17e-9), atol=0)
    assert np.allclose(mesh.region.pmax, (40e-9, 11e-9, 3e-9), atol=0)
    assert mesh.subregions["sr1"].allclose(
        df.Region(p1=(0, 1e-9, 1e-9), p2=(10e-9, 5e-9, 3e-9))
    )
    assert mesh.subregions["sr2"].allclose(
        df.Region(p1=(30e-9, 7e-9, -13e-9), p2=(33e-9, 9e-9, -7e-9))
    )

    # 4d region
    region = df.Region(p1=(0, 0, 0, 0), p2=(40e-9, 20e-9, 10e-9, 5e-9))
    mesh = df.Mesh(region=region, n=(2, 3, 4, 5))
    # subregion rotation
    mesh.subregions = {
        "sr1": df.Region(p1=(0, 0, 0, 0), p2=(20e-9, 20e-9, 10e-9, 5e-9)),
        "sr2": df.Region(p1=(20e-9, 0, 0, 0), p2=(40e-9, 20e-9, 10e-9, 5e-9)),
    }

    rotated = mesh.rotate90("x0", "x1")
    assert np.allclose(rotated.n, (3, 2, 4, 5))
    assert rotated.subregions["sr1"].allclose(
        df.Region(p1=(10e-9, -10e-9, 0, 0), p2=(30e-9, 10e-9, 10e-9, 5e-9))
    )
    assert rotated.subregions["sr2"].allclose(
        df.Region(p1=(10e-9, 10e-9, 0, 0), p2=(30e-9, 30e-9, 10e-9, 5e-9))
    )


def test_slider():
    p1 = (-10e-9, -5e-9, 10e-9)
    p2 = (10e-9, 5e-9, 0)
    cell = (1e-9, 2.5e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    assert isinstance(mesh, df.Mesh)

    x_slider = mesh.slider("x")
    assert isinstance(x_slider, ipywidgets.SelectionSlider)

    y_slider = mesh.slider("y", multiplier=1)
    assert isinstance(y_slider, ipywidgets.SelectionSlider)

    z_slider = mesh.slider("z", multiplier=1e3)
    assert isinstance(z_slider, ipywidgets.SelectionSlider)


def test_axis_selector():
    p1 = (-10e-9, -5e-9, 10e-9)
    p2 = (10e-9, 5e-9, 0)
    cell = (1e-9, 2.5e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    assert isinstance(mesh, df.Mesh)

    axis_widget = mesh.axis_selector()
    assert isinstance(axis_widget, ipywidgets.Dropdown)

    axis_widget = mesh.axis_selector(widget="radiobuttons")
    assert isinstance(axis_widget, ipywidgets.RadioButtons)

    axis_widget = mesh.axis_selector(description="something")
    assert isinstance(axis_widget, ipywidgets.Dropdown)

    with pytest.raises(ValueError):
        axis_widget = mesh.axis_selector(widget="something")


@pytest.mark.parametrize(
    "p1, p2, cell",
    [
        (0, 100, 10),
        (np.array([-100, -50]), np.array([100, 50]), (10, 5)),
        (np.array([0, 0, 0]), np.array([100, 60, 10]), (10, 10, 5)),
        (np.array([0, 0, 0, 0]), np.array([100, 60, 10, 20]), (10, 10, 5, 5)),
    ],
)
def test_save_load_subregions(p1, p2, cell, tmp_path):
    sr = {"r1": df.Region(p1=p2, p2=p2 / 2), "r2": df.Region(p1=p2 / 2, p2=p2)}
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=sr)
    assert isinstance(mesh, df.Mesh)

    mesh.save_subregions(tmp_path / "mesh.json")

    test_mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    assert test_mesh.subregions == {}
    test_mesh.load_subregions(tmp_path / "mesh.json")
    assert test_mesh.subregions == sr


def test_coordinate_field(valid_mesh):
    cfield = valid_mesh.coordinate_field()
    assert isinstance(cfield, df.Field)
    manually = df.Field(valid_mesh, nvdim=valid_mesh.region.ndim, value=lambda p: p)
    assert cfield.allclose(manually, atol=0)
    for dim in valid_mesh.region.dims:
        index = [0] * valid_mesh.region.ndim
        index[valid_mesh.region._dim2index(dim)] = slice(None)
        # extra index for vector dimension: vector component along the current direction
        index = tuple(index) + (valid_mesh.region._dim2index(dim),)
        assert np.allclose(cfield.array[index], getattr(valid_mesh.cells, dim), atol=0)


def test_sel_convert_intput():
    # 3d
    p1 = (0, 0, 0)
    p2 = (5, 10, 10)
    mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 2))

    # dim, dim_index, selection, selection_index
    assert mesh._sel_convert_input("x") == ("x", 0, 2.5, 2)
    assert mesh._sel_convert_input(y=4.1) == ("y", 1, 4.5, 4)
    assert mesh._sel_convert_input(y=(0, 2)) == ("y", 1, [0.5, 2.5], slice(0, 3))
    assert mesh._sel_convert_input(z=(2, 5)) == ("z", 2, [3, 5], slice(1, 3))

    # 1d
    mesh = df.Mesh(p1=0, p2=10e-9, n=5)
    assert mesh._sel_convert_input("x") == ("x", 0, 5e-9, 2)


def test_sel_string_1D():
    # Sel center of 1D mesh with no subregions
    p1 = 0
    p2 = 100
    cell = 10
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    # 0 dimensional regions/meshes are not "allowed"
    with pytest.raises(ValueError):
        mesh.sel("x")

    with pytest.raises(ValueError):
        mesh.sel(x=50)


@pytest.mark.parametrize(
    "p1, p2, dims, cell",
    [
        ([0, 0], [100, 50], list("ab"), (10, 5)),
        ([0, 0, 0], [100, 60, 30], list("abc"), (10, 10, 5)),
        ([0, 0, 0, 0], [100, 60, 30, 30], None, (10, 10, 5, 5)),
    ],
)
def test_sel_single(p1, p2, dims, cell):
    region = df.Region(p1=p1, p2=p2, dims=dims)
    # Sel center of mesh (>1d) with no subregions
    mesh = df.Mesh(region=region, cell=cell)
    for dim in mesh.region.dims:
        sub_mesh = mesh.sel(dim)
        assert isinstance(sub_mesh, df.Mesh)
        assert sub_mesh.region.ndim == mesh.region.ndim - 1
        bool_ = [i != dim for i in mesh.region.dims]
        assert np.allclose(sub_mesh.region.pmin, mesh.region.pmin[bool_], atol=0)
        assert np.allclose(sub_mesh.region.pmax, mesh.region.pmax[bool_], atol=0)
        assert sub_mesh.region.dims == tuple(d for d in mesh.region.dims if d != dim)
        assert np.all(sub_mesh.cell == mesh.cell[bool_])

    # Sel single plane of mesh (>1d) with no subregions
    for dim in mesh.region.dims:
        options = {dim: 10}
        sub_mesh = mesh.sel(**options)
        assert isinstance(sub_mesh, df.Mesh)
        assert sub_mesh.region.ndim == mesh.region.ndim - 1
        bool_ = [i != dim for i in mesh.region.dims]
        assert np.allclose(sub_mesh.region.pmin, mesh.region.pmin[bool_], atol=0)
        assert np.allclose(sub_mesh.region.pmax, mesh.region.pmax[bool_], atol=0)
        assert sub_mesh.region.dims == tuple([d for d in mesh.region.dims if d != dim])
        assert np.all(sub_mesh.cell == mesh.cell[bool_])
        with pytest.raises(ValueError):
            mesh.sel(dim, 10)
        with pytest.raises(ValueError):
            mesh.sel(dim, **options)
        with pytest.raises(ValueError):
            mesh.sel()
        with pytest.raises(ValueError):
            mesh.sel(10)
            # out-of-bounds test
            with pytest.raises(ValueError):
                mesh.sel(**{dim: 200})


@pytest.mark.parametrize(
    "p1, p2, dims, cell",
    [
        [0, 100, list("a"), 10],
        [(0, 0), (100, 50), list("ab"), (10, 10)],
        [(0, 0, 0), (100, 60, 50), list("abc"), (10, 10, 10)],
        [(0, 0, 0, 0), (100, 60, 50, 40), None, (10, 10, 10, 10)],
    ],
)
def test_sel_range(p1, p2, dims, cell):
    region = df.Region(p1=p1, p2=p2, dims=dims)
    # Sel range of mesh with no subregions
    mesh = df.Mesh(region=region, cell=cell)
    for dim in mesh.region.dims:
        options = {dim: (12.5, 29.5)}
        sub_mesh = mesh.sel(**options)
        assert isinstance(sub_mesh, df.Mesh)
        assert sub_mesh.region.ndim == mesh.region.ndim
        bool_ = [i != dim for i in mesh.region.dims]
        idx = mesh.region._dim2index(dim)
        assert np.isclose(sub_mesh.region.pmax[idx], 30.0, atol=0)
        assert np.isclose(sub_mesh.region.pmin[idx], 10.0, atol=0)
        assert np.allclose(sub_mesh.region.pmin[bool_], mesh.region.pmin[bool_], atol=0)
        assert np.allclose(sub_mesh.region.pmax[bool_], mesh.region.pmax[bool_], atol=0)
        assert sub_mesh.region.dims == mesh.region.dims
        assert all(sub_mesh.n[bool_] == mesh.n[bool_])
        assert sub_mesh.n[idx] == 2

        # Single layer
        options = {dim: np.array([2.5, 13.5])}
        sub_mesh = mesh.sel(**options)
        assert isinstance(sub_mesh, df.Mesh)
        assert sub_mesh.region.ndim == mesh.region.ndim
        bool_ = [i != dim for i in mesh.region.dims]
        idx = mesh.region._dim2index(dim)
        assert np.isclose(sub_mesh.region.pmax[idx], 20.0, atol=0)
        assert np.isclose(sub_mesh.region.pmin[idx], 0.0, atol=0)
        assert np.allclose(sub_mesh.region.pmin[bool_], mesh.region.pmin[bool_], atol=0)
        assert np.allclose(sub_mesh.region.pmax[bool_], mesh.region.pmax[bool_], atol=0)
        assert sub_mesh.region.dims == mesh.region.dims
        assert all(sub_mesh.n[bool_] == mesh.n[bool_])
        assert sub_mesh.n[idx] == 2

        # Too many values
        with pytest.raises(ValueError):
            options = {dim: (12.5, 29.5, 3.5)}
            mesh.sel(**options)

        # Out of bounds
        with pytest.raises(ValueError):
            options = {dim: (12.5, 1000)}
            mesh.sel(**options)

        # Wrong value type
        with pytest.raises(TypeError):
            options = {dim: (12.5, dim)}
            mesh.sel(**options)


def test_sel_subregions():
    p1 = (0, 0, 0)
    p2 = (20, 20, 20)
    region = df.Region(p1=p1, p2=p2, dims=["x0", "x1", "x2"])
    cell = (2, 2, 2)
    mesh = df.Mesh(region=region, cell=cell)

    sub_region = {
        "in": df.Region(p1=(2, 2, 2), p2=(8, 8, 8)),
        "out": df.Region(p1=(0, 0, 0), p2=(2, 2, 2)),
        "half": df.Region(p1=(4, 4, 4), p2=(12, 12, 12)),
    }
    mesh = df.Mesh(region=region, cell=cell, subregions=sub_region)
    sub_mesh = mesh.sel(x0=(3.6, 7.8))
    assert sorted(sub_mesh.subregions) == ["half", "in"]
    assert np.isclose(sub_mesh.subregions["in"].pmin[0], 2.0, atol=0)
    assert np.isclose(sub_mesh.subregions["in"].pmax[0], 8.0, atol=0)
    assert np.isclose(sub_mesh.subregions["half"].pmin[0], 4.0, atol=0)
    assert np.isclose(sub_mesh.subregions["half"].pmax[0], 8.0, atol=0)

    # reduce to single layer
    sub_mesh = mesh.sel("x0")
    assert sorted(sub_mesh.subregions) == ["half"]
    assert sub_mesh.subregions["half"].ndim == 2
    assert np.allclose(sub_mesh.subregions["half"].pmin, [4, 4], atol=0)
    assert np.allclose(sub_mesh.subregions["half"].pmax, [12, 12], atol=0)

    # No subregions in selection
    sub_region = {
        "out1": df.Region(p1=(6, 6, 6), p2=(10, 10, 10)),
        "out2": df.Region(p1=(0, 0, 0), p2=(2, 2, 2)),
    }
    mesh = df.Mesh(region=region, cell=cell, subregions=sub_region)
    sub_mesh = mesh.sel(x0=(4.5, 5.1))
    assert sub_mesh.subregions == {}
    sub_mesh = mesh.sel("x0")
    assert sub_mesh.subregions == {}

    # cell boundary and cell midpoint return the same selection
    assert mesh.sel(x0=6) == mesh.sel(x0=7)


def test_fftn_mesh():
    # 3D mesh with even number of cells
    p1 = (-10, -10, -5)
    p2 = (10, 10, 5)
    cell = (1, 1, 1)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    mesh_fft = mesh.fftn()

    assert mesh_fft.region.ndim == 3
    assert mesh_fft.region.dims == ("k_x", "k_y", "k_z")
    assert mesh_fft.region.units == ("(m)$^{-1}$", "(m)$^{-1}$", "(m)$^{-1}$")
    assert np.array_equal(mesh_fft.n, (20, 20, 10))
    assert np.allclose(mesh_fft.cell, (0.05, 0.05, 0.1))
    assert np.allclose(mesh_fft.region.pmin, (-0.525, -0.525, -0.55))
    assert np.allclose(mesh_fft.region.pmax, (0.475, 0.475, 0.45))

    mesh_fft = mesh.fftn(rfft=True)

    assert mesh_fft.region.ndim == 3
    assert mesh_fft.region.dims == ("k_x", "k_y", "k_z")
    assert mesh_fft.region.units == ("(m)$^{-1}$", "(m)$^{-1}$", "(m)$^{-1}$")
    assert np.array_equal(mesh_fft.n, (20, 20, 6))
    assert np.allclose(mesh_fft.cell, (0.05, 0.05, 0.1))
    assert np.allclose(mesh_fft.region.pmin, (-0.525, -0.525, -0.05))
    assert np.allclose(mesh_fft.region.pmax, (0.475, 0.475, 0.55))

    mesh_fft = mesh.fftn().ifftn()
    assert mesh_fft.region.units == ("m", "m", "m")
    assert mesh_fft.region.dims == ("x", "y", "z")
    assert mesh_fft.allclose(mesh)

    mesh_fft = mesh.fftn(rfft=True).ifftn(rfft=True)
    assert mesh_fft.region.units == ("m", "m", "m")
    assert mesh_fft.region.dims == ("x", "y", "z")
    assert mesh_fft.allclose(mesh)

    mesh_fft = mesh.fftn(rfft=True).ifftn(rfft=True, shape=mesh.n)
    assert mesh_fft.region.units == ("m", "m", "m")
    assert mesh_fft.region.dims == ("x", "y", "z")
    assert mesh_fft.allclose(mesh)

    # 3D mesh with odd number of cells
    p1 = (-10, -10, -5)
    p2 = (10, 10, 4)
    cell = (1, 1, 1)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    mesh_fft = mesh.fftn(rfft=True).ifftn(rfft=True, shape=mesh.n)
    assert mesh_fft.region.ndim == 3
    assert np.array_equal(mesh_fft.n, (20, 20, 9))
    assert np.allclose(mesh_fft.cell, (1, 1, 1))
    assert np.allclose(mesh_fft.region.pmin, (-10, -10, -4.5))
    assert np.allclose(mesh_fft.region.pmax, (10, 10, 4.5))

    # 1D mesh with even number of cells
    p1 = -10
    p2 = 10
    cell = 1
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    mesh_fft = mesh.fftn()
    assert mesh_fft.region.ndim == 1
    assert mesh_fft.region.dims == ("k_x",)
    assert mesh_fft.region.units == ("(m)$^{-1}$",)
    assert np.array_equal(mesh_fft.n, (20,))
    assert np.allclose(mesh_fft.cell, 0.05)
    assert np.allclose(mesh_fft.region.pmin, (-0.525,))
    assert np.allclose(mesh_fft.region.pmax, (0.475,))

    mesh_fft = mesh.fftn(rfft=True)
    assert mesh_fft.region.ndim == 1
    assert mesh_fft.region.dims == ("k_x",)
    assert mesh_fft.region.units == ("(m)$^{-1}$",)
    assert np.array_equal(mesh_fft.n, (11,))
    assert np.allclose(mesh_fft.cell, 0.05)
    assert np.allclose(mesh_fft.region.pmin, (-0.025,))
    assert np.allclose(mesh_fft.region.pmax, (0.525,))

    mesh_fft = mesh.fftn().ifftn()
    assert mesh_fft.region.units == ("m",)
    assert mesh_fft.region.dims == ("x",)
    assert mesh_fft.allclose(mesh)

    mesh_fft = mesh.fftn(rfft=True).ifftn(rfft=True)
    assert mesh_fft.region.units == ("m",)
    assert mesh_fft.region.dims == ("x",)
    assert mesh_fft.allclose(mesh)

    mesh_fft = mesh.fftn(rfft=True).ifftn(rfft=True, shape=mesh.n)
    assert mesh_fft.region.units == ("m",)
    assert mesh_fft.region.dims == ("x",)
    assert mesh_fft.allclose(mesh)

    # 1D mesh with odd number of cells
    p1 = -10
    p2 = 9
    cell = 1
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    mesh_fft = mesh.fftn(rfft=True).ifftn(rfft=True, shape=mesh.n)
    assert mesh_fft.region.ndim == 1
    assert np.array_equal(mesh_fft.n, (19,))
    assert np.allclose(mesh_fft.cell, (1,))
    assert mesh_fft.region.units == ("m",)
    assert mesh_fft.region.dims == ("x",)
    assert np.allclose(mesh_fft.region.pmin, (-9.5,))
    assert np.allclose(mesh_fft.region.pmax, (9.5,))


@pytest.mark.parametrize(
    "shape, error",
    [
        [(20, 20, 10), ValueError],
        [(20, 20, 7), ValueError],
        ["a", TypeError],
        [(20,), ValueError],
        [(20, 20, 10, 10), ValueError],
        [10, ValueError],
    ],
)
def test_irfftn_3d(shape, error):
    p1 = (-10, -10, -5)
    p2 = (10, 10, 4)
    cell = (1, 1, 1)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    mesh_fft = mesh.fftn(rfft=True)

    with pytest.raises(error):
        mesh_fft.ifftn(rfft=True, shape=shape)


@pytest.mark.parametrize(
    "shape, error",
    [
        [(20, 20), ValueError],
        ["a", TypeError],
        [(20,), ValueError],  # Too many cells
        [(17,), ValueError],  # Too few cells
        [(20, 20, 10, 10), ValueError],
        [17, ValueError],  # Too few cells
        [20, ValueError],  # Too many cells
    ],
)
def test_irfftn_1d(shape, error):
    p1 = -10
    p2 = 9
    cell = 1
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    mesh_fft = mesh.fftn(rfft=True)

    with pytest.raises(error):
        mesh_fft.ifftn(rfft=True, shape=shape)
