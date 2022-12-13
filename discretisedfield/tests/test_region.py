import re

import matplotlib.pyplot as plt
import numpy as np
import pytest

import discretisedfield as df
import discretisedfield.plotting.util as plot_util

html_re = (
    r"<strong>Region</strong>( <i>\w+</i>)?\s*"
    r"<ul>\s*"
    r"<li>pmin = \[.*\]</li>\s*"
    r"<li>pmax = \[.*\]</li>\s*"
    r"<li>dims = .*</li>\s*"
    r"<li>units = .*</li>\s*"
    r"</ul>"
)


@pytest.fixture
def region_3d():
    p1 = (-50e-9, -50e-9, 0)
    p2 = (50e-9, 50e-9, 20e-9)
    return df.Region(p1=p1, p2=p2)


@pytest.mark.parametrize(
    "p1, p2, ndim",
    [
        [0, 2e-10, 1],
        [(5e-9,), (2e-10,), 1],
        [(5e-9,), -2e-10, 1],
        [(0, 0), (5, 7), 2],
        [(3, -1), [0, 5], 2],
        [(0, 0), np.array([20e-9, 10e-9]), 2],
        [[1.5e-9, -2e-9], np.array((7.5e-9, 2e-9)), 2],
        [(0, 0, 0), (5, 5, 5), 3],
        [(-1, 0, -3), (5, 7, 5), 3],
        [(0, 0, 0), (5e-9, 5e-9, 5e-9), 3],
        [(-1.5e-9, -5e-9, 0), (1.5e-9, -15e-9, -10e-9), 3],
        [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0)), 3],
        [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6), 3],
        [(0, 125e-9, 0), (500e-9, 0, -3e-9), 3],
        [(0, 1, 2, 3), (10, 9, 8, 7), 4],
        [[0, 1, 2, 3, 4], [10, 9, 8, 7, 6], 5],
        [np.arange(10.0), np.arange(10.0, 20.0), 10],
    ],
)
def test_init_valid_args(p1, p2, ndim):
    region = df.Region(p1=p1, p2=p2)
    assert isinstance(region, df.Region)
    pattern = r"^Region\(pmin=\[.+\], pmax=\[.+\], dims=\[.+\], units=\[.+\]\)$"
    assert re.match(pattern, str(region))
    assert region.ndim == ndim


@pytest.mark.parametrize(
    "p1,p2,error",
    [
        [[], [], ValueError],
        [("1", 0, 0), (1, 1, 1), TypeError],
        [(-1.5e-9, -5e-9, "a"), (1.5e-9, 15e-9, 16e-9), TypeError],
        [(-1.5e-9, -5e-9, 0), (1.5e-9, 16e-9), ValueError],
        [-1.5e-9, (1.5e-9, 16e-9), ValueError],
        [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 1 + 2j), TypeError],
        ["string", (5, 1, 1e-9), TypeError],
        ["abc", "def", TypeError],
        [[(-1, 0, -3)], [(5, 7, 5)], TypeError],
        [
            np.array([-1, 0, -3]).reshape((-1, 1)),
            np.array([5, 7, 5]).reshape((-1, 1)),
            TypeError,
        ],
    ],
)
def test_init_invalid_args(p1, p2, error):
    with pytest.raises(error):
        df.Region(p1=p1, p2=p2)


@pytest.mark.parametrize(
    "p1,p2",
    [
        [1, 1.0],
        [(np.pi), [np.pi]],
        [(0, 0), (0, 0)],
        [(0, 100e-9, 1e-9), (150e-9, 100e-9, 6e-9)],
        [(0, 101e-9, -1), (150e-9, 101e-9, 0)],
        [(10e9, 10e3, 0), (0.01e12, 11e3, 5)],
        [(1e-9, 100e-9, 7e-9, 4e-6), (1e-9, 100e-9, 7e-9, 4e-6)],
    ],
)
def test_init_zero_edge_length(p1, p2):
    with pytest.raises(ValueError):
        df.Region(p1=p1, p2=p2)


@pytest.mark.parametrize(
    "p1, p2, pmin, pmax, edges, center, volume",
    [
        (0, 10, 0, 10, 10, 5, 10),
        (0, -3e-9, -3e-9, 0, 3e-9, -1.5e-9, 3e-9),
        [
            (1, 3e-6),
            (-1, 7e-5),
            (-1, 3e-6),
            (1, 7e-5),
            (2, 6.7e-5),
            (0, 3.65e-5),
            1.34e-4,
        ],
        [
            (0, -4, 16.5),
            (15, -6, 11),
            (0, -6, 11),
            (15, -4, 16.5),
            (15, 2, 5.5),
            (7.5, -5, 13.75),
            165,
        ],
        [
            (-10e6, 0, 0),
            (10e6, 1e6, 1e6),
            (-10e6, 0, 0),
            (10e6, 1e6, 1e6),
            (20e6, 1e6, 1e6),
            (0, 0.5e6, 0.5e6),
            2e19,
        ],
        [
            (-18.5e-9, 10e-9, 0),
            (10e-9, 5e-9, -10e-9),
            (-18.5e-9, 5e-9, -10e-9),
            (10e-9, 10e-9, 0),
            (28.5e-9, 5e-9, 10e-9),
            (-4.25e-9, 7.5e-9, -5e-9),
            1.425e-24,
        ],
        [
            np.arange(10),
            np.arange(1, 20, 2),
            np.arange(10),
            np.arange(1, 20, 2),
            np.arange(1, 11),
            np.arange(0.5, 14.1, 1.5),
            3628800,
        ],
    ],
)
def test_pmin_pmax_edges_center_volume(p1, p2, pmin, pmax, edges, center, volume):
    region = df.Region(p1=p1, p2=p2)

    assert isinstance(region, df.Region)
    assert np.allclose(region.pmin, pmin, atol=0)
    assert np.allclose(region.pmax, pmax, atol=0)
    assert np.allclose(region.edges, edges, atol=0)
    assert np.allclose(region.center, center, atol=0)
    assert np.isclose(region.volume, volume, atol=0)


@pytest.mark.parametrize(
    "p1, p2, units, dims, rstr",
    [
        [
            -10e-9,
            10e-9,
            "nm",
            "a",
            "Region(pmin=[-1e-08], pmax=[1e-08], dims=['a'], units=['nm'])",
        ],
        [
            (0, -10e-9),
            (100e-9, 10e-9),
            ["nm", "s"],
            ["space", "time"],
            (
                "Region(pmin=[0.0, -1e-08], pmax=[1e-07, 1e-08], dims=['space',"
                " 'time'], units=['nm', 's'])"
            ),
        ],
        [
            (-1, -4, 11),
            (15, 10.1, 12.5),
            None,
            None,
            (
                "Region(pmin=[-1.0, -4.0, 11.0], pmax=[15.0, 10.1, 12.5],"
                " dims=['x', 'y', 'z'], units=['m', 'm', 'm'])"
            ),
        ],
        [
            (0, 0, 0, 0),
            (1e-6, 1e-3, 1, 1e3),
            None,
            None,
            (
                "Region(pmin=[0.0, 0.0, 0.0, 0.0], pmax=[1e-06, 0.001, 1.0, 1000.0],"
                " dims=['x0', 'x1', 'x2', 'x3'], units=['m', 'm', 'm', 'm'])"
            ),
        ],
    ],
)
def test_repr(p1, p2, units, dims, rstr):
    region = df.Region(p1=p1, p2=p2, units=units, dims=dims)

    assert isinstance(region, df.Region)
    assert repr(region) == rstr
    assert re.match(html_re, region._repr_html_())


@pytest.mark.parametrize(
    "p1_1, p1_2, p2",
    [
        [5e-9, 6e-9, 10e-9],
        [(-100e-9, -10e-9), (-99e-9, -10e-9), (100e-9, 10e-9)],
        [(0, 0, 0), (3, 3, 3), (10, 10, 10)],
    ],
)
def test_eq(p1_1, p1_2, p2):
    region1 = df.Region(p1=p1_1, p2=p2)
    region2 = df.Region(p1=p1_1, p2=p2)
    region3 = df.Region(p1=p1_2, p2=p2)

    assert isinstance(region1, df.Region)
    assert isinstance(region2, df.Region)
    assert isinstance(region3, df.Region)
    assert region1 == region2
    assert not region1 != region2
    assert region1 != region3
    assert not region1 == region3


@pytest.mark.parametrize(
    "p1_1, p1_2, p2",
    [
        [5e-9, 6e-9, 10e-9],
        [(-100e-9, -10e-9), (-99e-9, -10e-9), (100e-9, 10e-9)],
        [(0, 0, 0), (3, 3, 3), (10, 10, 10)],
    ],
)
def test_allclose(p1_1, p1_2, p2, atol=0):
    region1 = df.Region(p1=p1_1, p2=p2)
    region2 = df.Region(p1=p1_1, p2=p2)
    region3 = df.Region(p1=p1_2, p2=p2)

    assert isinstance(region1, df.Region)
    assert isinstance(region2, df.Region)
    assert isinstance(region3, df.Region)
    assert region1.allclose(region2, atol=0)
    assert not region1.allclose(region3, atol=0)
    assert not region2.allclose(region3, atol=0)


def test_tolerance_factor():  # TODO
    p1 = (0, 0, 0)
    p2 = (100e-9, 100e-9, 100e-9)
    region = df.Region(p1=p1, p2=p2)
    assert np.isclose(region.tolerance_factor, 1e-12, atol=0)

    region = df.Region(p1=p1, p2=p2, tolerance_factor=1e-3)
    assert np.isclose(region.tolerance_factor, 1e-3, atol=0)
    region.tolerance_factor = 1e-6
    assert np.isclose(region.tolerance_factor, 1e-6, atol=0)


@pytest.mark.parametrize("factor", [None, 1.0])
def test_contains_1d(factor):
    region = df.Region(p1=0, p2=20e-9)
    assert isinstance(region, df.Region)

    if factor is not None:
        region.tolerance_factor = factor
    tol = np.min(region.edges) * region.tolerance_factor
    tol_in = tol / 2
    tol_out = tol * 2

    assert 0 in region
    assert 20e-9 in region

    assert -tol_in in region
    assert -tol_out not in region

    assert 20e-9 + tol_in in region
    assert 20e-9 + tol_out not in region


@pytest.mark.parametrize("factor", [None, 1.0])
@pytest.mark.parametrize(
    "p1, p2",
    [
        [(0,), (20e-9,)],
        [(0, 10e-9), (20e-9, 0)],
        [(0, 10e-9, 0), (10e-9, 0, 20e-9)],
        [(0, 10e-9, 0, -10e-9), (10e-9, 0, 20e-9, 30e-9)],
    ],
)
def test_contains(factor, p1, p2):
    region = df.Region(p1=p1, p2=p2)
    assert isinstance(region, df.Region)

    if factor is not None:
        region.tolerance_factor = factor
    tol = np.min(region.edges) * region.tolerance_factor
    tol_in = tol / 2
    tol_out = tol * 2

    assert p1 in region
    assert p2 in region

    for i in range(region.ndim):
        point = list(region.pmin)
        point[i] -= tol_in
        assert point in region

        point = list(region.pmin)
        point[i] -= tol_out
        assert point not in region

    for i in range(region.ndim):
        point = list(region.pmax)
        point[i] += tol_in
        assert point in region

        point = list(region.pmax)
        point[i] += tol_out
        assert point not in region

    point = list(region.center)
    point[-1] = region.pmax[-1] + tol_in
    assert point in region

    point = list(region.center)
    point[-1] = region.pmax[-1] + tol_out
    assert point not in region


@pytest.mark.parametrize(
    "p11, p12, p21, p22, expected",
    [
        # 1D
        [(0,), (10e-9,), (20e-9,), (40e-9,), "x"],
        # 2D
        [(0, 0), (10e-9, 50e-9), (20e-9, 0), (30e-9, 50e-9), "x"],
        [(0, 0), (10e-9, 50e-9), (0, -50e-9), (10e-9, -10e-9), "y"],
        # 3D
        [(0, 0, 0), (10e-9, 50e-9, 20e-9), (20e-9, 0, 0), (30e-9, 50e-9, 20e-9), "x"],
        [(0, 0, 0), (10e-9, 50e-9, 20e-9), (0, -50e-9, 0), (10e-9, -10e-9, 20e-9), "y"],
        [(0, 0, 0), (100e-9, 50e-9, 20e-9), (0, 0, 20e-9), (100e-9, 50e-9, 30e-9), "z"],
        # 4D
        [
            (0, 0, 0, 0),
            (10e-9, 50e-9, 20e-9, 30e-9),
            (20e-9, 0, 0, 0),
            (30e-9, 50e-9, 20e-9, 30e-9),
            "x0",
        ],
        [
            (0, 0, 0, 0),
            (10e-9, 50e-9, 20e-9, 30e-9),
            (0, -50e-9, 0, 0),
            (10e-9, -10e-9, 20e-9, 30e-9),
            "x1",
        ],
        [
            (0, 0, 0, 0),
            (100e-9, 50e-9, 20e-9, 30e-9),
            (0, 0, 20e-9, 0),
            (100e-9, 50e-9, 30e-9, 30e-9),
            "x2",
        ],
        [
            (0, 0, 0, 0),
            (100e-9, 50e-9, 20e-9, 30e-9),
            (0, 0, 0, 30e-9),
            (100e-9, 50e-9, 20e-9, 40e-9),
            "x3",
        ],
    ],
)
def test_facing_surface(p11, p12, p21, p22, expected):
    region1 = df.Region(p1=p11, p2=p12)
    region2 = df.Region(p1=p21, p2=p22)

    res = region1.facing_surface(region2)

    assert res[0] == expected
    if (
        region1.pmin[region1._dim2index(res[0])]
        < region2.pmin[region2._dim2index(res[0])]
    ):
        assert res[1] == region1
        assert res[2] == region2
    else:
        assert res[1] == region2
        assert res[2] == region1
    assert region1.facing_surface(region2) == region2.facing_surface(region1)


@pytest.mark.parametrize(
    "p11, p12, p21, p22",
    [
        [(0,), (10e-9,), (0,), (40e-9,)],
        [(0, 0), (10e-9, 50e-9), (0, 0), (10e-9, 70e-9)],
        [(0, 0, 0), (10e-9, 50e-9, 20e-9), (0, 0, 0), (70e-9, 50e-9, 70e-9)],
        [(0, 0, 0), (10e-9, 50e-9, 20e-9), (0, 0, 0), (10e-9, 70e-9, 20e-9)],
        [(0, 0, 0), (10e-9, 50e-9, 20e-9), (0, 0, 0), (70e-9, 50e-9, 20e-9)],
        [
            (0, 0, 0, 0),
            (10e-9, 50e-9, 20e-9, 30e-9),
            (0, 0, 0, 0),
            (70e-9, 50e-9, 20e-9, 30e-9),
        ],
        [
            (0, 0, 0, 0),
            (10e-9, 50e-9, 20e-9, 30e-9),
            (0, 0, 0, 0),
            (10e-9, 70e-9, 20e-9, 30e-9),
        ],
        [
            (0, 0, 0, 0),
            (10e-9, 50e-9, 20e-9, 30e-9),
            (0, 0, 0, 0),
            (10e-9, 50e-9, 70e-9, 30e-9),
        ],
        [
            (0, 0, 0, 0),
            (10e-9, 50e-9, 20e-9, 30e-9),
            (0, 0, 0, 0),
            (10e-9, 50e-9, 20e-9, 70e-9),
        ],
    ],
)
def test_facing_surface_error(p11, p12, p21, p22):
    # Exceptions
    region1 = df.Region(p1=p11, p2=p12)
    region2 = df.Region(p1=p21, p2=p22)

    with pytest.raises(ValueError):
        region1.facing_surface(region2)

    with pytest.raises(TypeError):
        region1.facing_surface(5)


@pytest.mark.parametrize(
    "region,multiplier",
    [
        [df.Region(p1=(-50e-9, -50e-9, 0), p2=(50e-9, 50e-9, 20e-9)), 1e-9],
        [df.Region(p1=(0, 0), p2=(1e-5, 1e-4)), 1e-6],
    ],
)
def test_multiplier(region, multiplier):  # TODO remove
    assert region.multiplier == multiplier


@pytest.mark.parametrize(
    "p1, p2, factor, pmin, pmax, edges",
    [
        [-5, 5, 2, -10, 10, 20],
        [0, 10, 2, -5, 15, 20],
        [0, 10, (2,), -5, 15, 20],
        [
            (0, 10e-9),
            (20e-9, 50e-9),
            0.5,
            (5e-9, 20e-9),
            (15e-9, 40e-9),
            (10e-9, 20e-9),
        ],
        [
            (-50e-9, -50e-9, 0),
            (50e-9, 50e-9, 20e-9),
            2,
            (-100e-9, -100e-9, -10e-9),
            (100e-9, 100e-9, 30e-9),
            (200e-9, 200e-9, 40e-9),
        ],
        [
            (-50e-9, -50e-9, 0),
            (50e-9, 50e-9, 20e-9),
            0.5,
            (-25e-9, -25e-9, 5e-9),
            (25e-9, 25e-9, 15e-9),
            (50e-9, 50e-9, 10e-9),
        ],
        [
            (-50e-9, -50e-9, 0),
            (50e-9, 50e-9, 20e-9),
            (1, 0.1, 4),
            (-50e-9, -5e-9, -30e-9),
            (50e-9, 5e-9, 50e-9),
            (100e-9, 10e-9, 80e-9),
        ],
        [
            (0, 10, -20, -30),
            (20, -10, 0, -20),
            2.5,
            (-15, -25, -35, -37.5),
            (35, 25, 15, -12.5),
            (50, 50, 50, 25),
        ],
    ],
)
def test_scale(p1, p2, factor, pmin, pmax, edges):
    region = df.Region(p1=p1, p2=p2)
    res = region.scale(factor)
    assert isinstance(res, df.Region)
    assert np.allclose(res.pmin, pmin, atol=0)
    assert np.allclose(res.pmax, pmax, atol=0)
    assert np.allclose(res.edges, edges, atol=0)

    region.scale(factor, inplace=True)
    assert np.allclose(region.pmin, pmin, atol=0)
    assert np.allclose(region.pmax, pmax, atol=0)
    assert np.allclose(region.edges, edges, atol=0)


@pytest.mark.parametrize(
    "p1, p2, factor, reference_point, pmin, pmax",
    [
        [0, 10, 2, None, -5, 15],
        [0, 10, 2, 0, 0, 20],
        [0, 10, 2, [10], -10, 10],
        [
            (-10e-9, -5e-9),
            (20e-9, 25e-9),
            (0.5, 2.5),
            (0, 5e-9),
            (-5e-9, -20e-9),
            (10e-9, 55e-9),
        ],
    ],
)
def test_scale_reference(p1, p2, factor, reference_point, pmin, pmax):
    region = df.Region(p1=p1, p2=p2)
    res = region.scale(factor, reference_point=reference_point)
    assert isinstance(res, df.Region)
    assert np.allclose(res.pmin, pmin, atol=0)
    assert np.allclose(res.pmax, pmax, atol=0)

    region.scale(factor, reference_point=reference_point, inplace=True)
    assert np.allclose(region.pmin, pmin, atol=0)
    assert np.allclose(region.pmax, pmax, atol=0)


@pytest.mark.parametrize(
    "p1, p2, factor, reference_point, error",
    [
        [1, 2, "two", None, TypeError],
        [1, 2, 1, (1, 1), ValueError],
        [1, 2, (2, 2), None, ValueError],
        [(0, 0, 0), (10, 10, 10), (1, 2), None, ValueError],
        [(0, 0, 0), (10, 10, 10), (1, "two", 3), None, TypeError],
        [(0, 0, 0), (10, 10, 10), (1, 2, 3), 1, ValueError],
    ],
)
def test_invalid_scale(p1, p2, factor, reference_point, error):
    region = df.Region(p1=p1, p2=p2)
    with pytest.raises(error):
        region.scale(factor, reference_point=reference_point)


@pytest.mark.parametrize(
    "p1, p2, vector, pmin, pmax, center, edges",
    [
        [10, 0, -10, -10, 0, -5, 10],
        [0, 10, [1.5], 1.5, 11.5, 6.5, 10],
        [(-3, -3.5), (3, 3.5), (1.5, 2), (-1.5, -1.5), (4.5, 5.5), (1.5, 2), (6, 7)],
        [
            (-50e-9, -50e-9, 0),
            (50e-9, 50e-9, 20e-9),
            (50e-9, 0, -10e-9),
            (0, -50e-9, -10e-9),
            (100e-9, 50e-9, 10e-9),
            (50e-9, 0, 0),
            (100e-9, 100e-9, 20e-9),
        ],
        [
            (-10e-9, 0, 10e-9, 20e-9),
            (0, 1, 20e-9, 50e-9),
            (0, 50e-9, 50e-9, 80e-9),
            (-10e-9, 50e-9, 60e-9, 100e-9),
            (0, 1 + 50e-9, 70e-9, 130e-9),
            (-5e-9, 0.5 + 25e-9, 65e-9, 115e-9),
            (10e-9, 1, 10e-9, 30e-9),
        ],
    ],
)
def test_translate(p1, p2, vector, pmin, pmax, center, edges):
    region = df.Region(p1=p1, p2=p2)
    res = region.translate(vector)
    assert isinstance(res, df.Region)
    assert np.allclose(res.pmin, pmin, atol=0)
    assert np.allclose(res.pmax, pmax, atol=0)
    assert np.allclose(res.center, center, atol=0)
    assert np.allclose(res.edges, edges, atol=0)

    region.translate(vector, inplace=True)
    assert np.allclose(region.pmin, pmin, atol=0)
    assert np.allclose(region.pmax, pmax, atol=0)
    assert np.allclose(region.center, center, atol=0)
    assert np.allclose(region.edges, edges, atol=0)


@pytest.mark.parametrize(
    "p1, p2, vector, error",
    [
        [0, 10, (1, 2), ValueError],
        [(0, 0), (10, 10), (1, 2, 3), ValueError],
        [(0, 0, 0), (10, 10, 10), (3, 3), ValueError],
        [(0, 0, 0), (10, 10, 10), 3, ValueError],
    ],
)
def test_invalid_translate(p1, p2, vector, error):
    region = df.Region(p1=p1, p2=p2)
    with pytest.raises(error):
        region.translate(vector)


@pytest.mark.parametrize(
    "p1, p2, custom_units, default_units",
    [
        [0, 1, "a", "m"],
        [0, 1, ["a"], ["m"]],
        [(0, 0), (1, 1), list("ab"), list("mm")],
        [(0, 0, 0), (1, 1, 1), list("abc"), list("mmm")],
    ],
)
def test_units(p1, p2, custom_units, default_units):
    region = df.Region(p1=p1, p2=p2, units=custom_units)
    assert isinstance(region, df.Region)
    assert region.units == tuple(custom_units)

    region = df.Region(p1=p1, p2=p2)
    assert isinstance(region, df.Region)
    assert region.units == tuple(default_units)

    region.units = custom_units
    assert region.units == tuple(custom_units)

    region.units = None
    assert region.units == tuple(default_units)


@pytest.mark.parametrize(
    "p1, p2, units, error",
    [
        ([0], [1], ["m", "m", "m", "m"], ValueError),
        ([0], [1], [1], TypeError),
        ([0], [1], 5, TypeError),
        ([0, 0], [1, 2], ["m"], ValueError),
        ([0, 0], [1, 2], ["m", "m", "m", "m"], ValueError),
        ([0, 0], [1, 2], ["m", 1], TypeError),
        ([0, 0], [1, 2], "m", ValueError),
        ([0, 0], [1, 2], 5, TypeError),
        ([0, 0, 0], [1, 2, 3], ["m"], ValueError),
        ([0, 0, 0], [1, 2, 3], ["m", "m", "m", "m"], ValueError),
        ([0, 0, 0], [1, 2, 3], ["m", 1, "m"], TypeError),
        ([0, 0, 0], [1, 2, 3], "m", ValueError),
        ([0, 0, 0], [1, 2, 3], 5, TypeError),
    ],
)
def test_units_errors(p1, p2, units, error):
    with pytest.raises(error):
        region = df.Region(p1=p1, p2=p2)
        region.units = units

    with pytest.raises(error):
        df.Region(p1=p1, p2=p2, units=units)


@pytest.mark.parametrize(
    "p1, p2, custom_dims, default_dims",
    [
        [0, 1, "a", "x"],
        [0, 1, ["a"], ["x"]],
        [(0, 0), (1, 1), list("ab"), list("xy")],
        [(0, 0, 0), (1, 1, 1), ["ab", "cd", "e"], list("xyz")],
        [(0, 0, 0, 0), (1, 1, 1, 1), list("abcd"), ["x0", "x1", "x2", "x3"]],
    ],
)
def test_dims(p1, p2, custom_dims, default_dims):
    region = df.Region(p1=p1, p2=p2, dims=custom_dims)
    assert isinstance(region, df.Region)
    assert region.dims == tuple(custom_dims)

    region = df.Region(p1=p1, p2=p2)
    assert region.dims == tuple(default_dims)

    region.dims = custom_dims
    assert region.dims == tuple(custom_dims)

    region.dims = None
    assert region.dims == tuple(default_dims)


@pytest.mark.parametrize(
    "p1, p2, dims, check_dims",
    [
        [0, 10, None, ("x",)],
        [0, 10, "a", ("a",)],
        [(0, 0), (1, 1), ("something", "else"), ("something", "else")],
        [(0, 0, 0), (1, 1, 1), list("abc"), tuple("abc")],
        [(0, 0, 0, 0), (1, 1, 1, 1), list("abcd"), tuple("abcd")],
    ],
)
def test_dim2index(p1, p2, dims, check_dims):
    region = df.Region(p1=p1, p2=p2, dims=dims)
    assert region.dims == check_dims
    for dim in check_dims:
        assert region._dim2index(dim) == check_dims.index(dim)

    with pytest.raises(ValueError):
        region._dim2index("wrong_dim_name")


@pytest.mark.parametrize(
    "p1, p2, dims, error",
    [
        ([0], [1], ["a", "b", "c", "m"], ValueError),
        ([0], [1], [1], TypeError),
        ([0], [1], 5, TypeError),
        ([0, 0], [1, 2], ["m"], ValueError),
        ([0, 0], [1, 2], ["m", "x", "y"], ValueError),
        ([0, 0], [1, 2], ["m", 1], TypeError),
        ([0, 0], [1, 2], ["m", "m"], ValueError),
        ([0, 0], [1, 2], "m", ValueError),
        ([0, 0], [1, 2], 5, TypeError),
        ([0, 0, 0], [1, 2, 3], ["m"], ValueError),
        ([0, 0, 0], [1, 2, 3], ["m", "x", "y", "z"], ValueError),
        ([0, 0, 0], [1, 2, 3], ["m", 1, "y"], TypeError),
        ([0, 0, 0], [1, 2, 3], ["m", "x", "x"], ValueError),
        ([0, 0, 0], [1, 2, 3], ["x", "x", "y", "z"], ValueError),
        ([0, 0, 0], [1, 2, 3], "m", ValueError),
        ([0, 0, 0], [1, 2, 3], 5, TypeError),
    ],
)
def test_dims_errors(p1, p2, dims, error):
    with pytest.raises(error):
        region = df.Region(p1=p1, p2=p2)
        region.dims = dims

    with pytest.raises(error):
        df.Region(p1=p1, p2=p2, dims=dims)


def test_pmin_pmax(region_3d):
    with pytest.raises(AttributeError):
        region_3d.pmin = (-100e-9, -100e-9, 0)

    with pytest.raises(AttributeError):
        region_3d.pmax = (100e-9, 100e-9, 40e-9)


@pytest.mark.parametrize("p1, p2", [[0, 1], [(0, 0), (1, 1)], [(0, 0, 0), (1, 1, 1)]])
def test_mpl(p1, p2, tmp_path):
    region = df.Region(p1=p1, p2=p2)

    if region.ndim != 3:
        pytest.xfail(reason="plotting only supports 3d")

    # Check if it runs.
    region.mpl()
    region.mpl(
        figsize=(10, 10),
        multiplier=1e-9,
        color=plot_util.cp_hex[1],
        linewidth=3,
        box_aspect=(1, 1.5, 2),
        linestyle="dashed",
    )

    region.mpl(filename=tmp_path / "figure.pdf")

    plt.close("all")


@pytest.mark.parametrize("p1, p2", [[0, 1], [(0, 0), (1, 1)], [(0, 0, 0), (1, 1, 1)]])
def test_k3d(p1, p2):
    region = df.Region(p1=p1, p2=p2)

    if region.ndim != 3:
        pytest.xfail(reason="plotting only supports 3d")

    # Check if runs.
    region.k3d()
    region.k3d(multiplier=1e9, color=plot_util.cp_int[3], wireframe=True)
