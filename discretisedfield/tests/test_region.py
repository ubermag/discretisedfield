import numbers
import os
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

import discretisedfield as df
import discretisedfield.util as dfu

html_re = (
    r"<strong>Region</strong>( <i>\w+</i>)?\s*"
    r"<ul>\s*"
    r"<li>p1 = .*</li>\s*"
    r"<li>p2 = .*</li>\s*"
    r"</ul>"
)


def check_region(region):
    assert isinstance(region.p1, tuple)
    assert len(region.p1) == 3
    assert all(isinstance(i, numbers.Real) for i in region.p1)
    assert region.p1 in region

    assert isinstance(region.p2, tuple)
    assert len(region.p2) == 3
    assert all(isinstance(i, numbers.Real) for i in region.p2)
    assert region.p2 in region

    assert isinstance(region.pmin, tuple)
    assert len(region.pmin) == 3
    assert all(isinstance(i, numbers.Real) for i in region.pmin)
    assert region.pmin in region

    assert isinstance(region.pmax, tuple)
    assert len(region.pmax) == 3
    assert all(isinstance(i, numbers.Real) for i in region.pmax)
    assert region.pmax in region

    assert isinstance(region.edges, tuple)
    assert len(region.edges) == 3
    assert all(isinstance(i, numbers.Real) for i in region.edges)

    assert isinstance(region.centre, tuple)
    assert len(region.centre) == 3
    assert all(isinstance(i, numbers.Real) for i in region.centre)
    assert region.centre in region

    assert isinstance(region.random_point(), tuple)
    assert len(region.random_point()) == 3
    assert all(isinstance(i, numbers.Real) for i in region.random_point())
    assert region.random_point() in region

    assert isinstance(region.volume, numbers.Real)

    assert isinstance(repr(region), str)
    pattern = r"^Region\(p1=\([\d\se.,-]+\), p2=\([\d\se.,-]+\)\)$"
    assert re.match(pattern, str(region))

    assert isinstance(region._repr_html_(), str)
    assert re.match(html_re, region._repr_html_())

    assert region == region
    assert not region != region
    assert region != 2
    assert region in region

    # There are no facing surfaces in region | region
    with pytest.raises(ValueError):
        region | region


class TestRegion:
    def setup(self):
        self.valid_args = [
            [(0, 0, 0), (5, 5, 5)],
            [(-1, 0, -3), (5, 7, 5)],
            [(0, 0, 0), (5e-9, 5e-9, 5e-9)],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, -15e-9, -10e-9)],
            [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0))],
            [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6)],
            [(0, 125e-9, 0), (500e-9, 0, -3e-9)],
        ]

        self.invalid_args = [
            [("1", 0, 0), (1, 1, 1)],
            [(-1.5e-9, -5e-9, "a"), (1.5e-9, 15e-9, 16e-9)],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, 16e-9)],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 1 + 2j)],
            ["string", (5, 1, 1e-9)],
        ]

    def test_init_valid_args(self):
        for p1, p2 in self.valid_args:
            region = df.Region(p1=p1, p2=p2)
            check_region(region)

    def test_init_invalid_args(self):
        for p1, p2 in self.invalid_args:
            with pytest.raises((TypeError, ValueError)):
                df.Region(p1=p1, p2=p2)  # Raised by descriptors.

    def test_init_zero_edge_length(self):
        args = [
            [(0, 100e-9, 1e-9), (150e-9, 100e-9, 6e-9)],
            [(0, 101e-9, -1), (150e-9, 101e-9, 0)],
            [(10e9, 10e3, 0), (0.01e12, 11e3, 5)],
        ]

        for p1, p2 in args:
            with pytest.raises(ValueError) as excinfo:
                df.Region(p1=p1, p2=p2)
            assert "is zero" in str(excinfo.value)

    def test_pmin_pmax_edges_centre_volume(self):
        p1 = (0, -4, 16.5)
        p2 = (15, -6, 11)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        assert region.pmin == (0, -6, 11)
        assert region.pmax == (15, -4, 16.5)
        assert region.edges == (15, 2, 5.5)
        assert region.centre == (7.5, -5, 13.75)
        assert region.volume == 165

        p1 = (-10e6, 0, 0)
        p2 = (10e6, 1e6, 1e6)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        assert region.pmin == (-10e6, 0, 0)
        assert region.pmax == (10e6, 1e6, 1e6)
        assert region.edges == (20e6, 1e6, 1e6)
        assert region.centre == (0, 0.5e6, 0.5e6)
        assert abs(region.volume - 20 * (1e6) ** 3) < 1

        p1 = (-18.5e-9, 10e-9, 0)
        p2 = (10e-9, 5e-9, -10e-9)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        assert np.allclose(region.pmin, (-18.5e-9, 5e-9, -10e-9))
        assert np.allclose(region.pmax, (10e-9, 10e-9, 0))
        assert np.allclose(region.edges, (28.5e-9, 5e-9, 10e-9))
        assert np.allclose(region.centre, (-4.25e-9, 7.5e-9, -5e-9))
        assert abs(region.volume - 1425 * (1e-9**3)) < 1e-30

    def test_repr(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        rstr = "Region(p1=(-1, -4, 11), p2=(15, 10.1, 12.5))"
        assert repr(region) == rstr

    def test_eq(self):
        region1 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region2 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region3 = df.Region(p1=(3, 3, 3), p2=(10, 10, 10))

        check_region(region1)
        check_region(region2)
        check_region(region3)
        assert region1 == region2
        assert not region1 != region2
        assert region1 != region3
        assert not region1 == region3

    def test_tolerance_factor(self):
        p1 = (0, 0, 0)
        p2 = (100e-9, 100e-9, 100e-9)
        region = df.Region(p1=p1, p2=p2)
        assert np.isclose(region.tolerance_factor, 1e-12)

        region = df.Region(p1=p1, p2=p2, tolerance_factor=1e-3)
        assert np.isclose(region.tolerance_factor, 1e-3)
        region.tolerance_factor = 1e-6
        assert np.isclose(region.tolerance_factor, 1e-6)

    def test_contains(self):
        p1 = (0, 10e-9, 0)
        p2 = (10e-9, 0, 20e-9)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)
        tol = np.min(region.edges) * region.tolerance_factor
        tol_in = tol / 2
        tol_out = tol * 2
        assert (0, 0, 0) in region
        assert (-tol_in, 0, 0) in region
        assert (0, -tol_in, 0) in region
        assert (0, 0, -tol_in) in region
        assert (10e-9, 10e-9, 20e-9) in region
        assert (10e-9 + tol_in, 10e-9, 20e-9) in region
        assert (10e-9, 10e-9 + tol_in, 20e-9) in region
        assert (10e-9, 10e-9, 20e-9 + tol_in) in region

        assert (-tol_out, 0, 0) not in region
        assert (0, -tol_out, 0) not in region
        assert (0, 0, -tol_out) not in region
        assert (10e-9 + tol_out, 10e-9, 20e-9) not in region
        assert (10e-9, 10e-9 + tol_out, 20e-9) not in region
        assert (10e-9, 10e-9, 20e-9 + tol_out) not in region

        region.tolerance_factor = 1.0
        tol = np.min(region.edges) * region.tolerance_factor
        tol_in = tol / 2
        tol_out = tol * 2
        assert (0, 0, 0) in region
        assert (-tol_in, 0, 0) in region
        assert (0, -tol_in, 0) in region
        assert (0, 0, -tol_in) in region
        assert (10e-9, 10e-9, 20e-9) in region
        assert (10e-9 + tol_in, 10e-9, 20e-9) in region
        assert (10e-9, 10e-9 + tol_in, 20e-9) in region
        assert (10e-9, 10e-9, 20e-9 + tol_in) in region

        assert (-tol_out, 0, 0) not in region
        assert (0, -tol_out, 0) not in region
        assert (0, 0, -tol_out) not in region
        assert (10e-9 + tol_out, 10e-9, 20e-9) not in region
        assert (10e-9, 10e-9 + tol_out, 20e-9) not in region
        assert (10e-9, 10e-9, 20e-9 + tol_out) not in region

    def test_or(self):
        # x-direction
        p11 = (0, 0, 0)
        p12 = (10e-9, 50e-9, 20e-9)
        region1 = df.Region(p1=p11, p2=p12)

        p21 = (20e-9, 0, 0)
        p22 = (30e-9, 50e-9, 20e-9)
        region2 = df.Region(p1=p21, p2=p22)

        res = region1 | region2

        assert res[0] == "x"
        assert res[1] == region1
        assert res[2] == region2
        assert region1 | region2 == region2 | region1

        # y-direction
        p11 = (0, 0, 0)
        p12 = (10e-9, 50e-9, 20e-9)
        region1 = df.Region(p1=p11, p2=p12)

        p21 = (0, -50e-9, 0)
        p22 = (10e-9, -10e-9, 20e-9)
        region2 = df.Region(p1=p21, p2=p22)

        res = region1 | region2

        assert res[0] == "y"
        assert res[1] == region2
        assert res[2] == region1
        assert region1 | region2 == region2 | region1

        # z-direction
        p11 = (0, 0, 0)
        p12 = (100e-9, 50e-9, 20e-9)
        region1 = df.Region(p1=p11, p2=p12)

        p21 = (0, 0, 20e-9)
        p22 = (100e-9, 50e-9, 30e-9)
        region2 = df.Region(p1=p21, p2=p22)

        res = region1 | region2

        assert res[0] == "z"
        assert res[1] == region1
        assert res[2] == region2
        assert region1 | region2 == region2 | region1

        # Exceptions
        p11 = (0, 0, 0)
        p12 = (100e-9, 50e-9, 20e-9)
        region1 = df.Region(p1=p11, p2=p12)

        p21 = (0, 0, 10e-9)
        p22 = (100e-9, 50e-9, 30e-9)
        region2 = df.Region(p1=p21, p2=p22)

        with pytest.raises(ValueError):
            res = region1 | region2

        with pytest.raises(TypeError):
            res = region1 | 5

    def test_multiplier(self):
        p1 = (-50e-9, -50e-9, 0)
        p2 = (50e-9, 50e-9, 20e-9)
        region = df.Region(p1=p1, p2=p2)

        assert region.multiplier == 1e-9

        p1 = (0, 0, 0)
        p2 = (1e-5, 1e-4, 1e-5)
        region = df.Region(p1=p1, p2=p2)

        assert region.multiplier == 1e-6

    def test_mul_truediv(self):
        p1 = (-50e-9, -50e-9, 0)
        p2 = (50e-9, 50e-9, 20e-9)
        region = df.Region(p1=p1, p2=p2)

        res = region * 2
        check_region(res)
        assert np.allclose(res.pmin, (-100e-9, -100e-9, 0))
        assert np.allclose(res.pmax, (100e-9, 100e-9, 40e-9))
        assert np.allclose(res.edges, (200e-9, 200e-9, 40e-9))

        res = region / 2
        check_region(res)
        assert np.allclose(res.pmin, (-25e-9, -25e-9, 0))
        assert np.allclose(res.pmax, (25e-9, 25e-9, 10e-9))
        assert np.allclose(res.edges, (50e-9, 50e-9, 10e-9))

        assert region * 2 == 2 * region == region / 0.5

        with pytest.raises(TypeError):
            res = region * region

        with pytest.raises(TypeError):
            res = 5 / region

    def test_mpl(self):
        p1 = (-50e-9, -50e-9, 0)
        p2 = (50e-9, 50e-9, 20e-9)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)

        # Check if it runs.
        region.mpl()
        region.mpl(
            figsize=(10, 10),
            multiplier=1e-9,
            color=dfu.cp_hex[1],
            linewidth=3,
            box_aspect=(1, 1.5, 2),
            linestyle="dashed",
        )

        filename = "figure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            region.mpl(filename=tmpfilename)

        plt.close("all")

    def test_k3d(self):
        p1 = (-50e9, -50e9, 0)
        p2 = (50e9, 50e9, 20e9)
        region = df.Region(p1=p1, p2=p2)

        check_region(region)

        # Check if runs.
        region.k3d()
        region.k3d(multiplier=1e9, color=dfu.cp_int[3], wireframe=True)
