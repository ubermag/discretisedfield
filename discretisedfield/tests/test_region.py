import os
import re
import tempfile

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
def test_region():
    p1 = (-50e-9, -50e-9, 0)
    p2 = (50e-9, 50e-9, 20e-9)
    return df.Region(p1=p1, p2=p2)


if True:  # temporary "fix" to keep the diff minimal; remove in the end

    @pytest.mark.parametrize(
        "p1, p2",
        [
            [(0, 0, 0), (5, 5, 5)],
            [(-1, 0, -3), (5, 7, 5)],
            [(0, 0, 0), (5e-9, 5e-9, 5e-9)],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, -15e-9, -10e-9)],
            [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0))],
            [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6)],
            [(0, 125e-9, 0), (500e-9, 0, -3e-9)],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 1 + 2j)],
        ],
    )
    def test_init_valid_args(p1, p2):
        region = df.Region(p1=p1, p2=p2)
        assert isinstance(region, df.Region)
        pattern = r"^Region\(pmin=\[.+\], pmax=\[.+\], dims=\[.+\], units=\[.+\]\)$"
        assert re.match(pattern, str(region))

    @pytest.mark.parametrize(
        "p1,p2,error",
        [
            [("1", 0, 0), (1, 1, 1), TypeError],
            [(-1.5e-9, -5e-9, "a"), (1.5e-9, 15e-9, 16e-9), TypeError],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, 16e-9), ValueError],
            ["string", (5, 1, 1e-9), TypeError],
        ],
    )
    def test_init_invalid_args(p1, p2, error):
        with pytest.raises(error):
            df.Region(p1=p1, p2=p2)

    @pytest.mark.parametrize(
        "p1,p2",
        [
            [(0, 100e-9, 1e-9), (150e-9, 100e-9, 6e-9)],
            [(0, 101e-9, -1), (150e-9, 101e-9, 0)],
            [(10e9, 10e3, 0), (0.01e12, 11e3, 5)],
        ],
    )
    def test_init_zero_edge_length(p1, p2):
        with pytest.raises(ValueError) as excinfo:
            df.Region(p1=p1, p2=p2)
            assert "is zero" in str(excinfo.value)

    def test_pmin_pmax_edges_center_volume():
        p1 = (0, -4, 16.5)
        p2 = (15, -6, 11)
        region = df.Region(p1=p1, p2=p2)

        assert isinstance(region, df.Region)
        assert np.allclose(region.pmin, (0, -6, 11))
        assert np.allclose(region.pmax, (15, -4, 16.5))
        assert np.allclose(region.edges, (15, 2, 5.5))
        assert np.allclose(region.center, (7.5, -5, 13.75))
        assert region.volume == 165

        p1 = (-10e6, 0, 0)
        p2 = (10e6, 1e6, 1e6)
        region = df.Region(p1=p1, p2=p2)

        assert isinstance(region, df.Region)
        assert np.allclose(region.pmin, (-10e6, 0, 0))
        assert np.allclose(region.pmax, (10e6, 1e6, 1e6))
        assert np.allclose(region.edges, (20e6, 1e6, 1e6))
        assert np.allclose(region.center, (0, 0.5e6, 0.5e6))
        assert abs(region.volume - 20 * (1e6) ** 3) < 1

        p1 = (-18.5e-9, 10e-9, 0)
        p2 = (10e-9, 5e-9, -10e-9)
        region = df.Region(p1=p1, p2=p2)

        assert isinstance(region, df.Region)
        assert np.allclose(region.pmin, (-18.5e-9, 5e-9, -10e-9))
        assert np.allclose(region.pmax, (10e-9, 10e-9, 0))
        assert np.allclose(region.edges, (28.5e-9, 5e-9, 10e-9))
        assert np.allclose(region.center, (-4.25e-9, 7.5e-9, -5e-9))
        assert abs(region.volume - 1425 * (1e-9**3)) < 1e-30

    def test_repr():
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        region = df.Region(p1=p1, p2=p2)

        assert isinstance(region, df.Region)
        rstr = (
            "Region(pmin=[-1.0, -4.0, 11.0], pmax=[15.0, 10.1, 12.5],"
            " dims=['x', 'y', 'z'], units=['m', 'm', 'm'])"
        )
        assert repr(region) == rstr
        assert re.match(html_re, region._repr_html_())

        region.units = ["nm", "nm", "s"]
        rstr = (
            "Region(pmin=[-1.0, -4.0, 11.0], pmax=[15.0, 10.1, 12.5],"
            " dims=['x', 'y', 'z'], units=['nm', 'nm', 's'])"
        )
        assert repr(region) == rstr
        assert re.match(html_re, region._repr_html_())

        region.dims = ["time", "space", "c"]
        rstr = (
            "Region(pmin=[-1.0, -4.0, 11.0], pmax=[15.0, 10.1, 12.5],"
            " dims=['time', 'space', 'c'], units=['nm', 'nm', 's'])"
        )
        assert repr(region) == rstr
        assert re.match(html_re, region._repr_html_())

    def test_eq():
        region1 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region2 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region3 = df.Region(p1=(3, 3, 3), p2=(10, 10, 10))

        assert isinstance(region1, df.Region)
        assert isinstance(region2, df.Region)
        assert isinstance(region3, df.Region)
        assert region1 == region2
        assert not region1 != region2
        assert region1 != region3
        assert not region1 == region3

    def test_tolerance_factor():
        p1 = (0, 0, 0)
        p2 = (100e-9, 100e-9, 100e-9)
        region = df.Region(p1=p1, p2=p2)
        assert np.isclose(region.tolerance_factor, 1e-12)

        region = df.Region(p1=p1, p2=p2, tolerance_factor=1e-3)
        assert np.isclose(region.tolerance_factor, 1e-3)
        region.tolerance_factor = 1e-6
        assert np.isclose(region.tolerance_factor, 1e-6)

    @pytest.mark.parametrize("factor", [None, 1.0])
    def test_contains(factor):
        p1 = (0, 10e-9, 0)
        p2 = (10e-9, 0, 20e-9)
        region = df.Region(p1=p1, p2=p2)

        assert isinstance(region, df.Region)
        if factor is not None:
            region.tolerance_factor = factor
        tol = np.min(region.edges) * region.tolerance_factor
        tol_in = tol / 2
        tol_out = tol * 2
        assert (0, 0, 0) in region
        assert (-tol_in, 0, 0) in region
        assert (0, -tol_in, 0) in region
        assert (0, 0, -tol_in) in region
        assert (10e-9, 10e-9, 20e-9) in region
        assert (10e-9 + tol_in, 10e-9, 10e-9) in region
        assert (10e-9, 10e-9 + tol_in, 10e-9) in region
        assert (1e-9, 3e-9, 20e-9 + tol_in) in region

        assert (-tol_out, 0, 0) not in region
        assert (0, -tol_out, 0) not in region
        assert (0, 0, -tol_out) not in region
        assert (10e-9 + tol_out, 10e-9, 20e-9) not in region
        assert (10e-9, 10e-9 + tol_out, 20e-9) not in region
        assert (10e-9, 10e-9, 20e-9 + tol_out) not in region

    def test_or():
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

    @pytest.mark.parametrize(
        "region,multiplier",
        [
            [df.Region(p1=(-50e-9, -50e-9, 0), p2=(50e-9, 50e-9, 20e-9)), 1e-9],
            [df.Region(p1=(0, 0, 0), p2=(1e-5, 1e-4, 1e-5)), 1e-6],
        ],
    )
    def test_multiplier(test_region, multiplier):
        assert test_region.multiplier == multiplier

    def test_scale(test_region):
        res = test_region.scale(2)
        assert isinstance(res, df.Region)
        assert np.allclose(res.pmin, (-100e-9, -100e-9, 0))
        assert np.allclose(res.pmax, (100e-9, 100e-9, 40e-9))
        assert np.allclose(res.edges, (200e-9, 200e-9, 40e-9))

        res = test_region.scale(0.5)
        assert isinstance(res, df.Region)
        assert np.allclose(res.pmin, (-25e-9, -25e-9, 0))
        assert np.allclose(res.pmax, (25e-9, 25e-9, 10e-9))
        assert np.allclose(res.edges, (50e-9, 50e-9, 10e-9))

        res = test_region.scale((1, 0.1, 4))
        assert isinstance(res, df.Region)
        assert np.allclose(res.pmin, (-50e-9, -5e-9, 0))
        assert np.allclose(res.pmax, (50e-9, 5e-9, 80e-9))
        assert np.allclose(res.edges, (100e-9, 10e-9, 80e-9))

        test_region.scale(2, inplace=True)
        assert np.allclose(test_region.pmin, (-100e-9, -100e-9, 0))
        assert np.allclose(test_region.pmax, (100e-9, 100e-9, 40e-9))
        assert np.allclose(test_region.edges, (200e-9, 200e-9, 40e-9))

        with pytest.raises(ValueError):
            test_region.scale((1, 2))

        with pytest.raises(TypeError):
            test_region.scale((1, "two", 3))

        with pytest.raises(TypeError):
            res = test_region.scale("two")

    def test_translate(test_region):
        res = test_region.translate((50e-9, 0, -10e-9))
        assert isinstance(res, df.Region)
        assert np.allclose(res.pmin, (0, -50e-9, -10e-9))
        assert np.allclose(res.pmax, (100e-9, 50e-9, 10e-9))
        assert np.allclose(res.edges, (100e-9, 100e-9, 20e-9))

        test_region.translate((50e-9, 0, -10e-9), inplace=True)
        assert np.allclose(test_region.pmin, (0, -50e-9, -10e-9))
        assert np.allclose(test_region.pmax, (100e-9, 50e-9, 10e-9))
        assert np.allclose(test_region.edges, (100e-9, 100e-9, 20e-9))

        with pytest.raises(ValueError):
            test_region.translate((3, 3))

        with pytest.raises(TypeError):
            test_region.translate(3)

    def test_units(test_region):
        p1 = test_region.pmin
        p2 = test_region.pmax
        units = ["a", "b", "c"]
        region = df.Region(p1=p1, p2=p2, units=units)
        assert isinstance(region, df.Region)
        assert region.units == tuple(units)

        region = df.Region(p1=p1, p2=p2)
        assert isinstance(region, df.Region)
        assert region.units == ("m", "m", "m")

        region.units = units
        assert region.units == tuple(units)

        region.units = None
        assert region.units == ("m", "m", "m")

    @pytest.mark.parametrize(
        "units, error",
        [
            (["m"], ValueError),
            (["m", "m", "m", "m"], ValueError),
            (["m", 1, "m"], TypeError),
            ("m", TypeError),
            (5, TypeError),
        ],
    )
    def test_units_errors(test_region, units, error):
        with pytest.raises(error):
            test_region.units = units

        with pytest.raises(error):
            df.Region(p1=test_region.p1, p2=test_region.p2, units=units)

    def test_ndim(test_region):
        ndim = 3
        assert isinstance(test_region, df.Region)
        assert test_region.ndim == ndim

    def test_dims(test_region):
        p1 = test_region.pmin
        p2 = test_region.pax
        dims = ["a", "b", "c"]
        region = df.Region(p1=p1, p2=p2, dims=dims)
        assert isinstance(region, df.Region)
        assert region.dims == tuple(dims)

        region = df.Region(p1=p1, p2=p2)
        assert region.dims == ("x", "y", "z")

        region.dims = dims
        assert region.dims == tuple(dims)

        region.dims = None
        assert region.dims == ("x", "y", "z")

    @pytest.mark.parametrize(
        "dims, error",
        [
            (["x", "y", "z", "t"], ValueError),
            (["x", 1, "z"], TypeError),
            ("x", TypeError),
            (5, TypeError),
        ],
    )
    def test_dims_errors(test_region, dims, error):
        with pytest.raises(error):
            test_region.dims = dims

        with pytest.raises(error):
            df.Region(p1=test_region.p1, p2=test_region.p2, dims=dims)

    def test_allclose():
        region1 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region2 = df.Region(p1=(0, 0, 0), p2=(10, 10, 10))
        region3 = df.Region(p1=(3, 3, 3), p2=(10, 10, 10))

        assert isinstance(region1, df.Region)
        assert isinstance(region2, df.Region)
        assert isinstance(region3, df.Region)
        assert region1.allclose(region2)
        assert not region1.allclose(region3)
        assert not region2.allclose(region3)

    # unit test for setting pmin and pmax
    def test_pmin_pmax(test_region):
        with pytest.raises(AttributeError):
            test_region.pmin = (-100e-9, -100e-9, 0)

        with pytest.raises(AttributeError):
            test_region.pmax = (100e-9, 100e-9, 40e-9)

    def test_mpl(test_region):
        # Check if it runs.
        test_region.mpl()
        test_region.mpl(
            figsize=(10, 10),
            multiplier=1e-9,
            color=plot_util.cp_hex[1],
            linewidth=3,
            box_aspect=(1, 1.5, 2),
            linestyle="dashed",
        )

        filename = "figure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            test_region.mpl(filename=tmpfilename)

        plt.close("all")

    def test_k3d(test_region):
        # Check if runs.
        test_region.k3d()
        test_region.k3d(multiplier=1e9, color=plot_util.cp_int[3], wireframe=True)
