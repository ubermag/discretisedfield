import itertools
import os
import random
import re
import tempfile
import types

import holoviews as hv
import k3d
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import discretisedfield as df

from .test_mesh import TestMesh
from .test_mesh import html_re as mesh_html_re

html_re = (
    r"<strong>Field</strong>\s*<ul>\s*"
    rf"<li>{mesh_html_re}</li>\s*"
    r"<li>nvdim = \d+</li>\s*"
    r"(<li>vdims:\s*<ul>(<li>.*</li>\s*)+</ul>\s*</li>)?\s*"
    r"(<li>unit = .+</li>)?\s*"
    r"</ul>"
)


def check_field(field):
    # TODO add explicit tests for the remaining checks in here and remove
    # this function
    assert isinstance(field.nvdim, int)

    assert field.array.shape == (*field.mesh.n, field.nvdim)

    rstr = repr(field)
    assert isinstance(rstr, str)
    pattern = (
        r"^Field\(Mesh\(Region\(pmin=\[.+\], pmax=\[.+\], .+\), .+\)," r" nvdim=\d+\)$"
    )
    if field.vdims:
        pattern = pattern[:-3] + r", vdims: \(.+\)\)$"
    if field.unit is not None:
        pattern = pattern[:-3] + r", unit=.+\)$"
    assert re.search(pattern, rstr)

    assert isinstance(field._repr_html_(), str)
    assert re.search(html_re, field._repr_html_(), re.DOTALL)

    assert isinstance(field.__iter__(), types.GeneratorType)
    assert len(list(field)) == len(field.mesh)


def check_hv(plot, types):
    # generate the first plot output to have enough data in plot.info
    hv.renderer("bokeh").get_plot(plot)
    # find strings like "    :DynamicMap [comp,z]" or "    :Image    [x,y]"
    # the number of spaces can vary
    assert sorted(
        re.findall(r"(?<=:)\w+ \[[^]]+\]", re.sub(r"\s+", " ", str(plot)))
    ) == sorted(types)


class TestField:
    def setup(self):
        # Get meshes using valid arguments from TestMesh.
        tm = TestMesh()
        tm.setup()
        self.meshes = []
        for p1, p2, n, cell in tm.valid_args:
            region = df.Region(p1=p1, p2=p2)
            mesh = df.Mesh(region=region, n=n, cell=cell)
            self.meshes.append(mesh)

        # Create lists of field values.
        # dtype is computed automatically for array_like
        self.consts = [
            [0, None],
            [-5.0, None],
            [np.pi, None],
            [1e-15, None],
            [1.2e12, None],
            [random.random(), None],
            [1 + 1j, None],
        ]
        self.iters = [
            [(0, 0, 1), None],
            [(0, -5.1, np.pi), None],
            [[70, 1e15, 2 * np.pi], None],
            [[5, random.random(), np.pi], None],
            [np.array([4, -1, 3.7]), None],
            [np.array([2.1, 0.0, -5 * random.random()]), None],
            [(1 + 1j, 1 + 1j, 1 + 1j), None],
            [(0, 0, 1j), None],
            [np.random.random(3) + np.random.random(3) * 1j, None],
        ]
        # dtype has to be specified for callable
        self.sfuncs = [
            [lambda c: 1, np.float64],
            [lambda c: -2.4, np.float64],
            [lambda c: -6.4e-15, np.float64],
            [lambda c: 1 + 2j, np.complex128],
            [lambda c: c[0] + c[1] + c[2] + 1, np.float64],
            [lambda c: (c[0] - 1) ** 2 - c[1] + 7 + c[2] * 0.1, np.float64],
            [lambda c: np.sin(c[0]) + np.cos(c[1]) - np.sin(2 * c[2]), np.float64],
        ]
        self.vfuncs = [
            [lambda c: (1, 2, 0), np.float64],
            [lambda c: (-2.4, 1e-3, 9), np.float64],
            [lambda c: (1 + 1j, 2 + 2j, 3 + 3j), np.complex128],
            [lambda c: (0, 1j, 1), np.complex128],
            [lambda c: (c[0], c[1], c[2] + 100), np.float64],
            [lambda c: (c[0] + c[2] + 10, c[1], c[2] + 1), np.float64],
            [lambda c: (c[0] - 1, c[1] + 70, c[2] * 0.1), np.float64],
            [lambda c: (np.sin(c[0]), np.cos(c[1]), -np.sin(2 * c[2])), np.float64],
        ]

        # Create a field for plotting tests
        mesh = df.Mesh(p1=(-5e-9, -5e-9, -5e-9), p2=(5e-9, 5e-9, 5e-9), n=(5, 5, 5))

        def norm_fun(point):
            x, y, z = point
            if x**2 + y**2 <= (5e-9) ** 2:
                return 1e5
            else:
                return 0

        def value_fun(point):
            x, y, z = point
            if x <= 0:
                return (0, 0, 1)
            else:
                return (0, 0, -1)

        self.pf = df.Field(
            mesh, nvdim=3, value=value_fun, norm=norm_fun, vdims=["a", "b", "c"]
        )

    def test_init_valid_args(self):
        for mesh in self.meshes:
            for value, dtype in self.consts + self.sfuncs:
                f = df.Field(mesh, nvdim=1, value=value, dtype=dtype)
                check_field(f)

                assert isinstance(f.mesh, df.Mesh)
                assert f.nvdim == 1
                assert isinstance(f.array, np.ndarray)

            for value, dtype in self.iters + self.vfuncs:
                f = df.Field(mesh, nvdim=3, value=value, dtype=dtype)
                check_field(f)

                assert isinstance(f.mesh, df.Mesh)
                assert f.nvdim == 3
                assert isinstance(f.array, np.ndarray)

    def test_init_invalid_args(self):
        with pytest.raises(TypeError):
            mesh = "meaningless_mesh_string"
            df.Field(mesh, nvdim=1)

        for mesh in self.meshes:
            for nvdim in [0, -1, "dim", (2, 3)]:
                with pytest.raises((ValueError, TypeError)):
                    df.Field(mesh, nvdim=nvdim)

        # wrong abc.Iterable
        with pytest.raises(TypeError):
            df.Field(self.meshes[0], nvdim=1, value="string")

        # all builtin types are numeric types or Iterable
        class WrongType:
            pass

        with pytest.raises(TypeError):
            df.Field(self.meshes[0], nvdim=1, value=WrongType())

    def test_set_with_ndarray(self):
        for mesh in self.meshes:
            f = df.Field(mesh, nvdim=3)
            f.update_field_values(np.ones((*f.mesh.n, f.nvdim)))

            check_field(f)
            assert np.allclose(f.mean(), (1, 1, 1))

            with pytest.raises(ValueError):
                f.update_field_values(np.ones((2, 2)))

    def test_set_with_callable(self):
        for mesh in self.meshes:
            for func, dtype in self.sfuncs:
                f = df.Field(mesh, nvdim=1, value=func, dtype=dtype)
                check_field(f)

                def random_point(f):
                    return (
                        np.random.random(3) * f.mesh.region.edges + f.mesh.region.pmin
                    )

                rp = random_point(f)
                # Make sure to be at the centre of the cell
                rp = f.mesh.index2point(f.mesh.point2index(rp))
                assert f(rp) == func(rp)

        for mesh in self.meshes:
            for func, dtype in self.vfuncs:
                f = df.Field(mesh, nvdim=3, value=func, dtype=dtype)
                check_field(f)

                rp = random_point(f)
                rp = f.mesh.index2point(f.mesh.point2index(rp))
                assert np.all(f(rp) == func(rp))

    def test_set_with_dict(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        subregions = {
            "r1": df.Region(p1=(0, 0, 0), p2=(4e-9, 10e-9, 10e-9)),
            "r2": df.Region(p1=(4e-9, 0, 0), p2=(10e-9, 10e-9, 10e-9)),
        }
        mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)

        field = df.Field(
            mesh, nvdim=3, value={"r1": (0, 0, 1), "r2": (0, 0, 2), "r1:r2": (0, 0, 5)}
        )
        assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1))
        assert np.all(field((8e-9, 2e-9, 9e-9)) == (0, 0, 2))

        subregions = {"r1": df.Region(p1=(0, 0, 0), p2=(4e-9, 10e-9, 10e-9))}
        mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)

        with pytest.raises(KeyError):
            field = df.Field(mesh, nvdim=3, value={"r1": (0, 0, 1)})

        field = df.Field(mesh, nvdim=3, value={"r1": (0, 0, 1), "default": (1, 1, 1)})
        assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1))
        assert np.all(field((8e-9, 2e-9, 9e-9)) == (1, 1, 1))

        field = df.Field(mesh, nvdim=3, value={"default": (1, 1, 1)})
        assert np.all(field.array == (1, 1, 1))

        # dtype has to be specified for isinstance(value, dict)
        field = df.Field(
            mesh,
            nvdim=3,
            value={"r1": (0, 0, 1 + 2j), "default": (1, 1, 1)},
            dtype=np.complex128,
        )
        assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1 + 2j))
        assert np.all(field((8e-9, 2e-9, 9e-9)) == (1, 1, 1))

    def test_set_exception(self):
        for mesh in self.meshes:
            with pytest.raises(TypeError):
                df.Field(mesh, nvdim=3, value="meaningless_string")

            with pytest.raises(ValueError):
                df.Field(mesh, nvdim=3, value=5 + 5j)

    def test_components(self):
        for mesh in self.meshes:
            valid_components = ["a", "b", "c", "d", "e", "f"]
            invalid_components = ["a", "grad", "b", "div", "array", "c"]
            for nvdim in range(2, 7):
                f = df.Field(
                    mesh,
                    nvdim=nvdim,
                    value=list(range(nvdim)),
                    vdims=valid_components[:nvdim],
                )
                assert f.vdims == valid_components[:nvdim]
                check_field(f)

                with pytest.raises(ValueError):
                    df.Field(
                        mesh,
                        nvdim=nvdim,
                        value=list(range(nvdim)),
                        vdims=invalid_components[:nvdim],
                    )

            # wrong number of components
            with pytest.raises(ValueError):
                df.Field(mesh, nvdim=3, value=(1, 1, 1), vdims=valid_components)
            with pytest.raises(ValueError):
                df.Field(mesh, nvdim=3, value=(1, 1, 1), vdims=["x", "y"])

            # components not unique
            with pytest.raises(ValueError):
                df.Field(mesh, nvdim=3, value=(1, 1, 1), vdims=["x", "y", "x"])

            # test lshift
            f1 = df.Field(mesh, nvdim=1, value=1)
            f2 = df.Field(mesh, nvdim=1, value=2)
            f3 = df.Field(mesh, nvdim=1, value=3)

            f12 = f1 << f2
            check_field(f12)
            assert np.allclose(f12.array[0, 0, 0, :], [1, 2])
            assert f12.x == f1
            assert f12.y == f2

            f123 = f1 << f2 << f3
            assert np.allclose(f123.array[0, 0, 0, :], [1, 2, 3])
            assert f123.x == f1
            assert f123.y == f2
            assert f123.z == f3

            fa = df.Field(mesh, nvdim=1, value=10, vdims=["a"])
            fb = df.Field(mesh, nvdim=1, value=20, vdims=["b"])

            # default components if not all fields have component labels
            f1a = f1 << fa
            check_field(f1a)
            assert f1a.vdims == ["x", "y"]

            # custom components if all fields have custom components
            fab = fa << fb
            check_field(fab)
            assert fab.vdims == ["a", "b"]

    def test_unit(self):
        assert self.pf.unit is None
        mesh = self.pf.mesh
        field = df.Field(mesh, nvdim=3, value=(1, 2, 3), unit="A/m")
        check_field(field)
        assert field.unit == "A/m"
        field.unit = "mT"
        assert field.unit == "mT"
        with pytest.raises(TypeError):
            field.unit = 3
        assert field.unit == "mT"
        field.unit = None
        assert field.unit is None

        with pytest.raises(TypeError):
            df.Field(mesh, nvdim=1, unit=1)

    def test_value(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, nvdim=3)
        f.update_field_values((1, 1, 1))

        assert np.allclose(f.mean(), (1, 1, 1))

    def test_average(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
        f = df.Field(mesh, nvdim=3, value=(2, 2, 2))
        with pytest.warns(DeprecationWarning):
            f.average

    def test_norm(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
        f = df.Field(mesh, nvdim=3, value=(2, 2, 2))

        assert np.allclose(f.norm.array, 2 * np.sqrt(3))
        assert np.allclose(f.array, 2)

        f.norm = 1
        assert np.allclose(f.norm.array, 1)
        assert np.allclose(f.array, 1 / np.sqrt(3))

        for mesh in self.meshes:
            for value, dtype in self.iters + self.vfuncs:
                for norm_value in [1, 2.1, 50, 1e-3, np.pi]:
                    f = df.Field(
                        mesh, nvdim=3, value=value, norm=norm_value, dtype=dtype
                    )

                    # TODO: Why is this included?
                    # Compute norm.
                    norm = f.array[..., 0] ** 2
                    norm += f.array[..., 1] ** 2
                    norm += f.array[..., 2] ** 2
                    norm = np.sqrt(norm)

                    assert np.all(norm.shape == f.mesh.n)
                    assert f.norm.array.shape == (*tuple(f.mesh.n), 1)
                    assert np.all(abs(f.norm.array - norm_value) < 1e-12)

        # Exception
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(1, 1, 1))
        f = df.Field(mesh, nvdim=1, value=-5)
        f.norm = 5

    def test_norm_is_not_preserved(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, nvdim=3)

        f.update_field_values((0, 3, 0))
        f.norm = 1
        assert np.all(f.norm.array == 1)

        f.update_field_values((0, 2, 0))
        assert np.all(f.norm.array == 2)

    def test_norm_zero_field(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, nvdim=3, value=(0, 0, 0))
        f.norm = 1  # Does not change the norm of zero field
        assert np.all(f.norm.array == 0)

    def test_zero(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, nvdim=1, value=1e-6)
        zf = f.zero

        assert f.mesh == zf.mesh
        assert f.nvdim == zf.nvdim
        assert not np.any(zf.array)

        f = df.Field(mesh, nvdim=3, value=(5, -7, 1e3))
        zf = f.zero

        assert f.mesh == zf.mesh
        assert f.nvdim == zf.nvdim
        assert not np.any(zf.array)

    def test_orientation(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (5e-9, 5e-9, 5e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # No zero-norm cells
        f = df.Field(mesh, nvdim=3, value=(2, 0, 0))
        assert isinstance(f.orientation, df.Field)
        assert np.allclose(f.orientation.mean(), (1, 0, 0))

        # With zero-norm cells
        def value_fun(point):
            x, y, z = point
            if x <= 0:
                return (0, 0, 0)
            else:
                return (3, 0, 4)

        f = df.Field(mesh, nvdim=3, value=value_fun)
        assert np.allclose(f.orientation((-1.5e-9, 3e-9, 0)), (0, 0, 0))
        assert np.allclose(f.orientation((1.5e-9, 3e-9, 0)), (0.6, 0, 0.8))

        f = df.Field(mesh, nvdim=1, value=0)
        with pytest.raises(ValueError):
            f.orientation

    def test_call(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (5e-9, 5e-9, 5e-9)
        n = (5, 5, 10)  # cell points in x, y at: [-4, -2, 0, 2, 4] * 1e-9
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, nvdim=3, value=lambda p: (p[0], p[1], 1))
        assert np.allclose(f((0, 0, 0)), (0, 0, 1))
        assert np.allclose(f((0.5e-9, 0.5e-9, 0.5e-9)), (0, 0, 1))
        assert np.allclose(f((2e-9, 2e-9, 2e-9)), (2e-9, 2e-9, 1))
        assert np.allclose(f((1.5e-9, 2.5e-9, 1.5e-9)), (2e-9, 2e-9, 1))
        assert np.allclose(f((1.5e-9, 3.5e-9, 1.5e-9)), (2e-9, 4e-9, 1))
        assert np.allclose(f((-5e-9, 5e-9, -5e-9)), (-4e-9, 4e-9, 1))

        with pytest.raises(ValueError):
            f((0, 1, 0))

        with pytest.raises(ValueError):
            f((0, 0))

    def test_mean(self):
        tol = 1e-12

        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (5e-9, 5e-9, 5e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, nvdim=1, value=2)
        assert abs(f.mean() - 2) < tol

        f = df.Field(mesh, nvdim=3, value=(0, 1, 2))
        assert np.allclose(f.mean(), (0, 1, 2))

    def test_field_component(self):
        for mesh in self.meshes:
            f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
            assert all(isinstance(getattr(f, i), df.Field) for i in "xyz")
            assert all(getattr(f, i).nvdim == 1 for i in "xyz")

            f = df.Field(mesh, nvdim=2, value=(1, 2))
            assert all(isinstance(getattr(f, i), df.Field) for i in "xy")
            assert all(getattr(f, i).nvdim == 1 for i in "xy")

            # Exception.
            f = df.Field(mesh, nvdim=1, value=1)
            with pytest.raises(AttributeError):
                f.x.nvdim

    def test_get_attribute_exception(self):
        for mesh in self.meshes:
            f = df.Field(mesh, nvdim=3)
            with pytest.raises(AttributeError) as excinfo:
                f.__getattr__("nonexisting_attribute")
            assert "has no attribute" in str(excinfo.value)

    def test_dir(self):
        for mesh in self.meshes:
            f = df.Field(mesh, nvdim=3, value=(5, 6, -9))
            assert all(attr in dir(f) for attr in ["x", "y", "z", "div"])
            assert "grad" not in dir(f)

            f = df.Field(mesh, nvdim=1, value=1)
            assert all(attr not in dir(f) for attr in ["x", "y", "z", "div"])
            assert "grad" in dir(f)

    def test_eq(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (15e-9, 5e-9, 5e-9)
        cell = (5e-9, 1e-9, 2.5e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f1 = df.Field(mesh, nvdim=1, value=0.2)
        f2 = df.Field(mesh, nvdim=1, value=0.2)
        f3 = df.Field(mesh, nvdim=1, value=3.1)
        f4 = df.Field(mesh, nvdim=3, value=(1, -6, 0))
        f5 = df.Field(mesh, nvdim=3, value=(1, -6, 0))

        assert f1 == f2
        assert not f1 != f2
        assert not f1 == f3
        assert f1 != f3
        assert not f2 == f4
        assert f2 != f4
        assert f4 == f5
        assert not f4 != f5
        assert not f1 == 0.2
        assert f1 != 0.2

    def test_allclose(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (15e-9, 5e-9, 5e-9)
        cell = (5e-9, 1e-9, 2.5e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f1 = df.Field(mesh, nvdim=1, value=0.2)
        f2 = df.Field(mesh, nvdim=1, value=0.2 + 1e-9)
        f3 = df.Field(mesh, nvdim=1, value=0.21)
        f4 = df.Field(mesh, nvdim=3, value=(1, -6, 0))
        f5 = df.Field(mesh, nvdim=3, value=(1, -6 + 1e-8, 0))
        f6 = df.Field(mesh, nvdim=3, value=(1, -6.01, 0))

        assert f1.allclose(f2)
        assert not f1.allclose(f3)
        assert not f1.allclose(f5)
        assert f4.allclose(f5)
        assert not f4.allclose(f6)

        with pytest.raises(TypeError):
            f1.allclose(2)

    def test_point_neg(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (5e-9, 5e-9, 5e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Scalar field
        f = df.Field(mesh, nvdim=1, value=3)
        res = -f
        check_field(res)
        assert res.mean() == -3
        assert f == +f
        assert f == -(-f)
        assert f == +(-(-f))

        # Vector field
        f = df.Field(mesh, nvdim=3, value=(1, 2, -3))
        res = -f
        check_field(res)
        assert np.allclose(res.mean(), (-1, -2, 3))
        assert f == +f
        assert f == -(-f)
        assert f == +(-(-f))

    def test_pow(self):
        p1 = (0, 0, 0)
        p2 = (15e-9, 6e-9, 6e-9)
        cell = (3e-9, 3e-9, 3e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Scalar field
        f = df.Field(mesh, nvdim=1, value=2)
        res = f**2
        assert res.mean() == 4
        res = f ** (-1)
        assert res.mean() == 0.5

        # Attempt vector field
        f = df.Field(mesh, nvdim=3, value=(1, 2, -2))
        res = f**2
        assert np.allclose(res.mean(), (1, 4, 4))

        # Attempt to raise to non numbers.Real
        f = df.Field(mesh, nvdim=1, value=2)
        with pytest.raises(TypeError):
            res = f ** "a"
        res = f**f
        assert res.mean() == 4

    def test_add_subtract(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 10e-9, -5e-9)
        n = (2, 2, 1)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        # Scalar fields
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        f2 = df.Field(mesh, nvdim=1, value=-0.2)
        res = f1 + f2
        assert res.mean() == 1
        res = f1 - f2
        assert res.mean() == 1.4
        f1 += f2
        assert f1.mean() == 1
        f1 -= f2
        assert f1.mean() == 1.2

        # Vector fields
        f1 = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        f2 = df.Field(mesh, nvdim=3, value=(-1, -3, -5))
        res = f1 + f2
        assert np.allclose(res.mean(), (0, -1, -2))
        res = f1 - f2
        assert np.allclose(res.mean(), (2, 5, 8))
        f1 += f2
        assert np.allclose(f1.mean(), (0, -1, -2))
        f1 -= f2
        assert np.allclose(f1.mean(), (1, 2, 3))

        # Artithmetic checks
        assert f1 + f2 + (1, 1, 1) == (1, 1, 1) + f2 + f1
        assert f1 - f2 - (0, 0, 0) == (0, 0, 0) - (f2 - f1)
        assert f1 + (f1 + f2) == (f1 + f1) + f2
        assert f1 - (f1 + f2) == f1 - f1 - f2
        assert f1 + f2 - f1 == f2 + (0, 0, 0)

        # Constants
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        f2 = df.Field(mesh, nvdim=3, value=(-1, -3, -5))
        res = f1 + 2
        assert res.mean() == 3.2
        res = f1 - 1.2
        assert res.mean() == 0
        f1 += 2.5
        assert f1.mean() == 3.7
        f1 -= 3.7
        assert f1.mean() == 0
        res = f2 + (1, 3, 5)
        assert np.allclose(res.mean(), (0, 0, 0))
        res = f2 - (1, 2, 3)
        assert np.allclose(res.mean(), (-2, -5, -8))
        f2 += (1, 1, 1)
        assert np.allclose(f2.mean(), (0, -2, -4))
        f2 -= (-1, -2, 3)
        assert np.allclose(f2.mean(), (1, 0, -7))

        # Exceptions
        with pytest.raises(TypeError):
            res = f1 + "2"

        # Fields with different dimensions
        res = f1 + f2
        assert np.allclose(res.mean(), (1, 0, -7))

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, nvdim=1, value=1.2)
        f2 = df.Field(mesh2, nvdim=1, value=1)
        with pytest.raises(ValueError):
            res = f1 + f2
        with pytest.raises(ValueError):
            f1 += f2
        with pytest.raises(ValueError):
            f1 -= f2

    def test_mul_truediv(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 5e-9, 5e-9)
        cell = (1e-9, 5e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Scalar fields
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        f2 = df.Field(mesh, nvdim=1, value=-2)
        res = f1 * f2
        assert res.mean() == -2.4
        res = f1 / f2
        assert res.mean() == -0.6
        f1 *= f2
        assert f1.mean() == -2.4
        f1 /= f2
        assert f1.mean() == 1.2

        # Scalar field with a constant
        f = df.Field(mesh, nvdim=1, value=5)
        res = f * 2
        assert res.mean() == 10
        res = 3 * f
        assert res.mean() == 15
        res = f * (1, 2, 3)
        assert np.allclose(res.mean(), (5, 10, 15))
        res = (1, 2, 3) * f
        assert np.allclose(res.mean(), (5, 10, 15))
        res = f / 2
        assert res.mean() == 2.5
        res = 10 / f
        assert res.mean() == 2
        res = (5, 10, 15) / f
        assert np.allclose(res.mean(), (1, 2, 3))
        f *= 10
        assert f.mean() == 50
        f /= 10
        assert f.mean() == 5

        # Scalar field with a vector field
        f1 = df.Field(mesh, nvdim=1, value=2)
        f2 = df.Field(mesh, nvdim=3, value=(-1, -3, 5))
        res = f1 * f2  # __mul__
        assert np.allclose(res.mean(), (-2, -6, 10))
        res = f2 * f1  # __rmul__
        assert np.allclose(res.mean(), (-2, -6, 10))
        res = f2 / f1  # __truediv__
        assert np.allclose(res.mean(), (-0.5, -1.5, 2.5))
        f2 *= f1  # __imul__
        assert np.allclose(f2.mean(), (-2, -6, 10))
        f2 /= f1  # __truediv__
        assert np.allclose(f2.mean(), (-1, -3, 5))
        res = f1 / f2  # __rtruediv__
        assert np.allclose(res.mean(), (-2, -2 / 3, 2 / 5))

        # Vector field with a scalar
        f = df.Field(mesh, nvdim=3, value=(1, 2, 0))
        res = f * 2
        assert np.allclose(res.mean(), (2, 4, 0))
        res = 5 * f
        assert np.allclose(res.mean(), (5, 10, 0))
        res = f / 2
        assert np.allclose(res.mean(), (0.5, 1, 0))
        f *= 2
        assert np.allclose(f.mean(), (2, 4, 0))
        f /= 2
        assert np.allclose(f.mean(), (1, 2, 0))
        res = 10 / f
        assert np.allclose(res.mean(), (10, 5, np.inf))

        # Further checks
        f1 = df.Field(mesh, nvdim=1, value=2)
        f2 = df.Field(mesh, nvdim=3, value=(-1, -3, -5))
        assert f1 * f2 == f2 * f1
        assert 1.3 * f2 == f2 * 1.3
        assert -5 * f2 == f2 * (-5)
        assert (1, 2.2, -1) * f1 == f1 * (1, 2.2, -1)
        assert f1 * (f1 * f2) == (f1 * f1) * f2
        assert f1 * f2 / f1 == f2
        assert np.allclose((f2 * f2).mean(), (1, 9, 25))
        assert np.allclose((f2 / f2).mean(), (1, 1, 1))

        # Exceptions
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        f2 = df.Field(mesh, nvdim=3, value=(-1, -3, -5))
        with pytest.raises(TypeError):
            res = f2 * "a"
        with pytest.raises(TypeError):
            res = "a" / f1
        with pytest.raises(TypeError):
            f2 *= "a"
        with pytest.raises(TypeError):
            f2 /= "a"

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, nvdim=1, value=1.2)
        f2 = df.Field(mesh2, nvdim=1, value=1)
        with pytest.raises(ValueError):
            res = f1 * f2
        with pytest.raises(ValueError):
            res = f1 / f2
        with pytest.raises(ValueError):
            f1 *= f2
        with pytest.raises(ValueError):
            f1 /= f2

    def test_dot(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Zero vectors
        f1 = df.Field(mesh, nvdim=3, value=(0, 0, 0))
        res = f1.dot(f1)
        assert res.nvdim == 1
        assert res.mean() == 0

        # Orthogonal vectors
        f1 = df.Field(mesh, nvdim=3, value=(1, 0, 0))
        f2 = df.Field(mesh, nvdim=3, value=(0, 1, 0))
        f3 = df.Field(mesh, nvdim=3, value=(0, 0, 1))
        assert (f1.dot(f3)).mean() == 0
        assert (f1.dot(f2)).mean() == 0
        assert (f2.dot(f3)).mean() == 0
        assert (f1.dot(f1)).mean() == 1
        assert (f2.dot(f2)).mean() == 1
        assert (f3.dot(f3)).mean() == 1

        # Check if commutative
        assert f1.dot(f2) == f2.dot(f1)

        # Vector field with a constant
        f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        res = f.dot([1, 1, 1])
        assert res.mean() == 6

        # Spatially varying vectors
        def value_fun1(point):
            x, y, z = point
            return (x, y, z)

        def value_fun2(point):
            x, y, z = point
            return (z, x, y)

        f1 = df.Field(mesh, nvdim=3, value=value_fun1)
        f2 = df.Field(mesh, nvdim=3, value=value_fun2)

        # Check if commutative
        assert f1.dot(f2) == f2.dot(f1)

        # The dot product should be x*z + y*x + z*y
        assert (f1.dot(f2))((1, 1, 1)) == 3
        assert (f1.dot(f2))((3, 1, 1)) == 7
        assert (f1.dot(f2))((5, 7, 1)) == 47

        # Check norm computed using dot product
        assert f1.norm == (f1.dot(f1)) ** (0.5)

        # Exceptions
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        f2 = df.Field(mesh, nvdim=3, value=(-1, -3, -5))
        with pytest.raises(ValueError):
            res = f1.dot(f2)
        with pytest.raises(ValueError):
            res = f1.dot(f2)
        with pytest.raises(TypeError):
            res = f1.dot(3)

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, nvdim=3, value=(1, 2, 3))
        f2 = df.Field(mesh2, nvdim=3, value=(3, 2, 1))
        with pytest.raises(ValueError):
            res = f1.dot(f2)

    def test_cross(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Zero vectors
        f1 = df.Field(mesh, nvdim=3, value=(0, 0, 0))
        res = f1.cross(f1)
        assert res.nvdim == 3
        assert np.allclose(res.mean(), (0, 0, 0))

        # Orthogonal vectors
        f1 = df.Field(mesh, nvdim=3, value=(1, 0, 0))
        f2 = df.Field(mesh, nvdim=3, value=(0, 1, 0))
        f3 = df.Field(mesh, nvdim=3, value=(0, 0, 1))
        assert np.allclose((f1.cross(f2)).mean(), (0, 0, 1))
        assert np.allclose((f1.cross(f3)).mean(), (0, -1, 0))
        assert np.allclose((f2.cross(f3)).mean(), (1, 0, 0))
        assert np.allclose((f1.cross(f1)).mean(), (0, 0, 0))
        assert np.allclose((f2.cross(f2)).mean(), (0, 0, 0))
        assert np.allclose((f3.cross(f3)).mean(), (0, 0, 0))

        # Constants
        assert np.allclose((f1.cross((0, 1, 0))).mean(), (0, 0, 1))

        # Check if not comutative
        assert f1.cross(f2) == -(f2.cross(f1))
        assert f1.cross(f3) == -(f3.cross(f1))
        assert f2.cross(f3) == -(f3.cross(f2))

        f1 = df.Field(mesh, nvdim=3, value=lambda point: (point[0], point[1], point[2]))
        f2 = df.Field(mesh, nvdim=3, value=lambda point: (point[2], point[0], point[1]))

        # The cross product should be
        # (y**2-x*z, z**2-x*y, x**2-y*z)
        assert np.allclose((f1.cross(f2))((1, 1, 1)), (0, 0, 0))
        assert np.allclose((f1.cross(f2))((3, 1, 1)), (-2, -2, 8))
        assert np.allclose((f2.cross(f1))((3, 1, 1)), (2, 2, -8))
        assert np.allclose((f1.cross(f2))((5, 7, 1)), (44, -34, 18))

        # Exceptions
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        f2 = df.Field(mesh, nvdim=3, value=(-1, -3, -5))
        with pytest.raises(TypeError):
            res = f1.cross(2)
        with pytest.raises(ValueError):
            res = f1.cross(f2)

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, nvdim=3, value=(1, 2, 3))
        f2 = df.Field(mesh2, nvdim=3, value=(3, 2, 1))
        with pytest.raises(ValueError):
            res = f1.cross(f2)

    def test_lshift(self):
        p1 = (0, 0, 0)
        p2 = (10e6, 10e6, 10e6)
        cell = (5e6, 5e6, 5e6)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f1 = df.Field(mesh, nvdim=1, value=1)
        f2 = df.Field(mesh, nvdim=1, value=-3)
        f3 = df.Field(mesh, nvdim=1, value=5)

        res = f1 << f2 << f3
        assert res.nvdim == 3
        assert np.allclose(res.mean(), (1, -3, 5))

        # Different dimensions
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        f2 = df.Field(mesh, nvdim=2, value=(-1, -3))
        res = f1 << f2
        assert np.allclose(res.mean(), (1.2, -1, -3))
        res = f2 << f1
        assert np.allclose(res.mean(), (-1, -3, 1.2))

        # Constants
        f1 = df.Field(mesh, nvdim=1, value=1.2)
        res = f1 << 2
        assert np.allclose(res.mean(), (1.2, 2))
        res = f1 << (1, -1)
        assert np.allclose(res.mean(), (1.2, 1, -1))
        res = 3 << f1
        assert np.allclose(res.mean(), (3, 1.2))
        res = (1.2, 3) << f1 << 3
        assert np.allclose(res.mean(), (1.2, 3, 1.2, 3))

        # Exceptions
        with pytest.raises(TypeError):
            res = "a" << f1
        with pytest.raises(TypeError):
            res = f1 << "a"

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, nvdim=1, value=1.2)
        f2 = df.Field(mesh2, nvdim=1, value=1)
        with pytest.raises(ValueError):
            res = f1 << f2

    def test_all_operators(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 5e-9, 10e-9)
        n = (2, 2, 1)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f1 = df.Field(mesh, nvdim=1, value=2)
        f2 = df.Field(mesh, nvdim=3, value=(-4, 0, 1))
        res = (
            ((+f1 / 2 + f2.x) ** 2 - 2 * f1 * 3) / (-f2.z)
            - 2 * f2.y
            + 1 / f2.z**2
            + f2.dot(f2)
        )
        assert np.all(res.array == 21)

        res = 1 + f1 + 0 * f2.x - 3 * f2.y / 3
        assert res.mean() == 3

    def test_pad(self):
        p1 = (0, 0, 0)
        p2 = (10, 8, 2)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        field = df.Field(mesh, nvdim=1, value=1)

        pf = field.pad({"x": (1, 1)}, mode="constant")  # zeros padded
        assert pf.array.shape == (12, 8, 2, 1)

    def test_derivative(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)

        # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, nvdim=1, value=0)

        check_field(f.diff("x"))
        assert np.allclose(f.diff("x", order=1).mean(), 0)
        assert np.allclose(f.diff("y", order=1).mean(), 0)
        assert np.allclose(f.diff("z", order=1).mean(), 0)
        assert np.allclose(f.diff("x", order=2).mean(), 0)
        assert np.allclose(f.diff("y", order=2).mean(), 0)
        assert np.allclose(f.diff("z", order=2).mean(), 0)

        # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x", order=1).mean(), 1)
        assert np.allclose(f.diff("y", order=1).mean(), 1)
        assert np.allclose(f.diff("z", order=1).mean(), 1)
        assert np.allclose(f.diff("x", order=2).mean(), 0)
        assert np.allclose(f.diff("y", order=2).mean(), 0)
        assert np.allclose(f.diff("z", order=2).mean(), 0)

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        def value_fun(point):
            x, y, z = point
            return x * y + 2 * y + x * y * z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x")((7, 5, 1)), 10)
        assert np.allclose(f.diff("y")((7, 5, 1)), 16)
        assert np.allclose(f.diff("z")((7, 5, 1)), 35)
        assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 0)
        assert np.allclose(f.diff("y", order=2)((1, 1, 1)), 0)
        assert np.allclose(f.diff("z", order=2)((1, 1, 1)), 0)

        # f(x, y, z) = (0, 0, 0)
        # -> dfdx = (0, 0, 0)
        # -> dfdy = (0, 0, 0)
        # -> dfdz = (0, 0, 0)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

        check_field(f.diff("y"))
        assert np.allclose(f.diff("x").mean(), (0, 0, 0))
        assert np.allclose(f.diff("y").mean(), (0, 0, 0))
        assert np.allclose(f.diff("z").mean(), (0, 0, 0))

        # f(x, y, z) = (x,  y,  z)
        # -> dfdx = (1, 0, 0)
        # -> dfdy = (0, 1, 0)
        # -> dfdz = (0, 0, 1)
        def value_fun(point):
            x, y, z = point
            return (x, y, z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.diff("x").mean(), (1, 0, 0))
        assert np.allclose(f.diff("y").mean(), (0, 1, 0))
        assert np.allclose(f.diff("z").mean(), (0, 0, 1))

        # f(x, y, z) = (x*y, y*z, x*y*z)
        # -> dfdx = (y, 0, y*z)
        # -> dfdy = (x, z, x*z)
        # -> dfdz = (0, y, x*y)
        def value_fun(point):
            x, y, z = point
            return (x * y, y * z, x * y * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.diff("x")((3, 1, 3)), (1, 0, 3))
        assert np.allclose(f.diff("y")((3, 1, 3)), (3, 3, 9))
        assert np.allclose(f.diff("z")((3, 1, 3)), (0, 1, 3))
        assert np.allclose(f.diff("x")((5, 3, 5)), (3, 0, 15))
        assert np.allclose(f.diff("y")((5, 3, 5)), (5, 5, 25))
        assert np.allclose(f.diff("z")((5, 3, 5)), (0, 3, 15))

        # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
        # -> dfdx = (y, 1, y*z)
        # -> dfdy = (x, -2, x*z)
        # -> dfdz = (0, 0, x*y)
        def value_fun(point):
            x, y, z = point
            return (3 + x * y, x - 2 * y, x * y * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.diff("x")((7, 5, 1)), (5, 1, 5))
        assert np.allclose(f.diff("y")((7, 5, 1)), (7, -2, 7))
        assert np.allclose(f.diff("z")((7, 5, 1)), (0, 0, 35))

        # f(x, y, z) = 2*x*x + 2*y*y + 3*z*z
        # -> grad(f) = (4, 4, 6)
        def value_fun(point):
            x, y, z = point
            return 2 * x * x + 2 * y * y + 3 * z * z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x", order=2).mean(), 4)
        assert np.allclose(f.diff("y", order=2).mean(), 4)
        assert np.allclose(f.diff("z", order=2).mean(), 6)

        # f(x, y, z) = (2*x*x, 2*y*y, 3*z*z)
        def value_fun(point):
            x, y, z = point
            return (2 * x * x, 2 * y * y, 3 * z * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.diff("x", order=2).mean(), (4, 0, 0))
        assert np.allclose(f.diff("y", order=2).mean(), (0, 4, 0))
        assert np.allclose(f.diff("z", order=2).mean(), (0, 0, 6))

        # Test invalid direction
        with pytest.raises(ValueError):
            f.diff("q")

    def test_derivative_small(self):
        p1 = (0, 0, 0)
        p2 = (3, 3, 3)
        n = (3, 3, 3)

        # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        f = df.Field(mesh, nvdim=1, value=0)

        check_field(f.diff("x"))
        assert np.allclose(f.diff("x", order=1).mean(), 0)
        assert np.allclose(f.diff("y", order=1).mean(), 0)
        assert np.allclose(f.diff("z", order=1).mean(), 0)

        # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x", order=1).mean(), 1)
        assert np.allclose(f.diff("y", order=1).mean(), 1)
        assert np.allclose(f.diff("z", order=1).mean(), 1)

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        def value_fun(point):
            x, y, z = point
            return x * y + 2 * y + x * y * z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x")((2.5, 1.5, 1.5)), 3.75)
        assert np.allclose(f.diff("y")((1.5, 2.5, 1.5)), 5.75)
        assert np.allclose(f.diff("z")((1.5, 1.5, 2.5)), 2.25)

        # f(x, y, z) = (0, 0, 0)
        # -> dfdx = (0, 0, 0)
        # -> dfdy = (0, 0, 0)
        # -> dfdz = (0, 0, 0)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

        assert np.allclose(f.diff("x").mean(), (0, 0, 0))
        assert np.allclose(f.diff("y").mean(), (0, 0, 0))
        assert np.allclose(f.diff("z").mean(), (0, 0, 0))

        # f(x, y, z) = (x,  y,  z)
        # -> dfdx = (1, 0, 0)
        # -> dfdy = (0, 1, 0)
        # -> dfdz = (0, 0, 1)
        def value_fun(point):
            x, y, z = point
            return (x, y, z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.diff("x").mean(), (1, 0, 0))
        assert np.allclose(f.diff("y").mean(), (0, 1, 0))
        assert np.allclose(f.diff("z").mean(), (0, 0, 1))

        # f(x, y, z) = (x*y, y*z, x*y*z)
        # -> dfdx = (y, 0, y*z)
        # -> dfdy = (x, z, x*z)
        # -> dfdz = (0, y, x*y)
        def value_fun(point):
            x, y, z = point
            return (x * y, y * z, x * y * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.diff("x")((1.5, 1.5, 1.5)), (1.5, 0, 2.25))
        assert np.allclose(f.diff("x")((2.5, 1.5, 1.5)), (1.5, 0, 2.25))
        assert np.allclose(f.diff("y")((1.5, 1.5, 1.5)), (1.5, 1.5, 2.25))
        assert np.allclose(f.diff("y")((1.5, 2.5, 1.5)), (1.5, 1.5, 2.25))
        assert np.allclose(f.diff("z")((1.5, 1.5, 1.5)), (0, 1.5, 2.25))
        assert np.allclose(f.diff("z")((1.5, 1.5, 2.5)), (0, 1.5, 2.25))

        p1 = (0, 0, 0)
        p2 = (4, 4, 4)
        n = (4, 4, 4)
        # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        f = df.Field(mesh, nvdim=1, value=0)

        assert np.allclose(f.diff("x", order=2).mean(), 0)
        assert np.allclose(f.diff("y", order=2).mean(), 0)
        assert np.allclose(f.diff("z", order=2).mean(), 0)

        # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x", order=2).mean(), 0)
        assert np.allclose(f.diff("y", order=2).mean(), 0)
        assert np.allclose(f.diff("z", order=2).mean(), 0)

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        def value_fun(point):
            x, y, z = point
            return x * y + 2 * y + x * y * z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x", order=2)((1.5, 1.5, 1.5)), 0)
        assert np.allclose(f.diff("y", order=2)((1.5, 1.5, 1.5)), 0)
        assert np.allclose(f.diff("z", order=2)((1.5, 1.5, 1.5)), 0)

        # f(x, y, z) = 2*x*x + 2*y*y + 3*z*z
        # -> grad(f) = (4, 4, 6)
        def value_fun(point):
            x, y, z = point
            return 2 * x * x + 2 * y * y + 3 * z * z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.diff("x", order=2).mean(), 4)
        assert np.allclose(f.diff("y", order=2).mean(), 4)
        assert np.allclose(f.diff("z", order=2).mean(), 6)

        # f(x, y, z) = (2*x*x, 2*y*y, 3*z*z)
        def value_fun(point):
            x, y, z = point
            return (2 * x * x, 2 * y * y, 3 * z * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.diff("x", order=2)((1.5, 1.5, 1.5)), (4, 0, 0))
        assert np.allclose(f.diff("x", order=2)((3.5, 1.5, 1.5)), (4, 0, 0))
        assert np.allclose(f.diff("y", order=2)((1.5, 1.5, 1.5)), (0, 4, 0))
        assert np.allclose(f.diff("y", order=2)((1.5, 3.5, 1.5)), (0, 4, 0))
        assert np.allclose(f.diff("z", order=2)((1.5, 1.5, 1.5)), (0, 0, 6))
        assert np.allclose(f.diff("z", order=2)((1.5, 1.5, 3.5)), (0, 0, 6))

    def test_derivative_pbc(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (12.0, 8.0, 6.0)
        cell = (2, 2, 2)

        mesh_nopbc = df.Mesh(p1=p1, p2=p2, cell=cell)
        mesh_pbc = df.Mesh(p1=p1, p2=p2, cell=cell, bc="xyz")

        # Scalar field
        def value_fun(point):
            x, y, z = point
            return x * y * z

        # No PBC
        f = df.Field(mesh_nopbc, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x")((11, 1, 1)), 1)
        assert np.allclose(f.diff("y")((1, 7, 1)), 1)
        assert np.allclose(f.diff("z")((1, 1, 5)), 1)

        # PBC
        f = df.Field(mesh_pbc, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x")((11, 1, 1)), -2)
        assert np.allclose(f.diff("y")((1, 7, 1)), -1)
        assert np.allclose(f.diff("z")((1, 1, 5)), -0.5)

        # Vector field
        def value_fun(point):
            x, y, z = point
            return (x * y * z,) * 3

        # No PBC
        f = df.Field(mesh_nopbc, nvdim=3, value=value_fun)
        assert np.allclose(f.diff("x")((11, 1, 1)), (1, 1, 1))
        assert np.allclose(f.diff("y")((1, 7, 1)), (1, 1, 1))
        assert np.allclose(f.diff("z")((1, 1, 5)), (1, 1, 1))

        # PBC
        f = df.Field(mesh_pbc, nvdim=3, value=value_fun)
        assert np.allclose(f.diff("x")((11, 1, 1)), (-2, -2, -2))
        assert np.allclose(f.diff("y")((1, 7, 1)), (-1, -1, -1))
        assert np.allclose(f.diff("z")((1, 1, 5)), (-0.5, -0.5, -0.5))

        # Higher order derivatives
        def value_fun(point):
            x, y, z = point
            return x**2

        f = df.Field(mesh_nopbc, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 2)

        f = df.Field(mesh_pbc, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 32)

    def test_derivative_neumann(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (12.0, 8.0, 6.0)
        cell = (2, 2, 2)

        mesh_noneumann = df.Mesh(p1=p1, p2=p2, cell=cell)
        mesh_neumann = df.Mesh(p1=p1, p2=p2, cell=cell, bc="neumann")

        # Scalar field
        def value_fun(point):
            return point[0] * point[1] * point[2]

        # No Neumann
        f1 = df.Field(mesh_noneumann, nvdim=1, value=value_fun)
        assert np.allclose(f1.diff("x")((11, 1, 1)), 1)
        assert np.allclose(f1.diff("y")((1, 7, 1)), 1)
        assert np.allclose(f1.diff("z")((1, 1, 5)), 1)

        # Neumann
        f2 = df.Field(mesh_neumann, nvdim=1, value=value_fun)
        assert np.allclose(
            f1.diff("x")(f1.mesh.region.center), f2.diff("x")(f2.mesh.region.center)
        )
        assert f1.diff("x")((1, 7, 1)) != f2.diff("x")((1, 7, 1))
        assert np.allclose(f2.diff("x")((11, 1, 1)), 0.5)
        assert np.allclose(f2.diff("x")((1, 1, 1)), 0.5)

        # Higher order derivatives
        def value_fun(point):
            x, y, z = point
            return x**2

        f = df.Field(mesh_noneumann, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 2)

        f = df.Field(mesh_neumann, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 2)

    def test_derivative_dirichlet(self):
        p1 = (0.0, 0.0, 0.0)
        p2 = (12.0, 8.0, 6.0)
        cell = (2, 2, 2)

        mesh_nodirichlet = df.Mesh(p1=p1, p2=p2, cell=cell)
        mesh_dirichlet = df.Mesh(p1=p1, p2=p2, cell=cell, bc="dirichlet")

        # Scalar field
        def value_fun(point):
            return point[0] * point[1] * point[2]

        # No Dirichlet
        f1 = df.Field(mesh_nodirichlet, nvdim=1, value=value_fun)
        assert np.allclose(f1.diff("x")((11, 1, 1)), 1)
        assert np.allclose(f1.diff("y")((1, 7, 1)), 1)
        assert np.allclose(f1.diff("z")((1, 1, 5)), 1)

        # Dirichlet
        f2 = df.Field(mesh_dirichlet, nvdim=1, value=value_fun)
        assert np.allclose(
            f1.diff("x")(f1.mesh.region.center), f2.diff("x")(f2.mesh.region.center)
        )
        assert f1.diff("x")((1, 7, 1)) != f2.diff("x")((1, 7, 1))
        assert np.allclose(f2.diff("x")((11, 1, 1)), -2.25)
        assert np.allclose(f2.diff("x")((1, 1, 1)), 0.75)

        # Higher order derivatives
        def value_fun(point):
            x, y, z = point
            return x**2

        f = df.Field(mesh_nodirichlet, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 2)

        f = df.Field(mesh_dirichlet, nvdim=1, value=value_fun)
        assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 1.75)

    def test_derivative_single_cell(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 2)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Scalar field: f(x, y, z) = x + y + z
        # -> grad(f) = (1, 1, 1)
        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        # only one cell in the z-direction
        assert f.plane("x").diff("x").mean() == 0
        assert f.plane("y").diff("y").mean() == 0
        assert f.diff("z").mean() == 0

        # Vector field: f(x, y, z) = (x, y, z)
        # -> grad(f) = (1, 1, 1)
        def value_fun(point):
            x, y, z = point
            return (x, y, z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        # only one cell in the z-direction
        assert np.allclose(f.plane("x").diff("x").mean(), (0, 0, 0))
        assert np.allclose(f.plane("y").diff("y").mean(), (0, 0, 0))
        assert np.allclose(f.diff("z").mean(), (0, 0, 0))

    def test_grad(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
        f = df.Field(mesh, nvdim=1, value=0)

        check_field(f.grad)
        assert np.allclose(f.grad.mean(), (0, 0, 0))

        # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.grad.mean(), (1, 1, 1))

        # f(x, y, z) = x*y + y + z -> grad(f) = (y, x+1, 1)
        def value_fun(point):
            x, y, z = point
            return x * y + y + z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.grad((3, 1, 3)), (1, 4, 1))
        assert np.allclose(f.grad((5, 3, 5)), (3, 6, 1))

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        def value_fun(point):
            x, y, z = point
            return x * y + 2 * y + x * y * z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.grad((7, 5, 1)), (10, 16, 35))
        assert f.grad.x == f.diff("x")
        assert f.grad.y == f.diff("y")
        assert f.grad.z == f.diff("z")

        # Exception
        f = df.Field(mesh, nvdim=3, value=(1, 2, 3))

        with pytest.raises(ValueError):
            f.grad

    def test_div_curl(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # f(x, y, z) = (0, 0, 0)
        # -> div(f) = 0
        # -> curl(f) = (0, 0, 0)
        f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

        check_field(f.div)
        assert f.div.nvdim == 1
        assert f.div.mean() == 0

        check_field(f.curl)
        assert f.curl.nvdim == 3
        assert np.allclose(f.curl.mean(), (0, 0, 0))

        # f(x, y, z) = (x, y, z)
        # -> div(f) = 3
        # -> curl(f) = (0, 0, 0)
        def value_fun(point):
            x, y, z = point
            return (x, y, z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert f.div.mean() == 3
        assert np.allclose(f.curl.mean(), (0, 0, 0))

        # f(x, y, z) = (x*y, y*z, x*y*z)
        # -> div(f) = y + z + x*y
        # -> curl(f) = (x*z-y, -y*z, -x)
        def value_fun(point):
            x, y, z = point
            return (x * y, y * z, x * y * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.div((3, 1, 3)), 7)
        assert np.allclose(f.div((5, 3, 5)), 23)

        assert np.allclose(f.curl((3, 1, 3)), (8, -3, -3))
        assert np.allclose(f.curl((5, 3, 5)), (22, -15, -5))

        # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
        # -> div(f) = y - 2 + x*y
        # -> curl(f) = (x*z, -y*z, 1-x)
        def value_fun(point):
            x, y, z = point
            return (3 + x * y, x - 2 * y, x * y * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.div((7, 5, 1)), 38)
        assert np.allclose(f.curl((7, 5, 1)), (7, -5, -6))

        # Exception
        f = df.Field(mesh, nvdim=1, value=3.11)

        with pytest.raises(ValueError):
            f.div
        with pytest.raises(ValueError):
            f.curl

    def test_laplace(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # f(x, y, z) = (0, 0, 0)
        # -> laplace(f) = 0
        f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

        check_field(f.laplace)
        assert f.laplace.nvdim == 3
        assert np.allclose(f.laplace.mean(), (0, 0, 0))

        # f(x, y, z) = x + y + z
        # -> laplace(f) = 0
        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, nvdim=1, value=value_fun)
        check_field(f.laplace)
        assert np.allclose(f.laplace.mean(), 0)

        # f(x, y, z) = 2*x*x + 2*y*y + 3*z*z
        # -> laplace(f) = 4 + 4 + 6 = 14
        def value_fun(point):
            x, y, z = point
            return 2 * x * x + 2 * y * y + 3 * z * z

        f = df.Field(mesh, nvdim=1, value=value_fun)

        assert np.allclose(f.laplace.mean(), 14)

        # f(x, y, z) = (2*x*x, 2*y*y, 3*z*z)
        # -> laplace(f) = (4, 4, 6)
        def value_fun(point):
            x, y, z = point
            return (2 * x * x, 2 * y * y, 3 * z * z)

        f = df.Field(mesh, nvdim=3, value=value_fun)

        assert np.allclose(f.laplace.mean(), (4, 4, 6))

    def test_integrate(self):
        # Volume integral.
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (0.5, 0.5, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, nvdim=1, value=0)
        assert f.integrate() == 0

        f = df.Field(mesh, nvdim=1, value=2)
        assert f.integrate() == 2000

        f = df.Field(mesh, nvdim=3, value=(-1, 0, 3))
        assert np.allclose(f.integrate(), (-1000, 0, 3000))

        def value_fun(point):
            x, y, z = point
            if x <= 5:
                return (-1, -2, -3)
            else:
                return (1, 2, 3)

        f = df.Field(mesh, nvdim=3, value=value_fun)
        assert np.allclose(f.integrate(), (0, 0, 0))
        assert np.allclose(f.integrate(), (0, 0, 0))

        # Surface integral.
        p1 = (0, 0, 0)
        p2 = (10, 5, 3)
        cell = (0.5, 0.5, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, nvdim=1, value=0)
        assert f.plane("x").integrate() == 0

        f = df.Field(mesh, nvdim=1, value=2)
        assert f.plane("x").integrate() == 30
        assert f.plane("y").integrate() == 60
        assert f.plane("z").integrate() == 100

        f = df.Field(mesh, nvdim=3, value=(-1, 0, 3))
        assert f.plane("x").dot([1, 0, 0]).integrate() == -15
        assert f.plane("y").dot([0, 1, 0]).integrate() == 0
        assert f.plane("z").dot([0, 0, 1]).integrate() == 150

        # TODO change when n dimensional meshes are supported
        # The next line currently fails because we cannot detect consecutive
        # .plane methods. Therefore, the calculated cell volume is wrong.
        # The test should "fail" once n dimensional meshes are implemented.
        # The value on the right-hand-site is the expected result.
        assert f.plane("z").plane("x").dot([1, 0, 0]).integrate() != -5

        # Directional integral
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (0.5, 0.5, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, nvdim=3, value=(1, 1, 1))

        res = f.integrate(direction="x")
        assert isinstance(res, df.Field)
        assert res.nvdim == 3
        assert np.array_equal(res.mesh.n, (1, 20, 20))
        assert np.allclose(res.mean(), (10, 10, 10))

        res = f.integrate(direction="x").integrate(direction="y")
        assert isinstance(res, df.Field)
        assert res.nvdim == 3
        assert np.array_equal(res.mesh.n, (1, 1, 20))
        assert np.allclose(res.mean(), (100, 100, 100))

        res = f.integrate("x").integrate("y").integrate("z")
        assert res.nvdim == 3
        assert np.array_equal(res.mesh.n, (1, 1, 1))
        assert np.allclose(res.mean(), (1000, 1000, 1000))

        assert np.allclose(
            f.integrate("x").integrate("y").integrate("z").mean(), f.integrate()
        )

        # Cumulative integral
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (0.5, 0.5, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, nvdim=3, value=(1, 1, 1))

        f_int = f.integrate(direction="x", cumulative=True)
        assert isinstance(f_int, df.Field)
        assert f_int.nvdim == 3
        assert np.array_equal(f_int.mesh.n, (20, 20, 20))
        assert np.allclose(f_int.mean(), (5, 5, 5))
        assert np.allclose(f_int((0, 0, 0)), (0.25, 0.25, 0.25))
        assert np.allclose(f_int((0.9, 0.9, 0.9)), (0.75, 0.75, 0.75))
        assert np.allclose(f_int((10, 10, 10)), (9.75, 9.75, 9.75))
        assert np.allclose(f_int.diff("x").array, f.array)

        for i, d in enumerate("xyz"):
            f = df.Field(mesh, nvdim=1, value=lambda p: p[i])
            assert np.allclose(f.integrate(d, cumulative=True).diff(d).array, f.array)
            assert np.allclose(f.diff(d).integrate(d, cumulative=True).array, f.array)

        # Exceptions
        with pytest.raises(ValueError):
            f.integrate(cumulative=True)

        with pytest.raises(ValueError):
            f.integrate(direction="a")

        with pytest.raises(TypeError):
            f.integrate(1)

    def test_abs(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, nvdim=1, value=-1)
        abs(f).mean() == 1

        f = df.Field(mesh, nvdim=3, value=(-1, -1, -1))
        np.allclose(abs(f).mean(), (1, 1, 1))

        f = df.Field(mesh, nvdim=1, value=-1j)
        abs(f).mean() == 1

    def test_line(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), n=(10, 10, 10))
        f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        check_field(f)

        line = f.line(p1=(0, 0, 0), p2=(5, 5, 5), n=20)
        assert isinstance(line, df.Line)

        assert line.n == 20
        assert line.dim == 3

    def test_plane(self):
        for mesh, direction in itertools.product(self.meshes, ["x", "y", "z"]):
            f = df.Field(mesh, nvdim=1, value=3)
            check_field(f)
            plane = f.plane(direction, n=(3, 3))
            assert isinstance(plane, df.Field)

            p = zip(*list(plane.mesh))
            assert len(p) == 9
            v = zip(*list(plane))
            assert len(v) == 9

    def test_resample(self):
        resampled = self.pf.resample(n=(10, 15, 20))
        assert np.allclose(resampled.mesh.n, (10, 15, 20))
        assert resampled.mesh.region == self.pf.mesh.region
        pmin = self.pf.mesh.region.pmin
        assert np.allclose(resampled(pmin), self.pf(pmin))

        resampled = self.pf.resample(n=(1, 1, 1))
        assert np.allclose(resampled.mesh.n, (1, 1, 1))
        assert resampled.mesh.region == self.pf.mesh.region
        pmin = self.pf.mesh.region.pmin
        assert np.allclose(resampled(pmin), self.pf((0, 0, 0)))

        with pytest.raises(TypeError):
            self.pf.resample((0, 1, 2))

    def test_getitem(self):
        p1 = (0, 0, 0)
        p2 = (90, 50, 10)
        cell = (5, 5, 5)
        subregions = {
            "r1": df.Region(p1=(0, 0, 0), p2=(30, 50, 10)),
            "r2": df.Region(p1=(30, 0, 0), p2=(90, 50, 10)),
        }
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        def value_fun(point):
            x, y, z = point
            if x <= 60:
                return (-1, -2, -3)
            else:
                return (1, 2, 3)

        f = df.Field(mesh, nvdim=3, value=value_fun)
        check_field(f)
        check_field(f["r1"])
        check_field(f["r2"])
        check_field(f[subregions["r1"]])
        check_field(f[subregions["r2"]])

        assert np.allclose(f["r1"].mean(), (-1, -2, -3))
        assert np.allclose(f["r2"].mean(), (0, 0, 0))
        assert np.allclose(f[subregions["r1"]].mean(), (-1, -2, -3))
        assert np.allclose(f[subregions["r2"]].mean(), (0, 0, 0))

        assert len(f["r1"].mesh) + len(f["r2"].mesh) == len(f.mesh)

        # Meshes are not aligned
        subregion = df.Region(p1=(1.1, 0, 0), p2=(9.9, 15, 5))

        assert f[subregion].array.shape == (2, 3, 1, 3)

    def test_project(self):
        p1 = (-5, -5, -5)
        p2 = (5, 5, 5)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Constant scalar field
        f = df.Field(mesh, nvdim=1, value=5)
        check_field(f)
        assert f.project("x").array.shape == (1, 10, 10, 1)
        assert f.project("y").array.shape == (10, 1, 10, 1)
        assert f.project("z").array.shape == (10, 10, 1, 1)

        # Constant vector field
        f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        assert f.project("x").array.shape == (1, 10, 10, 3)
        assert f.project("y").array.shape == (10, 1, 10, 3)
        assert f.project("z").array.shape == (10, 10, 1, 3)

        # Spatially varying scalar field
        def value_fun(point):
            x, y, z = point
            if z <= 0:
                return 1
            else:
                return -1

        f = df.Field(mesh, nvdim=1, value=value_fun)
        sf = f.project("z")
        assert sf.array.shape == (10, 10, 1, 1)
        assert sf.mean() == 0

        # Spatially varying vector field
        def value_fun(point):
            x, y, z = point
            if z <= 0:
                return (3, 2, 1)
            else:
                return (3, 2, -1)

        f = df.Field(mesh, nvdim=3, value=value_fun)
        sf = f.project("z")
        assert sf.array.shape == (10, 10, 1, 3)
        assert np.allclose(sf.mean(), (3, 2, 0))

    def test_angle(self):
        p1 = (0, 0, 0)
        p2 = (8e-9, 2e-9, 2e-9)
        cell = (2e-9, 2e-9, 2e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        f = df.Field(mesh, nvdim=3, value=(1.0, 0.0, 0.0))

        assert np.allclose(f.angle((1.0, 0.0, 0.0)).mean(), 0.0)
        assert np.allclose(f.angle((0.0, 1.0, 0.0)).mean(), np.pi / 2)

    def test_write_read_ovf(self, tmp_path):
        representations = ["txt", "bin4", "bin8"]
        filename = "testfile.ovf"
        p1 = (0, 0, 0)
        p2 = (8e-9, 5e-9, 3e-9)
        cell = (1e-9, 1e-9, 1e-9)
        subregions = {
            "sr1": df.Region(p1=p1, p2=(2e-9, 2e-9, 1e-9)),
            "sr2": df.Region(p1=(3e-9, 0, 0), p2=p2),
        }
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell, subregions=subregions)

        # Write/read
        for nvdim, value in [
            (1, lambda point: point[0] + point[1] + point[2]),
            (2, lambda point: (point[0], point[1] + point[2])),
            (3, lambda point: (point[0], point[1], point[2])),
        ]:
            f = df.Field(mesh, nvdim=nvdim, value=value, unit="A/m")
            for rep in representations:
                tmpfilename = tmp_path / filename
                f.to_file(tmpfilename, representation=rep)
                f_read = df.Field.from_file(tmpfilename)

                assert f.allclose(f_read)
                assert f_read.unit == "A/m"
                assert f.mesh.subregions == f_read.mesh.subregions

                tmpfilename = tmp_path / f"no_sr_{filename}"
                f.to_file(tmpfilename, representation=rep, save_subregions=False)
                f_read = df.Field.from_file(tmpfilename)

                assert f.allclose(f_read)
                assert f_read.unit == "A/m"
                assert f_read.mesh.subregions == {}

            # Directly write with wrong representation (no data is written)
            with pytest.raises(ValueError):
                df.io.field_to_ovf(f, "fname.ovf", representation="bin5")

        # multiple different units (not supported by discretisedfield)
        f = df.Field(mesh, nvdim=3, value=(1, 1, 1), unit="m s kg")
        tmpfilename = str(tmp_path / filename)
        f.to_file(tmpfilename, representation=rep)
        f_read = df.Field.from_file(tmpfilename)

        assert f.allclose(f_read)
        assert f_read.unit is None

        # Extend scalar
        for rep in representations:
            f = df.Field(
                mesh, nvdim=1, value=lambda point: point[0] + point[1] + point[2]
            )
            tmpfilename = tmp_path / filename
            f.to_file(tmpfilename, representation=rep, extend_scalar=True)
            f_read = df.Field.from_file(tmpfilename)

            assert f.allclose(f_read.x)

        # Read different OOMMF representations
        # (OVF1, OVF2) x (txt, bin4, bin8)
        filenames = [
            "oommf-ovf2-txt.omf",
            "oommf-ovf2-bin4.omf",
            "oommf-ovf2-bin8.omf",
            "oommf-ovf1-txt.omf",
            "oommf-ovf1-bin4.omf",
            "oommf-ovf1-bin8.omf",
        ]
        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        for filename in filenames:
            omffilename = os.path.join(dirname, filename)
            f_read = df.Field.from_file(omffilename)

            if "ovf2" in filename:
                # The magnetisation is in the x-direction in OVF2 files.
                assert abs(f_read.orientation.x.mean() - 1) < 1e-2
            else:
                # The norm of magnetisation is known.
                assert abs(f_read.norm.mean() - 1261566.2610100) < 1e-3

        # Read component names (single-word and multi-word with and without hyphen)
        # from OOMMF files
        assert df.Field.from_file(
            os.path.join(dirname, "oommf-ovf2-bin8.omf")
        ).vdims == ["x", "y", "z"]
        assert df.Field.from_file(
            os.path.join(dirname, "oommf-ovf2-bin8.ohf")
        ).vdims == ["x", "y", "z"]
        assert df.Field.from_file(
            os.path.join(dirname, "oommf-ovf2-bin8.oef")
        ).vdims == ["Total_energy_density"]

        # Read different mumax3 bin4 and txt files (made on linux and windows)
        filenames = [
            "mumax-bin4-linux.ovf",
            "mumax-bin4-windows.ovf",
            "mumax-txt-linux.ovf",
        ]
        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        for filename in filenames:
            omffilename = os.path.join(dirname, filename)
            f_read = df.Field.from_file(omffilename)

            # We know the saved magentisation.
            f_saved = df.Field(f_read.mesh, nvdim=3, value=(1, 0.1, 0), norm=1)
            assert f_saved.allclose(f_read)

    def test_to_file_read_vtk(self, tmp_path):
        filename = "testfile.vtk"

        p1 = (0, 0, 0)
        p2 = (5e-9, 2e-9, 1e-9)
        cell = (1e-9, 1e-9, 1e-9)
        subregions = {
            "sr1": df.Region(p1=p1, p2=(2e-9, 2e-9, 1e-9)),
            "sr2": df.Region(p1=(3e-9, 0, 0), p2=p2),
        }
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell, subregions=subregions)

        for nvdim, value, vdims in zip(
            [1, 2, 3, 4],
            [1.2, (1, 2.5), (1e-3, -5e6, 5e6), np.random.random(4)],
            [None, ("m_mag", "m_phase"), None, "abcd"],
        ):
            for repr in ["txt", "bin", "xml"]:
                f = df.Field(mesh, nvdim=nvdim, value=value, vdims=vdims)
                tmpfilename = tmp_path / filename
                f.to_file(tmpfilename, representation=repr)
                f_read = df.Field.from_file(tmpfilename)

                assert np.allclose(f.array, f_read.array)
                assert np.allclose(f.mesh.region.pmin, f_read.mesh.region.pmin)
                assert np.allclose(f.mesh.region.pmax, f_read.mesh.region.pmax)
                assert np.allclose(f.mesh.cell, f_read.mesh.cell)
                assert np.all(f.mesh.n == f_read.mesh.n)
                assert f.vdims == f_read.vdims
                assert f.mesh.subregions == f_read.mesh.subregions

                tmpfilename = tmp_path / f"no_sr_{filename}"
                f.to_file(tmpfilename, representation=repr, save_subregions=False)
                f_read = df.Field.from_file(tmpfilename)

                assert f.allclose(f_read)
                assert f_read.mesh.subregions == {}

        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        f = df.Field.from_file(os.path.join(dirname, "vtk-file.vtk"))
        check_field(f)
        assert np.all(f.mesh.n == (5, 1, 2))
        assert f.array.shape == (5, 1, 2, 3)
        assert f.nvdim == 3

        # test reading legacy vtk file (written with discretisedfield<=0.61.0)
        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        f = df.Field.from_file(os.path.join(dirname, "vtk-vector-legacy.vtk"))
        check_field(f)
        assert np.all(f.mesh.n == (8, 1, 1))
        assert f.array.shape == (8, 1, 1, 3)
        assert f.nvdim == 3

        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        f = df.Field.from_file(os.path.join(dirname, "vtk-scalar-legacy.vtk"))
        check_field(f)
        assert np.all(f.mesh.n == (5, 1, 2))
        assert f.array.shape == (5, 1, 2, 1)
        assert f.nvdim == 1

        # test invalid arguments
        f = df.Field(mesh, nvdim=3, value=(0, 0, 1))
        with pytest.raises(ValueError):
            f.to_file(str(tmp_path / filename), representation="wrong")
        f._vdims = None  # manually remove component labels
        with pytest.raises(AttributeError):
            f.to_file(str(tmp_path / filename))

    def test_to_file_read_hdf5(self, tmp_path):
        filenames = ["testfile.hdf5", "testfile.h5"]

        p1 = (0, 0, 0)
        p2 = (10e-12, 5e-12, 5e-12)
        cell = (1e-12, 1e-12, 1e-12)
        subregions = {
            "sr1": df.Region(p1=p1, p2=(2e-12, 2e-12, 1e-12)),
            "sr2": df.Region(p1=(3e-12, 0, 0), p2=p2),
        }
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell, subregions=subregions)

        for nvdim, value in [(1, -1.23), (3, (1e-3 + np.pi, -5e6, 6e6))]:
            f = df.Field(mesh, nvdim=nvdim, value=value)
            for filename in filenames:
                tmpfilename = tmp_path / filename
                f.to_file(tmpfilename)
                f_read = df.Field.from_file(tmpfilename)

                assert f == f_read

                tmpfilename = tmp_path / f"no_sr_{filename}"
                f.to_file(tmpfilename, save_subregions=False)
                f_read = df.Field.from_file(tmpfilename)
                assert f == f_read

    def test_read_to_file_invalid_extension(self):
        filename = "testfile.jpg"

        p1 = (0, 0, 0)
        p2 = (10e-12, 5e-12, 3e-12)
        cell = (1e-12, 1e-12, 1e-12)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        f = df.Field(mesh, nvdim=1, value=5e-12)
        with pytest.raises(ValueError):
            f.to_file(filename)
        with pytest.raises(ValueError):
            df.Field.from_file(filename)

    def test_fft(self):
        p1 = (-10, -10, -5)
        p2 = (10, 10, 5)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        def _init_random(p):
            return np.random.rand(3) * 2 - 1

        f = df.Field(mesh, nvdim=3, value=_init_random, norm=1)

        # 3d fft
        assert f.allclose(f.fftn.ifftn.real)
        assert df.Field(mesh, nvdim=3).allclose(f.fftn.ifftn.imag)

        assert f.allclose(f.rfftn.irfftn)

        # 2d fft
        for i in ["x", "y", "z"]:
            plane = f.plane(i)
            assert plane.allclose(plane.fftn.ifftn.real)
            assert df.Field(mesh, nvdim=3).plane(i).allclose(plane.fftn.ifftn.imag)

            assert plane.allclose(plane.rfftn.irfftn)

        # Fourier slice theoreme
        for i in "xyz":
            plane = f.integrate(i)
            assert plane.allclose(f.fftn.plane(**{i: 0}).ifftn.real)
            assert (
                df.Field(mesh, nvdim=3)
                .integrate(i)
                .allclose(f.fftn.plane(**{i: 0}).ifftn.imag)
            )

        assert f.integrate("x").allclose(f.rfftn.plane(x=0).irfftn)
        assert f.integrate("y").allclose(f.rfftn.plane(y=0).irfftn)
        # plane along z removes rfftn-freq axis => needs ifftn
        assert f.integrate("z").allclose(f.rfftn.plane(z=0).ifftn.real)

    def test_mpl_scalar(self):
        # No axes
        for comp in self.pf.vdims:
            getattr(self.pf, comp).plane("x", n=(3, 4)).mpl.scalar()

        # Axes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for comp in self.pf.vdims:
            getattr(self.pf, comp).plane("x", n=(3, 4)).mpl.scalar(ax=ax)

        # All arguments
        for comp in self.pf.vdims:
            getattr(self.pf, comp).plane("x").mpl.scalar(
                figsize=(10, 10),
                filter_field=self.pf.norm,
                colorbar=True,
                colorbar_label="something",
                multiplier=1e-6,
                cmap="hsv",
                clim=(-1, 1),
            )

        # Saving plot
        filename = "testfigure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            self.pf.a.plane("x", n=(3, 4)).mpl.scalar(filename=tmpfilename)

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.a.mpl.scalar()  # not sliced
        with pytest.raises(ValueError):
            self.pf.plane("z").mpl.scalar()  # vector field
        with pytest.raises(ValueError):
            # wrong filter field
            self.pf.a.plane("z").mpl.scalar(filter_field=self.pf)
        plt.close("all")

    def test_mpl_lightess(self):
        filenames = ["skyrmion.omf", "skyrmion-disk.omf"]
        for i in filenames:
            filename = os.path.join(os.path.dirname(__file__), "test_sample", i)

            field = df.Field.from_file(filename)
            for plane in ["z"]:  # TODO test all directions "xyz" (check samples first)
                field.plane(plane).mpl.lightness()
                field.plane(plane).mpl.lightness(
                    lightness_field=-field.z, filter_field=field.norm
                )
            fig, ax = plt.subplots()
            field.plane("z").mpl.lightness(
                ax=ax, clim=[0, 0.5], colorwheel_xlabel="mx", colorwheel_ylabel="my"
            )
            # Saving plot
            filename = "testfigure.pdf"
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfilename = os.path.join(tmpdir, filename)
                field.plane("z").mpl.lightness(filename=tmpfilename)

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.mpl.lightness()  # not sliced
        with pytest.raises(ValueError):
            # wrong filter field
            self.pf.plane("z").mpl.lightness(filter_field=self.pf)
        with pytest.raises(ValueError):
            # wrong lightness field
            self.pf.plane("z").mpl.lightness(lightness_field=self.pf)
        plt.close("all")

    def test_mpl_vector(self):
        # No axes
        self.pf.plane("x", n=(3, 4)).mpl.vector()

        # Axes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.pf.plane("x", n=(3, 4)).mpl.vector(ax=ax)

        # All arguments
        self.pf.plane("x").mpl.vector(
            figsize=(10, 10),
            color_field=self.pf.b,
            colorbar=True,
            colorbar_label="something",
            multiplier=1e-6,
            cmap="hsv",
            clim=(-1, 1),
        )

        # 2d vector field
        plane_2d = self.pf.plane("z").a << self.pf.plane("z").b
        plane_2d.vdims = ["a", "b"]
        with pytest.raises(ValueError):
            plane_2d.mpl.vector()
        plane_2d.mpl.vector(vdims=["a", "b"])
        plane_2d.mpl.vector(vdims=["a", None])
        plane_2d.mpl.vector(vdims=[None, "b"])
        with pytest.raises(ValueError):
            plane_2d.mpl.vector(vdims=[None, None])

        # Saving plot
        filename = "testfigure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            self.pf.plane("x", n=(3, 4)).mpl.vector(filename=tmpfilename)

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.mpl.vector()  # not sliced
        with pytest.raises(ValueError):
            self.pf.b.plane("z").mpl.vector()  # scalar field
        with pytest.raises(ValueError):
            # wrong color field
            self.pf.plane("z").mpl.vector(color_field=self.pf)

        plt.close("all")

    def test_mpl_contour(self):
        # No axes
        self.pf.plane("z").c.mpl.contour()

        # Axes
        fig, ax = plt.subplots()
        self.pf.plane("z").c.mpl.contour(ax=ax)

        # All arguments
        self.pf.plane("z").c.mpl.contour(
            figsize=(10, 10),
            multiplier=1e-6,
            filter_field=self.pf.norm,
            colorbar=True,
            colorbar_label="something",
        )

        # Saving plot
        filename = "testfigure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            self.pf.plane("z").c.mpl.contour(filename=tmpfilename)

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.mpl.contour()  # not sliced
        with pytest.raises(ValueError):
            self.pf.plane("z").mpl.contour()  # vector field
        with pytest.raises(ValueError):
            # wrong filter field
            self.pf.plane("z").c.mpl.contour(filter_field=self.pf)

        plt.close("all")

    def test_mpl(self):
        # No axes
        self.pf.plane("x", n=(3, 4)).mpl()

        # Axes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.pf.a.plane("x", n=(3, 4)).mpl(ax=ax)

        self.pf.c.plane("x").mpl(
            figsize=(12, 6),
            scalar_kw={
                "filter_field": self.pf.norm,
                "colorbar_label": "scalar",
                "cmap": "twilight",
            },
            vector_kw={
                "color_field": self.pf.b,
                "use_color": True,
                "colorbar": True,
                "colorbar_label": "vector",
                "cmap": "hsv",
                "clim": (0, 1e6),
            },
            multiplier=1e-12,
        )

        # Saving plot
        filename = "testfigure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            self.pf.plane("x", n=(3, 4)).mpl(filename=tmpfilename)

        # Exception
        with pytest.raises(ValueError):
            self.pf.mpl()

        plt.close("all")

    def test_hv_scalar(self):
        for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
            normal = (set("xyz") - set(kdims)).pop()
            kdim_str = f"[{','.join(kdims)}]"
            check_hv(
                self.pf.hv.scalar(kdims=kdims),
                [f"DynamicMap [{normal},comp]", f"Image {kdim_str}"],
            )
            check_hv(
                self.pf.hv.scalar(kdims=kdims, roi=self.pf.norm),
                [f"DynamicMap [{normal},comp]", f"Image {kdim_str}"],
            )

            # additional kwargs and plane
            check_hv(
                self.pf.plane(normal).hv.scalar(kdims=kdims, clim=(-1, 1)),
                ["DynamicMap [comp]", f"Image {kdim_str}"],
            )

            for c in self.pf.vdims:
                check_hv(
                    getattr(self.pf, c).hv.scalar(kdims=kdims),
                    [f"DynamicMap [{normal}]", f"Image {kdim_str}"],
                )
                check_hv(
                    getattr(self.pf, c).plane(normal).hv.scalar(kdims=kdims),
                    [f"Image {kdim_str}"],
                )

        with pytest.raises(ValueError):
            check_hv(self.pf.hv.scalar(kdims=["wrong_name", "x"]), ...)

        with pytest.raises(ValueError):
            check_hv(self.pf.hv.scalar(kdims=["x", "y", "z"]), ...)

        with pytest.raises(TypeError):
            check_hv(self.pf.hv.scalar(kdims=["x", "y"], roi="z"), ...)

        with pytest.raises(ValueError):
            check_hv(self.pf.hv.scalar(kdims=["x", "y"], roi=self.pf), ...)

        with pytest.raises(ValueError):
            check_hv(
                self.pf.plane("z").hv.scalar(kdims=["x", "y"], roi=self.pf.norm), ...
            )

        with pytest.raises(ValueError):
            check_hv(
                self.pf[
                    df.Region(p1=(-5e-9, -5e-9, -5e-9), p2=(5e-9, 5e-9, -1e-9))
                ].hv.scalar(kdims=["x", "y"], roi=self.pf.norm.plane(z=4e-9)),
                ...,
            )

    def test_hv_vector(self):
        for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
            normal = (set("xyz") - set(kdims)).pop()
            kdim_str = f"[{','.join(kdims)}]"
            check_hv(
                self.pf.hv.vector(kdims=kdims),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            check_hv(
                self.pf.plane(normal).hv.vector(kdims=kdims),
                [f"VectorField {kdim_str}"],
            )
            check_hv(
                self.pf.hv.vector(kdims=kdims, roi=self.pf.norm),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            check_hv(
                self.pf.hv.vector(kdims=kdims, n=(10, 10)),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )

            # additional kwargs
            check_hv(
                self.pf.hv.vector(kdims=kdims, use_color=False, color="blue"),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )

            for comp in self.pf.vdims:
                check_hv(
                    self.pf.hv.vector(kdims=kdims, cdim=comp),
                    [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
                )

            with pytest.raises(ValueError):
                check_hv(self.pf.hv.vector(kdims=kdims, cdim="wrong"), ...)

            with pytest.raises(TypeError):
                check_hv(self.pf.hv.vector(kdims=kdims, cdim=self.pf.norm), ...)

            with pytest.raises(ValueError):
                check_hv(self.pf.hv.vector(kdims=kdims, vdims=["a", "b", "c"]), ...)

            # 2d field
            with pytest.raises(ValueError):
                check_hv((self.pf.a << self.pf.b).hv.vector(kdims=kdims), ...)

            field_2d = self.pf.a << self.pf.b
            field_2d.vdims = ["a", "b"]
            check_hv(
                field_2d.hv.vector(kdims=kdims, vdims=["a", "b"]),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            check_hv(
                field_2d.hv.vector(kdims=kdims, vdims=[None, "b"]),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            check_hv(
                field_2d.hv.vector(kdims=kdims, vdims=["a", None]),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            with pytest.raises(ValueError):
                check_hv(field_2d.hv.vector(kdims=kdims, vdims=[None, None]), ...)

            # 4d field
            field_4d = self.pf.a << self.pf.b << self.pf.a << self.pf.b
            field_4d.vdims = ["a", "b", "c", "d"]
            with pytest.raises(ValueError):
                check_hv(field_4d.hv.vector(kdims=kdims), ...)
            check_hv(
                field_4d.hv.vector(kdims=kdims, vdims=["c", "d"]),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            check_hv(
                field_4d.hv.vector(kdims=kdims, vdims=["c", "d"]),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            check_hv(
                field_4d.hv.vector(kdims=kdims, vdims=[None, "b"]),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )

        with pytest.raises(ValueError):
            check_hv(self.pf.hv.contour(kdims=["wrong_name", "x"]), ...)

        with pytest.raises(ValueError):
            check_hv(self.pf.hv.vector(kdims=["x", "y"], n=(10, 10, 10)), ...)

        # scalar field
        with pytest.raises(ValueError):
            check_hv(field_2d.a.hv.vector(kdims=["x", "y"]), ...)

    def test_hv_contour(self):
        for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
            normal = (set("xyz") - set(kdims)).pop()
            kdim_str = f"[{','.join(kdims)}]"
            # Required for the tests (not in a notebook):
            # If not specified the plot creation for the test fails because the width
            # and height of the plot cannot be calculated (NaN values). By manually
            # setting frame_width and frame_height this can be avoided.
            # It is not fully clear why this does not happen for the other plots.
            # Presumably, because we use a different method to create the contour plot.
            opts = dict(frame_width=300, frame_height=300)
            check_hv(
                self.pf.hv.contour(kdims=kdims).opts(**opts),
                [f"DynamicMap [{normal},comp]", f"Contours {kdim_str}"],
            )
            check_hv(
                self.pf.hv.contour(kdims=kdims, roi=self.pf.norm).opts(**opts),
                [f"DynamicMap [{normal},comp]", f"Contours {kdim_str}"],
            )

            # additional kwargs
            check_hv(
                self.pf.plane(normal)
                .hv.contour(kdims=kdims, clim=(-1, 1))
                .opts(**opts),
                ["DynamicMap [comp]", f"Contours {kdim_str}"],
            )

            for c in self.pf.vdims:
                check_hv(
                    getattr(self.pf, c).hv.contour(kdims=kdims).opts(**opts),
                    [f"DynamicMap [{normal}]", f"Contours {kdim_str}"],
                )

        with pytest.raises(ValueError):
            check_hv(self.pf.hv.contour(kdims=["wrong_name", "x"]), ...)

    def test_hv(self):
        for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
            normal = (set("xyz") - set(kdims)).pop()
            kdim_str = f"[{','.join(kdims)}]"
            # 1d field
            check_hv(
                self.pf.a.hv(kdims=kdims),
                [f"DynamicMap [{normal}]", f"Image {kdim_str}"],
            )
            check_hv(self.pf.a.plane(normal).hv(kdims=kdims), [f"Image {kdim_str}"])

            # 2d field
            field_2d = self.pf.b << self.pf.c
            check_hv(
                field_2d.hv(kdims=kdims),
                [f"DynamicMap [{normal},comp]", f"Image {kdim_str}"],
            )
            check_hv(
                field_2d.hv(kdims=kdims, vdims=["x", "y"]),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )
            check_hv(
                field_2d.plane(normal).hv(kdims=kdims),
                ["DynamicMap [comp]", f"Image {kdim_str}"],
            )

            # 3d field
            check_hv(
                self.pf.hv(kdims=kdims),
                [
                    f"DynamicMap [{normal}]",
                    f"Image {kdim_str}",
                    f"VectorField {kdim_str}",
                ],
            )
            check_hv(
                self.pf.hv(kdims=kdims, vdims=["a", "b"]),
                [
                    f"DynamicMap [{normal}]",
                    f"Image {kdim_str}",
                    f"VectorField {kdim_str}",
                ],
            )
            check_hv(
                self.pf.plane(normal).hv(kdims=kdims),
                [f"Image {kdim_str}", f"VectorField {kdim_str}"],
            )

            # additional kwargs
            check_hv(
                self.pf.hv(
                    kdims=kdims,
                    scalar_kw={"clim": (-1, 1)},
                    vector_kw={"cmap": "cividis"},
                ),
                [
                    f"DynamicMap [{normal}]",
                    f"Image {kdim_str}",
                    f"VectorField {kdim_str}",
                ],
            )

            # 4d field
            field_4d = self.pf.b << self.pf.c << self.pf.a << self.pf.a
            field_4d.vdims = ["v1", "v2", "v3", "v4"]
            check_hv(
                field_4d.hv(kdims=kdims),
                [f"DynamicMap [{normal},comp]", f"Image {kdim_str}"],
            )

            check_hv(
                field_4d.hv(kdims=kdims, vdims=["v2", "v1"]),
                [
                    f"DynamicMap [{normal},comp]",
                    f"Image {kdim_str}",
                    f"VectorField {kdim_str}",
                ],
            )

            check_hv(
                field_4d.plane(normal).hv(kdims=kdims),
                ["DynamicMap [comp]", f"Image {kdim_str}"],
            )

            check_hv(
                field_4d.plane(normal).hv(
                    kdims=kdims, vdims=["v2", "v1"], vector_kw={"cdim": "v4"}
                ),
                [
                    "DynamicMap [comp]",
                    f"Image {kdim_str}",
                    f"VectorField {kdim_str}",
                ],
            )

    def test_k3d_nonzero(self):
        # Default
        self.pf.norm.k3d.nonzero()

        # Color
        self.pf.a.k3d.nonzero(color=0xFF00FF)

        # Multiplier
        self.pf.b.k3d.nonzero(color=0xFF00FF, multiplier=1e-6)

        # Interactive field
        self.pf.c.plane("z").k3d.nonzero(
            color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf
        )

        # kwargs
        self.pf.a.plane("z").k3d.nonzero(
            color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf, wireframe=True
        )

        # Plot
        plot = k3d.plot()
        plot.display()
        self.pf.b.plane(z=0).k3d.nonzero(
            plot=plot, color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf
        )

        # Continuation for interactive plot testing.
        self.pf.c.plane(z=1e-9).k3d.nonzero(
            plot=plot, color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf
        )

        assert len(plot.objects) == 2

        with pytest.raises(ValueError):
            self.pf.k3d.nonzero()

    def test_k3d_scalar(self):
        # Default
        self.pf.a.k3d.scalar()

        # Filter field
        self.pf.b.k3d.scalar(filter_field=self.pf.norm)

        # Colormap
        self.pf.c.k3d.scalar(filter_field=self.pf.norm, cmap="hsv", color=0xFF00FF)

        # Multiplier
        self.pf.a.k3d.scalar(filter_field=self.pf.norm, color=0xFF00FF, multiplier=1e-6)

        # Interactive field
        self.pf.b.k3d.scalar(
            filter_field=self.pf.norm,
            color=0xFF00FF,
            multiplier=1e-6,
            interactive_field=self.pf,
        )

        # kwargs
        self.pf.c.k3d.scalar(
            filter_field=self.pf.norm,
            color=0xFF00FF,
            multiplier=1e-6,
            interactive_field=self.pf,
            wireframe=True,
        )

        # Plot
        plot = k3d.plot()
        plot.display()
        self.pf.a.plane(z=0).k3d.scalar(
            plot=plot,
            filter_field=self.pf.norm,
            color=0xFF00FF,
            multiplier=1e-6,
            interactive_field=self.pf,
        )

        # Continuation for interactive plot testing.
        self.pf.b.plane(z=1e-9).k3d.scalar(
            plot=plot,
            filter_field=self.pf.norm,
            color=0xFF00FF,
            multiplier=1e-6,
            interactive_field=self.pf,
        )

        assert len(plot.objects) == 2

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.k3d.scalar()
        with pytest.raises(ValueError):
            self.pf.c.k3d.scalar(filter_field=self.pf)  # filter field nvdim=3

    def test_k3d_vector(self):
        # Default
        self.pf.k3d.vector()

        # Color field
        self.pf.k3d.vector(color_field=self.pf.a)

        # Colormap
        self.pf.k3d.vector(color_field=self.pf.norm, cmap="hsv")

        # Head size
        self.pf.k3d.vector(color_field=self.pf.norm, cmap="hsv", head_size=3)

        # Points
        self.pf.k3d.vector(
            color_field=self.pf.norm, cmap="hsv", head_size=3, points=False
        )

        # Point size
        self.pf.k3d.vector(
            color_field=self.pf.norm,
            cmap="hsv",
            head_size=3,
            points=False,
            point_size=1,
        )

        # Vector multiplier
        self.pf.k3d.vector(
            color_field=self.pf.norm,
            cmap="hsv",
            head_size=3,
            points=False,
            point_size=1,
            vector_multiplier=1,
        )

        # Multiplier
        self.pf.k3d.vector(
            color_field=self.pf.norm,
            cmap="hsv",
            head_size=3,
            points=False,
            point_size=1,
            vector_multiplier=1,
            multiplier=1e-6,
        )

        # Interactive field
        self.pf.plane("z").k3d.vector(
            color_field=self.pf.norm,
            cmap="hsv",
            head_size=3,
            points=False,
            point_size=1,
            vector_multiplier=1,
            multiplier=1e-6,
            interactive_field=self.pf,
        )

        # Plot
        plot = k3d.plot()
        plot.display()
        self.pf.plane(z=0).k3d.vector(plot=plot, interactive_field=self.pf)

        # Continuation for interactive plot testing.
        self.pf.plane(z=1e-9).k3d.vector(plot=plot, interactive_field=self.pf)

        assert len(plot.objects) == 3

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.a.k3d.vector()
        with pytest.raises(ValueError):
            self.pf.k3d.vector(color_field=self.pf)  # filter field nvdim=3

    def test_plot_large_sample(self):
        p1 = (0, 0, 0)
        p2 = (50e9, 50e9, 50e9)
        cell = (25e9, 25e9, 25e9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        value = (1e6, 1e6, 1e6)
        field = df.Field(mesh, nvdim=3, value=value)

        field.plane("z").mpl()
        field.norm.k3d.nonzero()
        field.x.k3d.scalar()
        field.k3d.vector()

    def test_complex(self):
        mesh = df.Mesh(p1=(-5e-9, -5e-9, -5e-9), p2=(5e-9, 5e-9, 5e-9), n=(5, 5, 5))

        # real field
        real_field = self.pf.real
        check_field(real_field)
        assert np.allclose(real_field((-2e-9, 0, 0)), (0, 0, 1e5))
        assert np.allclose(real_field((2e-9, 0, 0)), (0, 0, -1e5))

        imag_field = self.pf.imag
        check_field(imag_field)
        assert df.Field(mesh, nvdim=3).allclose(imag_field)
        assert df.Field(mesh, nvdim=3).allclose(np.mod(self.pf.phase, np.pi))

        # complex field
        field = df.Field(mesh, nvdim=1, value=1 + 1j)
        real_field = field.real
        check_field(real_field)
        assert df.Field(mesh, nvdim=1, value=1).allclose(real_field)

        imag_field = field.imag
        check_field(imag_field)
        assert df.Field(mesh, nvdim=1, value=1).allclose(imag_field)
        assert df.Field(mesh, nvdim=1, value=np.pi / 4).allclose(field.phase)

    def test_numpy_ufunc(self):
        assert np.allclose(np.sin(self.pf).array, np.sin(self.pf.array))
        assert np.sum([self.pf, self.pf]).allclose(self.pf + self.pf)
        assert np.multiply(self.pf.a, self.pf.b).allclose(self.pf.a * self.pf.b)
        assert np.power(self.pf.c, 2).allclose(self.pf.c**2)

        # self.pf contains values of 1e5 and exp of this,produces an overflow
        field = df.Field(
            self.pf.mesh, nvdim=3, value=lambda _: np.random.random(3) * 2 - 1
        )
        assert np.allclose(
            np.exp(field.orientation).array, np.exp(field.orientation.array)
        )

    def test_to_xarray_valid_args(self):
        for mesh in self.meshes:
            for value, dtype in self.vfuncs:
                f = df.Field(mesh, nvdim=3, value=value, dtype=dtype)
                fxa = f.to_xarray()
                assert isinstance(fxa, xr.DataArray)
                assert f.nvdim == fxa["comp"].size
                assert sorted([*fxa.attrs]) == ["cell", "pmax", "pmin", "units"]
                assert np.allclose(fxa.attrs["cell"], f.mesh.cell)
                assert np.allclose(fxa.attrs["pmin"], f.mesh.region.pmin)
                assert np.allclose(fxa.attrs["pmax"], f.mesh.region.pmax)
                for i in "xyz":
                    assert np.array_equal(getattr(f.mesh.points, i), fxa[i].values)
                    assert (
                        fxa[i].attrs["units"]
                        == f.mesh.region.units[f.mesh.region.dims.index(i)]
                    )
                assert all(fxa["comp"].values == f.vdims)
                assert np.array_equal(f.array, fxa.values)

            for value, dtype in self.sfuncs:
                f = df.Field(mesh, nvdim=1, value=value, dtype=dtype)
                fxa = f.to_xarray()
                assert isinstance(fxa, xr.DataArray)
                assert sorted([*fxa.attrs]) == ["cell", "pmax", "pmin", "units"]
                assert np.allclose(fxa.attrs["cell"], f.mesh.cell)
                assert np.allclose(fxa.attrs["pmin"], f.mesh.region.pmin)
                assert np.allclose(fxa.attrs["pmax"], f.mesh.region.pmax)
                for i in "xyz":
                    assert np.array_equal(getattr(f.mesh.points, i), fxa[i].values)
                    assert (
                        fxa[i].attrs["units"]
                        == f.mesh.region.units[f.mesh.region.dims.index(i)]
                    )
                assert "comp" not in fxa.dims
                assert np.array_equal(f.array.squeeze(axis=-1), fxa.values)

        f6d = self.pf << self.pf
        f6d_xa = f6d.to_xarray()
        assert f6d_xa["comp"].size == 6
        assert "comp" not in f6d_xa.coords
        f6d.vdims = ["a", "c", "b", "e", "d", "f"]
        f6d_xa2 = f6d.to_xarray()
        assert "comp" in f6d_xa2.coords
        assert [*f6d_xa2["comp"].values] == ["a", "c", "b", "e", "d", "f"]

        # test name and units defaults
        f3d_xa = self.pf.to_xarray()
        assert f3d_xa.name == "field"
        assert f3d_xa.attrs["units"] is None

        # test name and units
        f3d_xa_2 = self.pf.to_xarray(name="m", unit="A/m")
        assert f3d_xa_2.name == "m"
        assert f3d_xa_2.attrs["units"] == "A/m"

    def test_to_xarray_invalid_args(self):
        args = [
            ["m", 42.0],
            [21.0, 42],
            [21, "A/m"],
            [{"name": "m"}, {"unit": "A/m"}],
            [["m"], ["A/m"]],
            [["m", "A/m"], None],
            [("m", "A/m"), None],
            [{"name": "m", "unit": "A/m"}, None],
        ]

        for name, unit in args:
            with pytest.raises(TypeError):
                self.pf.to_xarray(name, unit)

    def test_from_xarray_valid_args(self):
        for mesh in self.meshes:
            for value, dtype in self.vfuncs:
                f = df.Field(mesh, nvdim=3, value=value, dtype=dtype)
                fxa = f.to_xarray()
                f_new = df.Field.from_xarray(fxa)
                assert f_new == f  # or use allclose()

            for value, dtype in self.sfuncs:
                f = df.Field(mesh, nvdim=1, value=value, dtype=dtype)
                fxa = f.to_xarray()
                f_new = df.Field.from_xarray(fxa)
                assert f_new == f  # or use allclose()

        f_plane = self.pf.plane("z")
        f_plane_xa = f_plane.to_xarray()
        f_plane_new = df.Field.from_xarray(f_plane_xa)
        assert f_plane_new == f_plane  # or use allclose()

        f6d = self.pf << self.pf
        f6d_xa = f6d.to_xarray()
        f6d_new = df.Field.from_xarray(f6d_xa)
        assert f6d_new == f6d  # or use allclose()

        good_darray1 = xr.DataArray(
            np.ones((20, 20, 5, 3)),
            dims=["x", "y", "z", "comp"],
            coords=dict(
                x=np.arange(0, 20),
                y=np.arange(0, 20),
                z=np.arange(0, 5),
                comp=["x", "y", "z"],
            ),
            name="mag",
            attrs=dict(units="A/m"),
        )

        good_darray2 = xr.DataArray(
            np.ones((20, 20, 1, 3)),
            dims=["x", "y", "z", "comp"],
            coords=dict(
                x=np.arange(0, 20), y=np.arange(0, 20), z=[5.0], comp=["x", "y", "z"]
            ),
            name="mag",
            attrs=dict(units="A/m", cell=[1.0, 1.0, 1.0]),
        )

        good_darray3 = xr.DataArray(
            np.ones((20, 20, 1, 3)),
            dims=["x", "y", "z", "comp"],
            coords=dict(
                x=np.arange(0, 20), y=np.arange(0, 20), z=[5.0], comp=["x", "y", "z"]
            ),
            name="mag",
            attrs=dict(
                units="A/m",
                cell=[1.0, 1.0, 1.0],
                p1=[1.0, 1.0, 1.0],
                p2=[21.0, 21.0, 2.0],
            ),
        )

        fg_1 = df.Field.from_xarray(good_darray1)
        check_field(fg_1)
        fg_2 = df.Field.from_xarray(good_darray2)
        check_field(fg_2)
        fg_3 = df.Field.from_xarray(good_darray3)
        check_field(fg_3)

    def test_from_xarray_invalid_args_and_DataArrays(self):
        args = [
            int(),
            float(),
            str(),
            list(),
            dict(),
            xr.Dataset(),
            np.empty((20, 20, 20, 3)),
        ]

        bad_dim_no = xr.DataArray(
            np.ones((20, 20, 20, 5, 3), dtype=float),
            dims=["x", "y", "z", "a", "comp"],
            coords=dict(
                x=np.arange(0, 20),
                y=np.arange(0, 20),
                z=np.arange(0, 20),
                a=np.arange(0, 5),
                comp=["x", "y", "z"],
            ),
            name="mag",
            attrs=dict(units="A/m"),
        )

        bad_dim_no2 = xr.DataArray(
            np.ones((20, 20), dtype=float),
            dims=["x", "y"],
            coords=dict(x=np.arange(0, 20), y=np.arange(0, 20)),
            name="mag",
            attrs=dict(units="A/m"),
        )

        bad_dim3 = xr.DataArray(
            np.ones((20, 20, 5), dtype=float),
            dims=["a", "b", "c"],
            coords=dict(a=np.arange(0, 20), b=np.arange(0, 20), c=np.arange(0, 5)),
            name="mag",
            attrs=dict(units="A/m"),
        )

        bad_dim4 = xr.DataArray(
            np.ones((20, 20, 5, 3), dtype=float),
            dims=["x", "y", "z", "c"],
            coords=dict(
                x=np.arange(0, 20),
                y=np.arange(0, 20),
                z=np.arange(0, 5),
                c=["x", "y", "z"],
            ),
            name="mag",
            attrs=dict(units="A/m"),
        )

        bad_attrs = xr.DataArray(
            np.ones((20, 20, 1, 3), dtype=float),
            dims=["x", "y", "z", "comp"],
            coords=dict(
                x=np.arange(0, 20), y=np.arange(0, 20), z=[5.0], comp=["x", "y", "z"]
            ),
            name="mag",
            attrs=dict(units="A/m"),
        )

        def bad_coord_gen():
            rng = np.random.default_rng()
            for coord in "xyz":
                coord_dict = {coord: rng.normal(size=20)}
                for other_coord in "xyz".translate({ord(coord): None}):
                    coord_dict[other_coord] = np.arange(0, 20)
                coord_dict["comp"] = ["x", "y", "z"]

                yield xr.DataArray(
                    np.ones((20, 20, 20, 3), dtype=float),
                    dims=["x", "y", "z", "comp"],
                    coords=coord_dict,
                    name="mag",
                    attrs=dict(units="A/m"),
                )

        for arg in args:
            with pytest.raises(TypeError):
                df.Field.from_xarray(arg)
        with pytest.raises(ValueError):
            df.Field.from_xarray(bad_dim_no)
        with pytest.raises(ValueError):
            df.Field.from_xarray(bad_dim_no2)
        with pytest.raises(ValueError):
            df.Field.from_xarray(bad_dim3)
        with pytest.raises(ValueError):
            df.Field.from_xarray(bad_dim4)
        for bad_coord_geo in bad_coord_gen():
            with pytest.raises(ValueError):
                df.Field.from_xarray(bad_coord_geo)
        with pytest.raises(KeyError):
            df.Field.from_xarray(bad_attrs)
