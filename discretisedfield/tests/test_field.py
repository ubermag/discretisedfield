import itertools
import numbers
import os
import random
import re
import tempfile
import types

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
    r"<li>dim = \d+</li>\s*"
    r"(<li>components:\s*<ul>(<li>.*</li>\s*)+</ul>\s*</li>)?\s*"
    r"(<li>units = .+</li>)?\s*"
    r"</ul>"
)


def check_field(field):
    assert isinstance(field.mesh, df.Mesh)

    assert isinstance(field.dim, int)
    assert field.dim > 0

    assert isinstance(field.array, np.ndarray)
    assert field.array.shape == (*field.mesh.n, field.dim)

    average = field.average
    assert isinstance(average, (tuple, numbers.Complex))

    rstr = repr(field)
    assert isinstance(rstr, str)
    pattern = r"^Field\(Mesh\(Region\(p1=\(.+\), p2=\(.+\)\), .+\)," r" dim=\d+\)$"
    if field.components:
        pattern = pattern[:-3] + r", components: \(.+\)\)$"
    if field.units is not None:
        pattern = pattern[:-3] + r", units=.+\)$"
    assert re.search(pattern, rstr)

    assert isinstance(field._repr_html_(), str)
    assert re.search(html_re, field._repr_html_(), re.DOTALL)

    assert isinstance(field.__iter__(), types.GeneratorType)
    assert len(list(field)) == len(field.mesh)

    line = field.line(p1=field.mesh.region.pmin, p2=field.mesh.region.pmax, n=5)
    assert isinstance(line, df.Line)
    assert line.n == 5

    plane = field.plane("z", n=(2, 2))
    assert isinstance(plane, df.Field)
    assert len(plane.mesh) == 4
    assert plane.mesh.n == (2, 2, 1)

    project = field.project("z")
    assert isinstance(project, df.Field)
    assert project.mesh.n[2] == 1

    assert isinstance(field(field.mesh.region.centre), (tuple, numbers.Complex))
    assert isinstance(field(field.mesh.region.random_point()), (tuple, numbers.Complex))

    assert field == field
    assert not field != field

    assert +field == field
    assert -(-field) == field
    assert field + field == 2 * field
    assert field - (-field) == field + field
    assert 1 * field == field
    assert -1 * field == -field
    assert field.units is None or isinstance(field.units, str)

    if field.dim == 1:
        grad = field.grad
        assert isinstance(grad, df.Field)
        assert grad.dim == 3

        assert all(i not in dir(field) for i in "xyz")

        assert isinstance((field * df.dx).integral(), numbers.Complex)
        assert isinstance((field * df.dy).integral(), numbers.Complex)
        assert isinstance((field * df.dz).integral(), numbers.Complex)
        assert isinstance((field * df.dV).integral(), numbers.Complex)
        assert isinstance((field.plane("z") * df.dS).integral(), tuple)
        assert isinstance((field.plane("z") * abs(df.dS)).integral(), numbers.Complex)

    if field.dim == 3:
        norm = field.norm
        assert isinstance(norm, df.Field)
        assert norm == abs(field)
        assert norm.dim == 1

        div = field.div
        assert isinstance(div, df.Field)
        assert div.dim == 1

        curl = field.curl
        assert isinstance(curl, df.Field)
        assert curl.dim == 3

        field.plane("z")

        assert isinstance((field * df.dx).integral(), tuple)
        assert isinstance((field * df.dy).integral(), tuple)
        assert isinstance((field * df.dz).integral(), tuple)
        assert isinstance((field * df.dV).integral(), tuple)
        assert isinstance((field.plane("z") @ df.dS).integral(), numbers.Complex)
        assert isinstance((field.plane("z") * abs(df.dS)).integral(), tuple)

        orientation = field.orientation
        assert isinstance(orientation, df.Field)
        assert orientation.dim == 3

    if field.dim > 1 and field.components is not None:
        for comp in field.components:
            assert isinstance(getattr(field, comp), df.Field)
            assert getattr(field, comp).dim == 1


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

        self.pf = df.Field(mesh, dim=3, value=value_fun, norm=norm_fun)

    def test_init_valid_args(self):
        for mesh in self.meshes:
            for value, dtype in self.consts + self.sfuncs:
                f = df.Field(mesh, dim=1, value=value, dtype=dtype)
                check_field(f)

            for value, dtype in self.iters + self.vfuncs:
                f = df.Field(mesh, dim=3, value=value, dtype=dtype)
                check_field(f)

    def test_init_invalid_args(self):
        with pytest.raises(TypeError):
            mesh = "meaningless_mesh_string"
            df.Field(mesh, dim=1)

        for mesh in self.meshes:
            for dim in [0, -1, "dim", (2, 3)]:
                with pytest.raises((ValueError, TypeError)):
                    df.Field(mesh, dim=dim)

        # wrong abc.Iterable
        with pytest.raises(TypeError):
            df.Field(self.meshes[0], dim=1, value="string")

        # all builtin types are numeric types or Iterable
        class WrongType:
            pass

        with pytest.raises(TypeError):
            df.Field(self.meshes[0], dim=1, value=WrongType())

    def test_set_with_ndarray(self):
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3)
            f.value = np.ones(
                (
                    *f.mesh.n,
                    f.dim,
                )
            )

            check_field(f)
            assert isinstance(f.value, np.ndarray)
            assert f.average == (1, 1, 1)

            with pytest.raises(ValueError):
                f.value = np.ones((2, 2))

    def test_set_with_callable(self):
        for mesh in self.meshes:
            for func, dtype in self.sfuncs:
                f = df.Field(mesh, dim=1, value=func, dtype=dtype)
                check_field(f)

                rp = f.mesh.region.random_point()
                # Make sure to be at the centre of the cell
                rp = f.mesh.index2point(f.mesh.point2index(rp))
                assert f(rp) == func(rp)

        for mesh in self.meshes:
            for func, dtype in self.vfuncs:
                f = df.Field(mesh, dim=3, value=func, dtype=dtype)
                check_field(f)

                rp = f.mesh.region.random_point()
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
            mesh, dim=3, value={"r1": (0, 0, 1), "r2": (0, 0, 2), "r1:r2": (0, 0, 5)}
        )
        assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1))
        assert np.all(field((8e-9, 2e-9, 9e-9)) == (0, 0, 2))

        subregions = {"r1": df.Region(p1=(0, 0, 0), p2=(4e-9, 10e-9, 10e-9))}
        mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)

        with pytest.raises(KeyError):
            field = df.Field(mesh, dim=3, value={"r1": (0, 0, 1)})

        field = df.Field(mesh, dim=3, value={"r1": (0, 0, 1), "default": (1, 1, 1)})
        assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1))
        assert np.all(field((8e-9, 2e-9, 9e-9)) == (1, 1, 1))

        field = df.Field(mesh, dim=3, value={"default": (1, 1, 1)})
        assert np.all(field.array == (1, 1, 1))

        # dtype has to be specified for isinstance(value, dict)
        field = df.Field(
            mesh,
            dim=3,
            value={"r1": (0, 0, 1 + 2j), "default": (1, 1, 1)},
            dtype=np.complex128,
        )
        assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1 + 2j))
        assert np.all(field((8e-9, 2e-9, 9e-9)) == (1, 1, 1))

    def test_set_exception(self):
        for mesh in self.meshes:
            with pytest.raises(TypeError):
                df.Field(mesh, dim=3, value="meaningless_string")

            with pytest.raises(ValueError):
                df.Field(mesh, dim=3, value=5 + 5j)

    def test_coordinate_field(self):
        for mesh in self.meshes:
            cfield = df.Field.coordinate_field(mesh)
            check_field(cfield)
            manually = df.Field(mesh, dim=3, value=lambda p: p)
            assert cfield.allclose(manually)
            assert np.allclose(cfield.array[:, 0, 0, 0], mesh.midpoints.x)
            assert np.allclose(cfield.array[0, :, 0, 1], mesh.midpoints.y)
            assert np.allclose(cfield.array[0, 0, :, 2], mesh.midpoints.z)

    def test_components(self):
        for mesh in self.meshes:
            valid_components = ["a", "b", "c", "d", "e", "f"]
            invalid_components = ["a", "grad", "b", "div", "array", "c"]
            for dim in range(2, 7):
                f = df.Field(
                    mesh,
                    dim=dim,
                    value=list(range(dim)),
                    components=valid_components[:dim],
                )
                assert f.components == valid_components[:dim]
                check_field(f)

                with pytest.raises(ValueError):
                    df.Field(
                        mesh,
                        dim=dim,
                        value=list(range(dim)),
                        components=invalid_components[:dim],
                    )

            # wrong number of components
            with pytest.raises(ValueError):
                df.Field(mesh, dim=3, value=(1, 1, 1), components=valid_components)
            with pytest.raises(ValueError):
                df.Field(mesh, dim=3, value=(1, 1, 1), components=["x", "y"])

            # components not unique
            with pytest.raises(ValueError):
                df.Field(mesh, dim=3, value=(1, 1, 1), components=["x", "y", "x"])

            # test lshift
            f1 = df.Field(mesh, dim=1, value=1)
            f2 = df.Field(mesh, dim=1, value=2)
            f3 = df.Field(mesh, dim=1, value=3)

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

            fa = df.Field(mesh, dim=1, value=10, components=["a"])
            fb = df.Field(mesh, dim=1, value=20, components=["b"])

            # default components if not all fields have component labels
            f1a = f1 << fa
            check_field(f1a)
            assert f1a.components == ["x", "y"]

            # custom components if all fields have custom components
            fab = fa << fb
            check_field(fab)
            assert fab.components == ["a", "b"]

    def test_units(self):
        assert self.pf.units is None
        mesh = self.pf.mesh
        field = df.Field(mesh, dim=3, value=(1, 2, 3), units="A/m")
        check_field(field)
        assert field.units == "A/m"
        field.units = "mT"
        assert field.units == "mT"
        with pytest.raises(TypeError):
            field.units = 3
        assert field.units == "mT"
        field.units = None
        assert field.units is None

        with pytest.raises(TypeError):
            df.Field(mesh, dim=1, units=1)

    def test_value(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, dim=3)
        f.value = (1, 1, 1)

        assert f.value == (1, 1, 1)

        f.array[0, 0, 0, 0] = 3
        assert isinstance(f.value, np.ndarray)

    def test_norm(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
        f = df.Field(mesh, dim=3, value=(2, 2, 2))

        assert np.all(f.norm.value == 2 * np.sqrt(3))
        assert np.all(f.norm.array == 2 * np.sqrt(3))
        assert np.all(f.array == 2)

        f.norm = 1
        assert np.all(f.norm.value == 1)
        assert np.all(f.norm.array == 1)
        assert np.all(f.array == 1 / np.sqrt(3))

        f.array[0, 0, 0, 0] = 3
        assert isinstance(f.norm.value, np.ndarray)
        assert not np.all(f.norm.value == 1)

        for mesh in self.meshes:
            for value, dtype in self.iters + self.vfuncs:
                for norm_value in [1, 2.1, 50, 1e-3, np.pi]:
                    f = df.Field(mesh, dim=3, value=value, norm=norm_value, dtype=dtype)

                    # TODO: Why is this included?
                    # Compute norm.
                    norm = f.array[..., 0] ** 2
                    norm += f.array[..., 1] ** 2
                    norm += f.array[..., 2] ** 2
                    norm = np.sqrt(norm)

                    assert norm.shape == f.mesh.n
                    assert f.norm.array.shape == (*f.mesh.n, 1)
                    assert np.all(abs(f.norm.array - norm_value) < 1e-12)

        # Exception
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(1, 1, 1))
        f = df.Field(mesh, dim=1, value=-5)
        with pytest.raises(ValueError):
            f.norm = 5

    def test_norm_is_not_preserved(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, dim=3)

        f.value = (0, 3, 0)
        f.norm = 1
        assert np.all(f.norm.array == 1)

        f.value = (0, 2, 0)
        assert np.all(f.norm.value != 1)
        assert np.all(f.norm.array == 2)

    def test_norm_zero_field_exception(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, dim=3, value=(0, 0, 0))
        with pytest.raises(ValueError):
            f.norm = 1

    def test_zero(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f = df.Field(mesh, dim=1, value=1e-6)
        zf = f.zero

        assert f.mesh == zf.mesh
        assert f.dim == zf.dim
        assert not np.any(zf.array)

        f = df.Field(mesh, dim=3, value=(5, -7, 1e3))
        zf = f.zero

        assert f.mesh == zf.mesh
        assert f.dim == zf.dim
        assert not np.any(zf.array)

    def test_orientation(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (5e-9, 5e-9, 5e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # No zero-norm cells
        f = df.Field(mesh, dim=3, value=(2, 0, 0))
        assert f.orientation.average == (1, 0, 0)

        # With zero-norm cells
        def value_fun(point):
            x, y, z = point
            if x <= 0:
                return (0, 0, 0)
            else:
                return (3, 0, 4)

        f = df.Field(mesh, dim=3, value=value_fun)
        assert np.allclose(f.orientation((-1.5e-9, 3e-9, 0)), (0, 0, 0))
        assert np.allclose(f.orientation((1.5e-9, 3e-9, 0)), (0.6, 0, 0.8))

        f = df.Field(mesh, dim=1, value=0)
        with pytest.raises(ValueError):
            f.orientation

    def test_average(self):
        tol = 1e-12

        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (5e-9, 5e-9, 5e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, dim=1, value=2)
        assert abs(f.average - 2) < tol

        f = df.Field(mesh, dim=3, value=(0, 1, 2))
        assert np.allclose(f.average, (0, 1, 2))

    def test_field_component(self):
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3, value=(1, 2, 3))
            assert all(isinstance(getattr(f, i), df.Field) for i in "xyz")
            assert all(getattr(f, i).dim == 1 for i in "xyz")

            f = df.Field(mesh, dim=2, value=(1, 2))
            assert all(isinstance(getattr(f, i), df.Field) for i in "xy")
            assert all(getattr(f, i).dim == 1 for i in "xy")

            # Exception.
            f = df.Field(mesh, dim=1, value=1)
            with pytest.raises(AttributeError):
                f.x.dim

    def test_get_attribute_exception(self):
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3)
            with pytest.raises(AttributeError) as excinfo:
                f.__getattr__("nonexisting_attribute")
            assert "has no attribute" in str(excinfo.value)

    def test_dir(self):
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3, value=(5, 6, -9))
            assert all(attr in dir(f) for attr in ["x", "y", "z", "div"])
            assert "grad" not in dir(f)

            f = df.Field(mesh, dim=1, value=1)
            assert all(attr not in dir(f) for attr in ["x", "y", "z", "div"])
            assert "grad" in dir(f)

    def test_eq(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (15e-9, 5e-9, 5e-9)
        cell = (5e-9, 1e-9, 2.5e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f1 = df.Field(mesh, dim=1, value=0.2)
        f2 = df.Field(mesh, dim=1, value=0.2)
        f3 = df.Field(mesh, dim=1, value=3.1)
        f4 = df.Field(mesh, dim=3, value=(1, -6, 0))
        f5 = df.Field(mesh, dim=3, value=(1, -6, 0))

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

        f1 = df.Field(mesh, dim=1, value=0.2)
        f2 = df.Field(mesh, dim=1, value=0.2 + 1e-9)
        f3 = df.Field(mesh, dim=1, value=0.21)
        f4 = df.Field(mesh, dim=3, value=(1, -6, 0))
        f5 = df.Field(mesh, dim=3, value=(1, -6 + 1e-8, 0))
        f6 = df.Field(mesh, dim=3, value=(1, -6.01, 0))

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
        f = df.Field(mesh, dim=1, value=3)
        res = -f
        check_field(res)
        assert res.average == -3
        assert f == +f
        assert f == -(-f)
        assert f == +(-(-f))

        # Vector field
        f = df.Field(mesh, dim=3, value=(1, 2, -3))
        res = -f
        check_field(res)
        assert res.average == (-1, -2, 3)
        assert f == +f
        assert f == -(-f)
        assert f == +(-(-f))

    def test_pow(self):
        p1 = (0, 0, 0)
        p2 = (15e-9, 6e-9, 6e-9)
        cell = (3e-9, 3e-9, 3e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Scalar field
        f = df.Field(mesh, dim=1, value=2)
        res = f**2
        assert res.average == 4
        res = f ** (-1)
        assert res.average == 0.5

        # Attempt vector field
        f = df.Field(mesh, dim=3, value=(1, 2, -2))
        with pytest.raises(ValueError):
            res = f**2

        # Attempt to raise to non numbers.Real
        f = df.Field(mesh, dim=1, value=2)
        with pytest.raises(TypeError):
            res = f ** "a"
        with pytest.raises(TypeError):
            res = f**f

    def test_add_subtract(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 10e-9, -5e-9)
        n = (2, 2, 1)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        # Scalar fields
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=1, value=-0.2)
        res = f1 + f2
        assert res.average == 1
        res = f1 - f2
        assert res.average == 1.4
        f1 += f2
        assert f1.average == 1
        f1 -= f2
        assert f1.average == 1.2

        # Vector fields
        f1 = df.Field(mesh, dim=3, value=(1, 2, 3))
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        res = f1 + f2
        assert res.average == (0, -1, -2)
        res = f1 - f2
        assert res.average == (2, 5, 8)
        f1 += f2
        assert f1.average == (0, -1, -2)
        f1 -= f2
        assert f1.average == (1, 2, 3)

        # Artithmetic checks
        assert f1 + f2 + (1, 1, 1) == (1, 1, 1) + f2 + f1
        assert f1 - f2 - (0, 0, 0) == (0, 0, 0) - (f2 - f1)
        assert f1 + (f1 + f2) == (f1 + f1) + f2
        assert f1 - (f1 + f2) == f1 - f1 - f2
        assert f1 + f2 - f1 == f2 + (0, 0, 0)

        # Constants
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        res = f1 + 2
        assert res.average == 3.2
        res = f1 - 1.2
        assert res.average == 0
        f1 += 2.5
        assert f1.average == 3.7
        f1 -= 3.7
        assert f1.average == 0
        res = f2 + (1, 3, 5)
        assert res.average == (0, 0, 0)
        res = f2 - (1, 2, 3)
        assert res.average == (-2, -5, -8)
        f2 += (1, 1, 1)
        assert f2.average == (0, -2, -4)
        f2 -= (-1, -2, 3)
        assert f2.average == (1, 0, -7)

        # Exceptions
        with pytest.raises(TypeError):
            res = f1 + "2"

        # Fields with different dimensions
        with pytest.raises(ValueError):
            res = f1 + f2

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, dim=1, value=1.2)
        f2 = df.Field(mesh2, dim=1, value=1)
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
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=1, value=-2)
        res = f1 * f2
        assert res.average == -2.4
        res = f1 / f2
        assert res.average == -0.6
        f1 *= f2
        assert f1.average == -2.4
        f1 /= f2
        assert f1.average == 1.2

        # Scalar field with a constant
        f = df.Field(mesh, dim=1, value=5)
        res = f * 2
        assert res.average == 10
        res = 3 * f
        assert res.average == 15
        res = f * (1, 2, 3)
        assert res.average == (5, 10, 15)
        res = (1, 2, 3) * f
        assert res.average == (5, 10, 15)
        res = f / 2
        assert res.average == 2.5
        res = 10 / f
        assert res.average == 2
        res = (5, 10, 15) / f
        assert res.average == (1, 2, 3)
        f *= 10
        assert f.average == 50
        f /= 10
        assert f.average == 5

        # Scalar field with a vector field
        f1 = df.Field(mesh, dim=1, value=2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, 5))
        res = f1 * f2  # __mul__
        assert res.average == (-2, -6, 10)
        res = f2 * f1  # __rmul__
        assert res.average == (-2, -6, 10)
        res = f2 / f1  # __truediv__
        assert res.average == (-0.5, -1.5, 2.5)
        f2 *= f1  # __imul__
        assert f2.average == (-2, -6, 10)
        f2 /= f1  # __truediv__
        assert f2.average == (-1, -3, 5)
        with pytest.raises(ValueError):
            res = f1 / f2  # __rtruediv__

        # Vector field with a scalar
        f = df.Field(mesh, dim=3, value=(1, 2, 0))
        res = f * 2
        assert res.average == (2, 4, 0)
        res = 5 * f
        assert res.average == (5, 10, 0)
        res = f / 2
        assert res.average == (0.5, 1, 0)
        f *= 2
        assert f.average == (2, 4, 0)
        f /= 2
        assert f.average == (1, 2, 0)
        with pytest.raises(ValueError):
            res = 10 / f

        # Further checks
        f1 = df.Field(mesh, dim=1, value=2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        assert f1 * f2 == f2 * f1
        assert 1.3 * f2 == f2 * 1.3
        assert -5 * f2 == f2 * (-5)
        assert (1, 2.2, -1) * f1 == f1 * (1, 2.2, -1)
        assert f1 * (f1 * f2) == (f1 * f1) * f2
        assert f1 * f2 / f1 == f2

        # Exceptions
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        with pytest.raises(TypeError):
            res = f2 * "a"
        with pytest.raises(TypeError):
            res = "a" / f1
        with pytest.raises(ValueError):
            res = f2 * f2
        with pytest.raises(ValueError):
            res = f2 / f2
        with pytest.raises(ValueError):
            res = 1 / f2
        with pytest.raises(ValueError):
            res = f1 / f2
        with pytest.raises(TypeError):
            f2 *= "a"
        with pytest.raises(TypeError):
            f2 /= "a"
        with pytest.raises(ValueError):
            f1 /= f2

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, dim=1, value=1.2)
        f2 = df.Field(mesh2, dim=1, value=1)
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
        f1 = df.Field(mesh, dim=3, value=(0, 0, 0))
        res = f1 @ f1
        assert res.dim == 1
        assert res.average == 0

        # Orthogonal vectors
        f1 = df.Field(mesh, dim=3, value=(1, 0, 0))
        f2 = df.Field(mesh, dim=3, value=(0, 1, 0))
        f3 = df.Field(mesh, dim=3, value=(0, 0, 1))
        assert (f1 @ f2).average == 0
        assert (f1 @ f3).average == 0
        assert (f2 @ f3).average == 0
        assert (f1 @ f1).average == 1
        assert (f2 @ f2).average == 1
        assert (f3 @ f3).average == 1

        # Check if commutative
        assert f1 @ f2 == f2 @ f1
        assert f1 @ (-1, 3, 2.2) == (-1, 3, 2.2) @ f1

        # Vector field with a constant
        f = df.Field(mesh, dim=3, value=(1, 2, 3))
        res = (1, 1, 1) @ f
        assert res.average == 6
        res = f @ [1, 1, 1]
        assert res.average == 6

        # Spatially varying vectors
        def value_fun1(point):
            x, y, z = point
            return (x, y, z)

        def value_fun2(point):
            x, y, z = point
            return (z, x, y)

        f1 = df.Field(mesh, dim=3, value=value_fun1)
        f2 = df.Field(mesh, dim=3, value=value_fun2)

        # Check if commutative
        assert f1 @ f2 == f2 @ f1

        # The dot product should be x*z + y*x + z*y
        assert (f1 @ f2)((1, 1, 1)) == 3
        assert (f1 @ f2)((3, 1, 1)) == 7
        assert (f1 @ f2)((5, 7, 1)) == 47

        # Check norm computed using dot product
        assert f1.norm == (f1 @ f1) ** (0.5)

        # Exceptions
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        with pytest.raises(ValueError):
            res = f1 @ f2
        with pytest.raises(ValueError):
            res = f1 @ f2
        with pytest.raises(TypeError):
            res = f1 @ 3

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, dim=3, value=(1, 2, 3))
        f2 = df.Field(mesh2, dim=3, value=(3, 2, 1))
        with pytest.raises(ValueError):
            res = f1 @ f2

    def test_cross(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Zero vectors
        f1 = df.Field(mesh, dim=3, value=(0, 0, 0))
        res = f1 & f1
        assert res.dim == 3
        assert res.average == (0, 0, 0)

        # Orthogonal vectors
        f1 = df.Field(mesh, dim=3, value=(1, 0, 0))
        f2 = df.Field(mesh, dim=3, value=(0, 1, 0))
        f3 = df.Field(mesh, dim=3, value=(0, 0, 1))
        assert (f1 & f2).average == (0, 0, 1)
        assert (f1 & f3).average == (0, -1, 0)
        assert (f2 & f3).average == (1, 0, 0)
        assert (f1 & f1).average == (0, 0, 0)
        assert (f2 & f2).average == (0, 0, 0)
        assert (f3 & f3).average == (0, 0, 0)

        # Constants
        assert (f1 & (0, 1, 0)).average == (0, 0, 1)
        assert ((0, 1, 0) & f1).average == (0, 0, 1)

        # Check if not comutative
        assert f1 & f2 == -(f2 & f1)
        assert f1 & f3 == -(f3 & f1)
        assert f2 & f3 == -(f3 & f2)

        f1 = df.Field(mesh, dim=3, value=lambda point: (point[0], point[1], point[2]))
        f2 = df.Field(mesh, dim=3, value=lambda point: (point[2], point[0], point[1]))

        # The cross product should be
        # (y**2-x*z, z**2-x*y, x**2-y*z)
        assert (f1 & f2)((1, 1, 1)) == (0, 0, 0)
        assert (f1 & f2)((3, 1, 1)) == (-2, -2, 8)
        assert (f2 & f1)((3, 1, 1)) == (2, 2, -8)
        assert (f1 & f2)((5, 7, 1)) == (44, -34, 18)

        # Exceptions
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        with pytest.raises(TypeError):
            res = f1 & 2
        with pytest.raises(ValueError):
            res = f1 & f2

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, dim=3, value=(1, 2, 3))
        f2 = df.Field(mesh2, dim=3, value=(3, 2, 1))
        with pytest.raises(ValueError):
            res = f1 & f2

    def test_lshift(self):
        p1 = (0, 0, 0)
        p2 = (10e6, 10e6, 10e6)
        cell = (5e6, 5e6, 5e6)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f1 = df.Field(mesh, dim=1, value=1)
        f2 = df.Field(mesh, dim=1, value=-3)
        f3 = df.Field(mesh, dim=1, value=5)

        res = f1 << f2 << f3
        assert res.dim == 3
        assert res.average == (1, -3, 5)

        # Different dimensions
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=2, value=(-1, -3))
        res = f1 << f2
        assert res.average == (1.2, -1, -3)
        res = f2 << f1
        assert res.average == (-1, -3, 1.2)

        # Constants
        f1 = df.Field(mesh, dim=1, value=1.2)
        res = f1 << 2
        assert res.average == (1.2, 2)
        res = f1 << (1, -1)
        assert res.average == (1.2, 1, -1)
        res = 3 << f1
        assert res.average == (3, 1.2)
        res = (1.2, 3) << f1 << 3
        assert res.average == (1.2, 3, 1.2, 3)

        # Exceptions
        with pytest.raises(TypeError):
            res = "a" << f1
        with pytest.raises(TypeError):
            res = f1 << "a"

        # Fields defined on different meshes
        mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
        mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
        f1 = df.Field(mesh1, dim=1, value=1.2)
        f2 = df.Field(mesh2, dim=1, value=1)
        with pytest.raises(ValueError):
            res = f1 << f2

    def test_all_operators(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 5e-9, 10e-9)
        n = (2, 2, 1)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f1 = df.Field(mesh, dim=1, value=2)
        f2 = df.Field(mesh, dim=3, value=(-4, 0, 1))
        res = (
            ((+f1 / 2 + f2.x) ** 2 - 2 * f1 * 3) / (-f2.z)
            - 2 * f2.y
            + 1 / f2.z**2
            + f2 @ f2
        )
        assert np.all(res.array == 21)

        res = 1 + f1 + 0 * f2.x - 3 * f2.y / 3
        assert res.average == 3

    def test_pad(self):
        p1 = (0, 0, 0)
        p2 = (10, 8, 2)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        field = df.Field(mesh, dim=1, value=1)

        pf = field.pad({"x": (1, 1)}, mode="constant")  # zeros padded
        assert pf.array.shape == (12, 8, 2, 1)

    def test_derivative(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)

        # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, dim=1, value=0)

        check_field(f.derivative("x"))
        assert f.derivative("x", n=1).average == 0
        assert f.derivative("y", n=1).average == 0
        assert f.derivative("z", n=1).average == 0
        assert f.derivative("x", n=2).average == 0
        assert f.derivative("y", n=2).average == 0
        assert f.derivative("z", n=2).average == 0

        # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.derivative("x", n=1).average == 1
        assert f.derivative("y", n=1).average == 1
        assert f.derivative("z", n=1).average == 1
        assert f.derivative("x", n=2).average == 0
        assert f.derivative("y", n=2).average == 0
        assert f.derivative("z", n=2).average == 0

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        def value_fun(point):
            x, y, z = point
            return x * y + 2 * y + x * y * z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.derivative("x")((7, 5, 1)) == 10
        assert f.derivative("y")((7, 5, 1)) == 16
        assert f.derivative("z")((7, 5, 1)) == 35
        assert f.derivative("x", n=2)((1, 1, 1)) == 0
        assert f.derivative("y", n=2)((1, 1, 1)) == 0
        assert f.derivative("z", n=2)((1, 1, 1)) == 0

        # f(x, y, z) = (0, 0, 0)
        # -> dfdx = (0, 0, 0)
        # -> dfdy = (0, 0, 0)
        # -> dfdz = (0, 0, 0)
        # No BC
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, dim=3, value=(0, 0, 0))

        check_field(f.derivative("y"))
        assert f.derivative("x").average == (0, 0, 0)
        assert f.derivative("y").average == (0, 0, 0)
        assert f.derivative("z").average == (0, 0, 0)

        # f(x, y, z) = (x,  y,  z)
        # -> dfdx = (1, 0, 0)
        # -> dfdy = (0, 1, 0)
        # -> dfdz = (0, 0, 1)
        def value_fun(point):
            x, y, z = point
            return (x, y, z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.derivative("x").average == (1, 0, 0)
        assert f.derivative("y").average == (0, 1, 0)
        assert f.derivative("z").average == (0, 0, 1)

        # f(x, y, z) = (x*y, y*z, x*y*z)
        # -> dfdx = (y, 0, y*z)
        # -> dfdy = (x, z, x*z)
        # -> dfdz = (0, y, x*y)
        def value_fun(point):
            x, y, z = point
            return (x * y, y * z, x * y * z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.derivative("x")((3, 1, 3)) == (1, 0, 3)
        assert f.derivative("y")((3, 1, 3)) == (3, 3, 9)
        assert f.derivative("z")((3, 1, 3)) == (0, 1, 3)
        assert f.derivative("x")((5, 3, 5)) == (3, 0, 15)
        assert f.derivative("y")((5, 3, 5)) == (5, 5, 25)
        assert f.derivative("z")((5, 3, 5)) == (0, 3, 15)

        # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
        # -> dfdx = (y, 1, y*z)
        # -> dfdy = (x, -2, x*z)
        # -> dfdz = (0, 0, x*y)
        def value_fun(point):
            x, y, z = point
            return (3 + x * y, x - 2 * y, x * y * z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.derivative("x")((7, 5, 1)) == (5, 1, 5)
        assert f.derivative("y")((7, 5, 1)) == (7, -2, 7)
        assert f.derivative("z")((7, 5, 1)) == (0, 0, 35)

        # f(x, y, z) = 2*x*x + 2*y*y + 3*z*z
        # -> grad(f) = (4, 4, 6)
        def value_fun(point):
            x, y, z = point
            return 2 * x * x + 2 * y * y + 3 * z * z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.derivative("x", n=2).average == 4
        assert f.derivative("y", n=2).average == 4
        assert f.derivative("z", n=2).average == 6

        # f(x, y, z) = (2*x*x, 2*y*y, 3*z*z)
        def value_fun(point):
            x, y, z = point
            return (2 * x * x, 2 * y * y, 3 * z * z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.derivative("x", n=2).average == (4, 0, 0)
        assert f.derivative("y", n=2).average == (0, 4, 0)
        assert f.derivative("z", n=2).average == (0, 0, 6)

        with pytest.raises(NotImplementedError):
            f.derivative("x", n=3)

    def test_derivative_pbc(self):
        p1 = (0, 0, 0)
        p2 = (10, 8, 6)
        cell = (2, 2, 2)

        mesh_nopbc = df.Mesh(p1=p1, p2=p2, cell=cell)
        mesh_pbc = df.Mesh(p1=p1, p2=p2, cell=cell, bc="xyz")

        # Scalar field
        def value_fun(point):
            return point[0] * point[1] * point[2]

        # No PBC
        f = df.Field(mesh_nopbc, dim=1, value=value_fun)
        assert f.derivative("x")((9, 1, 1)) == 1
        assert f.derivative("y")((1, 7, 1)) == 1
        assert f.derivative("z")((1, 1, 5)) == 1

        # PBC
        f = df.Field(mesh_pbc, dim=1, value=value_fun)
        assert f.derivative("x")((9, 1, 1)) == -1.5
        assert f.derivative("y")((1, 7, 1)) == -1
        assert f.derivative("z")((1, 1, 5)) == -0.5

        # Vector field
        def value_fun(point):
            return (point[0] * point[1] * point[2],) * 3

        # No PBC
        f = df.Field(mesh_nopbc, dim=3, value=value_fun)
        assert f.derivative("x")((9, 1, 1)) == (1, 1, 1)
        assert f.derivative("y")((1, 7, 1)) == (1, 1, 1)
        assert f.derivative("z")((1, 1, 5)) == (1, 1, 1)

        # PBC
        f = df.Field(mesh_pbc, dim=3, value=value_fun)
        assert f.derivative("x")((9, 1, 1)) == (-1.5, -1.5, -1.5)
        assert f.derivative("y")((1, 7, 1)) == (-1, -1, -1)
        assert f.derivative("z")((1, 1, 5)) == (-0.5, -0.5, -0.5)

    def test_derivative_neumann(self):
        p1 = (0, 0, 0)
        p2 = (10, 8, 6)
        cell = (2, 2, 2)

        mesh_noneumann = df.Mesh(p1=p1, p2=p2, cell=cell)
        mesh_neumann = df.Mesh(p1=p1, p2=p2, cell=cell, bc="neumann")

        # Scalar field
        def value_fun(point):
            return point[0] * point[1] * point[2]

        # No Neumann
        f1 = df.Field(mesh_noneumann, dim=1, value=value_fun)
        assert f1.derivative("x")((9, 1, 1)) == 1
        assert f1.derivative("y")((1, 7, 1)) == 1
        assert f1.derivative("z")((1, 1, 5)) == 1

        # Neumann
        f2 = df.Field(mesh_neumann, dim=1, value=value_fun)
        assert f1.derivative("x")(f1.mesh.region.centre) == f2.derivative("x")(
            f2.mesh.region.centre
        )
        assert f1.derivative("x")((1, 7, 1)) != f2.derivative("x")((1, 7, 1))

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

        f = df.Field(mesh, dim=1, value=value_fun)

        # only one cell in the z-direction
        assert f.plane("x").derivative("x").average == 0
        assert f.plane("y").derivative("y").average == 0
        assert f.derivative("z").average == 0

        # Vector field: f(x, y, z) = (x, y, z)
        # -> grad(f) = (1, 1, 1)
        def value_fun(point):
            x, y, z = point
            return (x, y, z)

        f = df.Field(mesh, dim=3, value=value_fun)

        # only one cell in the z-direction
        assert f.plane("x").derivative("x").average == (0, 0, 0)
        assert f.plane("y").derivative("y").average == (0, 0, 0)
        assert f.derivative("z").average == (0, 0, 0)

    def test_grad(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
        f = df.Field(mesh, dim=1, value=0)

        check_field(f.grad)
        assert f.grad.average == (0, 0, 0)

        # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.grad.average == (1, 1, 1)

        # f(x, y, z) = x*y + y + z -> grad(f) = (y, x+1, 1)
        def value_fun(point):
            x, y, z = point
            return x * y + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.grad((3, 1, 3)) == (1, 4, 1)
        assert f.grad((5, 3, 5)) == (3, 6, 1)

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        def value_fun(point):
            x, y, z = point
            return x * y + 2 * y + x * y * z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.grad((7, 5, 1)) == (10, 16, 35)
        assert f.grad.x == f.derivative("x")
        assert f.grad.y == f.derivative("y")
        assert f.grad.z == f.derivative("z")

        # Exception
        f = df.Field(mesh, dim=3, value=(1, 2, 3))

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
        f = df.Field(mesh, dim=3, value=(0, 0, 0))

        check_field(f.div)
        assert f.div.dim == 1
        assert f.div.average == 0

        check_field(f.curl)
        assert f.curl.dim == 3
        assert f.curl.average == (0, 0, 0)

        # f(x, y, z) = (x, y, z)
        # -> div(f) = 3
        # -> curl(f) = (0, 0, 0)
        def value_fun(point):
            x, y, z = point
            return (x, y, z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.div.average == 3
        assert f.curl.average == (0, 0, 0)

        # f(x, y, z) = (x*y, y*z, x*y*z)
        # -> div(f) = y + z + x*y
        # -> curl(f) = (x*z-y, -y*z, -x)
        def value_fun(point):
            x, y, z = point
            return (x * y, y * z, x * y * z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.div((3, 1, 3)) == 7
        assert f.div((5, 3, 5)) == 23

        assert f.curl((3, 1, 3)) == (8, -3, -3)
        assert f.curl((5, 3, 5)) == (22, -15, -5)

        # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
        # -> div(f) = y - 2 + x*y
        # -> curl(f) = (x*z, -y*z, 1-x)
        def value_fun(point):
            x, y, z = point
            return (3 + x * y, x - 2 * y, x * y * z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.div((7, 5, 1)) == 38
        assert f.curl((7, 5, 1)) == (7, -5, -6)

        # Exception
        f = df.Field(mesh, dim=1, value=3.11)

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
        f = df.Field(mesh, dim=3, value=(0, 0, 0))

        check_field(f.laplace)
        assert f.laplace.dim == 3
        assert f.laplace.average == (0, 0, 0)

        # f(x, y, z) = x + y + z
        # -> laplace(f) = 0
        def value_fun(point):
            x, y, z = point
            return x + y + z

        f = df.Field(mesh, dim=1, value=value_fun)
        check_field(f.laplace)
        assert f.laplace.average == 0

        # f(x, y, z) = 2*x*x + 2*y*y + 3*z*z
        # -> laplace(f) = 4 + 4 + 6 = 14
        def value_fun(point):
            x, y, z = point
            return 2 * x * x + 2 * y * y + 3 * z * z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.laplace.average == 14

        # f(x, y, z) = (2*x*x, 2*y*y, 3*z*z)
        # -> laplace(f) = (4, 4, 6)
        def value_fun(point):
            x, y, z = point
            return (2 * x * x, 2 * y * y, 3 * z * z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.laplace.average == (4, 4, 6)

    def test_integral(self):
        # Volume integral.
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, dim=1, value=0)
        assert (f * df.dV).integral() == 0
        assert (f * df.dx * df.dy * df.dz).integral() == 0

        f = df.Field(mesh, dim=1, value=2)
        assert (f * df.dV).integral() == 2000
        assert (f * df.dx * df.dy * df.dz).integral() == 2000

        f = df.Field(mesh, dim=3, value=(-1, 0, 3))
        assert (f * df.dV).integral() == (-1000, 0, 3000)
        assert (f * df.dx * df.dy * df.dz).integral() == (-1000, 0, 3000)

        def value_fun(point):
            x, y, z = point
            if x <= 5:
                return (-1, -2, -3)
            else:
                return (1, 2, 3)

        f = df.Field(mesh, dim=3, value=value_fun)
        assert (f * df.dV).integral() == (0, 0, 0)
        assert (f * df.dx * df.dy * df.dz).integral() == (0, 0, 0)

        # Surface integral.
        p1 = (0, 0, 0)
        p2 = (10, 5, 3)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, dim=1, value=0)
        assert (f.plane("x") * abs(df.dS)).integral() == 0
        assert (f.plane("x") * df.dy * df.dz).integral() == 0

        f = df.Field(mesh, dim=1, value=2)
        assert (f.plane("x") * abs(df.dS)).integral() == 30
        assert (f.plane("x") * df.dy * df.dz).integral() == 30
        assert (f.plane("y") * abs(df.dS)).integral() == 60
        assert (f.plane("y") * df.dx * df.dz).integral() == 60
        assert (f.plane("z") * abs(df.dS)).integral() == 100
        assert (f.plane("z") * df.dx * df.dy).integral() == 100

        f = df.Field(mesh, dim=3, value=(-1, 0, 3))
        assert (f.plane("x") * abs(df.dS)).integral() == (-15, 0, 45)
        assert (f.plane("y") * abs(df.dS)).integral() == (-30, 0, 90)
        assert (f.plane("z") * abs(df.dS)).integral() == (-50, 0, 150)

        f = df.Field(mesh, dim=3, value=(-1, 0, 3))
        assert df.integral(f.plane("x") @ df.dS) == -15
        assert df.integral(f.plane("y") @ df.dS) == 0
        assert df.integral(f.plane("z") @ df.dS) == 150

        # Directional integral
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, dim=3, value=(1, 1, 1))

        f = f.integral(direction="x")
        assert isinstance(f, df.Field)
        assert f.dim == 3
        assert f.mesh.n == (1, 10, 10)
        assert f.average == (10, 10, 10)

        f = f.integral(direction="x").integral(direction="y")
        assert isinstance(f, df.Field)
        assert f.dim == 3
        assert f.mesh.n == (1, 1, 10)
        assert f.average == (100, 100, 100)

        f = f.integral("x").integral("y").integral("z")
        assert f.dim == 3
        assert f.mesh.n == (1, 1, 1)
        assert f.average == (1000, 1000, 1000)

        assert f.integral("x").integral("y").integral("z").average == f.integral()

        # Improper integral
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        f = df.Field(mesh, dim=3, value=(1, 1, 1))

        f = f.integral(direction="x", improper=True)
        assert isinstance(f, df.Field)
        assert f.dim == 3
        assert f.mesh.n == (10, 10, 10)
        assert f.average == (5.5, 5.5, 5.5)
        assert f((0, 0, 0)) == (1, 1, 1)
        assert f((10, 10, 10)) == (10, 10, 10)

        # Exceptions
        with pytest.raises(ValueError):
            f.integral(direction="xy", improper=True)

    def test_line(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), n=(10, 10, 10))
        f = df.Field(mesh, dim=3, value=(1, 2, 3))
        check_field(f)

        line = f.line(p1=(0, 0, 0), p2=(5, 5, 5), n=20)
        assert isinstance(line, df.Line)

        assert line.n == 20
        assert line.dim == 3

    def test_plane(self):
        for mesh, direction in itertools.product(self.meshes, ["x", "y", "z"]):
            f = df.Field(mesh, dim=1, value=3)
            check_field(f)
            plane = f.plane(direction, n=(3, 3))
            assert isinstance(plane, df.Field)

            p, v = zip(*list(plane))
            assert len(p) == 9
            assert len(v) == 9

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

        f = df.Field(mesh, dim=3, value=value_fun)
        check_field(f)
        check_field(f["r1"])
        check_field(f["r2"])
        check_field(f[subregions["r1"]])
        check_field(f[subregions["r2"]])

        assert f["r1"].average == (-1, -2, -3)
        assert f["r2"].average == (0, 0, 0)
        assert f[subregions["r1"]].average == (-1, -2, -3)
        assert f[subregions["r2"]].average == (0, 0, 0)

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
        f = df.Field(mesh, dim=1, value=5)
        check_field(f)
        assert f.project("x").array.shape == (1, 10, 10, 1)
        assert f.project("y").array.shape == (10, 1, 10, 1)
        assert f.project("z").array.shape == (10, 10, 1, 1)

        # Constant vector field
        f = df.Field(mesh, dim=3, value=(1, 2, 3))
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

        f = df.Field(mesh, dim=1, value=value_fun)
        sf = f.project("z")
        assert sf.array.shape == (10, 10, 1, 1)
        assert sf.average == 0

        # Spatially varying vector field
        def value_fun(point):
            x, y, z = point
            if z <= 0:
                return (3, 2, 1)
            else:
                return (3, 2, -1)

        f = df.Field(mesh, dim=3, value=value_fun)
        sf = f.project("z")
        assert sf.array.shape == (10, 10, 1, 3)
        assert sf.average == (3, 2, 0)

    def test_angle(self):
        p1 = (0, 0, 0)
        p2 = (8e-9, 2e-9, 2e-9)
        cell = (2e-9, 2e-9, 2e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        def value_fun(point):
            x, y, z = point
            if x < 2e-9:
                return (1, 1, 1)
            elif 2e-9 <= x < 4e-9:
                return (1, -1, 0)
            elif 4e-9 <= x < 6e-9:
                return (-1, -1, 0)
            elif 6e-9 <= x < 8e-9:
                return (-1, 1, 0)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert abs(f.plane("z").angle((1e-9, 2e-9, 2e-9)) - np.pi / 4) < 1e-3
        assert abs(f.plane("z").angle((3e-9, 2e-9, 2e-9)) - 7 * np.pi / 4) < 1e-3
        assert abs(f.plane("z").angle((5e-9, 2e-9, 2e-9)) - 5 * np.pi / 4) < 1e-3
        assert abs(f.plane("z").angle((7e-9, 2e-9, 2e-9)) - 3 * np.pi / 4) < 1e-3

        # Exception
        with pytest.raises(ValueError):
            f.angle  # the field is not sliced

    def test_write_read_ovf(self):
        representations = ["txt", "bin4", "bin8"]
        filename = "testfile.ovf"
        p1 = (0, 0, 0)
        p2 = (8e-9, 5e-9, 3e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        # Write/read
        for dim, value in [
            (1, lambda point: point[0] + point[1] + point[2]),
            (2, lambda point: (point[0], point[1] + point[2])),
            (3, lambda point: (point[0], point[1], point[2])),
        ]:
            f = df.Field(mesh, dim=dim, value=value, units="A/m")
            for rep in representations:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpfilename = os.path.join(tmpdir, filename)
                    f.write(tmpfilename, representation=rep)
                    f_read = df.Field.fromfile(tmpfilename)

                    assert f.allclose(f_read)
                    assert f_read.units == "A/m"

            # Directly write with wrong representation (no data is written)
            with pytest.raises(ValueError):
                f._writeovf("fname.ovf", representation="bin5")

        # multiple different units (not supported by discretisedfield)
        f = df.Field(mesh, dim=3, value=(1, 1, 1), units="m s kg")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            f.write(tmpfilename, representation=rep)
            f_read = df.Field.fromfile(tmpfilename)

            assert f.allclose(f_read)
            assert f_read.units is None

        # Extend scalar
        for rep in representations:
            f = df.Field(
                mesh, dim=1, value=lambda point: point[0] + point[1] + point[2]
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfilename = os.path.join(tmpdir, filename)
                f.write(tmpfilename, representation=rep, extend_scalar=True)
                f_read = df.Field.fromfile(tmpfilename)

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
            f_read = df.Field.fromfile(omffilename)

            if "ovf2" in filename:
                # The magnetisation is in the x-direction in OVF2 files.
                assert abs(f_read.orientation.x.average - 1) < 1e-2
            else:
                # The norm of magnetisation is known.
                assert abs(f_read.norm.average - 1261566.2610100) < 1e-3

        # Read component names (single-word and multi-word with and without hyphen)
        # from OOMMF files
        assert df.Field.fromfile(
            os.path.join(dirname, "oommf-ovf2-bin8.omf")
        ).components == ["x", "y", "z"]
        assert df.Field.fromfile(
            os.path.join(dirname, "oommf-ovf2-bin8.ohf")
        ).components == ["x", "y", "z"]
        assert df.Field.fromfile(
            os.path.join(dirname, "oommf-ovf2-bin8.oef")
        ).components == ["Total_energy_density"]

        # Read different mumax3 bin4 files (made on linux and windows)
        filenames = ["mumax-bin4-linux.ovf", "mumax-bin4-windows.ovf"]
        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        for filename in filenames:
            omffilename = os.path.join(dirname, filename)
            f_read = df.Field.fromfile(omffilename)

            # We know the saved magentisation.
            f_saved = df.Field(f_read.mesh, dim=3, value=(1, 0.1, 0), norm=1)
            assert f_saved.allclose(f_read)

    def test_write_read_vtk(self, tmp_path):
        filename = "testfile.vtk"

        p1 = (0, 0, 0)
        p2 = (5e-9, 2e-9, 1e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        for dim, value, components in zip(
            [1, 2, 3, 4],
            [1.2, (1, 2.5), (1e-3, -5e6, 5e6), np.random.random(4)],
            [None, ("m_mag", "m_phase"), None, "abcd"],
        ):
            for representation in ["txt", "bin", "xml"]:
                f = df.Field(mesh, dim=dim, value=value, components=components)
                tmpfilename = str(tmp_path / filename)
                f.write(tmpfilename, representation=representation)
                f_read = df.Field.fromfile(tmpfilename)

                assert np.allclose(f.array, f_read.array)
                assert np.allclose(f.mesh.region.pmin, f_read.mesh.region.pmin)
                assert np.allclose(f.mesh.region.pmax, f_read.mesh.region.pmax)
                assert np.allclose(f.mesh.cell, f_read.mesh.cell)
                assert f.mesh.n == f_read.mesh.n
                assert f.components == f_read.components

        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        f = df.Field.fromfile(os.path.join(dirname, "vtk-file.vtk"))
        check_field(f)
        assert f.mesh.n == (5, 1, 2)
        assert f.array.shape == (5, 1, 2, 3)
        assert f.dim == 3

        # test reading legacy vtk file (written with discretisedfield<=0.61.0)
        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        f = df.Field.fromfile(os.path.join(dirname, "vtk-vector-legacy.vtk"))
        check_field(f)
        assert f.mesh.n == (8, 1, 1)
        assert f.array.shape == (8, 1, 1, 3)
        assert f.dim == 3

        dirname = os.path.join(os.path.dirname(__file__), "test_sample")
        f = df.Field.fromfile(os.path.join(dirname, "vtk-scalar-legacy.vtk"))
        check_field(f)
        assert f.mesh.n == (5, 1, 2)
        assert f.array.shape == (5, 1, 2, 1)
        assert f.dim == 1

        # test invalid arguments
        f = df.Field(mesh, dim=3, value=(0, 0, 1))
        with pytest.raises(ValueError):
            f.write(str(tmp_path / filename), representation="wrong")
        f._components = None  # manually remove component labels
        with pytest.raises(AttributeError):
            f.write(str(tmp_path / filename))

    def test_write_read_hdf5(self):
        filenames = ["testfile.hdf5", "testfile.h5"]

        p1 = (0, 0, 0)
        p2 = (10e-12, 5e-12, 5e-12)
        cell = (1e-12, 1e-12, 1e-12)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        for dim, value in [(1, -1.23), (3, (1e-3 + np.pi, -5e6, 6e6))]:
            f = df.Field(mesh, dim=dim, value=value)
            for filename in filenames:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpfilename = os.path.join(tmpdir, filename)
                    f.write(tmpfilename)
                    f_read = df.Field.fromfile(tmpfilename)

                    assert f == f_read

    def test_read_write_invalid_extension(self):
        filename = "testfile.jpg"

        p1 = (0, 0, 0)
        p2 = (10e-12, 5e-12, 3e-12)
        cell = (1e-12, 1e-12, 1e-12)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        f = df.Field(mesh, dim=1, value=5e-12)
        with pytest.raises(ValueError):
            f.write(filename)
        with pytest.raises(ValueError):
            df.Field.fromfile(filename)

    def test_fft(self):
        p1 = (-10, -10, -5)
        p2 = (10, 10, 5)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        def _init_random(p):
            return np.random.rand(3) * 2 - 1

        f = df.Field(mesh, dim=3, value=_init_random, norm=1)

        # 3d fft
        assert f.allclose(f.fftn.ifftn.real)
        assert df.Field(mesh, dim=3).allclose(f.fftn.ifftn.imag)

        assert f.allclose(f.rfftn.irfftn)

        # 2d fft
        for i in ["x", "y", "z"]:
            plane = f.plane(i)
            assert plane.allclose(plane.fftn.ifftn.real)
            assert df.Field(mesh, dim=3).plane(i).allclose(plane.fftn.ifftn.imag)

            assert plane.allclose(plane.rfftn.irfftn)

        # Fourier slice theoreme
        for i, di in [["x", df.dx], ["y", df.dy], ["z", df.dz]]:
            plane = (f * di).integral(i)
            assert plane.allclose(f.fftn.plane(**{i: 0}).ifftn.real)
            assert (
                (df.Field(mesh, dim=3) * df.dz)
                .integral(i)
                .allclose(f.fftn.plane(**{i: 0}).ifftn.imag)
            )

        assert (f * df.dx).integral("x").allclose(f.rfftn.plane(x=0).irfftn)
        assert (f * df.dy).integral("y").allclose(f.rfftn.plane(y=0).irfftn)
        # plane along z removes rfftn-freq axis => needs ifftn
        assert (f * df.dz).integral("z").allclose(f.rfftn.plane(z=0).ifftn.real)

    def test_mpl_scalar(self):
        # No axes
        self.pf.x.plane("x", n=(3, 4)).mpl.scalar()

        # Axes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.pf.x.plane("x", n=(3, 4)).mpl.scalar(ax=ax)

        # All arguments
        self.pf.x.plane("x").mpl.scalar(
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
            self.pf.x.plane("x", n=(3, 4)).mpl.scalar(filename=tmpfilename)

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.x.mpl.scalar()  # not sliced
        with pytest.raises(ValueError):
            self.pf.plane("z").mpl.scalar()  # vector field
        with pytest.raises(ValueError):
            # wrong filter field
            self.pf.x.plane("z").mpl.scalar(filter_field=self.pf)
        plt.close("all")

    def test_mpl_lightess(self):
        filenames = ["skyrmion.omf", "skyrmion-disk.omf"]
        for i in filenames:
            filename = os.path.join(os.path.dirname(__file__), "test_sample", i)

            field = df.Field.fromfile(filename)
            field.plane("z").mpl.lightness()
            field.plane("z").mpl.lightness(
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
        # TODO Filtering for lightness plots
        # with pytest.raises(ValueError) as excinfo:
        #     # wrong filter field
        #     self.pf.plane('z').mpl.lightness(filter_field=self.pf)
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
            color_field=self.pf.y,
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
            self.pf.plane("x", n=(3, 4)).mpl.vector(filename=tmpfilename)

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.mpl.vector()  # not sliced
        with pytest.raises(ValueError):
            self.pf.y.plane("z").mpl.vector()  # scalar field
        with pytest.raises(ValueError):
            # wrong color field
            self.pf.plane("z").mpl.vector(color_field=self.pf)

        plt.close("all")

    def test_mpl_contour(self):
        # No axes
        self.pf.plane("z").z.mpl.contour()

        # Axes
        fig, ax = plt.subplots()
        self.pf.plane("z").z.mpl.contour(ax=ax)

        # All arguments
        self.pf.plane("z").z.mpl.contour(
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
            self.pf.plane("z").z.mpl.contour(filename=tmpfilename)

        # Exceptions
        with pytest.raises(ValueError):
            self.pf.mpl.contour()  # not sliced
        with pytest.raises(ValueError):
            self.pf.plane("z").mpl.contour()  # vector field
        with pytest.raises(ValueError):
            # wrong filter field
            self.pf.plane("z").mpl.contour(filter_field=self.pf)

        plt.close("all")

    def test_mpl(self):
        # No axes
        self.pf.plane("x", n=(3, 4)).mpl()

        # Axes
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.pf.x.plane("x", n=(3, 4)).mpl(ax=ax)

        self.pf.z.plane("x").mpl(
            figsize=(12, 6),
            scalar_kw={
                "filter_field": self.pf.norm,
                "colorbar_label": "scalar",
                "cmap": "twilight",
            },
            vector_kw={
                "color_field": self.pf.y,
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

    def test_k3d_nonzero(self):
        # Default
        self.pf.norm.k3d.nonzero()

        # Color
        self.pf.x.k3d.nonzero(color=0xFF00FF)

        # Multiplier
        self.pf.x.k3d.nonzero(color=0xFF00FF, multiplier=1e-6)

        # Interactive field
        self.pf.x.plane("z").k3d.nonzero(
            color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf
        )

        # kwargs
        self.pf.x.plane("z").k3d.nonzero(
            color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf, wireframe=True
        )

        # Plot
        plot = k3d.plot()
        plot.display()
        self.pf.x.plane(z=0).k3d.nonzero(
            plot=plot, color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf
        )

        # Continuation for interactive plot testing.
        self.pf.x.plane(z=1e-9).k3d.nonzero(
            plot=plot, color=0xFF00FF, multiplier=1e-6, interactive_field=self.pf
        )

        assert len(plot.objects) == 2

        with pytest.raises(ValueError):
            self.pf.k3d.nonzero()

    def test_k3d_scalar(self):
        # Default
        self.pf.y.k3d.scalar()

        # Filter field
        self.pf.y.k3d.scalar(filter_field=self.pf.norm)

        # Colormap
        self.pf.x.k3d.scalar(filter_field=self.pf.norm, cmap="hsv", color=0xFF00FF)

        # Multiplier
        self.pf.y.k3d.scalar(filter_field=self.pf.norm, color=0xFF00FF, multiplier=1e-6)

        # Interactive field
        self.pf.y.k3d.scalar(
            filter_field=self.pf.norm,
            color=0xFF00FF,
            multiplier=1e-6,
            interactive_field=self.pf,
        )

        # kwargs
        self.pf.y.k3d.scalar(
            filter_field=self.pf.norm,
            color=0xFF00FF,
            multiplier=1e-6,
            interactive_field=self.pf,
            wireframe=True,
        )

        # Plot
        plot = k3d.plot()
        plot.display()
        self.pf.y.plane(z=0).k3d.scalar(
            plot=plot,
            filter_field=self.pf.norm,
            color=0xFF00FF,
            multiplier=1e-6,
            interactive_field=self.pf,
        )

        # Continuation for interactive plot testing.
        self.pf.y.plane(z=1e-9).k3d.scalar(
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
            self.pf.x.k3d.scalar(filter_field=self.pf)  # filter field dim=3

    def test_k3d_vector(self):
        # Default
        self.pf.k3d.vector()

        # Color field
        self.pf.k3d.vector(color_field=self.pf.x)

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
            self.pf.x.k3d.vector()
        with pytest.raises(ValueError):
            self.pf.k3d.vector(color_field=self.pf)  # filter field dim=3

    def test_plot_large_sample(self):
        p1 = (0, 0, 0)
        p2 = (50e9, 50e9, 50e9)
        cell = (25e9, 25e9, 25e9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        value = (1e6, 1e6, 1e6)
        field = df.Field(mesh, dim=3, value=value)

        field.plane("z").mpl()
        field.norm.k3d.nonzero()
        field.x.k3d.scalar()
        field.k3d.vector()

    def test_complex(self):
        mesh = df.Mesh(p1=(-5e-9, -5e-9, -5e-9), p2=(5e-9, 5e-9, 5e-9), n=(5, 5, 5))

        # real field
        real_field = self.pf.real
        check_field(real_field)
        assert real_field((-2e-9, 0, 0)) == (0, 0, 1e5)
        assert real_field((2e-9, 0, 0)) == (0, 0, -1e5)

        imag_field = self.pf.imag
        check_field(imag_field)
        assert df.Field(mesh, dim=3).allclose(imag_field)
        assert df.Field(mesh, dim=3).allclose(np.mod(self.pf.phase, np.pi))

        # complex field
        field = df.Field(mesh, dim=1, value=1 + 1j)
        real_field = field.real
        check_field(real_field)
        assert df.Field(mesh, dim=1, value=1).allclose(real_field)

        imag_field = field.imag
        check_field(imag_field)
        assert df.Field(mesh, dim=1, value=1).allclose(imag_field)
        assert df.Field(mesh, dim=1, value=np.pi / 4).allclose(field.phase)

    def test_numpy_ufunc(self):
        assert np.allclose(np.sin(self.pf).array, np.sin(self.pf.array))
        assert np.sum([self.pf, self.pf]).allclose(self.pf + self.pf)
        assert np.multiply(self.pf.x, self.pf.y).allclose(self.pf.x * self.pf.y)
        assert np.power(self.pf.z, 2).allclose(self.pf.z**2)

        # self.pf contains values of 1e5 and exp of this,produces an overflow
        field = df.Field(
            self.pf.mesh, dim=3, value=lambda _: np.random.random(3) * 2 - 1
        )
        assert np.allclose(
            np.exp(field.orientation).array, np.exp(field.orientation.array)
        )

    def test_to_xarray_valid_args(self):
        for mesh in self.meshes:
            for value, dtype in self.vfuncs:
                f = df.Field(mesh, dim=3, value=value, dtype=dtype)
                fxa = f.to_xarray()
                assert isinstance(fxa, xr.DataArray)
                assert f.dim == fxa["comp"].size
                assert sorted([*fxa.attrs]) == ["cell", "p1", "p2", "units"]
                assert fxa.attrs["cell"] == f.mesh.cell
                assert fxa.attrs["p1"] == f.mesh.region.p1
                assert fxa.attrs["p2"] == f.mesh.region.p2
                for i in "xyz":
                    assert np.array_equal(getattr(f.mesh.midpoints, i), fxa[i].values)
                    assert fxa[i].attrs["units"] == f.mesh.attributes["unit"]
                assert all(fxa["comp"].values == f.components)
                assert np.array_equal(f.array, fxa.values)

            for value, dtype in self.sfuncs:
                f = df.Field(mesh, dim=1, value=value, dtype=dtype)
                fxa = f.to_xarray()
                assert isinstance(fxa, xr.DataArray)
                assert sorted([*fxa.attrs]) == ["cell", "p1", "p2", "units"]
                assert fxa.attrs["cell"] == f.mesh.cell
                assert fxa.attrs["p1"] == f.mesh.region.p1
                assert fxa.attrs["p2"] == f.mesh.region.p2
                for i in "xyz":
                    assert np.array_equal(getattr(f.mesh.midpoints, i), fxa[i].values)
                    assert fxa[i].attrs["units"] == f.mesh.attributes["unit"]
                assert "comp" not in fxa.dims
                assert np.array_equal(f.array.squeeze(axis=-1), fxa.values)

        f6d = self.pf << self.pf
        f6d_xa = f6d.to_xarray()
        assert f6d_xa["comp"].size == 6
        assert "comp" not in f6d_xa.coords
        f6d.components = ["a", "c", "b", "e", "d", "f"]
        f6d_xa2 = f6d.to_xarray()
        assert "comp" in f6d_xa2.coords
        assert [*f6d_xa2["comp"].values] == ["a", "c", "b", "e", "d", "f"]

        # test name and units defaults
        f3d_xa = self.pf.to_xarray()
        assert f3d_xa.name == "field"
        assert f3d_xa.attrs["units"] is None

        # test name and units
        f3d_xa_2 = self.pf.to_xarray(name="m", units="A/m")
        assert f3d_xa_2.name == "m"
        assert f3d_xa_2.attrs["units"] == "A/m"

    def test_to_xarray_invalid_args(self):
        args = [
            ["m", 42.0],
            [21.0, 42],
            [21, "A/m"],
            [{"name": "m"}, {"units": "A/m"}],
            [["m"], ["A/m"]],
            [["m", "A/m"], None],
            [("m", "A/m"), None],
            [{"name": "m", "units": "A/m"}, None],
        ]

        for name, units in args:
            with pytest.raises(TypeError):
                self.pf.to_xarray(name, units)

    def test_from_xarray_valid_args(self):
        for mesh in self.meshes:
            for value, dtype in self.vfuncs:
                f = df.Field(mesh, dim=3, value=value, dtype=dtype)
                fxa = f.to_xarray()
                f_new = df.Field.from_xarray(fxa)
                assert f_new == f  # or use allclose()

            for value, dtype in self.sfuncs:
                f = df.Field(mesh, dim=1, value=value, dtype=dtype)
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
