import numbers
import os
import re
import tempfile
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
    r"\s*<li>attributes:\s*<ul>\s*"
    r"(\s*<li>(.*:.*|.*Mesh.*)</li>)+\s*"
    r"</ul>\s*</li>\s*"
    r"</ul>"
)


class TestMesh:
    def setup(self):
        self.valid_args = [
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
        ]

        self.invalid_args = [
            [(0, 0, 0), (5, 5, 5), None, (-1, 1, 1)],
            [(0, 0, 0), (5, 5, 5), (-1, 1, 1), None],
            [(0, 0, 0), (5, 5, 5), "n", None],
            [(0, 0, 0), (5, 5, 5), (1, 2, 2 + 1j), None],
            [(0, 0, 0), (5, 5, 5), (1, 2, "2"), None],
            [("1", 0, 0), (1, 1, 1), None, (0, 0, 1e-9)],
            [(-1.5e-9, -5e-9, "a"), (1.5e-9, 15e-9, 16e-9), None, (5, 1, -1e-9)],
            [(-1.5e-9, -5e-9, "a"), (1.5e-9, 15e-9, 16e-9), (5, 1, -1), None],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, 16e-9), None, (0.1e-9, 0.1e-9, 1e-9)],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 1 + 2j), None, (5, 1, 1e-9)],
            ["string", (5, 1, 1e-9), None, "string"],
            [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 16e-9), None, 2 + 2j],
        ]

    def test_init_valid_args(self):
        for p1, p2, n, cell in self.valid_args:
            mesh1 = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)
            assert isinstance(mesh1, df.Mesh)

            assert isinstance(mesh1.region, df.Region)
            if n is not None:
                assert np.allclose(mesh1.n, n)
            if cell is not None:
                assert np.allclose(mesh1.cell, cell)

            mesh2 = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            assert isinstance(mesh2, df.Mesh)

            assert isinstance(mesh2.region, df.Region)
            if n is not None:
                assert np.allclose(mesh2.n, n)
            if cell is not None:
                assert np.allclose(mesh2.cell, cell)

            assert mesh1 == mesh2

    def test_init_invalid_args(self):
        for p1, p2, n, cell in self.invalid_args:
            with pytest.raises((TypeError, ValueError)):
                df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)

            with pytest.raises((TypeError, ValueError)):
                df.Mesh(p1=p1, p2=p2, n=n, cell=cell)

    def test_init_subregions(self):
        p1 = (0, 0, 0)
        p2 = (100, 50, 10)
        cell = (10, 10, 10)
        subregions = {
            "r1": df.Region(p1=(0, 0, 0), p2=(50, 50, 10)),
            "r2": df.Region(p1=(50, 0, 0), p2=(100, 50, 10)),
        }
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        assert isinstance(mesh, df.Mesh)
        assert mesh.subregions == subregions

        # Invalid subregions.
        p1 = (0, 0, 0)
        p2 = (100e-9, 50e-9, 10e-9)
        cell = (10e-9, 10e-9, 10e-9)

        # Subregion not an aggregate.
        subregions = {"r1": df.Region(p1=(0, 0, 0), p2=(45e-9, 50e-9, 10e-9))}
        with pytest.raises(ValueError):
            df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        # Subregion not aligned.
        subregions = {"r1": df.Region(p1=(5e-9, 0, 0), p2=(45e-9, 50e-9, 10e-9))}
        with pytest.raises(ValueError):
            df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        # Subregion not in the mesh region.
        subregions = {"r1": df.Region(p1=(0, 0, 0), p2=(45e-9, 50e-9, 200e-9))}
        with pytest.raises(ValueError):
            df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        with pytest.raises(TypeError):
            df.Mesh(p1=p1, p2=p2, cell=cell, subregions=["a", "b"])

        subregions = {1: df.Region(p1=(0, 0, 0), p2=(45e-9, 50e-9, 200e-9))}
        with pytest.raises(TypeError):
            df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        subregions = {"r1": "top half of the region"}
        with pytest.raises(TypeError):
            df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

    def test_init_with_region_and_points(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        region = df.Region(p1=p1, p2=p2)
        n = (10, 10, 10)

        with pytest.raises(ValueError) as excinfo:
            df.Mesh(region=region, p1=p1, p2=p2, n=n)
        assert "not both." in str(excinfo.value)

    def test_init_with_n_and_cell(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        n = (15, 141, 11)
        cell = (1, 0.1, 0.5)

        with pytest.raises(ValueError) as excinfo:
            df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
        assert "not both." in str(excinfo.value)

    def test_region_not_aggregate_of_cell(self):
        args = [
            [(0, 100e-9, 1e-9), (150e-9, 120e-9, 6e-9), (4e-9, 1e-9, 1e-9)],
            [(0, 100e-9, 0), (150e-9, 104e-9, 1e-9), (2e-9, 1.5e-9, 0.1e-9)],
            [(10e9, 10e3, 0), (11e9, 11e3, 5), (1e9, 1e3, 1.5)],
        ]

        for p1, p2, cell in args:
            with pytest.raises(ValueError):
                df.Mesh(p1=p1, p2=p2, cell=cell)

    def test_cell_greater_than_domain(self):
        p1 = (0, 0, 0)
        p2 = (1e-9, 1e-9, 1e-9)
        args = [
            (2e-9, 1e-9, 1e-9),
            (1e-9, 2e-9, 1e-9),
            (1e-9, 1e-9, 2e-9),
            (1e-9, 5e-9, 0.1e-9),
        ]

        for cell in args:
            with pytest.raises(ValueError):
                df.Mesh(p1=p1, p2=p2, cell=cell)

    def test_cell_n(self):
        p1 = (0, 0, 0)
        p2 = (20e-9, 20e-9, 20e-9)
        cell = (2e-9, 4e-9, 1e-9)
        n = (10, 5, 20)

        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert mesh.n == n
        with pytest.raises(AttributeError):
            mesh.cell = (2e-9, 2e-9, 2e-9)
        with pytest.raises(AttributeError):
            mesh.n = (10, 10, 10)
        with pytest.raises(TypeError):
            df.Mesh(p1=p1, p2=p2, cell={"x": 2e-9, "y": 4e-9, "z": 1e-9})
        with pytest.raises(TypeError):
            df.Mesh(p1=p1, p2=p2, n={"x": 10, "y": 5, "z": 20})
        with pytest.raises(ValueError):
            df.Mesh(p1=p1, p2=p2, cell=(2e-9, 4e-9))
        with pytest.raises(ValueError):
            df.Mesh(p1=p1, p2=p2, n=(10, 5))
        with pytest.raises(ValueError):
            df.Mesh(p1=p1, p2=p2)
        with pytest.raises(ValueError):
            df.Mesh(p1=p1, p2=p2, cell=cell, n=n)

    def test_bc(self):
        p1 = (0, 0, 0)
        p2 = (20e-9, 20e-9, 20e-9)
        region = df.Region(p1=p1, p2=p2, dims=["x", "y", "z"])
        cell = (2e-9, 4e-9, 1e-9)

        for bc in ["x", "y", "yz", "zx", "xyz", "Neumann", "dirichlet"]:
            df.Mesh(region=region, cell=cell, bc=bc)

        with pytest.raises(TypeError):
            df.Mesh(region=region, cell=cell, bc=2)
        with pytest.raises(ValueError):
            df.Mesh(region=region, cell=cell, bc="user")
        with pytest.raises(ValueError):
            df.Mesh(region=region, cell=cell, bc="xxz")

    def test_len(self):
        p1 = (0, 0, 0)
        p2 = (5, 4, 3)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert isinstance(mesh, df.Mesh)

        assert len(mesh) == 5 * 4 * 3

    def test_indices_coordinates_iter(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        assert isinstance(mesh, df.Mesh)

        assert isinstance(mesh.indices, types.GeneratorType)
        assert len(list(mesh.indices)) == 125
        for index in mesh.indices:
            assert isinstance(index, tuple)
            assert len(index) == 3
            assert all(isinstance(i, int) for i in index)
            assert all([0 <= i <= 4 for i in index])

        assert isinstance(mesh.__iter__(), types.GeneratorType)
        assert len(list(mesh)) == 125
        for point in mesh:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert all(isinstance(i, numbers.Real) for i in point)
            assert all([1 <= i <= 9 for i in point])

    def test_eq(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        n = (1, 1, 1)
        mesh1 = df.Mesh(p1=p1, p2=p2, n=n)
        # NOTE: Why do we need to test mesh1 type here?
        # assert isinstance(mesh1, df.Mesh)
        mesh2 = df.Mesh(p1=p1, p2=p2, n=n)
        # assert isinstance(mesh2, df.Mesh)

        assert mesh1 == mesh2
        assert not mesh1 != mesh2
        assert mesh1 != 1
        assert not mesh2 == "mesh2"

        p1 = (0, 0, 0)
        p2 = (10 + 1e-12, 10 + 2e-13, 10 + 3e-12)
        n = (1, 1, 1)
        mesh3 = df.Mesh(p1=p1, p2=p2, n=n)
        # assert isinstance(mesh3, df.Mesh)

        assert not mesh1 == mesh3
        assert not mesh2 == mesh3
        assert mesh1 != mesh3
        assert mesh2 != mesh3

    def test_allclose(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        n = (1, 1, 1)
        mesh1 = df.Mesh(p1=p1, p2=p2, n=n)
        mesh2 = df.Mesh(p1=p1, p2=p2, n=n)

        assert mesh1.allclose(mesh2)

        p1 = (0, 0, 0)
        p2 = (10 + 1e-12, 10 + 2e-13, 10 + 3e-12)
        n = (1, 1, 1)
        atol = 1e-10
        rtol = 1e-8
        mesh3 = df.Mesh(p1=p1, p2=p2, n=n)

        assert mesh1.allclose(mesh3, rtol=rtol, atol=atol)

        p2 = (10 + 1e-9, 10 + 2e-7, 10 + 3e-8)
        mesh4 = df.Mesh(p1=p1, p2=p2, n=n)

        assert not mesh1.allclose(mesh4, rtol=rtol, atol=atol)

        with pytest.raises(TypeError):
            mesh1.allclose(df.Region(p1=p1, p2=p2))

        with pytest.raises(TypeError):
            mesh1.allclose(mesh3, rtol=rtol, atol="20")

        with pytest.raises(TypeError):
            mesh1.allclose(mesh3, rtol="1", atol=atol)

        with pytest.raises(TypeError):
            mesh1.allclose(mesh3, rtol="1", atol="20")

    def test_repr(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        cell = (1, 0.1, 0.5)

        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, bc="x")
        assert isinstance(mesh, df.Mesh)

        rstr = (
            "Mesh(Region(pmin=[-1.0, -4.0, 11.0], pmax=[15.0, 10.1, 12.5], "
            "dims=['x', 'y', 'z'], units=['m', 'm', 'm']), "
            "n=(16, 141, 3), bc=x, attributes: (unit: m, fourierspace: "
            "False, isplane: False))"
        )
        assert repr(mesh) == rstr
        assert re.match(html_re, mesh._repr_html_(), re.DOTALL)

    def test_index2point(self):
        p1 = (15, -4, 12.5)
        p2 = (-1, 10.1, 11)
        cell = (1, 0.1, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert isinstance(mesh, df.Mesh)

        assert mesh.index2point((5, 10, 1)) == (4.5, -2.95, 11.75)

        # Correct minimum index
        assert isinstance(mesh.index2point((0, 0, 0)), tuple)
        assert mesh.index2point((0, 0, 0)) == (-0.5, -3.95, 11.25)

        # Below minimum index
        with pytest.raises(ValueError):
            mesh.index2point((-1, 0, 0))
        with pytest.raises(ValueError):
            mesh.index2point((0, -1, 0))
        with pytest.raises(ValueError):
            mesh.index2point((0, 0, -1))

        # Correct maximum index
        assert isinstance(mesh.index2point((15, 140, 2)), tuple)
        assert mesh.index2point((15, 140, 2)) == (14.5, 10.05, 12.25)

        # Above maximum index
        with pytest.raises(ValueError):
            mesh.index2point((16, 0, 0))
        with pytest.raises(ValueError):
            mesh.index2point((0, 141, 0))
        with pytest.raises(ValueError):
            mesh.index2point((0, 0, 3))

    def test_point2index(self):
        p1 = (-10e-9, -5e-9, 10e-9)
        p2 = (10e-9, 5e-9, 0)
        cell = (1e-9, 5e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert isinstance(mesh, df.Mesh)

        # (0, 0, 0) cell
        assert mesh.point2index((-10e-9, -5e-9, 0)) == (0, 0, 0)
        assert mesh.point2index((-9.5e-9, -2.5e-9, 0.5e-9)) == (0, 0, 0)
        assert mesh.point2index((-9.01e-9, -0.1e-9, 0.9e-9)) == (0, 0, 0)

        # (19, 1, 9) cell
        assert mesh.point2index((10e-9, 5e-9, 10e-9)) == (19, 1, 9)
        assert mesh.point2index((9.5e-9, 2.5e-9, 9.5e-9)) == (19, 1, 9)
        assert mesh.point2index((9.1e-9, 0.1e-9, 9.1e-9)) == (19, 1, 9)

        # vicinity of (0, 0, 0) point
        assert mesh.point2index((1e-16, 1e-16, 0.99e-16)) == (10, 1, 0)
        assert mesh.point2index((-1e-16, -1e-16, 0.01e-16)) == (9, 0, 0)

        # Points outside the mesh.
        p1 = (-10, 5, 0)
        p2 = (10, -5, 10e-9)
        n = (10, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        assert isinstance(mesh, df.Mesh)

        tol = 1e-12  # picometer tolerance
        with pytest.raises(ValueError):
            mesh.point2index((-10 - tol, 0, 5))
        with pytest.raises(ValueError):
            mesh.point2index((-5, -5 - tol, 5))
        with pytest.raises(ValueError):
            mesh.point2index((-5, 0, -tol))
        with pytest.raises(ValueError):
            mesh.point2index((10 + tol, 0, 5))
        with pytest.raises(ValueError):
            mesh.point2index((6, 5 + tol, 5))
        with pytest.raises(ValueError):
            mesh.point2index((0, 0, 10e-9 + tol))

    def test_index2point_point2index_mutually_inverse(self):
        p1 = (15, -4, 12.5)
        p2 = (-1, 10.1, 11)
        cell = (1, 0.1, 0.5)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        assert isinstance(mesh, df.Mesh)

        for p in [(-0.5, -3.95, 11.25), (14.5, 10.05, 12.25)]:
            assert mesh.index2point(mesh.point2index(p)) == p

        for i in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
            assert mesh.point2index(mesh.index2point(i)) == i

    def test_region2slice(self):
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

    @pytest.mark.filterwarnings("ignore::FutureWarning")
    def test_axis_points(self):
        p1 = (0, 0, 0)
        p2 = (10, 6, 8)
        cell = (2, 2, 2)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert np.allclose(mesh.axis_points("x"), [1.0, 3.0, 5.0, 7.0, 9.0])
        assert np.allclose(mesh.axis_points("y"), [1.0, 3.0, 5.0])
        assert np.allclose(mesh.axis_points("z"), [1.0, 3.0, 5.0, 7.0])

    def test_points(self):
        p1 = (0, 0, 4)
        p2 = (10, 6, 0)
        cell = (2, 2, 1)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert np.allclose(mesh.points.x, [1.0, 3.0, 5.0, 7.0, 9.0])
        assert np.allclose(mesh.points.y, [1.0, 3.0, 5.0])
        assert np.allclose(mesh.points.z, [0.5, 1.5, 2.5, 3.5])

    def test_vertices(self):
        p1 = (0, 1, 0)
        p2 = (5, 0, 6)
        cell = (1, 1, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        assert np.allclose(mesh.vertices.x, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.allclose(mesh.vertices.y, [0.0, 1.0])
        assert np.allclose(mesh.vertices.z, [0.0, 2.0, 4.0, 6.0])

    def test_line(self):
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

    def test_or(self):
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

    def test_is_aligned(self):
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

        assert mesh1.is_aligned(mesh2) is True
        assert mesh1.is_aligned(mesh3) is False
        assert mesh1.is_aligned(mesh4) is False
        assert mesh1.is_aligned(mesh1) is True

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

        assert mesh5.is_aligned(mesh6, tol) is True
        assert mesh5.is_aligned(mesh7, tol) is False

        # Test exceptions
        with pytest.raises(TypeError):
            mesh5.is_aligned(mesh6.region, tol)
        with pytest.raises(TypeError):
            mesh5.is_aligned(mesh6, "1e-12")

    def test_getitem(self):
        # Subregions disctionary
        p1 = (0, 0, 0)
        p2 = (100, 50, 10)
        cell = (5, 5, 5)
        subregions = {
            "r1": df.Region(p1=(0, 0, 0), p2=(50, 50, 10)),
            "r2": df.Region(p1=(50, 0, 0), p2=(100, 50, 10)),
        }
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        # NOTE: Why do we need to check mesh type here?
        # assert isinstance(mesh, df.Mesh)

        submesh1 = mesh["r1"]
        assert isinstance(submesh1, df.Mesh)
        assert np.allclose(submesh1.region.pmin, (0, 0, 0))
        assert np.allclose(submesh1.region.pmax, (50, 50, 10))
        assert np.allclose(submesh1.cell, (5, 5, 5))

        submesh2 = mesh["r2"]
        assert isinstance(submesh2, df.Mesh)
        assert np.allclose(submesh2.region.pmin, (50, 0, 0))
        assert np.allclose(submesh2.region.pmax, (100, 50, 10))
        assert np.allclose(submesh2.cell, (5, 5, 5))

        assert len(submesh1) + len(submesh2) == len(mesh)

        # "newly-defined" region
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        # assert isinstance(mesh, df.Mesh)

        submesh = mesh[df.Region(p1=(0.1, 2.2, 4.01), p2=(4.9, 3.8, 5.7))]
        assert isinstance(submesh, df.Mesh)
        assert np.allclose(submesh.region.pmin, (0, 2, 4))
        assert np.allclose(submesh.region.pmax, (5, 4, 6))
        assert np.allclose(submesh.cell, cell)
        assert np.allclose(submesh.n, (5, 2, 2))
        assert mesh[mesh.region].allclose(mesh)

        p1 = (20e-9, 0, 15e-9)
        p2 = (-25e-9, 100e-9, -5e-9)
        cell = (5e-9, 5e-9, 5e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        # assert isinstance(mesh, df.Mesh)

        submesh = mesh[df.Region(p1=(11e-9, 22e-9, 1e-9), p2=(-9e-9, 79e-9, 14e-9))]
        assert isinstance(submesh, df.Mesh)
        assert np.allclose(
            submesh.region.pmin, (-10e-9, 20e-9, 0), atol=1e-15, rtol=1e-5
        )
        assert np.allclose(
            submesh.region.pmax, (15e-9, 80e-9, 15e-9), atol=1e-15, rtol=1e-5
        )
        assert submesh.cell == cell
        assert submesh.n == (5, 12, 3)
        assert mesh[mesh.region].allclose(mesh)

        with pytest.raises(ValueError):
            submesh = mesh[
                df.Region(p1=(11e-9, 22e-9, 1e-9), p2=(200e-9, 79e-9, 14e-9))
            ]

    def test_pad(self):
        p1 = (-1, 2, 7)
        p2 = (5, 9, 4)
        cell = (1, 1, 1)
        region = df.Region(p1=p1, p2=p2)
        mesh = df.Mesh(region=region, cell=cell)

        padded_mesh = mesh.pad({"x": (0, 1)})
        assert np.allclose(padded_mesh.region.pmin, (-1, 2, 4))
        assert np.allclose(padded_mesh.region.pmax, (6, 9, 7))
        assert np.allclose(padded_mesh.n, (7, 7, 3))

        padded_mesh = mesh.pad({"y": (1, 1)})
        assert np.allclose(padded_mesh.region.pmin, (-1, 1, 4))
        assert np.allclose(padded_mesh.region.pmax, (5, 10, 7))
        assert np.allclose(padded_mesh.n, (6, 9, 3))

        padded_mesh = mesh.pad({"z": (2, 3)})
        assert np.allclose(padded_mesh.region.pmin, (-1, 2, 2))
        assert np.allclose(padded_mesh.region.pmax, (5, 9, 10))
        assert np.allclose(padded_mesh.n, (6, 7, 8))

        padded_mesh = mesh.pad({"x": (1, 1), "y": (1, 1), "z": (1, 1)})
        assert np.allclose(padded_mesh.region.pmin, (-2, 1, 3))
        assert np.allclose(padded_mesh.region.pmax, (6, 10, 8))
        assert np.allclose(padded_mesh.n, (8, 9, 5))

    def test_getattr(self):
        p1 = (0, 0, 0)
        p2 = (100e-9, 80e-9, 10e-9)
        cell = (1e-9, 5e-9, 10e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert mesh.dx == 1e-9
        assert mesh.dy == 5e-9
        assert mesh.dz == 10e-9

        with pytest.raises(AttributeError):
            mesh.dk

    def test_dir(self):
        p1 = (0, 0, 0)
        p2 = (100e-9, 80e-9, 10e-9)
        cell = (1e-9, 5e-9, 10e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert all([i in dir(mesh) for i in ["dx", "dy", "dz"]])

    def test_dV(self):
        p1 = (0, 0, 0)
        p2 = (100, 80, 10)
        cell = (1, 2, 2.5)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert mesh.dV == 5

    def test_dS(self):
        p1 = (0, 0, 0)
        p2 = (100, 80, 10)
        cell = (1, 2, 2.5)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert np.allclose(mesh.plane("x").dS.mean(), (5, 0, 0))
        assert np.allclose(mesh.plane("y").dS.mean(), (0, 2.5, 0))
        assert np.allclose(mesh.plane("z").dS.mean(), (0, 0, 2))

        # Exception
        with pytest.raises(ValueError):
            mesh.dS

    def test_mpl(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)
            mesh.mpl()
            mesh.mpl(box_aspect=[1, 2, 3])

            filename = "figure.pdf"
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfilename = os.path.join(tmpdir, filename)
                mesh.mpl(filename=tmpfilename)

            plt.close("all")

    def test_k3d(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            mesh.k3d()
            mesh.plane("x").k3d()

    def test_k3d_mpl_subregions(self):
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

        filename = "figure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            mesh.mpl.subregions(filename=tmpfilename)

        plt.close("all")

        # k3d tests
        mesh.k3d.subregions()

    def test_slider(self):
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

    def test_axis_selector(self):
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

    def test_save_load_subregions(self, tmp_path):
        p1 = (0, 0, 0)
        p2 = (100, 50, 10)
        cell = (10, 10, 10)
        subregions = {
            "r1": df.Region(p1=(0, 0, 0), p2=(50, 50, 10)),
            "r2": df.Region(p1=(50, 0, 0), p2=(100, 50, 10)),
        }
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        assert isinstance(mesh, df.Mesh)

        mesh.save_subregions(str(tmp_path / "mesh.json"))

        mesh2 = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert mesh2.subregions == {}
        mesh2.load_subregions(str(tmp_path / "mesh.json"))
        assert mesh2.subregions == subregions

    def test_coordinate_field(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            cfield = mesh.coordinate_field()
            assert isinstance(cfield, df.Field)
            manually = df.Field(mesh, dim=3, value=lambda p: p)
            assert cfield.allclose(manually)
            assert np.allclose(cfield.array[:, 0, 0, 0], mesh.points.x)
            assert np.allclose(cfield.array[0, :, 0, 1], mesh.points.y)
            assert np.allclose(cfield.array[0, 0, :, 2], mesh.points.z)

    # ------------------ sel method test draft -----------------------------------------
    # def test_sel(self):
    #     p1 = (0, 0, 0)
    #     p2 = (20, 20, 20)
    #     cell = (2, 2, 2)
    #     mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    #     for dim in mesh.region.dims:
    #         sub_mesh = mesh.sel(f"{dim}")
    #         assert isinstance(sub_mesh, df.Mesh)
    #         assert sub_mesh.region.ndim == mesh.region.ndim - 1
    #         assert np.allclose(
    #             sub_mesh.region.pmin,
    #             mesh.region.pmin[mesh.region.dims != dim],
    #         )
    #         assert np.allclose(
    #             sub_mesh.region.pmax,
    #             mesh.region.pmax[mesh.region.dims != dim],
    #         )

    #     sub_mesh = mesh.sel(x=3.1)
    #     assert isinstance(sub_mesh, df.Mesh)
    #     assert sub_mesh.region.ndim == mesh.region.ndim - 1
    #     assert np.allclose(
    #         sub_mesh.region.pmin, mesh.region.pmin[mesh.region.dims != "x"]
    #     )
    #     assert np.allclose(
    #         sub_mesh.region.pmax, mesh.region.pmax[mesh.region.dims != "x"]
    #     )

    #     sub_mesh = mesh.sel(x=4, z=14)
    #     assert isinstance(sub_mesh, df.Mesh)
    #     assert sub_mesh.region.ndim == mesh.region.ndim - 2
    #     assert np.allclose(
    #         sub_mesh.region.pmin,
    #         mesh.region.pmin[mesh.region.dims != "x" and mesh.region.dims != "z"],
    #     )
    #     assert np.allclose(
    #         sub_mesh.region.pmax,
    #         mesh.region.pmax[mesh.region.dims != "x" and mesh.region.dims != "z"],
    #     )
