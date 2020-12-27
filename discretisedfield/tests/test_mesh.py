import os
import re
import types
import pytest
import numbers
import tempfile
import ipywidgets
import numpy as np
import discretisedfield as df
import matplotlib.pyplot as plt


def check_mesh(mesh):
    assert isinstance(mesh.region, df.Region)

    assert isinstance(mesh.cell, tuple)
    assert len(mesh.cell) == 3
    assert all(isinstance(i, numbers.Real) for i in mesh.cell)
    assert all(i > 0 for i in mesh.cell)

    assert isinstance(mesh.n, tuple)
    assert len(mesh.n) == 3
    assert all(isinstance(i, int) for i in mesh.n)
    assert all(i > 0 for i in mesh.n)

    assert isinstance(mesh.bc, str)

    assert isinstance(mesh.subregions, dict)
    assert all(isinstance(i, str) for i in mesh.subregions.keys())
    assert all(isinstance(i, df.Region) for i in mesh.subregions.values())

    assert isinstance(len(mesh), int)
    assert len(mesh) > 0

    assert isinstance(repr(mesh), str)
    pattern = r'^Mesh\(region=Region\(p1=\(.+\), p2=\(.+\)\), n=.+\)$'
    assert re.search(pattern, repr(mesh))

    assert isinstance(mesh.indices, types.GeneratorType)
    assert isinstance(mesh.__iter__(), types.GeneratorType)
    assert len(list(mesh.indices)) == len(mesh)
    assert len(list(mesh)) == len(mesh)

    line = mesh.line(p1=mesh.region.pmin, p2=mesh.region.pmax, n=3)
    assert isinstance(line, types.GeneratorType)
    assert len(list(line)) == 3
    assert all(isinstance(i, tuple) for i in line)
    assert all(i in mesh.region for i in line)

    plane_mesh = mesh.plane('z', n=(2, 2))
    assert isinstance(plane_mesh, df.Mesh)
    assert isinstance(plane_mesh.info, dict)
    assert plane_mesh.info
    assert 1 in plane_mesh.n
    assert len(plane_mesh) == 4
    assert all(isinstance(i, tuple) for i in plane_mesh)
    assert all(i in mesh.region for i in plane_mesh)

    assert mesh.point2index(mesh.index2point((0, 0, 0))) == (0, 0, 0)

    assert mesh == mesh
    assert not mesh != mesh

    assert mesh | mesh


class TestMesh:
    def setup(self):
        self.valid_args = [[(0, 0, 0), (5, 5, 5),
                            [1, 1, 1], None],
                           [(-1, 0, -3), (5, 7, 5),
                            None, (1, 1, 1)],
                           [(0, 0, 0), (5e-9, 5e-9, 5e-9),
                            None, (1e-9, 1e-9, 1e-9)],
                           [(0, 0, 0), (5e-9, 5e-9, 5e-9),
                            (5, 5, 5), None],
                           [(-1.5e-9, -5e-9, 0), (1.5e-9, -15e-9, -10e-9),
                            None, (1.5e-9, 0.5e-9, 10e-9)],
                           [(-1.5e-9, -5e-9, 0), (1.5e-9, -15e-9, -10e-9),
                            (3, 10, 2), None],
                           [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0)),
                            None, (0.5e-9, 1e-9, 5e-9)],
                           [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0)),
                            (5, 5, 7), None],
                           [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6),
                            None, (0.5e-6, 2e-6, 2.5e-6)],
                           [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6),
                            (1, 10, 20), None],
                           [(0, 125e-9, 0), (500e-9, 0, -3e-9),
                            None, (25e-9, 25e-9, 3e-9)]]

        self.invalid_args = [[(0, 0, 0), (5, 5, 5),
                              None, (-1, 1, 1)],
                             [(0, 0, 0), (5, 5, 5),
                              (-1, 1, 1), None],
                             [(0, 0, 0), (5, 5, 5),
                              'n', None],
                             [(0, 0, 0), (5, 5, 5),
                              (1, 2, 2+1j), None],
                             [(0, 0, 0), (5, 5, 5),
                              (1, 2, '2'), None],
                             [('1', 0, 0), (1, 1, 1),
                              None, (0, 0, 1e-9)],
                             [(-1.5e-9, -5e-9, 'a'), (1.5e-9, 15e-9, 16e-9),
                              None, (5, 1, -1e-9)],
                             [(-1.5e-9, -5e-9, 'a'), (1.5e-9, 15e-9, 16e-9),
                              (5, 1, -1), None],
                             [(-1.5e-9, -5e-9, 0), (1.5e-9, 16e-9),
                              None, (0.1e-9, 0.1e-9, 1e-9)],
                             [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 1+2j),
                              None, (5, 1, 1e-9)],
                             ['string', (5, 1, 1e-9),
                              None, 'string'],
                             [(-1.5e-9, -5e-9, 0), (1.5e-9, 15e-9, 16e-9),
                              None, 2+2j]]

    def test_init_valid_args(self):
        for p1, p2, n, cell in self.valid_args:
            mesh1 = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)
            check_mesh(mesh1)

            mesh2 = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            check_mesh(mesh2)

            assert mesh1 == mesh2

    def test_init_invalid_args(self):
        for p1, p2, n, cell in self.invalid_args:
            with pytest.raises((TypeError, ValueError)):
                mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)

            with pytest.raises((TypeError, ValueError)):
                mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)

    def test_init_subregions(self):
        p1 = (0, 0, 0)
        p2 = (100, 50, 10)
        cell = (10, 10, 10)
        subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 50, 10)),
                      'r2': df.Region(p1=(50, 0, 0), p2=(100, 50, 10))}
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        check_mesh(mesh)

        # Invalid subregions.
        p1 = (0, 0, 0)
        p2 = (100e-9, 50e-9, 10e-9)
        cell = (10e-9, 10e-9, 10e-9)

        # Subregion not an aggregate.
        subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(45e-9, 50e-9, 10e-9))}
        with pytest.raises(ValueError):
            mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        # Subregion not aligned.
        subregions = {'r1': df.Region(p1=(5e-9, 0, 0),
                                      p2=(45e-9, 50e-9, 10e-9))}
        with pytest.raises(ValueError):
            mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        # Subregion not in the mesh region.
        subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(45e-9, 50e-9, 200e-9))}
        with pytest.raises(ValueError):
            mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

    def test_init_with_region_and_points(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        region = df.Region(p1=p1, p2=p2)
        n = (10, 10, 10)

        with pytest.raises(ValueError) as excinfo:
            mesh = df.Mesh(region=region, p1=p1, p2=p2, n=n)
        assert 'not both.' in str(excinfo.value)

    def test_init_with_n_and_cell(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        n = (15, 141, 11)
        cell = (1, 0.1, 0.5)

        with pytest.raises(ValueError) as excinfo:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
        assert 'not both.' in str(excinfo.value)

    def test_region_not_aggregate_of_cell(self):
        args = [[(0, 100e-9, 1e-9),
                 (150e-9, 120e-9, 6e-9),
                 (4e-9, 1e-9, 1e-9)],
                [(0, 100e-9, 0),
                 (150e-9, 104e-9, 1e-9),
                 (2e-9, 1.5e-9, 0.1e-9)],
                [(10e9, 10e3, 0),
                 (11e9, 11e3, 5),
                 (1e9, 1e3, 1.5)]]

        for p1, p2, cell in args:
            with pytest.raises(ValueError) as excinfo:
                mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def test_cell_greater_than_domain(self):
        p1 = (0, 0, 0)
        p2 = (1e-9, 1e-9, 1e-9)
        args = [(2e-9, 1e-9, 1e-9),
                (1e-9, 2e-9, 1e-9),
                (1e-9, 1e-9, 2e-9),
                (1e-9, 5e-9, 0.1e-9)]

        for cell in args:
            with pytest.raises(ValueError) as excinfo:
                mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def test_len(self):
        p1 = (0, 0, 0)
        p2 = (5, 4, 3)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

        assert len(mesh) == 5*4*3

    def test_indices_coordinates_iter(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        n = (5, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        check_mesh(mesh)

        assert len(list(mesh.indices)) == 125
        for index in mesh.indices:
            assert isinstance(index, tuple)
            assert len(index) == 3
            assert all(isinstance(i, int) for i in index)
            assert all([0 <= i <= 4 for i in index])

        assert len(list(mesh)) == 125
        for point in mesh:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert all(isinstance(i, numbers.Real) for i in point)
            assert all([1 <= i <= 9 for i in point])

    def test_eq(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh1 = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh1)
        mesh2 = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh2)

        assert mesh1 == mesh2
        assert not mesh1 != mesh2
        assert mesh1 != 1
        assert not mesh2 == 'mesh2'

        p1 = (0, 0, 0)
        p2 = (10e-9, 5e-9, 3e-9)
        cell = (1e-9, 2.5e-9, 0.5e-9)
        mesh3 = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh3)

        assert not mesh1 == mesh3
        assert not mesh2 == mesh3
        assert mesh1 != mesh3
        assert mesh2 != mesh3

    def test_repr(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        cell = (1, 0.1, 0.5)

        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, bc='x')
        check_mesh(mesh)

        rstr = ('Mesh(region=Region(p1=(-1.0, -4.0, 11.0), '
                'p2=(15.0, 10.1, 12.5)), n=(16, 141, 3), '
                'bc=\'x\', subregions={})')
        assert repr(mesh) == rstr

    def test_index2point(self):
        p1 = (15, -4, 12.5)
        p2 = (-1, 10.1, 11)
        cell = (1, 0.1, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

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
        check_mesh(mesh)

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
        check_mesh(mesh)

        tol = 1e-12  # picometer tolerance
        with pytest.raises(ValueError):
            mesh.point2index((-10-tol, 0, 5))
        with pytest.raises(ValueError):
            mesh.point2index((-5, -5-tol, 5))
        with pytest.raises(ValueError):
            mesh.point2index((-5, 0, -tol))
        with pytest.raises(ValueError):
            mesh.point2index((10+tol, 0, 5))
        with pytest.raises(ValueError):
            mesh.point2index((6, 5+tol, 5))
        with pytest.raises(ValueError):
            mesh.point2index((0, 0, 10e-9+tol))

    def test_index2point_point2index_mutually_inverse(self):
        p1 = (15, -4, 12.5)
        p2 = (-1, 10.1, 11)
        cell = (1, 0.1, 0.5)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        check_mesh(mesh)

        for p in [(-0.5, -3.95, 11.25), (14.5, 10.05, 12.25)]:
            assert mesh.index2point(mesh.point2index(p)) == p

        for i in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]:
            assert mesh.point2index(mesh.index2point(i)) == i

    def test_axis_points(self):
        p1 = (0, 0, 0)
        p2 = (10, 6, 8)
        cell = (2, 2, 2)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert list(mesh.axis_points('x')) == [1.0, 3.0, 5.0, 7.0, 9.0]
        assert list(mesh.axis_points('y')) == [1.0, 3.0, 5.0]
        assert list(mesh.axis_points('z')) == [1.0, 3.0, 5.0, 7.0]

    def test_neighbours(self):
        p1 = (0, 0, 0)
        p2 = (5, 3, 2)
        n = (5, 3, 2)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)

        neighbours = mesh.neighbours((1, 1, 1))
        assert isinstance(neighbours, list)
        assert len(neighbours) == 5
        assert (0, 1, 1) in neighbours
        assert (2, 1, 1) in neighbours
        assert (1, 0, 1) in neighbours
        assert (1, 2, 1) in neighbours
        assert (1, 1, 0) in neighbours

        neighbours = mesh.neighbours((0, 0, 0))
        assert isinstance(neighbours, list)
        assert len(neighbours) == 3
        assert (1, 0, 0) in neighbours
        assert (0, 1, 0) in neighbours
        assert (0, 0, 1) in neighbours

        p1 = (0, 0, 0)
        p2 = (5, 5, 5)
        n = (5, 5, 5)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, bc='xy')

        neighbours = mesh.neighbours((0, 0, 0))
        assert isinstance(neighbours, list)
        assert len(neighbours) == 5
        assert (4, 0, 0) in neighbours
        assert (1, 0, 0) in neighbours
        assert (0, 1, 0) in neighbours
        assert (0, 4, 0) in neighbours
        assert (0, 0, 1) in neighbours

        neighbours = mesh.neighbours((4, 4, 4))
        assert isinstance(neighbours, list)
        assert len(neighbours) == 5
        assert (3, 4, 4) in neighbours
        assert (0, 4, 4) in neighbours
        assert (4, 0, 4) in neighbours
        assert (4, 3, 4) in neighbours
        assert (4, 4, 3) in neighbours

        with pytest.raises(ValueError):
            neighbours = mesh.neighbours((10, 4, 4))

    def test_line(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

        tol = 1e-12
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

    def test_plane(self):
        p1 = (0, 0, 0)
        p2 = (10, 5, 3)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

        plane = mesh.plane(z=1, n=(2, 2))
        check_mesh(plane)
        assert isinstance(plane, df.Mesh)
        assert len(list(plane)) == 4
        for point in plane:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert point[2] == 1

        plane = mesh.plane(y=4.2, n=(3, 2))
        check_mesh(plane)
        assert isinstance(plane, df.Mesh)
        assert len(list(plane)) == 6
        for point in plane:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert point[1] == 4.2

        plane = mesh.plane('x')
        check_mesh(plane)
        assert isinstance(plane, df.Mesh)
        assert len(list(plane)) == 15
        for point in plane:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert point[0] == 5

        plane = mesh.plane('y', n=(10, 10))
        check_mesh(plane)
        assert isinstance(plane, df.Mesh)
        assert len(list(plane)) == 100
        for point in plane:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert point[1] == 2.5

        with pytest.raises(ValueError):
            plane = list(mesh.plane(x=-1))

        with pytest.raises(ValueError):
            plane = list(mesh.plane(y=6))

        with pytest.raises(ValueError):
            plane = list(mesh.plane(z=-1e-9))

        with pytest.raises(ValueError):
            plane = list(mesh.plane('x', z=1))

        with pytest.raises(ValueError):
            plane = list(mesh.plane('z', z=1))

        with pytest.raises(ValueError):
            plane = list(mesh.plane(x=2, z=1))

        info = mesh.plane('x').info
        assert info['planeaxis'] == 0
        assert info['axis1'] == 1
        assert info['axis2'] == 2
        assert info['point'] == 5

        info = mesh.plane('y').info
        assert info['planeaxis'] == 1
        assert info['axis1'] == 0
        assert info['axis2'] == 2
        assert info['point'] == 2.5

        info = mesh.plane('z').info
        assert info['planeaxis'] == 2
        assert info['axis1'] == 0
        assert info['axis2'] == 1
        assert info['point'] == 1.5

        info = mesh.plane(x=0).info
        assert info['planeaxis'] == 0
        assert info['axis1'] == 1
        assert info['axis2'] == 2
        assert info['point'] == 0

        info = mesh.plane(y=0).info
        assert info['planeaxis'] == 1
        assert info['axis1'] == 0
        assert info['axis2'] == 2
        assert info['point'] == 0

        info = mesh.plane(z=0).info
        assert info['planeaxis'] == 2
        assert info['axis1'] == 0
        assert info['axis2'] == 1
        assert info['point'] == 0

        info = mesh.plane(x=5).info
        assert info['planeaxis'] == 0
        assert info['axis1'] == 1
        assert info['axis2'] == 2
        assert info['point'] == 5

        with pytest.raises(KeyError):
            plane_mesh = mesh.plane('xy')
        with pytest.raises(KeyError):
            plane_mesh = mesh.plane('zy')
        with pytest.raises(ValueError):
            plane_mesh = mesh.plane('y', 'x')
        with pytest.raises(KeyError):
            plane_mesh = mesh.plane('xzy')
        with pytest.raises(ValueError):
            plane_mesh = mesh.plane('z', x=3)
        with pytest.raises(ValueError):
            plane_mesh = mesh.plane('y', y=5)
        with pytest.raises(ValueError):
            plane_mesh = mesh.plane('z', x=5)

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

        assert mesh1 | mesh2 is True
        assert mesh1 | mesh3 is False
        assert mesh1 | mesh4 is False
        assert mesh1 | mesh1 is True

    def test_getitem(self):
        # Subregions disctionary
        p1 = (0, 0, 0)
        p2 = (100, 50, 10)
        cell = (5, 5, 5)
        subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 50, 10)),
                      'r2': df.Region(p1=(50, 0, 0), p2=(100, 50, 10))}
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        check_mesh(mesh)

        submesh1 = mesh['r1']
        check_mesh(submesh1)
        assert submesh1.region.pmin == (0, 0, 0)
        assert submesh1.region.pmax == (50, 50, 10)
        assert submesh1.cell == (5, 5, 5)

        submesh2 = mesh['r2']
        check_mesh(submesh2)
        assert submesh2.region.pmin == (50, 0, 0)
        assert submesh2.region.pmax == (100, 50, 10)
        assert submesh2.cell == (5, 5, 5)

        assert len(submesh1) + len(submesh2) == len(mesh)

        # "newly-defined" region
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

        submesh = mesh[df.Region(p1=(0.1, 2.2, 4.01), p2=(4.9, 3.8, 5.7))]
        check_mesh(submesh)
        assert submesh.region.pmin == (0, 2, 4)
        assert submesh.region.pmax == (5, 4, 6)
        assert submesh.cell == cell
        assert submesh.n == (5, 2, 2)
        assert mesh[mesh.region] == mesh

        p1 = (20e-9, 0, 15e-9)
        p2 = (-25e-9, 100e-9, -5e-9)
        cell = (5e-9, 5e-9, 5e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

        submesh = mesh[df.Region(p1=(11e-9, 22e-9, 1e-9),
                                 p2=(-9e-9, 79e-9, 14e-9))]
        check_mesh(submesh)
        assert np.allclose(submesh.region.pmin, (-10e-9, 20e-9, 0),
                           atol=1e-15, rtol=1e-5)
        assert np.allclose(submesh.region.pmax, (15e-9, 80e-9, 15e-9),
                           atol=1e-15, rtol=1e-5)
        assert submesh.cell == cell
        assert submesh.n == (5, 12, 3)
        assert mesh[mesh.region] == mesh

        with pytest.raises(ValueError):
            submesh = mesh[df.Region(p1=(11e-9, 22e-9, 1e-9),
                                     p2=(200e-9, 79e-9, 14e-9))]

    def test_pad(self):
        p1 = (-1, 2, 7)
        p2 = (5, 9, 4)
        cell = (1, 1, 1)
        region = df.Region(p1=p1, p2=p2)
        mesh = df.Mesh(region=region, cell=cell)

        padded_mesh = mesh.pad({'x': (0, 1)})
        assert padded_mesh.region.pmin == (-1, 2, 4)
        assert padded_mesh.region.pmax == (6, 9, 7)
        assert padded_mesh.n == (7, 7, 3)

        padded_mesh = mesh.pad({'y': (1, 1)})
        assert padded_mesh.region.pmin == (-1, 1, 4)
        assert padded_mesh.region.pmax == (5, 10, 7)
        assert padded_mesh.n == (6, 9, 3)

        padded_mesh = mesh.pad({'z': (2, 3)})
        assert padded_mesh.region.pmin == (-1, 2, 2)
        assert padded_mesh.region.pmax == (5, 9, 10)
        assert padded_mesh.n == (6, 7, 8)

        padded_mesh = mesh.pad({'x': (1, 1), 'y': (1, 1), 'z': (1, 1)})
        assert padded_mesh.region.pmin == (-2, 1, 3)
        assert padded_mesh.region.pmax == (6, 10, 8)
        assert padded_mesh.n == (8, 9, 5)

    def test_getattr(self):
        p1 = (0, 0, 0)
        p2 = (100e-9, 80e-9, 10e-9)
        cell = (1e-9, 5e-9, 10e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert mesh.dx == 1e-9
        assert mesh.dy == 5e-9
        assert mesh.dz == 10e-9

        with pytest.raises(AttributeError):
            res = mesh.dk

    def test_dir(self):
        p1 = (0, 0, 0)
        p2 = (100e-9, 80e-9, 10e-9)
        cell = (1e-9, 5e-9, 10e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        assert all([i in dir(mesh) for i in ['dx', 'dy', 'dz']])

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

        assert mesh.plane('x').dS.average == (5, 0, 0)
        assert mesh.plane('y').dS.average == (0, 2.5, 0)
        assert mesh.plane('z').dS.average == (0, 0, 2)

        # Exception
        with pytest.raises(ValueError):
            res = mesh.dS

    def test_mpl(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n, cell=cell)
            mesh.mpl()

            filename = 'figure.pdf'
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfilename = os.path.join(tmpdir, filename)
                mesh.mpl(filename=tmpfilename)

            plt.close('all')

    def test_k3d(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            mesh.k3d()
            mesh.plane('x').k3d()

    def test_k3d_mpl_subregions(self):
        p1 = (0, 0, 0)
        p2 = (100, 80, 10)
        cell = (100, 5, 10)
        subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(100, 10, 10)),
                      'r2': df.Region(p1=(0, 10, 0), p2=(100, 20, 10)),
                      'r3': df.Region(p1=(0, 20, 0), p2=(100, 30, 10)),
                      'r4': df.Region(p1=(0, 30, 0), p2=(100, 40, 10)),
                      'r5': df.Region(p1=(0, 40, 0), p2=(100, 50, 10)),
                      'r6': df.Region(p1=(0, 50, 0), p2=(100, 60, 10)),
                      'r7': df.Region(p1=(0, 60, 0), p2=(100, 70, 10)),
                      'r8': df.Region(p1=(0, 70, 0), p2=(100, 80, 10))}
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        # matplotlib tests
        mesh.mpl_subregions()

        filename = 'figure.pdf'
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            mesh.mpl_subregions(filename=tmpfilename)

        plt.close('all')

        # k3d tests
        mesh.k3d_subregions()

    def test_slider(self):
        p1 = (-10e-9, -5e-9, 10e-9)
        p2 = (10e-9, 5e-9, 0)
        cell = (1e-9, 2.5e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

        x_slider = mesh.slider('x')
        assert isinstance(x_slider, ipywidgets.SelectionSlider)

        y_slider = mesh.slider('y', multiplier=1)
        assert isinstance(x_slider, ipywidgets.SelectionSlider)

        z_slider = mesh.slider('z', multiplier=1e3)
        assert isinstance(x_slider, ipywidgets.SelectionSlider)

    def test_axis_selector(self):
        p1 = (-10e-9, -5e-9, 10e-9)
        p2 = (10e-9, 5e-9, 0)
        cell = (1e-9, 2.5e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        check_mesh(mesh)

        axis_widget = mesh.axis_selector()
        assert isinstance(axis_widget, ipywidgets.Dropdown)

        axis_widget = mesh.axis_selector(widget='radiobuttons')
        assert isinstance(axis_widget, ipywidgets.RadioButtons)

        axis_widget = mesh.axis_selector(description='something')
        assert isinstance(axis_widget, ipywidgets.Dropdown)

        with pytest.raises(ValueError):
            axis_widget = mesh.axis_selector(widget='something')
