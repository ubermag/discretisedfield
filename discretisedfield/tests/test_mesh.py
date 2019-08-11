import types
import pytest
import numpy as np
import discretisedfield as df


def check_mesh(mesh):
    assert isinstance(mesh.p1, tuple)
    assert len(mesh.p1) == 3
    assert isinstance(mesh.p2, tuple)
    assert len(mesh.p2) == 3
    assert isinstance(mesh.cell, tuple)
    assert len(mesh.cell) == 3
    assert isinstance(mesh.n, tuple)
    assert len(mesh.n) == 3
    assert all(isinstance(i, int) for i in mesh.n)
    assert isinstance(mesh.pmin, tuple)
    assert len(mesh.pmin) == 3
    assert isinstance(mesh.pmax, tuple)
    assert len(mesh.pmax) == 3
    assert isinstance(mesh.l, tuple)
    assert len(mesh.l) == 3
    assert isinstance(mesh.ntotal, int)
    assert isinstance(mesh.pbc, set)
    assert all(isinstance(i, str) for i in mesh.pbc)
    assert isinstance(mesh.regions, dict)
    assert isinstance(mesh.name, str)


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
                            (3, 50, 2), None],
                           [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0)),
                            None, (0.5e-9, 1e-9, 5e-9)],
                           [(-1.5e-9, -5e-9, -5e-9), np.array((0, 0, 0)),
                            (5, 5, 7), None],
                           [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6),
                            None, (0.5e-6, 1e-6, 2.5e-6)],
                           [[0, 5e-6, 0], (-1.5e-6, -5e-6, -5e-6),
                            (1, 10, 100), None],
                           [(0, 125e-9, 0), (500e-9, 0, -3e-9),
                            None, (2.5e-9, 2.5e-9, 3e-9)]]

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

    def test_init_with_cell(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        cell = (1, 0.1, 0.5)
        pbc = 'yxz'
        name = 'test_mesh'
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, pbc=pbc, name=name)

        check_mesh(mesh)
        assert mesh.p1 == p1
        assert mesh.p2 == p2
        assert mesh.cell == cell
        assert mesh.pbc == set(pbc)
        assert mesh.name == name
        assert mesh.pmin == (0, -4, 11)
        assert mesh.pmax == (15, 10.1, 16.5)
        assert mesh.l[0] == 15 - 0
        assert mesh.l[1] == 10.1 - (-4)
        assert mesh.l[2] == 16.5 - 11
        assert mesh.n[0] == (15 - 0) / 1.0
        assert mesh.n[1] == (10.1 - (-4)) / 0.1
        assert mesh.n[2] == (16.5 - 11) / 0.5
        assert mesh.ntotal == 23265

    def test_init_with_n(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        n = (15, 141, 11)
        pbc = 'yx'
        name = 'test_mesh'
        mesh = df.Mesh(p1=p1, p2=p2, n=n, pbc=pbc, name=name)

        check_mesh(mesh)
        assert mesh.p1 == p1
        assert mesh.p2 == p2
        assert mesh.pbc == set(pbc)
        assert mesh.name == name
        assert mesh.pmin == (0, -4, 11)
        assert mesh.pmax == (15, 10.1, 16.5)
        assert mesh.l[0] == 15 - 0
        assert mesh.l[1] == 10.1 - (-4)
        assert mesh.l[2] == 16.5 - 11
        assert mesh.cell[0] == (15 - 0) / 15
        assert mesh.cell[1] == (10.1 - (-4)) / 141
        assert mesh.cell[2] == (16.5 - 11) / 11
        assert mesh.ntotal == 15*141*11

    def test_init_with_n_and_cell(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        n = (15, 141, 11)
        cell = (1, 0.1, 0.5)
        pbc = 'x'
        name = 'test_mesh'
        with pytest.raises(ValueError) as excinfo:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell, pbc=pbc, name=name)
        assert 'One and only one' in str(excinfo.value)

    def test_init_valid_args(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)

            check_mesh(mesh)
            assert mesh.name == 'mesh'  # default name value
            assert mesh.pbc == set()  # default pbc value

    def test_init_invalid_args(self):
        for p1, p2, n, cell in self.invalid_args:
            with pytest.raises((TypeError, ValueError)):
                # Exceptions are raised by descriptors.
                mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)

    def test_init_invalid_name(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        n = (10, 100, 2)
        for name in ['mesh name', '2name', ' ', 5, '-name', '+mn']:
            with pytest.raises((TypeError, ValueError)):
                # Exception is raised by the descriptor for mesh.name.
                mesh = df.Mesh(p1=p1, p2=p2, n=n, name=name)

    def test_init_valid_pbc(self):
        for p1, p2, n, cell in self.valid_args:
            for pbc in ['x', 'z', 'zx', 'yxzz', 'yz']:
                mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell, pbc=pbc)

                check_mesh(mesh)
                assert mesh.pbc == set(pbc)

    def test_init_invalid_pbc(self):
        for p1, p2, n, cell in self.valid_args:
            for pbc in ['abc', 'a', '123', 5]:
                with pytest.raises((ValueError, TypeError)):
                    # Exception is raised by the descriptor for mesh.pbc.
                    mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell, pbc=pbc)

    def test_zero_domain_edge_length(self):
        args = [[(0, 100e-9, 1e-9), (150e-9, 100e-9, 6e-9),
                 None, (1e-9, 0.01e-9, 1e-9)],
                [(0, 100e-9, 0), (150e-9, 101e-9, 0),
                 None, (2e-9, 1e-9, 0.1e-9)],
                [(0, 101e-9, -1), (150e-9, 101e-9, 0),
                 (1, 1, 1), None],
                [(10e9, 10e3, 0), (0.01e12, 11e3, 5),
                 (5, 100, 30), None]]

        for p1, p2, n, cell in args:
            with pytest.raises(ValueError) as excinfo:
                mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            assert 'is zero' in str(excinfo.value)

    def test_domain_not_aggregate_of_cell(self):
        args = [[(0, 100e-9, 1e-9), (150e-9, 120e-9, 6e-9),
                 None, (4e-9, 1e-9, 1e-9)],
                [(0, 100e-9, 0), (150e-9, 104e-9, 1e-9),
                 None, (2e-9, 1.5e-9, 0.1e-9)],
                [(10e9, 10e3, 0), (11e9, 11e3, 5),
                 None, (1e9, 1e3, 1.5)]]

        for p1, p2, n, cell in args:
            with pytest.raises(ValueError) as excinfo:
                mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            assert 'not an aggregate' in str(excinfo.value)

    def test_cell_greater_than_domain(self):
        p1 = (0, 0, 0)
        p2 = (1e-9, 1e-9, 1e-9)
        args = [(2e-9, 1e-9, 1e-9), (1e-9, 2e-9, 1e-9),
                (1e-9, 1e-9, 2e-9), (1e-9, 5e-9, 0.1e-9)]
        for cell in args:
            with pytest.raises(ValueError) as excinfo:
                mymesh = df.Mesh(p1=p1, p2=p2, cell=cell)
            assert 'not an aggregate' in str(excinfo.value)

    def test_regions(self):
        p1 = (0, 0, 0)
        p2 = (100, 50, 10)
        cell = (1, 1, 1)
        regions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 50, 10)),
                   'r2': df.Region(p1=(50, 0, 0), p2=(100, 50, 10))}
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, regions=regions)

        assert (0, 0, 0) in mesh.regions['r1']
        assert (0, 0, 0) not in mesh.regions['r2']
        assert (25, 25, 5) in mesh.regions['r1']
        assert (25, 25, 5) not in mesh.regions['r2']
        assert (51, 10, 10) in mesh.regions['r2']
        assert (51, 10, 10) not in mesh.regions['r1']
        assert (100, 50, 10) in mesh.regions['r2']
        assert (100, 50, 10) not in mesh.regions['r1']

    def test_centre(self):
        p1 = (0, 0, 0)
        p2 = (100, 100, 100)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert mesh.centre == (50, 50, 50)
        assert mesh.centre in mesh

        p1 = (-18.5, 10, 0)
        p2 = (10, 5, 10)
        cell = (0.1, 0.25, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert mesh.centre == (-4.25, 7.5, 5)
        assert mesh.centre in mesh

        p1 = (500e-9, 125e-9, 3e-9)
        p2 = (0, 0, 0)
        n = (250, 125, 3)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        assert mesh.centre == (250e-9, 62.5e-9, 1.5e-9)
        assert mesh.centre in mesh

    def test_random_point(self):
        p1 = (-18.5, 5, 0)
        p2 = (10, -10, 10e-9)
        n = (100, 5, 10)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        for _ in range(50):
            random_point = mesh.random_point()
            assert isinstance(random_point, tuple)
            assert random_point in mesh

    def test_repr_no_pbc(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        cell = (1, 0.1, 0.5)
        name = 'meshname'
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name=name)

        rstr = ('Mesh(p1=(-1, -4, 11), p2=(15, 10.1, 12.5), '
                'cell=(1, 0.1, 0.5), pbc=set(), name=\'meshname\')')
        assert repr(mesh) == rstr

    def test_repr_pbc(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10, 12.5)
        n = (16, 140, 3)
        pbc = 'x'
        name = 'meshname'
        mesh = df.Mesh(p1=p1, p2=p2, n=n, pbc=pbc, name=name)

        rstr = ('Mesh(p1=(-1, -4, 11), p2=(15, 10, 12.5), '
                'cell=(1.0, 0.1, 0.5), pbc={\'x\'}, name=\'meshname\')')
        assert repr(mesh) == rstr

    def test_index2point(self):
        p1 = (15, -4, 12.5)
        p2 = (-1, 10.1, 11)
        cell = (1, 0.1, 0.5)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

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

        p1 = (-10, 5, 0)
        p2 = (10, -5, 10e-9)
        n = (10, 5, 5)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

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
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        for p in [(-0.5, -3.95, 11.25), (14.5, 10.05, 12.25)]:
            assert isinstance(mesh.index2point(mesh.point2index(p)), tuple)
            assert mesh.index2point(mesh.point2index(p)) == p

        for i in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            assert isinstance(mesh.point2index(mesh.index2point(i)), tuple)
            assert mesh.point2index(mesh.index2point(i)) == i

    def test_indices(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        n = (10, 10, 10)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        assert len(list(mesh.indices)) == 1000
        for index in mesh.indices:
            assert isinstance(index, tuple)
            assert len(index) == 3
            assert all([0 <= i <= 9 for i in index])

    def test_coordinates(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        assert len(list(mesh.coordinates)) == 1000
        for coord in mesh.coordinates:
            assert isinstance(coord, tuple)
            assert len(coord) == 3
            assert all([0.5 <= i <= 9.5 for i in coord])

    def test_iter(self):
        p1 = (1, 1, 1)
        p2 = (5, 5, 5)
        n = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        assert len(list(mesh)) == 8
        for coord in mesh:
            assert isinstance(coord, tuple)
            assert len(coord) == 3
            assert all([1.5 <= i <= 4.5 for i in coord])

    def test_contains(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 20e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        assert (0, 0, 0) in mesh
        assert (10e-9, 10e-9, 10e-9) in mesh
        assert (5e-9, 5e-9, 5e-9) in mesh
        assert (11e-9, 11e-9, 11e-9) not in mesh
        assert (-1e-9, -1e-9, -1e-9) not in mesh

    def test_line(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        tol = 1e-12
        line = mesh.line((0, 0, 0), (10, 10, 10), n=10)
        assert isinstance(line, types.GeneratorType)
        assert len(list(line))
        for point in line:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert all([0 <= i <= 10 for i in point])

        line = list(mesh.line((0, 0, 0), (10, 0, 0), n=30))
        assert len(line) == 30
        assert line[0] == (0, 0, 0)
        assert line[-1] == (10, 0, 0)

        with pytest.raises(ValueError):
            line = list(mesh.line((-1e-9, 0, 0), (10, 0, 0), n=30))

        with pytest.raises(ValueError):
            line = list(mesh.line((0, 0, 0), (11, 0, 0), n=30))

    def test_plane(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        plane = mesh.plane(z=3, n=(2, 2))
        assert isinstance(plane, df.Mesh)
        assert len(list(plane)) == 4
        for point in plane:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert point[2] == 3

        plane = mesh.plane(y=9.2, n=(3, 2))
        assert isinstance(plane, df.Mesh)
        assert len(list(plane)) == 6
        for point in plane:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert point[1] == 9.2

        plane = mesh.plane('x')
        assert isinstance(plane, df.Mesh)
        assert len(list(plane)) == 100
        for point in plane:
            assert isinstance(point, tuple)
            assert len(point) == 3
            assert point[0] == 5

        with pytest.raises(ValueError):
            plane = list(mesh.plane(x=-1))

        with pytest.raises(ValueError):
            plane = list(mesh.plane(y=11))

        with pytest.raises(ValueError):
            plane = list(mesh.plane(z=-1e-9))

        with pytest.raises(ValueError):
            plane = list(mesh.plane('x', z=-1e-9))

        with pytest.raises(ValueError):
            plane = list(mesh.plane('z', z=-1e-9))

    def test_mpl(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            mesh.mpl()

    def test_k3d(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            mesh.k3d()

    def test_k3d_points(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            mesh.k3d_points()

    def test_k3d_regions(self):
        p1 = (0, 0, 0)
        p2 = (100, 80, 10)
        cell = (2, 2, 2)
        regions = {'r1': df.Region(p1=(0, 0, 0), p2=(100, 10, 10)),
                   'r2': df.Region(p1=(0, 10, 0), p2=(100, 20, 10)),
                   'r3': df.Region(p1=(0, 20, 0), p2=(100, 30, 10)),
                   'r4': df.Region(p1=(0, 30, 0), p2=(100, 40, 10)),
                   'r5': df.Region(p1=(0, 40, 0), p2=(100, 50, 10)),
                   'r6': df.Region(p1=(0, 50, 0), p2=(100, 60, 10)),
                   'r7': df.Region(p1=(0, 60, 0), p2=(100, 70, 10)),
                   'r8': df.Region(p1=(0, 70, 0), p2=(100, 80, 10))}
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, regions=regions)
        mesh.k3d_regions()

    def test_script(self):
        for p1, p2, n, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, n=n, cell=cell)
            with pytest.raises(NotImplementedError):
                script = mesh._script

    def test_change_const_attributes(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        name = 'object_name'
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name=name)

        assert mesh.p1 == p1
        assert mesh.p2 == p2
        assert mesh.cell == cell
        assert mesh.name == name
        assert mesh.l == (10, 10, 10)
        assert mesh.n == (10, 10, 10)

        # Attempt to change attribute
        with pytest.raises(AttributeError):
            mesh.p2 = (15, 15, 15)
        with pytest.raises(AttributeError):
            mesh.name = 'new_object_name'

        # Make sure the attributes have not changed.
        assert mesh.p1 == p1
        assert mesh.p2 == p2
        assert mesh.cell == cell
        assert mesh.name == name
        assert mesh.l == (10, 10, 10)
        assert mesh.n == (10, 10, 10)
