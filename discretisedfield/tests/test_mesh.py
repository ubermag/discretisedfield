import pytest
import matplotlib
import numpy as np
import discretisedfield as df


class TestMesh(object):
    def setup(self):
        self.valid_args = [[(0, 0, 0),
                            (5, 5, 5),
                            [1, 1, 1]],
                           [(0, 0, 0),
                            (5e-9, 5e-9, 5e-9),
                            (1e-9, 1e-9, 1e-9)],
                           [(-1.5e-9, -5e-9, 0),
                            (1.5e-9, -15e-9, -160e-9),
                            (0.15e-9, 0.1e-9, 10e-9)],
                           [(-1.5e-9, -5e-9, -5e-9),
                            np.array((0, 0, 0)),
                            (0.5e-9, 1e-9, 5e-9)],
                           [[0, 5e-6, 0],
                            (-1.5e-6, -5e-6, -5e-6),
                            (0.5e-6, 1e-6, 5e-9)],
                           [(0, 125e-9, 0),
                            (500e-9, 0, -3e-9),
                            (2.5e-9, 2.5e-9, 3e-9)]]

        self.invalid_args = [[(0, 0, 0),
                              (5, 5, 5),
                              (-1, 1, 1)],
                             [("1", 0, 0),
                              (1, 1, 1),
                              (0, 0, 1e-9)],
                             [(-1.5e-9, -5e-9, "a"),
                              (1.5e-9, 15e-9, 16e-9),
                              (5, 1, -1e-9)],
                             [(-1.5e-9, -5e-9, 0),
                              (1.5e-9, 16e-9),
                              (0.1e-9, 0.1e-9, 1e-9)],
                             [(-1.5e-9, -5e-9, 0),
                              (1.5e-9, 15e-9, 1+2j),
                              (5, 1, 1e-9)],
                             ["string",
                              (5, 1, 1e-9),
                              "string"],
                             [(-1.5e-9, -5e-9, 0),
                              (1.5e-9, 15e-9, 16e-9),
                              2+2j]]

    def test_init(self):
        p1 = (0, -4, 16.5)
        p2 = (15, 10.1, 11)
        cell = (1, 0.1, 0.5)
        name = "test_mesh"
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name=name)

        assert isinstance(mesh.p1, tuple)
        assert mesh.p1 == p1
        assert isinstance(mesh.p2, tuple)
        assert mesh.p2 == p2
        assert isinstance(mesh.cell, tuple)
        assert mesh.cell == cell
        assert isinstance(mesh.name, str)
        assert mesh.name == name
        assert isinstance(mesh.pmin, tuple)
        assert mesh.pmin == (0, -4, 11)
        assert isinstance(mesh.pmax, tuple)
        assert mesh.pmax == (15, 10.1, 16.5)

        assert isinstance(mesh.l, tuple)
        assert len(mesh.l) == 3
        assert mesh.l[0] == 15 - 0
        assert mesh.l[1] == 10.1 - (-4)
        assert mesh.l[2] == 16.5 - 11

        assert isinstance(mesh.n, tuple)
        assert len(mesh.n) == 3
        assert mesh.n[0] == (15 - 0) / 1.0
        assert mesh.n[1] == (10.1 - (-4)) / 0.1
        assert mesh.n[2] == (16.5 - 11) / 0.5
        for i in range(3):
            assert isinstance(mesh.n[i], int)

    def test_init_valid_args(self):
        for p1, p2, cell in self.valid_args:
            mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

            assert isinstance(mesh.p1, tuple)
            assert mesh.p1 == tuple(p1)
            assert isinstance(mesh.p2, tuple)
            assert mesh.p2 == tuple(p2)
            assert isinstance(mesh.cell, tuple)
            assert mesh.cell == tuple(cell)
            assert isinstance(mesh.name, str)
            assert mesh.name == "mesh"  # default name value
            assert isinstance(mesh.l, tuple)
            assert isinstance(mesh.n, tuple)
            for i in range(3):
                assert isinstance(mesh.n[i], int)

    def test_init_invalid_args(self):
        for p1, p2, cell in self.invalid_args:
            with pytest.raises(TypeError):
                # Exceptions are raised by descriptors.
                mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def test_init_invalid_name(self):
        for name in ["mesh name", "2name", " ", 5, "-name", "+mn"]:
            with pytest.raises(TypeError):
                # Exception is raised by the descriptor (mesh.name).
                mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10),
                               cell=(1, 1, 1), name=name)

    def test_zero_domain_edge_length(self):
        args = [[(0, 100e-9, 1e-9),
                 (150e-9, 100e-9, 6e-9),
                 (1e-9, 0.01e-9, 1e-9)],
                [(0, 100e-9, 0),
                 (150e-9, 101e-9, 0),
                 (2e-9, 1e-9, 0.1e-9)],
                [(10e9, 10e3, 0),
                 (0.01e12, 11e3, 5),
                 (1e9, 1e3, 1)]]

        for p1, p2, cell in args:
            with pytest.raises(ValueError) as excinfo:
                mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
            assert "edge length is zero" in str(excinfo)

    def test_domain_not_aggregate_of_cell(self):
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
            assert "not an aggregate" in str(excinfo)

    def test_cell_greater_than_domain(self):
        p1 = (0, 0, 0)
        p2 = (1e-9, 1e-9, 1e-9)
        for cell in [(2e-9, 1e-9, 1e-9), (1e-9, 2e-9, 1e-9),
                     (1e-9, 1e-9, 2e-9), (1e-9, 5e-9, 0.1e-9)]:
            with pytest.raises(ValueError) as excinfo:
                mymesh = df.Mesh(p1=p1, p2=p2, cell=cell)
            assert "greater than the domain" in str(excinfo)

    def test_centre(self):
        p1 = (0, 0, 0)
        p2 = (100, 100, 100)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert mesh.centre == (50, 50, 50)

        p1 = (-18.5, 10, 0)
        p2 = (10, 5, 10)
        cell = (0.1, 0.25, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        assert mesh.centre == (-4.25, 7.5, 5)

        p1 = (500e-9, 125e-9, 3e-9)
        p2 = (0, 0, 0)
        cell = (10e-9, 5e-9, 1e-9)
        mesh = df.Mesh(p1, p2, cell)
        assert mesh.centre == (250e-9, 62.5e-9, 1.5e-9)

    def test_random_point(self):
        p1 = (-18.5, 5, 0)
        p2 = (10, -10, 10e-9)
        cell = (0.1e-9, 0.25, 2e-9)
        mesh = df.Mesh(p1, p2, cell)

        for _ in range(20):
            p = mesh.random_point()
            assert isinstance(p, tuple)
            for i in range(3):
                assert mesh.pmin[i] <= p[i] <= mesh.pmax[i]

    def test_repr(self):
        p1 = (-1, -4, 11)
        p2 = (15, 10.1, 12.5)
        cell = (1, 0.1, 0.5)
        name = "meshname"
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name=name)

        rstr = ("Mesh(p1=(-1, -4, 11), p2=(15, 10.1, 12.5), "
                "cell=(1, 0.1, 0.5), name=\"meshname\")")
        assert repr(mesh) == rstr

    def test_index2point(self):
        p1 = (15, -4, 12.5)
        p2 = (-1, 10.1, 11)
        cell = (1, 0.1, 0.5)
        mesh = df.Mesh(p1, p2, cell)

        assert mesh.index2point((0, 0, 0)) == (-0.5, -3.95, 11.25)
        assert mesh.index2point((5, 10, 1)) == (4.5, -2.95, 11.75)
        assert mesh.index2point((15, 140, 2)) == (14.5, 10.05, 12.25)

        # Correct minimum index
        assert isinstance(mesh.index2point((0, 0, 0)), tuple)
        # Below minimum index
        with pytest.raises(ValueError):
            mesh.index2point((-1, 0, 0))
        with pytest.raises(ValueError):
            mesh.index2point((0, -1, 0))
        with pytest.raises(ValueError):
            mesh.index2point((0, 0, -1))

        # Correct maximum index
        assert isinstance(mesh.index2point((15, 140, 2)), tuple)
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
        mesh = df.Mesh(p1, p2, cell)

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
        cell = (1, 5, 1e-9)
        mesh = df.Mesh(p1, p2, cell)

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
        mesh = df.Mesh(p1, p2, cell)

        for i in [(-0.5, -3.95, 11.25), (14.5, 10.05, 12.25)]:
            assert isinstance(mesh.index2point(mesh.point2index(i)), tuple)
            assert mesh.index2point(mesh.point2index(i)) == i

        for i in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            assert isinstance(mesh.point2index(mesh.index2point(i)), tuple)
            assert mesh.point2index(mesh.index2point(i)) == i

    def test_plot(self):
        for arg in self.valid_args:
            p1, p2, cell = arg
            mesh = df.Mesh(p1, p2, cell)

            assert isinstance(mesh.plot(), matplotlib.figure.Figure)

    def test_indices(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1, p2, cell)

        counter = 0
        for index in mesh.indices():
            assert isinstance(index, tuple)
            assert len(index) == 3
            for i in range(3):
                assert 0 <= index[i] <= 9
            counter += 1

        assert counter == 1000

    def test_coordinates(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1, p2, cell)

        counter = 0
        for coord in mesh.coordinates():
            assert isinstance(coord, tuple)
            assert len(coord) == 3
            for i in range(3):
                assert 0.5 <= coord[i] <= 9.5
            counter += 1

        assert counter == 1000

    def test_line_intersection(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1, p2, cell)

        tol = 1e-12
        li = mesh.line_intersection((1, 1, 1), (5, 5, 5), n=10)
        for point in li:
            assert isinstance(point, tuple)
            assert len(point) == 2

            d, p = point
            assert isinstance(d, float)
            assert 0 <= d <= 10*np.sqrt(3) + tol
            assert isinstance(p, tuple)
            assert len(p) == 3
            for j in range(3):
                assert 0 <= p[j] <= 10

        li = list(mesh.line_intersection((1, 0, 0), (0, 0, 0), n=30))

        assert len(li) == 30
        assert li[0][0] == 0
        assert abs(li[-1][0] - 10) < tol
        assert li[0][1] == (0, 0, 0)
        assert li[-1][1] == (10, 0, 0)

    def test_script(self):
        for arg in self.valid_args:
            p1, p2, cell = arg
            mesh = df.Mesh(p1, p2, cell)
            with pytest.raises(NotImplementedError):
                script = mesh.script
