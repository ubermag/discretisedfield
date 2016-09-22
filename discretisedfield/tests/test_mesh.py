import pytest
from discretisedfield.mesh import Mesh


class TestMesh(object):
    def setup(self):
        self.valid_args = [[(0, 0, 0),
                            (5, 5, 5),
                            (1, 1, 1)],
                           [(0, 0, 0),
                            [5e-9, 5e-9, 5e-9],
                            (1e-9, 1e-9, 1e-9)],
                           [(-1.5e-9, -5e-9, 0),
                            (1.5e-9, 15e-9, 16e-9),
                            (3e-9, 1e-9, 1e-9)],
                           [(-1.5e-9, -5e-9, -5e-9),
                            (0, 0, 0),
                            (0.5e-9, 1e-9, 5e-9)]]

        self.invalid_args = [[(0, 0, 0),
                              (5, 5, 5),
                              (-1, 1, 1)],
                             [(0, 0, 0),
                              (5e-9, 5e-9, 5e-9),
                              (2, 2, 2)],
                             ['1',
                              (1, 1, 1),
                              (0, 0, 1e-9)],
                             [(-1.5e-9, -5e-9, 'a'),
                              (1.5e-9, 15e-9, 16e-9),
                              (5, 1, -1e-9)],
                             [(-1.5e-9, -5e-9, 0),
                              (1.5e-9, 15e-9, 16e-9),
                              (-2e-9, 1, 1e-9)],
                             [(-1.5e-9, -5e-9, 0),
                              (1.5e-9, 16e-9),
                              (5, 1, 1e-9)],
                             [(-1.5e-9, -5e-9, 0),
                              (1.5e-9, 15e-9, 1+2j),
                              (5, 1, 1e-9)],
                             ['string', (5, 1, 1e-9), 'string'],
                             [(-1.5e-9, -5e-9, 0),
                              (1.5e-9, 15e-9, 16e-9),
                              1]]

    def test_init(self):
        c1 = (0, -4, 11)
        c2 = (15, 10.1, 16.5)
        d = (1, 0.1, 0.5)
        name = 'test_mesh'

        mesh = Mesh(c1, c2, d, name=name)

        assert mesh.l[0] == 15 - 0
        assert mesh.l[1] == 10.1 - (-4)
        assert mesh.l[2] == 16.5 - 11

        assert mesh.n[0] == (15 - 0) / 1.0
        assert mesh.n[1] == (10.1 - (-4)) / 0.1
        assert mesh.n[2] == (16.5 - 11) / 0.5

        assert isinstance(mesh.n[0], int)
        assert isinstance(mesh.n[1], int)
        assert isinstance(mesh.n[2], int)

        assert mesh.name == name

    def test_init_valid_args(self):
        for arg in self.valid_args:
            c1 = arg[0]
            c2 = arg[1]
            d = arg[2]

            mesh = Mesh(c1, c2, d)

            assert mesh.c1 == c1
            assert mesh.c2 == c2
            assert mesh.d == d

    def test_init_invalid_args(self):
        for arg in self.invalid_args:
            with pytest.raises(TypeError):
                c1 = arg[0]
                c2 = arg[1]
                d = arg[2]

                mesh = Mesh(c1, c2, d)

    def test_init_d_not_multiple_of_l(self):
        c1 = (0, -4, 11)
        c2 = (15, 10, 16)
        d = (1, 5, 1)
        name = 'test_field'

        with pytest.raises(TypeError):
            mesh = Mesh(c1, c2, d, name=name)

    def test_init_wrong_name(self):
        c1 = (0, -4, 11)
        c2 = (15, 10.1, 16.5)
        d = (1, 0.1, 0.5)
        name = 552

        with pytest.raises(TypeError):
            mesh = Mesh(c1, c2, d, name=name)

    def test_index2coord(self):
        c1 = (-1, -4, 11)
        c2 = (15, 10.1, 12.5)
        d = (1, 0.1, 0.5)
        name = 'test_field'

        mesh = Mesh(c1, c2, d, name=name)

        assert mesh.index2coord((0, 0, 0)) == (-0.5, -3.95, 11.25)
        assert mesh.index2coord((5, 10, 1)) == (4.5, -2.95, 11.75)

    def test_index2coord_exception(self):
        c1 = (-1, -4, 11)
        c2 = (15, 10.1, 12.5)
        d = (1, 0.1, 0.5)
        name = 'test_field'

        mesh = Mesh(c1, c2, d, name=name)

        with pytest.raises(ValueError):
            mesh.index2coord((-1, 0, 0))
            mesh.index2coord((500, 10, 1))

    def test_coord2index(self):
        c1 = (-10, -5, 0)
        c2 = (10, 5, 10)
        d = (1, 5, 1)
        name = 'test_field'

        mesh = Mesh(c1, c2, d, name=name)

        assert mesh.coord2index((-10, -5, 0)) == (0, 0, 0)
        assert mesh.n[0] == 20
        assert mesh.coord2index((10, 5, 10)) == (19, 1, 9)
        assert mesh.coord2index((0.0001, 0.0001, 5.0001)) == (10, 1, 5)
        assert mesh.coord2index((-0.0001, -0.0001, 4.9999)) == (9, 0, 4)

    def test_coord2index_exception(self):
        c1 = (-10, -5, 0)
        c2 = (10, 5, 10)
        d = (1, 5, 1)
        name = 'test_field'

        mesh = Mesh(c1, c2, d, name=name)

        with pytest.raises(ValueError):
            mesh.coord2index((-11, 0, 5))
            mesh.coord2index((-5, -5-1e-3, 5))
            mesh.coord2index((-5, 0, -0.01))
            mesh.coord2index((11, 0, 5))
            mesh.coord2index((6, 5+1e-6, 5))
            mesh.coord2index((0, 0, 10+1e-10))

    def test_domain_centre(self):
        c1 = (-18.5, 5, 0)
        c2 = (10, 10, 10)
        d = (0.1, 0.25, 2)
        name = 'test_field'

        mesh = Mesh(c1, c2, d, name=name)

        assert mesh.domain_centre() == (-4.25, 7.5, 5)

    def test_random_coord(self):
        c1 = (-18.5, 5, 0)
        c2 = (10, 10, 10)
        d = (0.1, 0.25, 2)
        name = 'test_field'

        mesh = Mesh(c1, c2, d, name=name)

        for j in range(500):
            c = mesh.random_coord()
            assert mesh.c1[0] <= c[0] <= mesh.c2[0]
            assert mesh.c1[1] <= c[1] <= mesh.c2[1]
            assert mesh.c1[2] <= c[2] <= mesh.c2[2]

    def test_plot_mesh(self):
        for arg in self.valid_args:
            c1 = arg[0]
            c2 = arg[1]
            d = arg[2]

            mesh = Mesh(c1, c2, d)

            mesh.plot_mesh()

    def test_script(self):
        for arg in self.valid_args:
            c1 = arg[0]
            c2 = arg[1]
            d = arg[2]

            mesh = Mesh(c1, c2, d)
            with pytest.raises(NotImplementedError):
                mesh.script()

    def test_discretirsation_greater_or_not_multiple_of_domain(self):
        c1 = (0, 0, 0)
        c2 = (1., 1., 1.)

        for d in [(2, 1, 1), (1, 2, 1), (1, 1, 3), (1, 5, 0.1)]:
            with pytest.raises(TypeError) as excinfo:
                mymesh = Mesh(c1, c2, d)
            assert 'Discretisation' in str(excinfo.value)
            assert 'cell' in str(excinfo.value)
            assert 'greater' in str(excinfo.value)
            assert 'multiple' in str(excinfo.value)
            assert 'domain' in str(excinfo.value)
