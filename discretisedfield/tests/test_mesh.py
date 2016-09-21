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
            print(arg)
            with pytest.raises(TypeError):
                c1 = arg[0]
                c2 = arg[1]
                d = arg[2]

                mesh = Mesh(c1, c2, d)

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

        for d in [(2, 1, 1), (1, 0.3, 1), (1, 1, 3), (1, 0.4, 0.4)]:
            with pytest.raises(TypeError) as excinfo:
                mymesh = Mesh(c1, c2, d)
            assert 'Discretisation' in str(excinfo.value)
            assert 'cell' in str(excinfo.value)
            assert 'greater' in str(excinfo.value)
            assert 'multiple' in str(excinfo.value)
            assert 'domain' in str(excinfo.value)
