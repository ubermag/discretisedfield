import os
import pytest
import random
import numpy as np
from discretisedfield import Field, read_oommf_file
import matplotlib


class TestField(object):
    def setup(self):
        self.scalar_fs = self.create_scalar_fs()
        self.vector_fs = self.create_vector_fs()
        self.constant_values = [0, -5., np.pi,
                                1e-15, 1.2e12, random.random()]
        self.tuple_values = [(0, 0, 1),
                             (0, -5.1, np.pi),
                             [70, 1e15, 2*np.pi],
                             [5, random.random(), np.pi],
                             np.array([4, -1, 3.7]),
                             np.array([2.1, 0.0, -5*random.random()])]
        self.scalar_pyfuncs = self.create_scalar_pyfuncs()
        self.vector_pyfuncs = self.create_vector_pyfuncs()

    def create_scalar_fs(self):
        cmin_list = [(0, 0, 0), (-5e-9, -8e-9, -10e-9), (10, -5, -80)]
        cmax_list = [(5e-9, 8e-9, 10e-9), (11e-9, 4e-9, 4e-9), (15, 10, 85)]
        d_list = [(1e-9, 1e-9, 1e-9), (1e-9, 2e-9, 1e-9), (5, 5, 2.5)]

        scalar_fs = []
        for i in range(len(cmin_list)):
            f = Field(cmin_list[i], cmax_list[i], d_list[i], dim=1)
            scalar_fs.append(f)

        return scalar_fs

    def create_vector_fs(self):
        cmin_list = [(0, 0, 0), (-5, -8, -10), (10, -5, -80)]
        cmax_list = [(5, 8, 10), (11, 4, 4), (15, 10, 85)]
        d_list = [(1, 1, 1), (1, 2, 1), (5, 5, 2.5)]

        vector_fs = []
        for i in range(len(cmin_list)):
            f = Field(cmin_list[i], cmax_list[i], d_list[i], dim=3)
            vector_fs.append(f)

        return vector_fs

    def create_scalar_pyfuncs(self):
        f = [lambda c: 1,
             lambda c: -2.4,
             lambda c: -6.4e-15,
             lambda c: c[0] + c[1] + c[2] + 1,
             lambda c: (c[0]-1)**2 - c[1]+7 + c[2]*0.1,
             lambda c: np.sin(c[0]) + np.cos(c[1]) - np.sin(2*c[2])]

        return f

    def create_vector_pyfuncs(self):
        f = [lambda c: (1, 2, 0),
             lambda c: (-2.4, 1e-3, 9),
             lambda c: (c[0], c[1], c[2] + 100),
             lambda c: (c[0]+c[2]+10, c[1], c[2]+1),
             lambda c: (c[0]-1, c[1]+70, c[2]*0.1),
             lambda c: (np.sin(c[0]), np.cos(c[1]), -np.sin(2*c[2]))]

        return f

    def test_init(self):
        cmin = (0, -4, 11)
        cmax = (15, 10.1, 16.5)
        d = (1, 0.1, 0.5)
        name = 'test_field'
        value = [1, 2]

        f = Field(cmin, cmax, d, dim=2, value=value, name=name)

        assert f.l[0] == 15 - 0
        assert f.l[1] == 10.1 - (-4)
        assert f.l[2] == 16.5 - 11

        assert f.n[0] == (15 - 0) / 1.0
        assert f.n[1] == (10.1 - (-4)) / 0.1
        assert f.n[2] == (16.5 - 11) / 0.5

        assert type(f.n[0]) is int
        assert type(f.n[1]) is int
        assert type(f.n[2]) is int

        assert f.name == name

        assert np.all(f.f[:, :, :, 0] == value[0])
        assert np.all(f.f[:, :, :, 1] == value[1])

    def test_init_wrong_cmin(self):
        cmin = 1
        cmax = (15, 10.1, 16.5)
        d = (1, 0.1, 0.5)
        name = 'test_field'

        with pytest.raises(TypeError):
            f = Field(cmin, cmax, d, dim=3, name=name)

    def test_init_wrong_cmax(self):
        cmin = (0, -4, 11)
        cmax = [15, 10.1, 16.5]
        d = (1, 0.1, 0.5)
        name = 'test_field'

        with pytest.raises(TypeError):
            f = Field(cmin, cmax, d, dim=3, name=name)

    def test_init_wrong_d(self):
        cmin = (0, -4, 11)
        cmax = (15, 10, 16)
        d = (0, 1, 1)
        name = 'test_field'

        with pytest.raises(TypeError):
            f = Field(cmin, cmax, d, dim=3, name=name)

    def test_init_wrong_dim(self):
        cmin = (0, -4, 11)
        cmax = (15, 10, 16)
        d = (0.1, 1, 1)
        name = 'test_field'

        with pytest.raises(TypeError):
            f = Field(cmin, cmax, d, dim=3.5, name=name)

    def test_init_d_not_multiple_of_l(self):
        cmin = (0, -4, 11)
        cmax = (15, 10, 16)
        d = (1, 5, 1)
        name = 'test_field'

        with pytest.raises(ValueError):
            f = Field(cmin, cmax, d, dim=3, name=name)

    def test_init_wrong_name(self):
        cmin = (0, -4, 11)
        cmax = (15, 10.1, 16.5)
        d = (1, 0.1, 0.5)
        name = 552

        with pytest.raises(TypeError):
            f = Field(cmin, cmax, d, dim=3, name=name)

    def test_index2coord(self):
        cmin = (-1, -4, 11)
        cmax = (15, 10.1, 12.5)
        d = (1, 0.1, 0.5)
        name = 'test_field'

        f = Field(cmin, cmax, d, dim=2, name=name)

        assert f.index2coord((0, 0, 0)) == (-0.5, -3.95, 11.25)
        assert f.index2coord((5, 10, 1)) == (4.5, -2.95, 11.75)

    def test_index2coord_exception(self):
        cmin = (-1, -4, 11)
        cmax = (15, 10.1, 12.5)
        d = (1, 0.1, 0.5)
        name = 'test_field'

        f = Field(cmin, cmax, d, dim=2, name=name)

        with pytest.raises(ValueError):
            f.index2coord((-1, 0, 0))
            f.index2coord((500, 10, 1))

    def test_coord2index(self):
        cmin = (-10, -5, 0)
        cmax = (10, 5, 10)
        d = (1, 5, 1)
        name = 'test_field'

        f = Field(cmin, cmax, d, dim=2, name=name)

        assert f.coord2index((-10, -5, 0)) == (0, 0, 0)
        assert f.n[0] == 20
        assert f.coord2index((10, 5, 10)) == (19, 1, 9)
        assert f.coord2index((0.0001, 0.0001, 5.0001)) == (10, 1, 5)
        assert f.coord2index((-0.0001, -0.0001, 4.9999)) == (9, 0, 4)

    def test_coord2index_exception(self):
        cmin = (-10, -5, 0)
        cmax = (10, 5, 10)
        d = (1, 5, 1)
        name = 'test_field'

        f = Field(cmin, cmax, d, dim=2, name=name)

        with pytest.raises(ValueError):
            f.coord2index((-11, 0, 5))
            f.coord2index((-5, -5-1e-3, 5))
            f.coord2index((-5, 0, -0.01))
            f.coord2index((11, 0, 5))
            f.coord2index((6, 5+1e-6, 5))
            f.coord2index((0, 0, 10+1e-10))

    def test_domain_centre(self):
        cmin = (-18.5, 5, 0)
        cmax = (10, 10, 10)
        d = (0.1, 0.25, 2)
        name = 'test_field'

        f = Field(cmin, cmax, d, dim=2, name=name)

        assert f.domain_centre() == (-4.25, 7.5, 5)

    def test_random_coord(self):
        cmin = (-18.5, 5, 0)
        cmax = (10, 10, 10)
        d = (0.1, 0.25, 2)
        name = 'test_field'

        f = Field(cmin, cmax, d, dim=2, name=name)

        for j in range(500):
            c = f.random_coord()
            assert f.cmin[0] <= c[0] <= f.cmax[0]
            assert f.cmin[1] <= c[1] <= f.cmax[1]
            assert f.cmin[2] <= c[2] <= f.cmax[2]

    def test_set_with_constant(self):
        for value in self.constant_values:
            for f in self.scalar_fs + self.vector_fs:
                f.set(value)

                # Check all values.
                assert np.all(f.f == value)

                # Check with sampling.
                assert np.all(f(f.random_coord()) == value)
                assert np.all(f.sample(f.random_coord()) == value)

    def test_set_with_tuple_list_ndarray(self):
        for value in self.tuple_values:
            for f in self.vector_fs:
                f.set(value, normalise=True)

                norm = (value[0]**2 + value[1]**2 + value[2]**2)**0.5
                for j in range(3):
                    c = f.random_coord()
                    assert np.all(f.f[:, :, :, j] == value[j]/norm)
                    assert np.all(f(c)[j] == value[j]/norm)
                    assert np.all(f.sample(c)[j] == value[j]/norm)

    def test_set_from_callable(self):
        # Test scalar fs.
        for f in self.scalar_fs:
            for pyfun in self.scalar_pyfuncs:
                f.set(pyfun)

                for j in range(10):
                    c = f.random_coord()
                    expected_value = pyfun(f.nearestcellcoord(c))
                    assert f(c) == expected_value
                    assert f.sample(c) == expected_value

        # Test vector fields.
        for f in self.vector_fs:
            for pyfun in self.vector_pyfuncs:
                f.set(pyfun)

                for j in range(10):
                    c = f.random_coord()
                    expected_value = pyfun(f.nearestcellcoord(c))
                    assert np.all(f(c) == expected_value)
                    assert np.all(f.sample(c) == expected_value)

    def test_set_exception(self):
        for f in self.vector_fs + self.scalar_fs:
            with pytest.raises(TypeError):
                f.set('string')

    def test_set_at_index(self):
        value = [1.1, -2e-9, 3.5]
        for f in self.vector_fs:
            f.set_at_index((0, 0, 0), value)
            assert f.f[0, 0, 0, 0] == value[0]
            assert f.f[0, 0, 0, 1] == value[1]
            assert f.f[0, 0, 0, 2] == value[2]

    def test_average_scalar_field(self):
        value = -1e-3 + np.pi
        tol = 1e-12
        for f in self.scalar_fs:
            f.set(value)
            average = f.average()

            assert abs(average - value) < tol

    def test_average_vector_field(self):
        value = np.array([1.1, -2e-3, np.pi])
        tol = 1e-12
        for f in self.vector_fs:
            f.set(value)
            average = f.average()

            assert abs(average[0] - value[0]) < tol
            assert abs(average[1] - value[1]) < tol
            assert abs(average[2] - value[2]) < tol

    def test_normalise(self):
        for f in self.vector_fs:
            for value in self.vector_pyfuncs + self.tuple_values:
                for norm_value in [1, 2.1, 50, 1e-3, np.pi]:
                    f.set(value)

                    f.normalise(norm=norm_value)

                    # Compute norm.
                    norm = 0
                    for j in range(f.dim):
                        norm += f.f[:, :, :, j]**2
                    norm = np.sqrt(norm)

                    assert norm.shape == (f.n[0], f.n[1], f.n[2])
                    diff = norm - norm_value

                    assert np.all(abs(norm - norm_value) < 1e-12)

    def test_normalise_scalar_field_exception(self):
        for f in self.scalar_fs:
            for value in self.scalar_pyfuncs:
                for norm_value in [1, 2.1, 50, 1e-3, np.pi]:
                    f.set(value)

                    with pytest.raises(NotImplementedError):
                        f.normalise(norm=norm_value)

    def test_slice_field(self):
        for s in 'xyz':
            for f in self.vector_fs + self.scalar_fs:
                if f.dim == 1:
                    funcs = self.scalar_pyfuncs
                elif f.dim == 3:
                    funcs = self.vector_pyfuncs

                for pyfun in funcs:
                    f.set(pyfun)
                    point = f.domain_centre()['xyz'.find(s)]
                    data = f.slice_field(s, point)
                    a1, a2, f_slice, cs = data

                    if s == 'x':
                        assert cs == (1, 2, 0)
                    elif s == 'y':
                        assert cs == (0, 2, 1)
                    elif s == 'z':
                        assert cs == (0, 1, 2)

                    tol = 1e-16
                    assert abs(a1[0] - (f.cmin[cs[0]] + f.d[cs[0]]/2.)) < tol
                    assert abs(a1[-1] - (f.cmax[cs[0]] - f.d[cs[0]]/2.)) < tol
                    assert abs(a2[0] - (f.cmin[cs[1]] + f.d[cs[1]]/2.)) < tol
                    assert abs(a2[-1] - (f.cmax[cs[1]] - f.d[cs[1]]/2.)) < tol
                    assert len(a1) == f.n[cs[0]]
                    assert len(a2) == f.n[cs[1]]
                    assert f_slice.shape == (f.n[cs[0]],
                                             f.n[cs[1]],
                                             f.dim)

                    for j in range(f.n[cs[0]]):
                        for k in range(f.n[cs[1]]):
                            c = list(f.domain_centre())
                            c[cs[0]] = a1[j]
                            c[cs[1]] = a2[k]
                            c = tuple(c)

                            assert np.all(f_slice[j, k] == f(c))

    def test_slice_field_wrong_axis(self):
        for f in self.vector_fs + self.scalar_fs:
            if f.dim == 1:
                funcs = self.scalar_pyfuncs
            elif f.dim == 3:
                funcs = self.vector_pyfuncs

            for pyfun in funcs:
                f.set(pyfun)
                point = f.domain_centre()[0]
                with pytest.raises(ValueError):
                    data = f.slice_field('xy', point)
                    data = f.slice_field('xyz', point)
                    data = f.slice_field('zy', point)
                    data = f.slice_field('string', point)
                    data = f.slice_field('point', point)

    def test_slice_field_wrong_point(self):
        for s in 'xyz':
            for f in self.vector_fs + self.scalar_fs:
                if f.dim == 1:
                    funcs = self.scalar_pyfuncs
                elif f.dim == 3:
                    funcs = self.vector_pyfuncs

                for pyfun in funcs:
                    f.set(pyfun)
                    point = f.domain_centre()['xyz'.find(s)] + 100
                    with pytest.raises(ValueError):
                        data = f.slice_field(s, point)

                    point = f.domain_centre()['xyz'.find(s)] - 100
                    with pytest.raises(ValueError):
                        data = f.slice_field(s, point)

    def test_plot_slice_vector_field(self):
        figname = 'test_slice_plot_figure.pdf'
        value = (1e-3 + np.pi, -5, 6)
        for f in self.vector_fs:
            f.set(value, normalise=True)
            point = f.domain_centre()['xyz'.find('y')]
            fig = f.plot_slice('y', point, axes=True)
            fig = f.plot_slice('y', point, axes=False)

    def test_plot_slice_vector_field_exception(self):
        value = (0, 0, 1)
        for f in self.vector_fs:
            f.set(value, normalise=True)
            point = f.domain_centre()['xyz'.find('z')]
            with pytest.raises(ValueError):
                fig = f.plot_slice('z', point)

    def test_write_read_oommf_file(self):
        tol = 1e-12
        filename = 'test_write_oommf_file.omf'
        value = (1e-3 + np.pi, -5, 6)
        for f in self.vector_fs:
            f.set(value)
            f.write_oommf_file(filename)

            f_loaded = read_oommf_file(filename)

            assert f.cmin == f_loaded.cmin
            assert f.cmax == f_loaded.cmax
            assert f.d == f_loaded.d
            assert f.d == f_loaded.d
            assert np.all(abs(f.f - f_loaded.f) < tol)

            os.system('rm {}'.format(filename))
