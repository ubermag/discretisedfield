import os
import re
import k3d
import types
import random
import pytest
import numbers
import tempfile
import itertools
import numpy as np
import discretisedfield as df
import matplotlib.pyplot as plt
from .test_mesh import TestMesh


def check_field(field):
    assert isinstance(field.mesh, df.Mesh)

    assert isinstance(field.dim, int)
    assert field.dim > 0

    assert isinstance(field.array, np.ndarray)
    assert field.array.shape == (*field.mesh.n, field.dim)

    average = field.average
    assert isinstance(average, (tuple, numbers.Real))

    rstr = repr(field)
    assert isinstance(rstr, str)
    pattern = (r'^Field\(mesh=Mesh\(region=Region\(p1=\(.+\), '
               r'p2=\(.+\)\), .+\), dim=\d+\)$')
    assert re.search(pattern, rstr)

    assert isinstance(field.__iter__(), types.GeneratorType)
    assert len(list(field)) == len(field.mesh)

    line = field.line(p1=field.mesh.region.pmin,
                      p2=field.mesh.region.pmax,
                      n=5)
    assert isinstance(line, df.Line)
    assert len(line.points) == 5
    assert len(line.values) == 5

    plane = field.plane('z', n=(2, 2))
    assert isinstance(plane, df.Field)
    assert len(plane.mesh) == 4
    assert plane.mesh.n == (2, 2, 1)

    project = field.project('z')
    assert isinstance(project, df.Field)
    assert project.mesh.n[2] == 1

    assert isinstance(field(field.mesh.region.centre), (tuple, numbers.Real))
    assert isinstance(field(field.mesh.region.random_point()),
                      (tuple, numbers.Real))

    assert field == field
    assert not field != field

    assert +field == field
    assert -(-field) == field
    assert field + field == 2*field
    assert field - (-field) == field + field
    assert 1*field == field
    assert -1*field == -field

    integral = field.integral
    assert isinstance(integral, (tuple, numbers.Real))

    if field.dim == 1:
        grad = field.grad
        assert isinstance(grad, df.Field)
        assert grad.dim == 3

        assert all(i not in dir(field) for i in 'xyz')

    if field.dim == 3:
        norm = field.norm
        assert isinstance(norm, df.Field)
        assert norm == abs(field)
        assert norm.dim == 1

        assert isinstance(field.x, df.Field)
        assert field.x.dim == 1

        assert isinstance(field.y, df.Field)
        assert field.y.dim == 1

        assert isinstance(field.z, df.Field)
        assert field.z.dim == 1

        div = field.div
        assert isinstance(div, df.Field)
        assert div.dim == 1

        curl = field.curl
        assert isinstance(curl, df.Field)
        assert curl.dim == 3

        field_plane = field.plane('z')
        assert isinstance(field_plane.topological_charge_density, df.Field)
        assert isinstance(field_plane.topological_charge(), numbers.Real)
        assert isinstance(field_plane.bergluescher, numbers.Real)

        orientation = field.orientation
        assert isinstance(orientation, df.Field)
        assert orientation.dim == 3

        assert all(i in dir(field) for i in 'xyz')


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
        self.consts = [0, -5., np.pi, 1e-15, 1.2e12, random.random()]
        self.iters = [(0, 0, 1),
                      (0, -5.1, np.pi),
                      [70, 1e15, 2*np.pi],
                      [5, random.random(), np.pi],
                      np.array([4, -1, 3.7]),
                      np.array([2.1, 0.0, -5*random.random()])]
        self.sfuncs = [lambda c: 1,
                       lambda c: -2.4,
                       lambda c: -6.4e-15,
                       lambda c: c[0] + c[1] + c[2] + 1,
                       lambda c: (c[0]-1)**2 - c[1]+7 + c[2]*0.1,
                       lambda c: np.sin(c[0]) + np.cos(c[1]) - np.sin(2*c[2])]
        self.vfuncs = [lambda c: (1, 2, 0),
                       lambda c: (-2.4, 1e-3, 9),
                       lambda c: (c[0], c[1], c[2] + 100),
                       lambda c: (c[0]+c[2]+10, c[1], c[2]+1),
                       lambda c: (c[0]-1, c[1]+70, c[2]*0.1),
                       lambda c: (np.sin(c[0]), np.cos(c[1]), -np.sin(2*c[2]))]

        # Create a field for plotting tests
        mesh = df.Mesh(p1=(-5e-9, -5e-9, -5e-9),
                       p2=(5e-9, 5e-9, 5e-9),
                       n=(5, 5, 5))

        def norm_fun(pos):
            x, y, z = pos
            if x**2 + y**2 <= (5e-9)**2:
                return 1
            else:
                return 0

        def value_fun(pos):
            x, y, z = pos
            if x <= 0:
                return (0, 0, 1)
            else:
                return (0, 0, -1)

        self.pf = df.Field(mesh, dim=3, value=value_fun, norm=norm_fun)

        # Make one vector point out-of-plane
        # self.pf.array[0, 0, 0, :] = (0, 0, 1)

    def test_init_valid_args(self):
        for mesh in self.meshes:
            for value in self.consts + self.sfuncs:
                f = df.Field(mesh, dim=1, value=value)
                check_field(f)

            for value in self.iters + self.vfuncs:
                f = df.Field(mesh, dim=3, value=value)
                check_field(f)

    def test_init_invalid_args(self):
        with pytest.raises(TypeError):
            mesh = 'meaningless_mesh_string'
            f = df.Field(mesh, dim=1)

        for mesh in self.meshes:
            for dim in [0, -1, 'dim', (2, 3)]:
                with pytest.raises((ValueError, TypeError)):
                    f = df.Field(mesh, dim=dim)

    def test_set_with_ndarray(self):
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3)
            f.value = np.ones((*f.mesh.n, f.dim,))

            check_field(f)
            assert isinstance(f.value, np.ndarray)
            assert f.average == (1, 1, 1)

            with pytest.raises(ValueError):
                f.value = np.ones((2, 2))

    def test_set_with_callable(self):
        for mesh in self.meshes:
            for func in self.sfuncs:
                f = df.Field(mesh, dim=1, value=func)
                check_field(f)

                rp = f.mesh.region.random_point()
                # Make sure to be at the centre of the cell
                rp = f.mesh.index2point(f.mesh.point2index(rp))
                assert f(rp) == func(rp)

        for mesh in self.meshes:
            for func in self.vfuncs:
                f = df.Field(mesh, dim=3, value=func)
                check_field(f)

                rp = f.mesh.region.random_point()
                rp = f.mesh.index2point(f.mesh.point2index(rp))
                assert np.all(f(rp) == func(rp))

    def test_set_with_dict(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 10e-9, 10e-9)
        n = (5, 5, 5)
        subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(5e-9, 10e-9, 10e-9)),
                      'r2': df.Region(p1=(5e-9, 0, 0),
                                      p2=(10e-9, 10e-9, 10e-9))}
        mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)

        field = df.Field(mesh, dim=3, value={'r1': (0, 0, 1),
                                             'r2': (0, 0, 2),
                                             'r1:r2': (0, 0, 5)})
        assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1))
        assert np.all(field((8e-9, 2e-9, 9e-9)) == (0, 0, 2))

    def test_set_exception(self):
        for mesh in self.meshes:
            with pytest.raises(ValueError):
                f = df.Field(mesh, dim=3, value='meaningless_string')

            with pytest.raises(ValueError):
                f = df.Field(mesh, dim=3, value=5+5j)

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

        assert np.all(f.norm.value == 2*np.sqrt(3))
        assert np.all(f.norm.array == 2*np.sqrt(3))
        assert np.all(f.array == 2)

        f.norm = 1
        assert np.all(f.norm.value == 1)
        assert np.all(f.norm.array == 1)
        assert np.all(f.array == 1/np.sqrt(3))

        f.array[0, 0, 0, 0] = 3
        assert isinstance(f.norm.value, np.ndarray)
        assert not np.all(f.norm.value == 1)

        for mesh in self.meshes:
            for value in self.iters + self.vfuncs:
                for norm_value in [1, 2.1, 50, 1e-3, np.pi]:
                    f = df.Field(mesh, dim=3, value=value, norm=norm_value)

                    # Compute norm.
                    norm = f.array[..., 0]**2
                    norm += f.array[..., 1]**2
                    norm += f.array[..., 2]**2
                    norm = np.sqrt(norm)

                    assert norm.shape == f.mesh.n
                    assert f.norm.array.shape == (*f.mesh.n, 1)
                    assert np.all(abs(norm - norm_value) < 1e-12)

        # Exception
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(1, 1, 1))
        f = df.Field(mesh, dim=1, value=-5)
        with pytest.raises(ValueError):
            norm = f.norm
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

    def test_orientation(self):
        p1 = (-5e-9, -5e-9, -5e-9)
        p2 = (5e-9, 5e-9, 5e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # No zero-norm cells
        f = df.Field(mesh, dim=3, value=(2, 0, 0))
        assert f.orientation.average == (1, 0, 0)

        # With zero-norm cells
        def value_fun(pos):
            x, y, z = pos
            if x <= 0:
                return (0, 0, 0)
            else:
                return (3, 0, 4)

        f = df.Field(mesh, dim=3, value=value_fun)
        assert f.orientation((-1.5e-9, 3e-9, 0)) == (0, 0, 0)
        assert f.orientation((1.5e-9, 3e-9, 0)) == (0.6, 0, 0.8)

        f = df.Field(mesh, dim=1, value=0)
        with pytest.raises(ValueError):
            of = f.orientation

    def test_average(self):
        value = -1e-3 + np.pi
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
            assert all(isinstance(getattr(f, i), df.Field) for i in 'xyz')
            assert all(getattr(f, i).dim == 1 for i in 'xyz')

            f = df.Field(mesh, dim=2, value=(1, 2))
            assert all(isinstance(getattr(f, i), df.Field) for i in 'xy')
            assert all(getattr(f, i).dim == 1 for i in 'xy')

            # Exception.
            f = df.Field(mesh, dim=1, value=1)
            with pytest.raises(AttributeError):
                fx = f.x.dim

    def test_get_attribute_exception(self):
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3)
            with pytest.raises(AttributeError) as excinfo:
                f.__getattr__('nonexisting_attribute')
            assert 'has no attribute' in str(excinfo.value)

    def test_dir(self):
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3, value=(5, 6, -9))
            assert all(attr in dir(f) for attr in ['x', 'y', 'z', 'div'])
            assert 'grad' not in dir(f)

            f = df.Field(mesh, dim=1, value=1)
            assert all(attr not in dir(f) for attr in ['x', 'y', 'z', 'div'])
            assert 'grad' in dir(f)

    def test_eq_ne(self):
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

    def test_pos_neg(self):
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
        res = f**(-1)
        assert res.average == 0.5

        # Attempt vector field
        f = df.Field(mesh, dim=3, value=(1, 2, -2))
        with pytest.raises(ValueError):
            res = f**2

        # Attempt to raise to non numbers.Real
        f = df.Field(mesh, dim=1, value=2)
        with pytest.raises(TypeError):
            res = f**'a'
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
        assert f1 + f2 == f2 + f1
        assert f1 - f2 == -(f2 - f1)
        assert f1 + (f1 + f2) == (f1 + f1) + f2
        assert f1 - (f1 + f2) == f1 - f1 - f2
        assert f1 + f2 - f1 == f2

        # Exceptions
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        with pytest.raises(TypeError):
            res = f1 + 2
        with pytest.raises(TypeError):
            f1 += 2
        with pytest.raises(TypeError):
            f1 -= 5.1
        with pytest.raises(TypeError):
            res = 2 + f2
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

        # Scalar field with a scalar
        f = df.Field(mesh, dim=1, value=5)
        res = f * 2  # __mul__
        assert res.average == 10
        res = 3 * f  # __rmul__
        assert res.average == 15
        res = f / 2  # __mul__
        assert res.average == 2.5
        res = 10 / f  # __rmul__
        assert res.average == 2
        f *= 10  # __imul__
        assert f.average == 50
        f /= 10  # __idiv__
        assert f.average == 5

        # Vector field with a scalar field
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
        assert f1 * (f1 * f2) == (f1 * f1) * f2
        assert f1 * f2 / f1 == f2

        # Exceptions
        f1 = df.Field(mesh, dim=1, value=1.2)
        f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
        with pytest.raises(TypeError):
            res = f2 * 'a'
        with pytest.raises(TypeError):
            res = 'a' / f1
        with pytest.raises(ValueError):
            res = f2 * f2
        with pytest.raises(ValueError):
            res = f2 / f2
        with pytest.raises(ValueError):
            res = 1 / f2
        with pytest.raises(ValueError):
            res = f1 / f2
        with pytest.raises(TypeError):
            f2 *= 'a'
        with pytest.raises(TypeError):
            f2 /= 'a'
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

    def test_matmul(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Zero vectors
        f1 = df.Field(mesh, dim=3, value=(0, 0, 0))
        res = f1@f1
        assert res.dim == 1
        assert res.array.shape == (5, 5, 5, 1)
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

        # Spatially varying vectors
        def value_fun1(pos):
            x, y, z = pos
            return (x, y, z)

        def value_fun2(pos):
            x, y, z = pos
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
        assert f1.norm == (f1 @ f1)**(0.5)

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

    def test_all_operators(self):
        p1 = (0, 0, 0)
        p2 = (5e-9, 5e-9, 10e-9)
        n = (2, 2, 1)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)

        f1 = df.Field(mesh, dim=1, value=2)
        f2 = df.Field(mesh, dim=3, value=(-4, 0, 1))
        res = ((+f1/2 + f2.x)**2 - 2*f1*3)/(-f2.z) - 2*f2.y + 1/f2.z**2 + f2@f2

        assert np.all(res.array == 21)

    def test_derivative(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
        f = df.Field(mesh, dim=1, value=0)

        check_field(f.derivative('x'))
        assert f.derivative('x').average == 0
        assert f.derivative('y').average == 0
        assert f.derivative('z').average == 0

        # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
        def value_fun(pos):
            x, y, z = pos
            return x + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.derivative('x').average == 1
        assert f.derivative('y').average == 1
        assert f.derivative('z').average == 1

        # f(x, y, z) = x*y + y + z -> grad(f) = (y, x+1, 1)
        def value_fun(pos):
            x, y, z = pos
            return x*y + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.derivative(0)((3, 1, 3)) == 1
        assert f.derivative(1)((3, 1, 3)) == 4
        assert f.derivative(2)((3, 1, 3)) == 1
        assert f.derivative(0)((5, 3, 5)) == 3
        assert f.derivative(1)((5, 3, 5)) == 6
        assert f.derivative(2)((5, 3, 5)) == 1

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        def value_fun(pos):
            x, y, z = pos
            return x*y + 2*y + x*y*z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.derivative('x')((7, 5, 1)) == 10
        assert f.derivative('y')((7, 5, 1)) == 16
        assert f.derivative('z')((7, 5, 1)) == 35

        # f(x, y, z) = (0, 0, 0)
        # -> dfdx = (0, 0, 0)
        # -> dfdy = (0, 0, 0)
        # -> dfdz = (0, 0, 0)
        f = df.Field(mesh, dim=3, value=(0, 0, 0))

        check_field(f.derivative('y'))
        assert f.derivative('x').average == (0, 0, 0)
        assert f.derivative('y').average == (0, 0, 0)
        assert f.derivative('z').average == (0, 0, 0)

        # f(x, y, z) = (x,  y,  z)
        # -> dfdx = (1, 0, 0)
        # -> dfdy = (0, 1, 0)
        # -> dfdz = (0, 0, 1)
        def value_fun(pos):
            x, y, z = pos
            return (x, y, z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.derivative('x').average == (1, 0, 0)
        assert f.derivative('y').average == (0, 1, 0)
        assert f.derivative('z').average == (0, 0, 1)

        # f(x, y, z) = (x*y, y*z, x*y*z)
        # -> dfdx = (y, 0, y*z)
        # -> dfdy = (x, z, x*z)
        # -> dfdz = (0, y, x*y)
        def value_fun(pos):
            x, y, z = pos
            return (x*y, y*z, x*y*z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.derivative(0)((3, 1, 3)) == (1, 0, 3)
        assert f.derivative(1)((3, 1, 3)) == (3, 3, 9)
        assert f.derivative(2)((3, 1, 3)) == (0, 1, 3)
        assert f.derivative(0)((5, 3, 5)) == (3, 0, 15)
        assert f.derivative(1)((5, 3, 5)) == (5, 5, 25)
        assert f.derivative(2)((5, 3, 5)) == (0, 3, 15)

        # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
        # -> dfdx = (y, 1, y*z)
        # -> dfdy = (x, -2, x*z)
        # -> dfdz = (0, 0, x*y)
        def value_fun(pos):
            x, y, z = pos
            return (3+x*y, x-2*y, x*y*z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.derivative('x')((7, 5, 1)) == (5, 1, 5)
        assert f.derivative('y')((7, 5, 1)) == (7, -2, 7)
        assert f.derivative('z')((7, 5, 1)) == (0, 0, 35)

    def test_derivative_single_cell(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 2)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Scalar field: f(x, y, z) = x + y + z
        # -> grad(f) = (1, 1, 1)
        def value_fun(pos):
            x, y, z = pos
            return x + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        # only one cell in the z-direction
        assert f.plane('x').derivative('x').average == 0
        assert f.plane('y').derivative('y').average == 0
        assert f.derivative('z').average == 0

        # Vector field: f(x, y, z) = (x, y, z)
        # -> grad(f) = (1, 1, 1)
        def value_fun(pos):
            x, y, z = pos
            return (x, y, z)

        f = df.Field(mesh, dim=3, value=value_fun)

        # only one cell in the z-direction
        assert f.plane('x').derivative('x').average == (0, 0, 0)
        assert f.plane('y').derivative('y').average == (0, 0, 0)
        assert f.derivative('z').average == (0, 0, 0)

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
        def value_fun(pos):
            x, y, z = pos
            return x + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.grad.average == (1, 1, 1)

        # f(x, y, z) = x*y + y + z -> grad(f) = (y, x+1, 1)
        def value_fun(pos):
            x, y, z = pos
            return x*y + y + z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.grad((3, 1, 3)) == (1, 4, 1)
        assert f.grad((5, 3, 5)) == (3, 6, 1)

        # f(x, y, z) = x*y + 2*y + x*y*z ->
        # grad(f) = (y+y*z, x+2+x*z, x*y)
        def value_fun(pos):
            x, y, z = pos
            return x*y + 2*y + x*y*z

        f = df.Field(mesh, dim=1, value=value_fun)

        assert f.grad((7, 5, 1)) == (10, 16, 35)
        assert f.grad.x == f.derivative('x')
        assert f.grad.y == f.derivative('y')
        assert f.grad.z == f.derivative('z')

        # Exception
        f = df.Field(mesh, dim=3, value=(1, 2, 3))

        with pytest.raises(ValueError):
            res = f.grad

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
        def value_fun(pos):
            x, y, z = pos
            return (x, y, z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.div.average == 3
        assert f.curl.average == (0, 0, 0)

        # f(x, y, z) = (x*y, y*z, x*y*z)
        # -> div(f) = y + z + x*y
        # -> curl(f) = (x*z-y, -y*z, -x)
        def value_fun(pos):
            x, y, z = pos
            return (x*y, y*z, x*y*z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.div((3, 1, 3)) == 7
        assert f.div((5, 3, 5)) == 23

        assert f.curl((3, 1, 3)) == (8, -3, -3)
        assert f.curl((5, 3, 5)) == (22, -15, -5)

        # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
        # -> div(f) = y - 2 + x*y
        # -> curl(f) = (x*z, -y*z, 1-x)
        def value_fun(pos):
            x, y, z = pos
            return (3+x*y, x-2*y, x*y*z)

        f = df.Field(mesh, dim=3, value=value_fun)

        assert f.div((7, 5, 1)) == 38
        assert f.curl((7, 5, 1)) == (7, -5, -6)

        # Exception
        f = df.Field(mesh, dim=1, value=3.11)

        with pytest.raises(ValueError):
            res = f.div
        with pytest.raises(ValueError):
            res = f.curl

    def test_integral(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        f = df.Field(mesh, dim=1, value=0)
        assert f.integral == 0

        f = df.Field(mesh, dim=1, value=2)
        assert f.integral == 2000

        f = df.Field(mesh, dim=3, value=(-1, 0, 3))
        assert f.integral == (-1000, 0, 3000)

        def value_fun(pos):
            x, y, z = pos
            if x <= 5:
                return (-1, -2, -3)
            else:
                return (1, 2, 3)

        f = df.Field(mesh, dim=3, value=value_fun)
        assert f.integral == (0, 0, 0)

    def test_topological_charge(self):
        p1 = (0, 0, 0)
        p2 = (10, 10, 10)
        cell = (2, 2, 2)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # f(x, y, z) = (0, 0, 0)
        # -> Q(f) = 0
        f = df.Field(mesh, dim=3, value=(0, 0, 0))

        q = f.plane('z').topological_charge_density
        check_field(q)
        assert q.dim == 1
        assert q.average == 0
        assert f.plane('z').topological_charge(method='continuous') == 0
        assert f.plane('z').topological_charge(method='berg-luescher') == 0

        # Skyrmion from a file
        test_filename = os.path.join(os.path.dirname(__file__),
                                     'test_sample/',
                                     'skyrmion.omf')
        f = df.Field.fromfile(test_filename)
        Qc = f.plane('z').topological_charge(method='continuous')
        Qbl = f.plane('z').topological_charge(method='berg-luescher')
        assert abs(Qc - 1) < 0.15
        assert abs(Qbl - 1) < 1e-3

        # Not sliced
        f = df.Field(mesh, dim=3, value=(1, 2, 3))
        with pytest.raises(ValueError):
            res = f.topological_charge(method='continuous')
        with pytest.raises(ValueError):
            res = f.topological_charge(method='berg-luescher')

        # Scalar field
        f = df.Field(mesh, dim=1, value=3.14)
        with pytest.raises(ValueError):
            res = f.plane('z').topological_charge(method='continuous')
        with pytest.raises(ValueError):
            res = f.plane('z').topological_charge(method='berg-luescher')

        # Method does not exist
        with pytest.raises(ValueError):
            res = f.plane('z').topological_charge(method='some-method')

    def test_line(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), n=(10, 10, 10))
        f = df.Field(mesh, dim=3, value=(1, 2, 3))
        check_field(f)

        line = f.line(p1=(0, 0, 0), p2=(5, 5, 5), n=20)
        assert isinstance(line, df.Line)

        assert len(line.points) == 20
        assert len(line.values) == 20
        assert line.points[0] == (0, 0, 0)
        assert line.points[-1] == (5, 5, 5)
        assert line.values[0] == (1, 2, 3)
        assert line.values[-1] == (1, 2, 3)

    def test_plane(self):
        for mesh, direction in itertools.product(self.meshes, ['x', 'y', 'z']):
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
        subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(30, 50, 10)),
                      'r2': df.Region(p1=(30, 0, 0), p2=(90, 50, 10))}
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

        def value_fun(pos):
            x, y, z = pos
            if x <= 60:
                return (-1, -2, -3)
            else:
                return (1, 2, 3)

        f = df.Field(mesh, dim=3, value=value_fun)
        check_field(f)
        check_field(f['r1'])
        check_field(f['r2'])

        assert f['r1'].average == (-1, -2, -3)
        assert f['r2'].average == (0, 0, 0)

        assert len(f['r1'].mesh) + len(f['r2'].mesh) == len(f.mesh)

    def test_project(self):
        p1 = (-5, -5, -5)
        p2 = (5, 5, 5)
        cell = (1, 1, 1)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        # Constant scalar field
        f = df.Field(mesh, dim=1, value=5)
        check_field(f)
        assert f.project('x').array.shape == (1, 10, 10, 1)
        assert f.project('y').array.shape == (10, 1, 10, 1)
        assert f.project('z').array.shape == (10, 10, 1, 1)

        # Constant vector field
        f = df.Field(mesh, dim=3, value=(1, 2, 3))
        assert f.project('x').array.shape == (1, 10, 10, 3)
        assert f.project('y').array.shape == (10, 1, 10, 3)
        assert f.project('z').array.shape == (10, 10, 1, 3)

        # Spatially varying scalar field
        def value_fun(pos):
            x, y, z = pos
            if z <= 0:
                return 1
            else:
                return -1

        f = df.Field(mesh, dim=1, value=value_fun)
        sf = f.project('z')
        assert sf.array.shape == (10, 10, 1, 1)
        assert sf.average == 0

        # Spatially varying vector field
        def value_fun(pos):
            x, y, z = pos
            if z <= 0:
                return (3, 2, 1)
            else:
                return (3, 2, -1)

        f = df.Field(mesh, dim=3, value=value_fun)
        sf = f.project('z')
        assert sf.array.shape == (10, 10, 1, 3)
        assert sf.average == (3, 2, 0)

    def test_write_read_ovf(self):
        representations = ['txt', 'bin4', 'bin8']
        tolerance = dict(zip(representations, (0, 1e-6, 1e-12)))
        filename = 'testfile.ovf'

        p1 = (0, 0, 0)
        p2 = (10e-9, 5e-9, 3e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        for dim, value in [(1, -1.23),
                           (3, (4, 2, -13e6)),
                           (3, lambda pos: (pos[0], pos[1], pos[2])),
                           (1, lambda pos: pos[0] + pos[1] + pos[2])]:
            f = df.Field(mesh, dim=dim, value=value)
            for rep in representations:
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpfilename = os.path.join(tmpdir, filename)
                    f.write(tmpfilename, representation=rep)
                    f_read = df.Field.fromfile(tmpfilename)

                    assert f.mesh == f_read.mesh
                    np.testing.assert_allclose(f.array, f_read.array,
                                               rtol=tolerance[rep])

        # Extend scalar
        for rep in representations:
            f = df.Field(mesh, dim=1, value=lambda pos: pos[0]+pos[1]+pos[2])
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpfilename = os.path.join(tmpdir, filename)
                f.write(tmpfilename, extend_scalar=True)
                f_read = df.Field.fromfile(tmpfilename)

                assert f.mesh == f_read.mesh
                assert f_read.dim == 3
                np.testing.assert_allclose(f.array, f_read.x.array,
                                           rtol=tolerance[rep])
                assert np.equal(f_read.y.array, 0).all()
                assert np.equal(f_read.z.array, 0).all()

        # Attempt to write dim=2 field.
        f = df.Field(mesh, dim=2, value=(1, 2))
        with pytest.raises(TypeError) as excinfo:
            f.write(filename)

    def test_write_read_vtk(self):
        filename = 'testfile.vtk'

        p1 = (0, 0, 0)
        p2 = (10e-9, 5e-9, 3e-9)
        cell = (1e-9, 1e-9, 1e-9)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        f = df.Field(mesh, dim=3, value=(1e6, 2e6, -5e6))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            f.write(tmpfilename)

            with open(tmpfilename, 'r') as f:
                for line in f.readlines():
                    if 'CELL_DATA' in line:
                        pattern = line.strip()
                        break
            assert pattern == f'CELL_DATA {len(mesh)}'

    def test_write_read_hdf5(self):
        filenames = ['testfile.hdf5', 'testfile.h5']

        p1 = (0, 0, 0)
        p2 = (10e-12, 5e-12, 3e-12)
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
        filename = 'testfile.jpg'

        p1 = (0, 0, 0)
        p2 = (10e-12, 5e-12, 3e-12)
        cell = (1e-12, 1e-12, 1e-12)
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        f = df.Field(mesh, dim=1, value=5e-12)
        with pytest.raises(ValueError) as excinfo:
            f.write(filename)
        with pytest.raises(ValueError) as excinfo:
            f = df.Field.fromfile(filename)

    def test_read_mumax3_ovffile(self):
        # Output file has been produced with Mumax ~3.1.0
        # (fd3a50233f0a6625086d390) using this script:
        #
        # SetGridsize(128, 32, 1)
        # SetCellsize(500e-9/128, 125e-9/32, 3e-9)
        #
        # Msat  = 800e3
        # Aex   = 13e-12
        # alpha = 0.02
        #
        # m = uniform(1, .1, 0)
        # save(m)
        filenames = ['mumax-output-linux.ovf', 'mumax-output-win.ovf']
        dirname = os.path.join(os.path.dirname(__file__), 'test_sample')

        for f in filenames:
            path = os.path.join(dirname, f)

            f = df.Field.fromfile(path)

            # Compare with the human readable part of file.
            assert f.dim == 3
            assert len(f.mesh) == 4096
            assert f.mesh.region.pmin == (0., 0., 0.)
            assert f.mesh.region.pmax == (5e-07, 1.25e-07, 3e-09)
            assert f.array.shape == (128, 32, 1, 3)

            # Compare with vector field (we know from the script
            # shown above). m vector in mumax (uses 4 bytes).
            m = np.array([1, 0.1, 0], dtype=np.float32)

            # magnetisation is normalised before saving
            v = m / sum(m**2)**0.5

            assert np.equal(f.array[..., :], v).all()

    def test_mpl(self):
        self.pf.plane('z', n=(3, 4)).mpl()
        self.pf.z.plane('x', n=(3, 4)).mpl()

        with pytest.raises(ValueError) as excinfo:
            self.pf.mpl()

    def test_imshow(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.pf.x.plane('x', n=(3, 4)).imshow(ax=ax)
        self.pf.x.plane('y', n=(1, 1)).imshow(ax=ax, cmap='viridis')
        self.pf.x.plane('z').imshow(ax=ax, filter_field=self.pf.norm)

        with pytest.raises(ValueError) as excinfo:
            self.pf.imshow(ax=ax)
        with pytest.raises(ValueError) as excinfo:
            self.pf.plane('z').imshow(ax=ax)

    def test_quiver(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        self.pf.plane('z', n=(3, 4)).quiver(ax=ax)
        self.pf.plane('x', n=(3, 4)).quiver(ax=ax, color_field=self.pf.y)

        with pytest.raises(ValueError) as excinfo:
            self.pf.quiver(ax=ax)
        with pytest.raises(ValueError) as excinfo:
            self.pf.x.plane('y').quiver(ax=ax)

    def test_colorbar(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        coloredplot = self.pf.x.plane('x', n=(3, 4)).imshow(ax=ax)
        self.pf.colorbar(ax=ax, coloredplot=coloredplot)

    def test_k3d_nonzero(self):
        self.pf.norm.k3d_nonzero()
        self.pf.x.k3d_nonzero()
        self.pf.z.k3d_nonzero(field=self.pf)
        self.pf.norm.plane('z').k3d_nonzero(field=self.pf, interactive=True)

        with pytest.raises(ValueError) as excinfo:
            self.pf.k3d_nonzero()

    def test_k3d_voxels(self):
        self.pf.x.k3d_voxels()
        self.pf.y.k3d_voxels(filter_field=self.pf.norm)
        self.pf.z.k3d_voxels(field=self.pf)
        self.pf.norm.plane('z').k3d_voxels(field=self.pf, interactive=True)

        # Exceptions
        with pytest.raises(ValueError) as excinfo:
            self.pf.k3d_voxels()
        with pytest.raises(ValueError):
            self.pf.x.k3d_voxels(filter_field=self.pf)  # filter field dim=3
        with pytest.raises(ValueError):
            self.pf.x.k3d_voxels(filter_field=self.pf.norm, n=300)  # n > 256

    def test_k3d_vectors(self):
        self.pf.k3d_vectors()
        self.pf.k3d_vectors(color_field=self.pf.z)
        self.pf.k3d_vectors(points=False)
        self.pf.k3d_vectors(field=self.pf)
        self.pf.plane('z').k3d_vectors(field=self.pf, interactive=True)

        # Simulate interactive plotting
        plot = k3d.plot()
        self.pf.plane('z').k3d_vectors(field=self.pf, interactive=True,
                                       plot=plot)
        self.pf.plane('y').k3d_vectors(field=self.pf, interactive=True,
                                       plot=plot)

        # Exceptions
        with pytest.raises(ValueError) as excinfo:
            self.pf.x.k3d_vectors()  # dim=1
        with pytest.raises(ValueError):
            self.pf.k3d_vectors(color_field=self.pf)  # color field dim=3

    def test_plot_large_sample(self):
        p1 = (0, 0, 0)
        p2 = (50e9, 50e9, 50e9)
        cell = (25e9, 25e9, 25e9)
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        value = (1e6, 1e6, 1e6)
        field = df.Field(mesh, dim=3, value=value)

        field.plane('z').mpl()
        field.norm.k3d_nonzero()
        field.x.k3d_voxels()
        field.k3d_vectors()
