import pytest
import discretisedfield as df


def test_cross():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # Zero vectors
    f1 = df.Field(mesh, dim=3, value=(0, 0, 0))
    res = df.cross(f1, f1)
    assert res.dim == 3
    assert res.average == (0, 0, 0)

    # Orthogonal vectors
    f1 = df.Field(mesh, dim=3, value=(1, 0, 0))
    f2 = df.Field(mesh, dim=3, value=(0, 1, 0))
    f3 = df.Field(mesh, dim=3, value=(0, 0, 1))
    assert df.cross(f1, f2).average == (0, 0, 1)
    assert df.cross(f1, f3).average == (0, -1, 0)
    assert df.cross(f2, f3).average == (1, 0, 0)
    assert df.cross(f1, f1).average == (0, 0, 0)
    assert df.cross(f2, f2).average == (0, 0, 0)
    assert df.cross(f3, f3).average == (0, 0, 0)

    # Check if not comutative
    assert df.cross(f1, f2) == -df.cross(f2, f1)
    assert df.cross(f1, f3) == -df.cross(f3, f1)
    assert df.cross(f2, f3) == -df.cross(f3, f2)

    f1 = df.Field(mesh, dim=3, value=lambda pos: (pos[0], pos[1], pos[2]))
    f2 = df.Field(mesh, dim=3, value=lambda pos: (pos[2], pos[0], pos[1]))

    # The cross product should be
    # (y**2-x*z, z**2-x*y, x**2-y*z)
    assert df.cross(f1, f2)((1, 1, 1)) == (0, 0, 0)
    assert df.cross(f1, f2)((3, 1, 1)) == (-2, -2, 8)
    assert df.cross(f2, f1)((3, 1, 1)) == (2, 2, -8)
    assert df.cross(f1, f2)((5, 7, 1)) == (44, -34, 18)

    # Exceptions
    f1 = df.Field(mesh, dim=1, value=1.2)
    f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
    with pytest.raises(TypeError):
        res = df.cross(f1, 2)
    with pytest.raises(ValueError):
        res = df.cross(f1, f2)

    # Fields defined on different meshes
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
    f1 = df.Field(mesh1, dim=3, value=(1, 2, 3))
    f2 = df.Field(mesh2, dim=3, value=(3, 2, 1))
    with pytest.raises(ValueError):
        res = df.cross(f1, f2)


def test_stack():
    p1 = (0, 0, 0)
    p2 = (10e6, 10e6, 10e6)
    cell = (1e5, 1e5, 1e5)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    f1 = df.Field(mesh, dim=1, value=1)
    f2 = df.Field(mesh, dim=1, value=-3)
    f3 = df.Field(mesh, dim=1, value=5)

    res = df.stack([f1, f2, f3])

    assert res.dim == 3
    assert res.average == (1, -3, 5)

    # Exceptions
    f1 = df.Field(mesh, dim=1, value=1.2)
    f2 = df.Field(mesh, dim=3, value=(-1, -3, -5))
    with pytest.raises(TypeError):
        res = df.stack([f1, 2])
    with pytest.raises(ValueError):
        res = df.stack([f1, f2])

    # Fields defined on different meshes
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
    f1 = df.Field(mesh1, dim=1, value=1.2)
    f2 = df.Field(mesh2, dim=1, value=1)
    with pytest.raises(ValueError):
        res = df.stack([f1, f2])
