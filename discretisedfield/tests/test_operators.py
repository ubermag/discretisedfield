import pytest
import discretisedfield as df


def test_stack():
    p1 = (0, 0, 0)
    p2 = (10e6, 10e6, 10e6)
    cell = (5e6, 5e6, 5e6)
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
