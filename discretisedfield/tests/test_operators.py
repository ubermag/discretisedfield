import discretisedfield as df


def test_dot():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # Zero vectors
    f1 = df.Field(mesh, dim=3, value=(0, 0, 0))
    res = df.dot(f1, f1)
    assert res.dim == 1
    assert res.array.shape == (5, 5, 5, 1)
    assert res.average == (0,)

    # Orthogonal vectors
    f1 = df.Field(mesh, dim=3, value=(1, 0, 0))
    f2 = df.Field(mesh, dim=3, value=(0, 1, 0))
    f3 = df.Field(mesh, dim=3, value=(0, 0, 1))
    assert df.dot(f1, f2).average == (0,)
    assert df.dot(f1, f3).average == (0,)
    assert df.dot(f2, f3).average == (0,)
    assert df.dot(f1, f1).average == (1,)
    assert df.dot(f2, f2).average == (1,)
    assert df.dot(f3, f3).average == (1,)

    # Spatially varying vectors
    def value_fun1(pos):
        x, y, z = pos
        return (x, y, z)

    def value_fun2(pos):
        x, y, z = pos
        return (z, x, y)

    f1 = df.Field(mesh, dim=3, value=value_fun1)
    f2 = df.Field(mesh, dim=3, value=value_fun2)

    # The dot product should be x*z + y*x + z*y
    assert df.dot(f1, f2)((1, 1, 1)) == (3,)
    assert df.dot(f1, f2)((3, 1, 1)) == (7,)
    assert df.dot(f1, f2)((5, 7, 1)) == (47,)


def test_cross():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # Zero vectors
    f1 = df.Field(mesh, dim=3, value=(0, 0, 0))
    res = df.cross(f1, f1)
    assert res.dim == 3
    assert res.array.shape == (5, 5, 5, 3)
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

    # Spatially varying vectors
    def value_fun1(pos):
        x, y, z = pos
        return (x, y, z)

    def value_fun2(pos):
        x, y, z = pos
        return (z, x, y)

    f1 = df.Field(mesh, dim=3, value=value_fun1)
    f2 = df.Field(mesh, dim=3, value=value_fun2)

    # The cross product should be
    # (y**2-x*z, z**2-x*y, x**2-y*z)
    assert df.cross(f1, f2)((1, 1, 1)) == (0, 0, 0)
    assert df.cross(f1, f2)((3, 1, 1)) == (-2, -2, 8)
    assert df.cross(f2, f1)((3, 1, 1)) == (2, 2, -8)
    assert df.cross(f1, f2)((5, 7, 1)) == (44, -34, 18)

