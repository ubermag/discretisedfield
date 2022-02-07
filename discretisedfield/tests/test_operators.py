import pytest
import discretisedfield as df


def test_instances():
    p1 = (0, 0, 0)
    p2 = (100, 200, 300)
    cell = (1, 2, 3)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, cell=cell)
    field = df.Field(mesh, dim=3, value=(1, 2, 3))

    assert df.dx(field) == 1
    assert df.dy(field) == 2
    assert df.dz(field) == 3
    assert df.dS(field.plane('z')).average == (0, 0, 2)
    assert df.dV(field) == 6


def test_integral():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (1, 1, 1)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, cell=cell)
    field = df.Field(mesh, dim=3, value=(1, -2, 3))

    for attr in ['dx', 'dy', 'dz', 'dV']:
        assert df.integral(field*getattr(df, attr)) == (1000, -2000, 3000)
        assert df.integral(getattr(df, attr)*field) == (1000, -2000, 3000)
        assert df.integral(field*abs(getattr(df, attr))) == (1000, -2000, 3000)

    assert df.integral(field * (2 * df.dx)) == (2000, -4000, 6000)
    assert df.integral(field * (df.dx * 2)) == (2000, -4000, 6000)

    assert df.integral(field.plane('z') @ df.dS) == 300
    assert df.integral(df.dS @ field.plane('z')) == 300

    assert df.integral(field.plane('z') * (df.dS @ df.dS)) == (100, -200, 300)

    assert (field.plane('z') * (df.dS @ (0, 0, 1))).average == (1, -2, 3)
    assert (field.plane('z') * ((0, 0, 1) @ df.dS)).average == (1, -2, 3)

    dV = df.dx*df.dy*df.dz
    assert df.integral(field * dV) == df.integral(field * df.dV)

    with pytest.raises(TypeError):
        res = df.dx * 'dy'

    with pytest.raises(TypeError):
        res = df.dS @ 'dy'
