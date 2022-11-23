import numpy as np

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
    assert np.allclose(df.dS(field.plane("z")).mean(), (0, 0, 2))
    assert df.dV(field) == 6


def test_integrate():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (1, 1, 1)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, cell=cell)
    field = df.Field(mesh, dim=3, value=(1, -2, 3))

    assert np.allclose(df.integrate(field), (1000, -2000, 3000))
    assert np.allclose(df.integrate(field * 2), (2000, -4000, 6000))

    assert df.integrate(field.plane("z").dot([0, 0, 1])) == 300
