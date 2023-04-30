import numpy as np
import pytest

import discretisedfield as df
import discretisedfield.tools as dft


def test_hopf_index():
    # Define Hopfion H=1 texture using stereographic projection
    def psi1(x, y, z):
        return 2 * x / (1 + x**2 + y**2 + z**2)

    def psi2(x, y, z):
        return 2 * y / (1 + x**2 + y**2 + z**2)

    def psi3(x, y, z):
        return 2 * z / (1 + x**2 + y**2 + z**2)

    def psi4(x, y, z):
        return (x**2 + y**2 + z**2 - 1) / (x**2 + y**2 + z**2 + 1)

    def mx(X, Y, Z):
        return 2 * (psi1(X, Y, Z)*psi4(X, Y, Z) + psi2(X, Y, Z)*psi3(X, Y, Z))

    def my(X, Y, Z):
        return 2*(psi2(X, Y, Z)*psi4(X, Y, Z) - psi1(X, Y, Z)*psi3(X, Y, Z))

    def mz(X, Y, Z):
        return psi4(X, Y, Z)**2 + psi3(X, Y, Z)**2 - psi2(X, Y, Z)**2 - psi1(X, Y, Z)**2

    def m_init(pos):
        x, y, z = pos
        return (mx(x, y, z), my(x, y, z), mz(x, y, z))

    p1 = (-5, -5, -5)
    p2 = (5, 5, 5)
    n = (100, 100, 100)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    # Test H=1 texture
    field = df.Field(mesh, nvdim=3, value=m_init)
    H = dft.hopf_index(field)
    assert 0.9 < H < 1.1

    # Test H=0 uniform texture
    field = df.Field(mesh, nvdim=3, value=(0, 0, 1))
    H = dft.hopf_index(field)
    assert abs(H) < 1e-3

    # Attempt to calculate Hopf index of a field with less than three dimensions
    for n_dims in (1, 2):
        field = df.Field(mesh, nvdim=n_dims)
        with pytest.raises(ValueError):
            dft.hopf_index(field)
