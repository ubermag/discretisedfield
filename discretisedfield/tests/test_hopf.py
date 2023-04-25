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

    x = np.linspace(-5, 5, 100)
    y = np.copy(x)
    z = np.copy(x)
    X, Y, Z = np.meshgrid(x, y, z)

    p1 = (-5, -5, -5)
    p2 = (5, 5, 5)
    n = (100, 100, 100)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    # Test H=1 texture
    m = np.zeros(shape=(X.shape[0], X.shape[1], X.shape[2], 3), dtype=float)
    m[:, :, :, 0] = 2 * (psi1(X, Y, Z) * psi3(X, Y, Z) + psi2(X, Y, Z) * psi4(X, Y, Z))
    m[:, :, :, 1] = 2 * (psi2(X, Y, Z) * psi3(X, Y, Z) - psi1(X, Y, Z) * psi4(X, Y, Z))
    m[:, :, :, 2] = (
        psi1(X, Y, Z) ** 2
        + psi2(X, Y, Z) ** 2
        - psi3(X, Y, Z) ** 2
        - psi4(X, Y, Z) ** 2
    )
    field = df.Field(mesh, nvdim=3, value=m)
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
            dft.hopf_idx(field)
