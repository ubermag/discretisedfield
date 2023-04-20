import numpy as np

import pytest

import discretisedfield as df
import discretisedfield.tools as dft


def test_hopf_index_H1():

    # Define Hopfion H=1 texture using stereographic projection
    def psi1(x, y, z):
        return 2*x / (1 + x**2 + y**2 + z**2)

    def psi2(x, y, z):
        return 2*y / (1 + x**2 + y**2 + z**2)

    def psi3(x, y, z):
        return 2*z / (1 + x**2 + y**2 + z**2)

    def psi4(x, y, z):
        return (x**2 + y**2 + z**2 - 1) / (x**2 + y**2 + z**2 + 1)

    x = np.linspace(-5, 5, 100)
    y = np.copy(x)
    z = np.copy(x)

    X, Y, Z = np.meshgrid(x, y, z)

    mx = 2 * (psi1(X, Y, Z)*psi3(X, Y, Z) + psi2(X, Y, Z)*psi4(X, Y, Z))
    my = 2 * (psi2(X, Y, Z)*psi3(X, Y, Z) - psi1(X, Y, Z)*psi4(X, Y, Z))
    mz = psi1(X, Y, Z)**2 + psi2(X, Y, Z)**2 - psi3(X, Y, Z)**2 - psi4(X, Y, Z)**2

    m = np.zeros(shape=(mx.shape[0], mx.shape[1], mx.shape[2], 3), dtype=float)
    m[:, :, :, 0] = mx
    m[:, :, :, 1] = my
    m[:, :, :, 2] = mz

    p1 = (-5, -5, -5)
    p2 = (5, 5, 5)
    n = (100, 100, 100)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    field = df.Field(mesh, nvdim=3, value=m)
    
    H = field.hopf_index()
    assert 0.99 < H < 1.01
