import os

import numpy as np
import pytest

import discretisedfield as df
import discretisedfield.tools as dft


def test_topological_charge():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # f(x, y, z) = (0, 0, 0)
    # -> Q(f) = 0
    f = df.Field(mesh, dim=3, value=(0, 0, 0))

    for method in ["continuous", "berg-luescher"]:
        q = dft.topological_charge_density(f.plane("z"), method=method)

        assert q.dim == 1
        assert q.average == 0
        for absolute in [True, False]:
            assert (
                dft.topological_charge(f.plane("z"), method=method, absolute=absolute)
                == 0
            )

    # Skyrmion (with PBC) from a file
    test_filename = os.path.join(
        os.path.dirname(__file__), "test_sample/", "skyrmion.omf"
    )
    f = df.Field.fromfile(test_filename)

    for method in ["continuous", "berg-luescher"]:
        q = dft.topological_charge_density(f.plane("z"), method=method)

        assert q.dim == 1
        assert q.average > 0
        for absolute in [True, False]:
            Q = dft.topological_charge(f.plane("z"), method=method, absolute=absolute)
            assert abs(Q) < 1 and abs(Q - 1) < 0.15

    # Not sliced
    f = df.Field(mesh, dim=3, value=(1, 2, 3))
    for function in ["topological_charge_density", "topological_charge"]:
        for method in ["continuous", "berg-luescher"]:
            with pytest.raises(ValueError):
                getattr(dft, function)(f, method=method)

    # Scalar field
    f = df.Field(mesh, dim=1, value=3.14)
    for function in ["topological_charge_density", "topological_charge"]:
        for method in ["continuous", "berg-luescher"]:
            with pytest.raises(ValueError):
                getattr(dft, function)(f.plane("z"), method=method)

    # Method does not exist
    f = df.Field(mesh, dim=3, value=(1, 2, 3))
    for function in ["topological_charge_density", "topological_charge"]:
        with pytest.raises(ValueError):
            getattr(dft, function)(f.plane("z"), method="wrong")


def test_emergent_magnetic_field():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # f(x, y, z) = (0, 0, 0)
    # -> F(f) = 0
    f = df.Field(mesh, dim=3, value=(0, 0, 0))

    assert dft.emergent_magnetic_field(f).dim == 3
    assert dft.emergent_magnetic_field(f).average == (0, 0, 0)

    with pytest.raises(ValueError):
        dft.emergent_magnetic_field(f.x)


def test_neigbouring_cell_angle():
    p1 = (0, 0, 0)
    p2 = (100, 100, 100)
    n = (10, 10, 10)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    field = df.Field(mesh, dim=3, value=(0, 1, 0))

    for direction in "xyz":
        for units in ["rad", "deg"]:
            sa = dft.neigbouring_cell_angle(field, direction=direction, units=units)
            assert sa.average == 0

    # Exceptions
    scalar_field = df.Field(mesh, dim=1, value=5)
    with pytest.raises(ValueError):
        dft.neigbouring_cell_angle(scalar_field, direction="x")

    with pytest.raises(ValueError):
        dft.neigbouring_cell_angle(field, direction="l")

    with pytest.raises(ValueError):
        dft.neigbouring_cell_angle(field, direction="x", units="wrong")


def test_max_neigbouring_cell_angle():
    p1 = (0, 0, 0)
    p2 = (100, 100, 100)
    n = (10, 10, 10)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    field = df.Field(mesh, dim=3, value=(0, 1, 0))

    for units in ["rad", "deg"]:
        assert dft.max_neigbouring_cell_angle(field, units=units).average == 0


def test_count_lange_cell_angle_regions():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    n = (10, 10, 10)
    ps1 = (3, 3, 0)
    ps2 = (6, 4, 10)
    subregions = {"sub": df.Region(p1=ps1, p2=ps2)}
    mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
    field = df.Field(mesh, dim=3, value={"sub": (0, 0, 1), "default": (0, 0, -1)})

    for direction, res in [["x", 2], ["y", 1], ["z", 0]]:
        assert (
            dft.count_large_cell_angle_regions(field, min_angle=1, direction=direction)
            == res
        )

    assert dft.count_large_cell_angle_regions(field, min_angle=1) == 1


def test_count_bps():
    # TODO use BP field from file
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    n = (10, 10, 10)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    field = df.Field(mesh, dim=3, value=(0, 0, 1))

    result = dft.count_bps(field)
    assert result["bp_number"] == 0
    assert result["bp_number_hh"] == 0
    assert result["bp_number_tt"] == 0
    assert result["bp_pattern_x"] == "[[0.0, 10]]"


def test_demag_tensor():
    L = 2e-9
    mesh = df.Mesh(p1=(-L, -L, -L), p2=(L, L, L), cell=(1e-9, 1e-9, 1e-9))
    # The second method is very slow and only intended for demonstration
    # purposes as it is easier to understand. It is not exposed anywhere.
    assert dft.demag_tensor(mesh).allclose(
        df.tools.tools._demag_tensor_field_based(mesh)
    )

    mesh = df.Mesh(p1=(-0.5, -0.5, -0.5), p2=(19.5, 9.5, 2.5), cell=(1, 1, 1))
    tensor = dft.demag_tensor(mesh)
    # differences to oommf:
    # - an additional minus sign in the tensor definiton
    # - the tensor is in fourier space
    # - the tensor is padded as required for the calculation of the demag field
    rtensor = -tensor.ifftn.real[
        df.Region(p1=(-0.5 + 1e-12, -0.5 + 1e-12, -0.5), p2=(19.5, 9.5, 2.5))
    ]
    # the tensor computed with oommf is obtained with Oxs_SimpleDemag
    oommf_tensor = os.path.join(
        os.path.dirname(__file__), "test_sample", "demag_tensor_oommf.omf"
    )
    assert rtensor.allclose(df.Field.fromfile(oommf_tensor))


def test_demag_field_sphere():
    L = 10e-9
    mesh = df.Mesh(p1=(-L, -L, -L), p2=(L, L, L), cell=(1e-9, 1e-9, 1e-9))

    def norm(p):
        x, y, z = p
        if x**2 + y**2 + z**2 < L**2:
            return 1
        return 0

    f = df.Field(mesh, dim=3, value=(0, 0, 1), norm=norm)

    tensor = dft.demag_tensor(mesh)
    assert np.allclose(
        dft.demag_field(f, tensor)((0, 0, 0)),
        (0, 0, -1 / 3),
        atol=1e-4,  # non-accurate sphere approximation
    )

    oommf_sphere = os.path.join(
        os.path.dirname(__file__), "test_sample", "demag_field_sphere.omf"
    )
    assert dft.demag_field(f, tensor).allclose(df.Field.fromfile(oommf_sphere))


def test_demag_field_plane():
    L = 100
    mesh = df.Mesh(p1=(-L, -L, -1 / 2), p2=(L, L, 1 / 2), cell=(1, 1, 1))
    f = df.Field(mesh, dim=3, value=(0, 0, 1), norm=1)
    tensor = dft.demag_tensor(mesh)
    assert np.allclose(
        dft.demag_field(f, tensor)((0, 0, 0)),
        (0, 0, -1),
        rtol=5e-3,  # plane is not really infinite
    )

    oommf_plane = os.path.join(
        os.path.dirname(__file__), "test_sample", "demag_field_plane.omf"
    )
    assert dft.demag_field(f, tensor).allclose(df.Field.fromfile(oommf_plane))
