import os

import numpy as np
import pytest

import discretisedfield as df
import discretisedfield.tools as dft


@pytest.mark.parametrize("method", ["continuous", "berg-luescher"])
def test_topological_charge(method):
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # f(x, y, z) = (0, 0, 0)
    # -> Q(f) = 0
    f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

    for normal_direction in "xyz":
        q = dft.topological_charge_density(f.sel(normal_direction), method=method)
        assert q.nvdim == 1
        assert np.allclose(q.mean(), 0)

        for absolute in [True, False]:
            assert np.allclose(
                dft.topological_charge(
                    f.sel(normal_direction), method=method, absolute=absolute
                ),
                0,
            )

    # Skyrmion (with PBC) from a file
    test_filename = os.path.join(
        os.path.dirname(__file__), "test_sample/", "skyrmion.omf"
    )
    f = df.Field.from_file(test_filename)

    q = dft.topological_charge_density(f.sel("z"), method=method)

    assert q.nvdim == 1
    assert q.mean() > 0
    for absolute in [True, False]:
        Q = dft.topological_charge(f.sel("z"), method=method, absolute=absolute)
        assert abs(Q - 1) < 0.15

    # Test valid with hald skyrmion
    f.valid = True
    f.valid[: f.valid.shape[0] // 2] = False

    q = dft.topological_charge_density(f.sel("z"), method=method)

    assert q.nvdim == 1
    assert q.mean() > 0
    assert np.array_equal(q.valid, f.sel("z").valid)
    for absolute in [True, False]:
        Q = dft.topological_charge(f.sel("z"), method=method, absolute=absolute)
        assert abs(Q - 0.5) < 0.15

    # Not sliced
    f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
    for function in ["topological_charge_density", "topological_charge"]:
        with pytest.raises(ValueError):
            getattr(dft, function)(f, method=method)

    # Scalar field
    f = df.Field(mesh, nvdim=1, value=3.14)
    for function in ["topological_charge_density", "topological_charge"]:
        with pytest.raises(ValueError):
            getattr(dft, function)(f.sel("z"), method=method)

    # Method does not exist
    f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
    for function in ["topological_charge_density", "topological_charge"]:
        with pytest.raises(ValueError):
            getattr(dft, function)(f.sel("z"), method="wrong")


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_emergent_magnetic_field(ndim, nvdim):
    p1 = (0,) * ndim
    p2 = (10,) * ndim
    cell = (2,) * ndim
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # f(x, y, z) = (0, 0, 0)
    # -> F(f) = 0
    f = df.Field(mesh, nvdim=nvdim)

    if ndim != 3 or nvdim != 3:
        with pytest.raises(ValueError):
            dft.emergent_magnetic_field(f)
    else:
        assert dft.emergent_magnetic_field(f).nvdim == 3
        assert np.allclose(dft.emergent_magnetic_field(f).mean(), (0, 0, 0))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_neighbouring_cell_angle(ndim, nvdim):
    p1 = (0,) * ndim
    p2 = (100,) * ndim
    n = (10,) * ndim
    value = (1,) * nvdim
    region = df.Region(p1=p1, p2=p2, dims=[f"g{i}" for i in range(1, ndim + 1)])
    mesh = df.Mesh(region=region, n=n)
    field = df.Field(mesh, nvdim=nvdim, value=value)

    # Exception
    if nvdim != 3:
        with pytest.raises(ValueError):
            dft.neighbouring_cell_angle(field, direction="g1")
        return

    with pytest.raises(ValueError):
        dft.neighbouring_cell_angle(field, direction="x")

    with pytest.raises(ValueError):
        dft.neighbouring_cell_angle(field, direction="g1", units="wrong")

    for direction in field.mesh.region.dims:
        for units in ["rad", "deg"]:
            sa = dft.neighbouring_cell_angle(field, direction=direction, units=units)
            assert sa.mean() == 0

    # Check for a value of angle
    arr_x_coord = mesh.coordinate_field().g1.array
    arr = np.concatenate(
        [
            np.full((*mesh.n, 1), 0),
            np.cos(arr_x_coord * np.pi / 20),
            np.sin(arr_x_coord * np.pi / 20),
        ],
        axis=-1,
    )
    fied_piby2 = df.Field(mesh, nvdim=3, value=arr)
    assert np.isclose(
        dft.neighbouring_cell_angle(fied_piby2, direction="g1", units="rad").mean(),
        np.pi / 2,
    )


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_max_neighbouring_cell_angle(ndim):
    p1 = (0,) * ndim
    p2 = (100,) * ndim
    n = (10,) * ndim
    regoin = df.Region(p1=p1, p2=p2, dims=[f"g{i}" for i in range(1, ndim + 1)])
    mesh = df.Mesh(region=regoin, n=n)
    field = df.Field(mesh, nvdim=3, value=(0, 1, 0))

    for units in ["rad", "deg"]:
        assert dft.max_neighbouring_cell_angle(field, units=units).mean() == 0

    # Check for a value of max angle
    def val(point):
        g1, *_ = point
        if g1 < 50:
            return (0, 0, 1)
        else:
            return (0, 0, -1)

    field.update_field_values(val)

    # Here the mean is 2*pi/10 because 2 cells will have pi values
    assert np.isclose(dft.max_neighbouring_cell_angle(field).mean(), 2 * np.pi / 10)


def test_count_lange_cell_angle_regions():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    n = (10, 10, 10)
    ps1 = (3, 3, 0)
    ps2 = (6, 4, 10)
    subregions = {"sub": df.Region(p1=ps1, p2=ps2)}
    mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
    field = df.Field(mesh, nvdim=3, value={"sub": (0, 0, 1), "default": (0, 0, -1)})

    for direction, res in [["x", 2], ["y", 1], ["z", 0]]:
        assert (
            dft.count_large_cell_angle_regions(field, min_angle=1, direction=direction)
            == res
        )

    assert dft.count_large_cell_angle_regions(field, min_angle=1) == 1


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_count_bps(ndim, nvdim):
    # TODO use BP field from file
    p1 = (0,) * ndim
    p2 = (10,) * ndim
    n = (10,) * ndim
    value = (1,) * nvdim
    dims = [f"g{i}" for i in range(ndim)]
    region = df.Region(p1=p1, p2=p2, dims=dims)
    mesh = df.Mesh(region=region, n=n)
    field = df.Field(mesh, nvdim=nvdim, value=value)

    if ndim != 3 or nvdim != 3:
        with pytest.raises(ValueError):
            dft.count_bps(field, "g1")
    else:
        with pytest.raises(ValueError):
            dft.count_bps(field, "wrong")
        result = dft.count_bps(field, "g1")
        assert result["bp_number"] == 0
        assert result["bp_number_hh"] == 0
        assert result["bp_number_tt"] == 0
        assert result["bp_pattern_g1"] == "[[0.0, 10]]"


@pytest.mark.filterwarnings("ignore:This method is still experimental.")
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
    rtensor = -tensor.ifftn().real[
        df.Region(p1=(-0.5 + 1e-12, -0.5 + 1e-12, -0.5), p2=(19.5, 9.5, 2.5))
    ]
    # the tensor computed with oommf is obtained with Oxs_SimpleDemag
    oommf_tensor = os.path.join(
        os.path.dirname(__file__), "test_sample", "demag_tensor_oommf.omf"
    )
    assert rtensor.allclose(df.Field.from_file(oommf_tensor))


@pytest.mark.filterwarnings("ignore:This method is still experimental.")
def test_demag_field_sphere():
    L = 10e-9
    mesh = df.Mesh(p1=(-L, -L, -L), p2=(L, L, L), cell=(1e-9, 1e-9, 1e-9))

    def norm(p):
        x, y, z = p
        if x**2 + y**2 + z**2 < L**2:
            return 1
        return 0

    f = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=norm)

    tensor = dft.demag_tensor(mesh)
    assert np.allclose(
        dft.demag_field(f, tensor)((0, 0, 0)),
        (0, 0, -1 / 3),
        atol=1e-4,  # non-accurate sphere approximation
    )

    oommf_sphere = os.path.join(
        os.path.dirname(__file__), "test_sample", "demag_field_sphere.omf"
    )
    assert dft.demag_field(f, tensor).allclose(df.Field.from_file(oommf_sphere))


@pytest.mark.filterwarnings("ignore:This method is still experimental.")
def test_demag_field_plane():
    L = 100
    mesh = df.Mesh(p1=(-L, -L, -1 / 2), p2=(L, L, 1 / 2), cell=(1, 1, 1))
    f = df.Field(mesh, nvdim=3, value=(0, 0, 1), norm=1)
    tensor = dft.demag_tensor(mesh)
    assert np.allclose(
        dft.demag_field(f, tensor)((0, 0, 0)),
        (0, 0, -1),
        rtol=5e-3,  # plane is not really infinite
    )

    oommf_plane = os.path.join(
        os.path.dirname(__file__), "test_sample", "demag_field_plane.omf"
    )
    assert dft.demag_field(f, tensor).allclose(df.Field.from_file(oommf_plane))
