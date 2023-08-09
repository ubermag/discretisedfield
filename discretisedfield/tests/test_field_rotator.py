import re

import numpy as np
import pytest

import discretisedfield as df

from .test_field import html_re as field_html_re

html_re = (
    r"<strong>FieldRotator</strong>\s*<ul>\s*"
    rf"<li>unrotated {field_html_re}</li>\s*"
    r"<li>rotation_quaternion:\s*<pre>.*</pre>\s*</li>\s*</ul>"
)


def check_rotator(rotator):
    assert isinstance(rotator.field, df.Field)

    assert isinstance(repr(rotator), str)
    pattern = r"^FieldRotator\(unrotatedField\(.+\), rotation_quaternion:.*\)$"
    assert re.match(pattern, repr(rotator))

    assert isinstance(rotator._repr_html_(), str)
    assert re.match(html_re, rotator._repr_html_(), re.DOTALL)


@pytest.fixture
def mesh():
    return df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))


@pytest.fixture(params=[(1, None), (3, 1)])
def field(mesh, request):
    nvdim, norm = request.param
    return df.Field(
        mesh, nvdim=nvdim, value=np.random.random((*mesh.n, nvdim)) * 2 - 1, norm=norm
    )


def test_valid_rotation(field):
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    fr.rotate("from_quat", [0, 0, 1, 1])
    check_rotator(fr)

    matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    fr.rotate("from_matrix", matrix)
    check_rotator(fr)

    fr.rotate("from_rotvec", rotvec=np.pi / 2 * np.array([0, 0, 1]))
    check_rotator(fr)

    fr.rotate("from_mrp", [0, 0, np.pi / 2])
    check_rotator(fr)

    fr.rotate("from_euler", seq="x", angles=np.pi / 2)
    check_rotator(fr)
    fr.rotate("from_euler", seq="xyz", angles=(np.pi / 2, np.pi / 4, np.pi / 6))
    check_rotator(fr)
    fr.rotate("from_euler", seq="XYZ", angles=(np.pi / 2, np.pi / 4, np.pi / 6))
    check_rotator(fr)

    fr.rotate("align_vector", initial=(1, 0, 1), final=(0, 0.2, -3))
    check_rotator(fr)


def test_valid_vdim_mapping(mesh):
    field = df.Field(
        mesh,
        nvdim=3,
        value=(1, 0, 0),
        vdims=("v0", "v1", "v2"),
        vdim_mapping={"v0": "z", "v1": "y", "v2": "x"},
    )
    fr = df.FieldRotator(field)
    fr.rotate("from_euler", seq="z", angles=np.pi / 2, n=(1, 1, 1))
    assert np.allclose(fr.field.v0.array, 1)
    assert np.allclose(fr.field.v1.array, 0)
    assert np.allclose(fr.field.v2.array, 0)

    fr = df.FieldRotator(field)
    fr.rotate("from_euler", seq="x", angles=np.pi / 2, n=(1, 1, 1))
    assert np.allclose(fr.field.v0.array, 0)
    assert np.allclose(fr.field.v1.array, -1)
    assert np.allclose(fr.field.v2.array, 0)


def test_n(field):
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    n = (10, 10, 10)
    fr.rotate("from_euler", seq="x", angles=np.pi / 6, n=n)
    check_rotator(fr)
    assert np.all(fr.field.mesh.n == n)


def test_rotation_inverse_rotation(field):
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    fr.rotate("align_vector", initial=(0, 0, 1), final=(1, 1, 1))
    check_rotator(fr)
    fr.rotate("align_vector", initial=(1, 1, 1), final=(0, 0, 1))
    check_rotator(fr)
    # field.allclose needs '==' for the mesh
    assert np.allclose(field.array, fr.field.array)


def test_scalar_cube():
    # multiples of 90 degree rotations should return the same array
    mesh = df.Mesh(p1=(-5, -5, -5), p2=(5, 5, 5), cell=(1, 1, 1))
    field = df.Field(mesh, nvdim=1, value=1)
    fr = df.FieldRotator(field)
    for s in ["x", "y", "z"]:
        for pref in range(1, 5):
            fr.rotate("from_euler", seq=s, angles=pref * np.pi / 2)
            check_rotator(fr)
            assert np.allclose(field.array, fr.field.array)
            fr.clear_rotation()
    check_rotator(fr)
    # no rotation => field should be the same
    assert field == fr.field


def test_macrospin_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(1, 1, 1), n=(1, 1, 1))
    field = df.Field(mesh, nvdim=3, value=(0, 0, 1))

    fr = df.FieldRotator(field)

    fr.rotate("from_euler", seq="z", angles=np.pi)
    assert np.allclose(fr.field.array.squeeze(), [0, 0, 1])

    fr.rotate("from_euler", seq="x", angles=np.pi / 2)
    assert np.allclose(fr.field.array.squeeze(), [0, -1, 0])

    fr.clear_rotation()
    fr.rotate("from_euler", seq="y", angles=np.pi / 2)
    assert np.allclose(fr.field.array.squeeze(), [1, 0, 0])
    fr.rotate("from_euler", seq="z", angles=np.pi)
    assert np.allclose(fr.field.array.squeeze(), [-1, 0, 0])

    fr.clear_rotation()
    fr.rotate("align_vector", initial=(0, 0, 1), final=(1, 1, 1), n=(1, 1, 1))
    assert np.allclose(
        fr.field.array.squeeze(), [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]
    )


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_invalid_field(valid_mesh, nvdim):
    field = df.Field(valid_mesh, nvdim=nvdim)
    if nvdim not in [1, 3] or valid_mesh.region.ndim != 3:
        with pytest.raises(ValueError):
            df.FieldRotator(field)


def test_invalid_vdim_mapping():
    mesh = df.Mesh(p1=(-5, -5, -5), p2=(5, 5, 5), cell=(1, 1, 1))
    field = df.Field(mesh, nvdim=3)
    field.vdim_mapping = {}
    with pytest.raises(ValueError):
        df.FieldRotator(field)

    field.vdim_mapping = {"x": None, "y": "y", "z": "z"}
    with pytest.raises(ValueError):
        df.FieldRotator(field)


def test_invalid_method(field):
    fr = df.FieldRotator(field)
    with pytest.raises(ValueError):
        fr.rotate("unknown method")
