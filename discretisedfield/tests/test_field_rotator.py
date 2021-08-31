import numpy as np
import pytest
import discretisedfield as df
from .test_field import check_field


def test_vector_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    fr.rotate('align_vector', initial=(0, 0, 1), final=(1, 1, 1))
    check_field(fr.field)
    fr.rotate('align_vector', initial=(1, 1, 1), final=(0, 0, 1))
    check_field(fr.field)
    # field.allclose needs '==' for the mesh
    assert np.allclose(field.array, fr.field.array)


def test_scalar_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(1) * 2 - 1

    field = df.Field(mesh, dim=1, value=init_m)
    fr = df.FieldRotator(field)
    assert fr.field == field

    fr.rotate('align_vector', initial=(0, 0, 1), final=(1, 1, 1))
    check_field(fr.field)
    fr.rotate('align_vector', initial=(1, 1, 1), final=(0, 0, 1))
    check_field(fr.field)
    assert np.allclose(field.array, fr.field.array)


def test_scalar_cube():
    mesh = df.Mesh(p1=(-5, -5, -5), p2=(5, 5, 5), cell=(1, 1, 1))
    field = df.Field(mesh, dim=1, value=1)
    fr = df.FieldRotator(field)
    for s in ['x', 'y', 'z']:
        for pref in range(1, 5):
            fr.rotate('from_euler', seq=s, angles=pref * np.pi/2)
            check_field(fr.field)
            assert np.allclose(field.array, fr.field.array)
            fr.clear_rotation()
    check_field(fr.field)
    # no rotation => field should be the same
    assert field == fr.field


def test_from_quat_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    fr.rotate('from_quat', [0, 0, 1, 1])
    check_field(fr.field)


def test_from_matrix_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    matrix = [[0, -1, 0],
              [1, 0, 0],
              [0, 0, 1]]
    fr.rotate('from_matrix', matrix)
    check_field(fr.field)


def test_from_rotvec_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    fr.rotate('from_rotvec', rotvec=np.pi/2 * np.array([0, 0, 1]))
    check_field(fr.field)


def test_from_mrp_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    fr.rotate('from_mrp', [0, 0, np.pi/2])
    check_field(fr.field)


def test_from_euler_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    fr.rotate('from_euler', seq='x', angles=np.pi/2)
    check_field(fr.field)
    fr.rotate('from_euler', seq='xyz', angles=(np.pi/2, np.pi/4, np.pi/6))
    check_field(fr.field)
    fr.rotate('from_euler', seq='XYZ', angles=(np.pi/2, np.pi/4, np.pi/6))
    check_field(fr.field)


def test_n():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    # no rotation => field should be the same
    assert fr.field == field

    n = (10, 10, 10)
    fr.rotate('from_euler', seq='x', angles=np.pi/6, n=n)
    check_field(fr.field)
    assert fr.field.mesh.n == n


def test_invalid_field():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))
    field = df.Field(mesh, dim=2, value=(1, 1))
    with pytest.raises(ValueError):
        df.FieldRotator(field)


def test_invalid_method():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))
    field = df.Field(mesh, dim=3, value=(1, 1, 1))
    fr = df.FieldRotator(field)
    with pytest.raises(ValueError):
        fr.rotate('unknown method')
