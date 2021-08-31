import discretisedfield as df
import numpy as np
from .test_field import check_field


def test_vector_rotation():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

    def init_m(p):
        return np.random.random(3) * 2 - 1

    field = df.Field(mesh, dim=3, value=init_m, norm=1)
    fr = df.FieldRotator(field)
    assert fr.field == field

    fr.rotate('align_vector', initial=(0, 0, 1), final=(1, 1, 1))
    check_field(fr.field)
    fr.rotate('align_vector', initial=(1, 1, 1), final=(0, 0, 1))
    check_field(fr.field)
    assert field == fr.field


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
    assert field == fr.field


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
    assert field == fr.field
