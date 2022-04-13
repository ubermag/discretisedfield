import re

import numpy as np
import pytest

import discretisedfield as df

from .test_field import check_field
from .test_field import html_re as field_html_re

html_re = (
    r"<strong>FieldRotator</strong>\s*<ul>\s*"
    rf"<li>unrotated {field_html_re}</li>\s*"
    r"<li>rotation_quaternion:\s*<pre>.*</pre>\s*</li>\s*</ul>"
)


def check_rotator(rotator):
    check_field(rotator.field)

    assert isinstance(repr(rotator), str)
    pattern = r"^FieldRotator\(unrotatedField\(.+\), rotation_quaternion:.*\)$"
    assert re.match(pattern, repr(rotator))

    assert isinstance(rotator._repr_html_(), str)
    assert re.match(html_re, rotator._repr_html_(), re.DOTALL)


class TestFieldRotator:
    def setup(self):
        self.mesh = df.Mesh(p1=(0, 0, 0), p2=(20, 10, 5), cell=(1, 1, 1))

        def vector(p):
            return np.random.random(3) * 2 - 1

        def scalar(p):
            return np.random.random(1) * 2 - 1

        scalar_field = df.Field(self.mesh, dim=1, value=scalar)
        vector_field = df.Field(self.mesh, dim=3, value=vector, norm=1)
        self.fields = [scalar_field, vector_field]

    def test_valid_rotation(self):
        for field in self.fields:
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

    def test_n(self):
        for field in self.fields:
            fr = df.FieldRotator(field)
            # no rotation => field should be the same
            assert fr.field == field

            n = (10, 10, 10)
            fr.rotate("from_euler", seq="x", angles=np.pi / 6, n=n)
            check_rotator(fr)
            assert fr.field.mesh.n == n

    def test_rotation_inverse_rotation(self):
        for field in self.fields:
            fr = df.FieldRotator(field)
            # no rotation => field should be the same
            assert fr.field == field

            fr.rotate("align_vector", initial=(0, 0, 1), final=(1, 1, 1))
            check_rotator(fr)
            fr.rotate("align_vector", initial=(1, 1, 1), final=(0, 0, 1))
            check_rotator(fr)
            # field.allclose needs '==' for the mesh
            assert np.allclose(field.array, fr.field.array)

    def test_scalar_cube(self):
        # multiples of 90 degree rotations should return the same array
        mesh = df.Mesh(p1=(-5, -5, -5), p2=(5, 5, 5), cell=(1, 1, 1))
        field = df.Field(mesh, dim=1, value=1)
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

    def test_macrospin_rotation(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(1, 1, 1), n=(1, 1, 1))
        field = df.Field(mesh, dim=3, value=(0, 0, 1))

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

    def test_invalid_field(self):
        field = df.Field(self.mesh, dim=2, value=(1, 1))
        with pytest.raises(ValueError):
            df.FieldRotator(field)

    def test_invalid_method(self):
        for field in self.fields:
            fr = df.FieldRotator(field)
            with pytest.raises(ValueError):
                fr.rotate("unknown method")
