import numpy as np

import discretisedfield as df
import discretisedfield.plotting.util as plot_util


def test_inplane_angle():
    field = df.Mesh(p1=(-1, -1), p2=(1, 1), n=(2, 2)).coordinate_field()
    angles = plot_util.inplane_angle(field, "x", "y")
    assert isinstance(angles, df.Field)
    # The field vectors start at (-1, -1) with 180° + 45°
    # then first y increases -> 90° + 45°
    # then x and afterwards y increases -> 270° + 45° and 0° + 45°
    assert np.allclose(
        angles.array.flat, np.array([2, 1, 3, 0]) * np.pi / 2 + np.pi / 4
    )
