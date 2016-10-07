import numpy as np
import discretisedfield.util as dfu


def test_plane_line_intersection():
    assert dfu.plane_line_intersection((0, 1, 0), (0, 1, 0),
                                       (0, 1, 0), (0, 0, 0)) == (0, 1, 0)
    assert dfu.plane_line_intersection((1, 0, 0), (1, 0, 0),
                                       (1, 1, 1), (0, 0, 0)) == (1, 1, 1)
    assert dfu.plane_line_intersection((1, 0, 0), (1, 0, 0),
                                       (1, -3, 1), (0, 0, 0)) == (1, -3, 1)

    # Special case 1: The intersection point should be (0, 0, 0),
    # however the equation results in d=0, implying there is not
    # a single intersection point.
    assert dfu.plane_line_intersection((1, 0, 0), (0, 0, 0),
                                       (1, 0, 0), (0, 0, 0)) == (0, 0, 0)
    assert dfu.plane_line_intersection((0, 1, 0), (0, 0, 0),
                                       (0, 1, 0), (0, 0, 0)) == (0, 0, 0)
    assert dfu.plane_line_intersection((0, 0, 1), (0, 0, 0),
                                       (0, 0, 1), (0, 0, 0)) == (0, 0, 0)
    assert dfu.plane_line_intersection((1, 0, 0), (0, 0, 0),
                                       (1, 1, 1), (0, 0, 0)) == (0, 0, 0)

    # Special case 2: n and l are perpendicular to each other,
    # which means that line and plane are parallel to each other.
    assert dfu.plane_line_intersection((1, 0, 0), (0, 0, 0),
                                       (0, 0, 1), (0, 0, 0)) is False
    assert dfu.plane_line_intersection((0, 1, 0), (0, 0, 0),
                                       (0, 0, 1), (0, 0, 0)) is False
    assert dfu.plane_line_intersection((1, 0, 0), (5, 1, 3),
                                       (0, 0, 1), (11, 9, -6)) is False


def test_box_line_intersection():
    points = dfu.box_line_intersection((0, 0, 0), (1, 1, 1),
                                       (1, 0, 0), (0, 0, 0))
    assert (0, 0, 0) in points
    assert (1, 0, 0) in points

    points = dfu.box_line_intersection((0, 0, 0), (1, 1, 1),
                                       (0, 1, 0), (0, 0, 0))
    assert (0, 0, 0) in points
    assert (0, 1, 0) in points

    points = dfu.box_line_intersection((0, 0, 0), (1, 1, 1),
                                       (0, 0, 1), (0, 0, 0))
    assert (0, 0, 0) in points
    assert (0, 0, 1) in points

    points = dfu.box_line_intersection((0, 0, 0), (5, 1, 1),
                                       (1, 0, 0), (0, 0, 0))
    assert (0, 0, 0) in points
    assert (5, 0, 0) in points

    points = dfu.box_line_intersection((0, 0, 0), (1, 1, 1),
                                       (1, 1, 1), (0, 0, 0))
    assert (0, 0, 0) in points
    assert (1, 1, 1) in points
