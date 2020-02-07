import pytest
import numpy as np
import discretisedfield as df
import discretisedfield.util as dfu


def test_array2tuple():
    dfu.array2tuple(np.array([1, 2, 3])) == (1, 2, 3)


def test_bergluescher_angle():
    # 1/8 of the full angle
    v1 = (1, 0, 0)
    v2 = (0, 1, 0)
    v3 = (0, 0, 1)

    angle = dfu.bergluescher_angle(v1, v2, v3)
    # CCW orientation
    assert dfu.bergluescher_angle(v1, v2, v3) == 1/8
    assert dfu.bergluescher_angle(v2, v3, v1) == 1/8
    assert dfu.bergluescher_angle(v3, v1, v2) == 1/8
    # CW orientation
    assert dfu.bergluescher_angle(v3, v2, v1) == -1/8
    assert dfu.bergluescher_angle(v2, v1, v3) == -1/8
    assert dfu.bergluescher_angle(v1, v3, v2) == -1/8

    # 0 of the full angle
    v1 = (1, 0, 0)
    v2 = (1, 0, 0)
    v3 = (0, 0, 1)

    angle = dfu.bergluescher_angle(v1, v2, v3)
    assert dfu.bergluescher_angle(v1, v2, v3) == 0
    assert dfu.bergluescher_angle(v2, v3, v1) == 0
    assert dfu.bergluescher_angle(v3, v1, v2) == 0


def test_assemble_index():
    index_dict = {0: 5, 1: 3, 2: 4}
    assert dfu.assemble_index(index_dict) == (5, 3, 4)
    index_dict = {2: 4}
    assert dfu.assemble_index(index_dict) == (0, 0, 4)
    index_dict = {1: 5, 2: 3, 0: 4}
    assert dfu.assemble_index(index_dict) == (4, 5, 3)
    index_dict = {1: 3, 2: 4}
    assert dfu.assemble_index(index_dict) == (0, 3, 4)


def test_voxels():
    plot_array = np.ones((5, 6, 7))
    pmin = (0, 0, 0)
    pmax = (5e6, 6e6, 7e6)
    color_palette = dfu.color_palette('cividis', 2, 'int')

    dfu.voxels(plot_array, pmin, pmax, color_palette)


def test_points():
    coordinates = np.array([(0, 0, 0)])
    color = dfu.color_palette('cividis', 1, 'int')[0]
    point_size = 2

    dfu.points(coordinates, color, point_size)


def test_vectors():
    coordinates = np.array([(0, 0, 0)])
    vectors = np.array([(1, 1, 1)])
    colors = [100]

    dfu.vectors(coordinates, vectors, colors)
