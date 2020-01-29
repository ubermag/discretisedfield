import pytest
import numpy as np
import discretisedfield as df
import discretisedfield.util as dfu


def test_rescale():
    tol = 1e-6
    assert (dfu.rescale(1e-9)[0] - 1) < tol
    assert dfu.rescale(1e-9)[1] == 'n'
    assert (dfu.rescale(50e-9)[0] - 50) < tol
    assert dfu.rescale(50e-9)[1] == 'n'
    assert (dfu.rescale(100e-9)[0] - 100) < tol
    assert dfu.rescale(100e-9)[1] == 'n'
    assert (dfu.rescale(1001e-9)[0] - 1.001) < tol
    assert dfu.rescale(1001e-9)[1] == 'u'
    assert (dfu.rescale(0)[0] - 0) < tol
    assert dfu.rescale(0)[1] == ''
    assert (dfu.rescale(1e3)[0] - 1) < tol
    assert dfu.rescale(1e3)[1] == 'k'
    assert (dfu.rescale(0.5e-9)[0] - 500) < tol
    assert dfu.rescale(0.5e-9)[1] == 'p'
    assert (dfu.rescale(0.5)[0] - 500) < tol
    assert dfu.rescale(0.5)[1] == 'm'
    assert (dfu.rescale(0.05)[0] - 50) < tol
    assert dfu.rescale(0.05)[1] == 'm'
    assert (dfu.rescale(5)[0] - 5) < tol
    assert dfu.rescale(5)[1] == ''
    assert (dfu.rescale(50)[0] - 50) < tol
    assert dfu.rescale(50)[1] == ''
    assert (dfu.rescale(500)[0] - 500) < tol
    assert dfu.rescale(500)[1] == ''


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
