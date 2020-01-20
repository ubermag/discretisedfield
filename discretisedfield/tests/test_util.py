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
    
    
def test_plane_info():
    info = dfu.plane_info('x')
    assert info['planeaxis'] == 0
    assert info['axis1'] == 1
    assert info['axis2'] == 2
    assert info['point'] is None

    info = dfu.plane_info('y')
    assert info['planeaxis'] == 1
    assert info['axis1'] == 0
    assert info['axis2'] == 2
    assert info['point'] is None

    info = dfu.plane_info('z')
    assert info['planeaxis'] == 2
    assert info['axis1'] == 0
    assert info['axis2'] == 1
    assert info['point'] is None

    info = dfu.plane_info(x=0)
    assert info['planeaxis'] == 0
    assert info['axis1'] == 1
    assert info['axis2'] == 2
    assert info['point'] == 0

    info = dfu.plane_info(y=0)
    assert info['planeaxis'] == 1
    assert info['axis1'] == 0
    assert info['axis2'] == 2
    assert info['point'] == 0

    info = dfu.plane_info(z=0)
    assert info['planeaxis'] == 2
    assert info['axis1'] == 0
    assert info['axis2'] == 1
    assert info['point'] == 0

    info = dfu.plane_info(x=5)
    assert info['planeaxis'] == 0
    assert info['axis1'] == 1
    assert info['axis2'] == 2
    assert info['point'] == 5

    with pytest.raises(ValueError):
        info = dfu.plane_info('xy')
    with pytest.raises(ValueError):
        info = dfu.plane_info('zy')
    with pytest.raises(ValueError):
        info = dfu.plane_info('y', 'x')
    with pytest.raises(ValueError):
        info = dfu.plane_info('xzy')
    with pytest.raises(ValueError):
        info = dfu.plane_info('z', x=3)
    with pytest.raises(ValueError):
        info = dfu.plane_info('y', y=5)
    with pytest.raises(ValueError):
        info = dfu.plane_info(x=0, z=3)
