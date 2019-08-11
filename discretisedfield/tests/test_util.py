import pytest
import numpy as np
import discretisedfield.util as dfu


def test_array2tuple():
    dfu.array2tuple(np.array([1, 2, 3])) == (1, 2, 3)


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
