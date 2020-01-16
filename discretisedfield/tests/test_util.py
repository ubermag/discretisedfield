import pytest
import numpy as np
import discretisedfield as df
import discretisedfield.util as dfu


def test_array2tuple():
    dfu.array2tuple(np.array([1, 2, 3])) == (1, 2, 3)


def test_bergluescher_angle():
    v1 = (1, 0, 0)
    v2 = (0, 1, 0)
    v3 = (0, 0, 1)

    angle = dfu.bergluescher_angle(v1, v2, v3)
    print(angle)
    assert dfu.bergluescher_angle(v1, v2, v3)
    

def test_assemble_index():
    index_dict = {0: 5, 1: 3, 2: 4}
    assert dfu.assemble_index(index_dict) == (5, 3, 4)
    index_dict = {2: 4}
    assert dfu.assemble_index(index_dict) == (0, 0, 4)
    index_dict = {1: 5, 2: 3, 0: 4}
    assert dfu.assemble_index(index_dict) == (4, 5, 3)
    index_dict = {1: 3, 2: 4}
    assert dfu.assemble_index(index_dict) == (0, 3, 4)


def test_compatible():
    # One of the operands is not a field.
    mesh = df.Mesh(p1=(0, 0, 0), p2=(2, 2, 2), cell=(1, 1, 1))

    f1 = df.Field(mesh, dim=1, value=5)
    f2 = 5.1

    with pytest.raises(TypeError):
        dfu.compatible(f1, f2)

    # Fields defined on different meshes.
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(2, 2, 2), cell=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))

    f1 = df.Field(mesh1, dim=3, value=(0, 0, 0))
    f2 = df.Field(mesh2, dim=3, value=(0, 0, 0))

    with pytest.raises(ValueError):
        dfu.compatible(f1, f2)

    # Fields have different dimensions.
    mesh = df.Mesh(p1=(0, 0, 0), p2=(2, 2, 2), cell=(1, 1, 1))

    f1 = df.Field(mesh1, dim=1, value=5)
    f2 = df.Field(mesh2, dim=3, value=(0, 0, 0))

    with pytest.raises(ValueError):
        dfu.compatible(f1, f2)
    
    
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
