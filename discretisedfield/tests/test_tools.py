import pytest
import discretisedfield as df
import discretisedfield.tools as dft


def test_neigbouring_cell_angle():
    p1 = (0, 0, 0)
    p2 = (100, 100, 100)
    n = (10, 10, 10)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    field = df.Field(mesh, dim=3, value=(0, 1, 0))

    for direction in 'xyz':
        for units in ['rad', 'deg']:
            sa = dft.neigbouring_cell_angle(field,
                                            direction=direction,
                                            units=units)
            assert sa.average == 0

    # Exceptions
    scalar_field = df.Field(mesh, dim=1, value=5)
    with pytest.raises(ValueError):
        res = dft.neigbouring_cell_angle(scalar_field, direction='x')

    with pytest.raises(ValueError):
        res = dft.neigbouring_cell_angle(field, direction='l')

    with pytest.raises(ValueError):
        res = dft.neigbouring_cell_angle(field, direction='x', units='wrong')

def test_max_neigbouring_cell_angle():
    p1 = (0, 0, 0)
    p2 = (100, 100, 100)
    n = (10, 10, 10)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    field = df.Field(mesh, dim=3, value=(0, 1, 0))

    for units in ['rad', 'deg']:
        assert dft.max_neigbouring_cell_angle(field, units=units).average == 0
