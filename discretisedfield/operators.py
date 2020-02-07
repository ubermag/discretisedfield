import numpy as np
import discretisedfield as df
import discretisedfield.util as dfu


def cross(f1, f2):
    """Cross product.

    This function computes the cross product between two fields. Both fields
    must be three-dimensional (``dim=3``) and defined on the same mesh.

    Parameters
    ----------
    f1/f2 : discretisedfield.Field

        Operands.

    Returns
    -------
    discretisedfield.Field

        Resulting field.

    Raises
    ------
    ValueError, TypeError

        If the operator cannot be applied.

    Example
    -------
    1. Compute the cross product of two vector fields.

    >>> import discretisedfield as df
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (10, 10, 10)
    >>> cell = (2, 2, 2)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    ...
    >>> f1 = df.Field(mesh, dim=3, value=(1, 0, 0))
    >>> f2 = df.Field(mesh, dim=3, value=(0, 1, 0))
    >>> df.cross(f1, f2).average
    (0.0, 0.0, 1.0)

    """
    if not isinstance(f1, df.Field) or not isinstance(f2, df.Field):
        msg = (f'Unsupported operand type(s) for discretisedfield.cross: '
               f'{type(f1)} and {type(f2)}.')
        raise TypeError(msg)
    if f1.dim != 3 or f2.dim != 3:
        msg = (f'Cannot compute the cross product on '
               f'dim={f1.dim} and dim={f2.dim} fields.')
        raise ValueError(msg)
    if f1.mesh != f2.mesh:
        msg = ('Cannot compute the cross product of '
               'fields defined on different meshes.')
        raise ValueError(msg)

    res_array = np.cross(f1.array, f2.array)
    return df.Field(f1.mesh, dim=3, value=res_array)


def stack(fields):
    """Stacks multiple scalar fields in a single vector field.

    This method takes a list of scalar (``dim=1``) fields and returns a vector
    field, whose components are defined by the scalar fields passed. If any of
    the fields passed has ``dim!=1` or they are not defined on the same mesh,
    an exception is raised. The dimension of the resulting field is equal to
    the length of the passed list.

    Parameters
    ----------
    fields : list

        List of ``discretisedfield.Field`` objects with ``dim=1``.

    Returns
    -------
    disrectisedfield.Field

        Resulting field.

    Raises
    ------
    ValueError

        If the dimension of any of the fields is not 1, or the fields passed
        are not defined on the same mesh.

    Example
    -------
    1. Stack 3 scalar fields.

    >>> import discretisedfield as df
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (10, 10, 10)
    >>> cell = (2, 2, 2)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    ...
    >>> f1 = df.Field(mesh, dim=1, value=1)
    >>> f2 = df.Field(mesh, dim=1, value=5)
    >>> f3 = df.Field(mesh, dim=1, value=-3)
    ...
    >>> f = df.stack([f1, f2, f3])
    >>> f.average
    (1.0, 5.0, -3.0)
    >>> f.dim
    3
    >>> f.x == f1
    True
    >>> f.y == f2
    True
    >>> f.z == f3
    True

    """
    if not all(isinstance(f, df.Field) for f in fields):
        msg = 'Only discretisedfield.Field objects can be stacked.'
        raise TypeError(msg)
    if not all(f.dim == 1 for f in fields):
        msg = 'Only dim=1 fields can be stacked.'
        raise ValueError(msg)
    if not all(f.mesh == fields[0].mesh for f in fields):
        msg = 'Only fields defined on the same mesh can be stacked.'
        raise ValueError(msg)

    array_list = [f.array[..., 0] for f in fields]
    return df.Field(fields[0].mesh, dim=len(fields),
                    value=np.stack(array_list, axis=3))
