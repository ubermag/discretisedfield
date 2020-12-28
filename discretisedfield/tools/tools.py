import numpy as np
import discretisedfield as df
import discretisedfield.util as dfu


def neigbouring_cell_angle(field, /, direction, units='rad'):
    """Calculate angles between neighbouring cells.

    This method calculates the angle between magnetic moments in all
    neighbouring cells. The calculation is only possible for vector fields
    (``dim=3``). Angles are computed in degrees if ``units='deg'`` and in
    radians if ``units='rad'``.

    The resulting field has one discretisation cell less in the specified
    direction.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    direction : str

        The direction in which the angles are calculated. Can be ``'x'``,
        ``'y'`` or ``'z'``.

    units : str, optional

        Angles are computed in degrees if ``units='deg'`` and in radians if
        ``units='rad'``. Defaults to ``'rad'``.

    Returns
    -------
    discretisedfield.Field

        A scalar field with angles. In the given direction the number of cells
        is reduced by one compared to the given field.

    Raises
    ------
    ValueError

        If ``field`` is not a vector field, or ``direction`` or ``units`` is
        invalid.

    Examples
    --------
    1. Computing the angle between neighbouring cells in z-direction.

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> field = df.Field(mesh, dim=3, value=(0, 1, 0))
    ...
    >>> dft.neigbouring_cell_angle(field, direction='z')
    Field(...)

    """
    if not field.dim == 3:
        msg = f'Cannot compute spin angles for a field with {field.dim=}.'
        raise ValueError(msg)

    if direction not in dfu.axesdict.keys():
        msg = f'Cannot compute spin angles for direction {direction=}.'
        raise ValueError(msg)

    if units not in ['rad', 'deg']:
        msg = f'Units {units=} not supported.'
        raise ValueError(msg)

    # Orientation field
    fo = field.orientation

    if direction == 'x':
        dot_product = np.einsum('...j,...j->...',
                                fo.array[:-1, ...],
                                fo.array[1:, ...])
        delta_p = np.divide((field.mesh.dx, 0, 0), 2)

    elif direction == 'y':
        dot_product = np.einsum('...j,...j->...',
                                fo.array[:, :-1, ...],
                                fo.array[:, 1:, ...])
        delta_p = np.divide((0, field.mesh.dy, 0), 2)

    elif direction == 'z':
        dot_product = np.einsum('...j,...j->...',
                                fo.array[..., :-1, :],
                                fo.array[..., 1:, :])
        delta_p = np.divide((0, 0, field.mesh.dz), 2)

    # Define new mesh.
    p1 = np.add(field.mesh.region.pmin, delta_p)
    p2 = np.subtract(field.mesh.region.pmax, delta_p)
    mesh = df.Mesh(p1=p1, p2=p2, cell=field.mesh.cell)

    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
    if units == 'deg':
        angles = np.degrees(angles)

    return df.Field(mesh, dim=1, value=angles.reshape(*angles.shape, 1))


def max_neigbouring_cell_angle(field, /, units='rad'):
    """Calculate maximum angle between neighbouring cells in all directions.

    This function computes an angle between a cell and all its six neighbouring
    cells and assigns the maximum to that cell. The calculation is only
    possible for vector fields (``dim=3``). Angles are computed in degrees if
    ``units='deg'`` and in radians if ``units='rad'``.

    The resulting field has one discretisation cell less in the specified
    direction.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    units : str, optional

        Angles are computed in degrees if ``units='deg'`` and in radians if
        ``units='rad'``. Defaults to ``'rad'``.

    Returns
    -------
    discretisedfield.Field

        A scalar field with maximum angles.

    Examples
    --------
    1. Computing the maximum angle between neighbouring cells.

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> field = df.Field(mesh, dim=3, value=(0, 1, 0))
    ...
    >>> dft.max_neigbouring_cell_angle(field)
    Field(...)

    """
    x_angles = neigbouring_cell_angle(field, 'x', units=units).array.squeeze()
    y_angles = neigbouring_cell_angle(field, 'y', units=units).array.squeeze()
    z_angles = neigbouring_cell_angle(field, 'z', units=units).array.squeeze()

    max_angles = np.zeros((*field.array.shape[:-1], 6), dtype=np.float64)
    max_angles[1:, :, :, 0] = x_angles
    max_angles[:-1, :, :, 1] = x_angles
    max_angles[:, 1:, :, 2] = y_angles
    max_angles[:, :-1, :, 3] = y_angles
    max_angles[:, :, 1:, 4] = z_angles
    max_angles[:, :, :-1, 5] = z_angles
    max_angles = max_angles.max(axis=-1, keepdims=True)

    return df.Field(field.mesh, dim=1, value=max_angles)
