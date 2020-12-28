import itertools
import numpy as np
import discretisedfield as df
import discretisedfield.util as dfu


def topological_charge_density(field, /, method='continuous'):
    """Topological charge density.

    This method computes the topological charge density for a vector field
    (``dim=3``). Two different methods are available and can be selected using
    ``method``:

    1. Continuous method:

        .. math::

            q = \\frac{1}{4\\pi} \\mathbf{n} \\cdot \\left(\\frac{\\partial
            \\mathbf{n}}{\\partial x} \\times \\frac{\\partial
            \\mathbf{n}}{\\partial x} \\right),

        where :math:`\\mathbf{n}` is the orientation field.

    2. Berg-Luescher method. Details can be found in:

        1. B. Berg and M. Luescher. Definition and statistical distributions of
        a topological number in the lattice O(3) sigma-model. Nuclear Physics B
        190 (2), 412-424 (1981).

        2. J.-V. Kim and J. Mulkers. On quantifying the topological charge in
        micromagnetics using a lattice-based approach. IOP SciNotes 1, 025211
        (2020).

    Topological charge is defined on two-dimensional samples only. Therefore,
    the field must be "sliced" using the ``discretisedfield.Field.plane``
    method. If the field is not three-dimensional or the field is not sliced,
    ``ValueError`` is raised.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    method : str, optional

        Method how the topological charge is computed. It can be ``continuous``
        or ``berg-luescher``. Defaults to ``continuous``.

    Returns
    -------
    discretisedfield.Field

        Topological charge density scalar field.

    Raises
    ------
    ValueError

        If the field is not three-dimensional or the field is not sliced.

    Example
    -------
    1. Compute topological charge density of a spatially constant vector field.

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (10, 10, 10)
    >>> cell = (2, 2, 2)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    >>> f = df.Field(mesh, dim=3, value=(1, 1, -1))
    ...
    >>> dft.topological_charge_density(f.plane('z'))
    Field(...)
    >>> dft.topological_charge_density(f.plane('z'), method='berg-luescher')
    Field(...)

    2. An attempt to compute the topological charge density of a scalar field.

    >>> f = df.Field(mesh, dim=1, value=12)
    >>> dft.topological_charge_density(f.plane('z'))
    Traceback (most recent call last):
    ...
    ValueError: ...

    3. Attempt to compute the topological charge density of a vector field,
    which is not sliced.

    >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
    >>> dft.topological_charge_density(f)
    Traceback (most recent call last):
    ...
    ValueError: ...

    .. seealso:: :py:func:`~discretisedfield.tools.topological_charge`

    """
    if field.dim != 3:
        msg = (f'Cannot compute topological charge density '
               f'for {field.dim=} field.')
        raise ValueError(msg)

    if not hasattr(field.mesh, 'info'):
        msg = ('The field must be sliced before the topological '
               'charge density can be computed.')
        raise ValueError(msg)

    if method not in ['continuous', 'berg-luescher']:
        msg = 'Method can be either continuous or berg-luescher'
        raise ValueError(msg)

    axis1 = field.mesh.info['axis1']
    axis2 = field.mesh.info['axis2']
    of = field.orientation  # unit field - orientation field

    if method == 'continuous':
        return 1/(4*np.pi) * of @ (of.derivative(dfu.raxesdict[axis1]) &
                                   of.derivative(dfu.raxesdict[axis2]))

    elif method == 'berg-luescher':
        q = field.__class__(field.mesh, dim=1, value=0)

        # Area of a single triangle
        area = 0.5 * field.mesh.cell[axis1] * field.mesh.cell[axis2]

        for i, j in itertools.product(range(of.mesh.n[axis1]),
                                      range(of.mesh.n[axis2])):
            index = dfu.assemble_index(0, 3, {axis1: i, axis2: j})
            v0 = of.array[index]

            # Extract 4 neighbouring vectors (if they exist)
            v1 = v2 = v3 = v4 = None
            if i + 1 < of.mesh.n[axis1]:
                v1 = of.array[dfu.assemble_index(0, 3, {axis1: i+1,
                                                        axis2: j})]
            if j + 1 < of.mesh.n[axis2]:
                v2 = of.array[dfu.assemble_index(0, 3, {axis1: i,
                                                        axis2: j+1})]
            if i - 1 >= 0:
                v3 = of.array[dfu.assemble_index(0, 3, {axis1: i-1,
                                                        axis2: j})]
            if j - 1 >= 0:
                v4 = of.array[dfu.assemble_index(0, 3, {axis1: i,
                                                        axis2: j-1})]

            charge = 0
            triangle_count = 0

            if v1 is not None and v2 is not None:
                triangle_count += 1
                charge += dfu.bergluescher_angle(v0, v1, v2)

            if v2 is not None and v3 is not None:
                triangle_count += 1
                charge += dfu.bergluescher_angle(v0, v2, v3)

            if v3 is not None and v4 is not None:
                triangle_count += 1
                charge += dfu.bergluescher_angle(v0, v3, v4)

            if v4 is not None and v1 is not None:
                triangle_count += 1
                charge += dfu.bergluescher_angle(v0, v4, v1)

            if triangle_count > 0:
                q.array[index] = charge / (area * triangle_count)
            else:
                # If the cell has no neighbouring cells
                q.array[index] = 0

        return q


def topological_charge(field, /, method='continuous', absolute=False):
    """Topological charge.

    This function computes topological charge for a vector field (``dim=3``).
    There are two possible methods, which can be chosen using ``method``
    parameter. For details on method, please refer to
    :py:func:`~discretisedfield.tools.topological_charge_density`. Absolute
    topological charge given as integral over the absolute values of the
    topological charge density can be computed by passing ``absolute=True``.

    Topological charge is defined on two-dimensional samples. Therefore,
    the field must be "sliced" using ``discretisedfield.Field.plane``
    method. If the field is not three-dimensional or the field is not
    sliced and ``ValueError`` is raised.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    method : str, optional

        Method how the topological charge is computed. It can be ``continuous``
        or ``berg-luescher``. Defaults to ``continuous``.

    absolute : bool, optional

        If ``True`` the absolute topological charge is computed.
        Defaults to ``False``.

    Returns
    -------
    float

        Topological charge.

    Raises
    ------
    ValueError

        If the field is not three-dimensional or the field is not sliced.

    Example
    -------
    1. Compute the topological charge of a spatially constant vector field
    (zero value is expected).

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (10, 10, 10)
    >>> cell = (2, 2, 2)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    ...
    >>> f = df.Field(mesh, dim=3, value=(1, 1, -1))
    >>> dft.topological_charge(f.plane('z'), method='continuous')
    0.0
    >>> dft.topological_charge(f.plane('z'), method='continuous',
    ...                                      absolute=True)
    0.0
    >>> dft.topological_charge(f.plane('z'), method='berg-luescher')
    0.0
    >>> dft.topological_charge(f.plane('z'), method='berg-luescher',
    ...                                      absolute=True)
    0.0

    2. Attempt to compute the topological charge of a scalar field.

    >>> f = df.Field(mesh, dim=1, value=12)
    >>> dft.topological_charge(f.plane('z'))
    Traceback (most recent call last):
    ...
    ValueError: ...

    3. Attempt to compute the topological charge of a vector field, which
    is not sliced.

    >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
    >>> dft.topological_charge(f)
    Traceback (most recent call last):
    ...
    ValueError: ...

    .. seealso::

        :py:func:`~discretisedfield.tools.topological_charge_density`

    """
    if field.dim != 3:
        msg = f'Cannot compute topological charge for {field.dim=} field.'
        raise ValueError(msg)

    if not hasattr(field.mesh, 'info'):
        msg = ('The field must be sliced before the '
               'topological charge can be computed.')
        raise ValueError(msg)

    if method not in ['continuous', 'berg-luescher']:
        msg = 'Method can be either continuous or berg-luescher'
        raise ValueError(msg)

    if method == 'continuous':
        q = topological_charge_density(field, method=method)
        if absolute:
            return df.integral(abs(q) * abs(df.dS))
        else:
            return df.integral(q * abs(df.dS))

    elif method == 'berg-luescher':
        axis1 = field.mesh.info['axis1']
        axis2 = field.mesh.info['axis2']
        of = field.orientation

        topological_charge = 0
        for i, j in itertools.product(range(of.mesh.n[axis1] - 1),
                                      range(of.mesh.n[axis2] - 1)):
            v1 = of.array[dfu.assemble_index(0, 3, {axis1: i,
                                                    axis2: j})]
            v2 = of.array[dfu.assemble_index(0, 3, {axis1: i + 1,
                                                    axis2: j})]
            v3 = of.array[dfu.assemble_index(0, 3, {axis1: i + 1,
                                                    axis2: j + 1})]
            v4 = of.array[dfu.assemble_index(0, 3, {axis1: i,
                                                    axis2: j + 1})]

            triangle1 = dfu.bergluescher_angle(v1, v2, v4)
            triangle2 = dfu.bergluescher_angle(v2, v3, v4)

            if absolute:
                triangle1 = abs(triangle1)
                triangle2 = abs(triangle2)

            topological_charge += triangle1 + triangle2

        return topological_charge


def emergent_magnetic_field(field):
    """Emergent magnetic field.

    PUT EQUATION HERE and a reference.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    Returns
    -------
    discretisedfield.Field

        Emergent magnetic field.

    Raises
    ------
    ValueError

        If the field is not three-dimensional.

    Example
    -------
    1. Compute topological charge density of a spatially constant vector field.

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (10, 10, 10)
    >>> cell = (2, 2, 2)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    >>> f = df.Field(mesh, dim=3, value=(1, 1, -1))
    ...
    >>> dft.emergent_magnetic_field(f)
    Field(...)

    """
    if field.dim != 3:
        msg = (f'Cannot compute emergent magnetic field '
               f'for {field.dim=} field.')
        raise ValueError(msg)

    Fx = field @ (field.derivative('y') & field.derivative('z'))
    Fy = field @ (field.derivative('x') & field.derivative('y'))
    Fz = field @ (field.derivative('z') & field.derivative('x'))

    return Fx << Fy << Fz


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
