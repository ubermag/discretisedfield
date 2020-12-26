def spin_angle(self, direction, degrees=False):
    """Calculate spin angles between neighbouring cells.

    This method calculates the angle between the magnetic moments in all
    neighbouring cells. The calculation is only possible for fields with
    ``dim=3``. Angles between neighbouring cells in the given direction
    are calculated. Angles are can be in radians or degrees depending on
    ``degrees``.

    Parameters
    ----------
    direction : stream

        The direction in which the angles are calculated. Can be ``x``,
        ``y`` or ``z``.

    degrees : bool

        If ``True`` angles are given in degrees else in radians.

    Returns
    -------
    discretisedfield.Field

        A one-dimensional field with angles. In the given direction the
        number of cells is reduced by one compared to the given field.

    Raises
    ------
    ValueError

        If the field has not ``dim=3`` or the direction is invalid.

    Example
    -------
    1. Computing the angle between neighbouring cells in ``z``-direction.

    >>> import discretisedfield as df
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> field = df.Field(mesh, dim=3, value=(0, 1, 0))
    ...

    """
    if not self.dim == 3:
        msg = (f'Calculation of spin angles is not possible for a field'
               f' with dim = {self.dim}.')
        raise ValueError(msg)
    if direction == 'x':
        dot_product = np.einsum('...j,...j->...',
                                self.orientation.array[:-1, :, :, :],
                                self.orientation.array[1:, :, :, :])
        delta_p = np.array((self.mesh.cell[0], 0, 0)) / 2
    elif direction == 'y':
        dot_product = np.einsum('...j,...j->...',
                                self.orientation.array[:, :-1, :, :],
                                self.orientation.array[:, 1:, :, :])
        delta_p = np.array((0, self.mesh.cell[1], 0)) / 2
    elif direction == 'z':
        dot_product = np.einsum('...j,...j->...',
                                self.orientation.array[:, :, :-1, :],
                                self.orientation.array[:, :, 1:, :])
        delta_p = np.array((0, 0, self.mesh.cell[2])) / 2
    else:
        msg = f'Direction "{direction}" is not a valid direction.'
        raise ValueError(msg)
    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
    mesh = df.Mesh(p1=(np.array(self.mesh.region.p1) + delta_p),
                   p2=(np.array(self.mesh.region.p2) - delta_p),
                   cell=self.mesh.cell)
    if degrees:
        angles = np.degrees(angles)
    return self.__class__(mesh, dim=1,
                          value=angles.reshape(*angles.shape, 1))


def max_spin_angle(self, degrees=False):
    """Calculate the maximum spin angle for each cell.

    This method for each cell computes the spin angle to the six
    neighbouring cells and takes the maximum angle. The calculation is
    only possible for fields with ``dim=3``. Angles are can be in radians
    or degrees depending on ``degrees``.

    Parameters
    ----------
    degrees : bool

        If ``True`` angles are given in degrees else in radians.

    Returns
    -------
    discretisedfield.Field

        A one-dimensional field with angles with the same number of cells
        as the given field.

    Raises
    ------
    ValueError

        If the field has not ``dim=3`` or the direction is invalid.

    Example
    -------
    1. Computing the angle between neighbouring cells in ``z``-direction.

    >>> import discretisedfield as df
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> field = df.Field(mesh, dim=3, value=(0, 1, 0))

    """
    x_angles = self.spin_angle('x', degrees).array.squeeze()
    y_angles = self.spin_angle('y', degrees).array.squeeze()
    z_angles = self.spin_angle('z', degrees).array.squeeze()

    max_angles = np.zeros((*self.array.shape[:-1], 6), dtype=np.float64)
    max_angles[1:, :, :, 0] = x_angles
    max_angles[:-1, :, :, 1] = x_angles
    max_angles[:, 1:, :, 2] = y_angles
    max_angles[:, :-1, :, 3] = y_angles
    max_angles[:, :, 1:, 4] = z_angles
    max_angles[:, :, :-1, 5] = z_angles
    max_angles = max_angles.max(axis=-1, keepdims=True)

    return self.__class__(self.mesh, dim=1, value=max_angles)
