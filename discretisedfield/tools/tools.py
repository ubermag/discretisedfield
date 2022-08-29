import itertools
import warnings

import numpy as np
from scipy import ndimage

import discretisedfield as df
import discretisedfield.util as dfu


def topological_charge_density(field, /, method="continuous"):
    r"""Topological charge density.

    This method computes the topological charge density for a vector field
    (``dim=3``). Two different methods are available and can be selected using
    ``method``:

    1. Continuous method:

        .. math::

            q = \frac{1}{4\pi} \mathbf{n} \cdot \left(\frac{\partial
            \mathbf{n}}{\partial x} \times \frac{\partial
            \mathbf{n}}{\partial x} \right),

        where :math:`\mathbf{n}` is the orientation field.

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
        msg = f"Cannot compute topological charge density for {field.dim=} field."
        raise ValueError(msg)

    if not field.mesh.attributes["isplane"]:
        msg = (
            "The field must be sliced before the topological "
            "charge density can be computed."
        )
        raise ValueError(msg)

    if method not in ["continuous", "berg-luescher"]:
        msg = "Method can be either continuous or berg-luescher"
        raise ValueError(msg)

    axis1 = field.mesh.attributes["axis1"]
    axis2 = field.mesh.attributes["axis2"]
    of = field.orientation  # unit field - orientation field

    if method == "continuous":
        return (
            1
            / (4 * np.pi)
            * of
            @ (
                of.derivative(dfu.raxesdict[axis1])
                & of.derivative(dfu.raxesdict[axis2])
            )
        )

    elif method == "berg-luescher":
        q = df.Field(field.mesh, dim=1)

        # Area of a single triangle
        area = 0.5 * field.mesh.cell[axis1] * field.mesh.cell[axis2]

        for i, j in itertools.product(range(of.mesh.n[axis1]), range(of.mesh.n[axis2])):
            index = dfu.assemble_index(0, 3, {axis1: i, axis2: j})
            v0 = of.array[index]

            # Extract 4 neighbouring vectors (if they exist)
            v1 = v2 = v3 = v4 = None
            if i + 1 < of.mesh.n[axis1]:
                v1 = of.array[dfu.assemble_index(0, 3, {axis1: i + 1, axis2: j})]
            if j + 1 < of.mesh.n[axis2]:
                v2 = of.array[dfu.assemble_index(0, 3, {axis1: i, axis2: j + 1})]
            if i - 1 >= 0:
                v3 = of.array[dfu.assemble_index(0, 3, {axis1: i - 1, axis2: j})]
            if j - 1 >= 0:
                v4 = of.array[dfu.assemble_index(0, 3, {axis1: i, axis2: j - 1})]

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


def topological_charge(field, /, method="continuous", absolute=False):
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
        msg = f"Cannot compute topological charge for {field.dim=} field."
        raise ValueError(msg)

    if not field.mesh.attributes["isplane"]:
        msg = "The field must be sliced before the topological charge can be computed."
        raise ValueError(msg)

    if method not in ["continuous", "berg-luescher"]:
        msg = "Method can be either continuous or berg-luescher"
        raise ValueError(msg)

    if method == "continuous":
        q = topological_charge_density(field, method=method)
        if absolute:
            return df.integral(abs(q) * abs(df.dS))
        else:
            return df.integral(q * abs(df.dS))

    elif method == "berg-luescher":
        axis1 = field.mesh.attributes["axis1"]
        axis2 = field.mesh.attributes["axis2"]
        of = field.orientation

        topological_charge = 0
        for i, j in itertools.product(
            range(of.mesh.n[axis1] - 1), range(of.mesh.n[axis2] - 1)
        ):
            v1 = of.array[dfu.assemble_index(0, 3, {axis1: i, axis2: j})]
            v2 = of.array[dfu.assemble_index(0, 3, {axis1: i + 1, axis2: j})]
            v3 = of.array[dfu.assemble_index(0, 3, {axis1: i + 1, axis2: j + 1})]
            v4 = of.array[dfu.assemble_index(0, 3, {axis1: i, axis2: j + 1})]

            triangle1 = dfu.bergluescher_angle(v1, v2, v4)
            triangle2 = dfu.bergluescher_angle(v2, v3, v4)

            if absolute:
                triangle1 = abs(triangle1)
                triangle2 = abs(triangle2)

            topological_charge += triangle1 + triangle2

        return topological_charge


def emergent_magnetic_field(field):
    r"""Emergent magnetic field.

    Emergent magnetic field for a (magnetic) unit vector field
    :math:`\boldsymbol{m}` is defined as:

    .. math::

        F_{kl} = \boldsymbol{m} \cdot (\partial_k \boldsymbol{m}
        \times \partial_l \boldsymbol{m})

    Details are given in Volovik, G. E., Rysti, J., Mäkinen, J. T. & Eltsov,
    V. B. Spin, Orbital, Weyl and Other Glasses in Topological Superfluids. J
    Low Temp Phys 196, 82–101 (2019).

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
        msg = f"Cannot compute emergent magnetic field for {field.dim=} field."
        raise ValueError(msg)

    Fx = field @ (field.derivative("y") & field.derivative("z"))
    Fy = field @ (field.derivative("z") & field.derivative("x"))
    Fz = field @ (field.derivative("x") & field.derivative("y"))

    return Fx << Fy << Fz


def neigbouring_cell_angle(field, /, direction, units="rad"):
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
        msg = f"Cannot compute spin angles for a field with {field.dim=}."
        raise ValueError(msg)

    if direction not in dfu.axesdict.keys():
        msg = f"Cannot compute spin angles for direction {direction=}."
        raise ValueError(msg)

    if units not in ["rad", "deg"]:
        msg = f"Units {units=} not supported."
        raise ValueError(msg)

    # Orientation field
    fo = field.orientation

    if direction == "x":
        dot_product = np.einsum("...j,...j->...", fo.array[:-1, ...], fo.array[1:, ...])
        delta_p = np.divide((field.mesh.dx, 0, 0), 2)

    elif direction == "y":
        dot_product = np.einsum(
            "...j,...j->...", fo.array[:, :-1, ...], fo.array[:, 1:, ...]
        )
        delta_p = np.divide((0, field.mesh.dy, 0), 2)

    elif direction == "z":
        dot_product = np.einsum(
            "...j,...j->...", fo.array[..., :-1, :], fo.array[..., 1:, :]
        )
        delta_p = np.divide((0, 0, field.mesh.dz), 2)

    # Define new mesh.
    p1 = np.add(field.mesh.region.pmin, delta_p)
    p2 = np.subtract(field.mesh.region.pmax, delta_p)
    mesh = df.Mesh(p1=p1, p2=p2, cell=field.mesh.cell)

    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
    if units == "deg":
        angles = np.degrees(angles)

    return df.Field(mesh, dim=1, value=angles.reshape(*angles.shape, 1))


def max_neigbouring_cell_angle(field, /, units="rad"):
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
    x_angles = neigbouring_cell_angle(field, "x", units=units).array.squeeze()
    y_angles = neigbouring_cell_angle(field, "y", units=units).array.squeeze()
    z_angles = neigbouring_cell_angle(field, "z", units=units).array.squeeze()

    max_angles = np.zeros((*field.array.shape[:-1], 6))
    max_angles[1:, :, :, 0] = x_angles
    max_angles[:-1, :, :, 1] = x_angles
    max_angles[:, 1:, :, 2] = y_angles
    max_angles[:, :-1, :, 3] = y_angles
    max_angles[:, :, 1:, 4] = z_angles
    max_angles[:, :, :-1, 5] = z_angles
    max_angles = max_angles.max(axis=-1, keepdims=True)

    return df.Field(field.mesh, dim=1, value=max_angles)


def count_large_cell_angle_regions(field, /, min_angle, direction=None, units="rad"):
    """Count regions with large angles between neighbouring cells.

    This method counts regions, where the angle between neighbouring
    cells is above the given threshold. If ``direction`` is not specified
    the maximum of all neighbouring cells is used, otherwise only neighbouring
    cells in the given direction are taken into account. The minimum angle can
    be specified both in radians and degrees, depending on ``units``.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    min_angle : numbers.Real

        Minimum angle to count. Can be either radians or degrees depending
        on ``units``.

    direction : str, optional

        Direction of neighbouring cells. Can be ``None`` or one of ``x``,
        ``y``, or ``z``. If ``None``, all directions are taken into account.
        Defaults to ``None``.

    units : str, optional

        Unit of ``min_angle``. Can be ``rad`` for radians or ``deg`` for
        degrees. Defaults to ``rad``.

    Returns
    -------
    int

        Number of regions.

    Examples
    --------
    1. Counting regions depending on all directions.

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> field = df.Field(mesh, dim=3, \
                         value=lambda p: (0, 0, 1) if p[0] < 50 \
                         else (0, 0, -1))
    ...
    >>> dft.count_large_cell_angle_regions(field, min_angle=90, units='deg')
    1

    2. Counting regions depending on a single direction.

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> field = df.Field(mesh, dim=3, \
                         value=lambda p: (0, 0, 1) if p[0] < 50 \
                                                   else (0, 0, -1))
    ...
    >>> dft.count_large_cell_angle_regions(field, min_angle=90, units='deg', \
                                           direction='x')
    1
    >>> dft.count_large_cell_angle_regions(field, min_angle=90, units='deg', \
                                           direction='y')
    0
    """
    if direction is None:
        cell_angles = max_neigbouring_cell_angle(field, units=units).array
    else:
        cell_angles = neigbouring_cell_angle(
            field, direction=direction, units=units
        ).array
    _, num_features = ndimage.label(cell_angles > min_angle)
    return num_features


def count_bps(field, /, direction="x"):
    """Bloch point count and arrangement.

    Function to obtain information about Bloch point number and arrangement.
    The calculations are based on emergent magnetic field. The normalised
    volume integral over subvolumes, increasing cell by cell in the given
    ``direction`` is computed to obtain the local number of Bloch points at
    each point in the given ``direction``. Bloch point count and arangement
    are obtained by summing jumps in the local number of Bloch points.

    The results are:

    - Total number of Bloch points.
    - Number of head-to-head Bloch points.
    - Number of tail-to-tail Bloch points.
    - Arrangement of Bloch points in the given ``direction``. Starting from the
      lower end the local Bloch point count and the number of cells over which
      it stays constant are reported.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    direction : str, optional

        Direction in which to compute arrangement. Can be one of ``x``, ``y``,
        or ``z``. Defaults to ``x``.

    Returns
    -------
    dict

        Dictionary containing information about BPs.

    Examples
    --------
    """
    F_div = emergent_magnetic_field(field.orientation).div

    d_vals = {"x": df.dx, "y": df.dy, "z": df.dz}
    averaged = str.replace("xyz", direction, "")
    dF = d_vals[averaged[0]] * d_vals[averaged[1]]
    dl = d_vals[direction]

    F_red = df.integral(F_div * dF, direction=averaged)
    F_int = df.integral(F_red * dl, direction=direction, improper=True)
    bp_number = (F_int / (4 * np.pi)).array.squeeze().round()
    bp_count = bp_number[1:] - bp_number[:-1]

    results = {}
    results["bp_number"] = abs(bp_count).sum()
    results["bp_number_hh"] = abs(bp_count[bp_count < 0].sum())
    results["bp_number_tt"] = bp_count[bp_count > 0].sum()

    # pattern = list([<local BP_count>, <repetitions>])
    pattern = [[bp_number[0], 1]]
    for q_val in bp_number[1:]:
        if q_val == pattern[-1][0]:
            pattern[-1][1] += 1
        else:
            pattern.append([q_val, 1])
    results[f"bp_pattern_{direction}"] = str(pattern)

    return results


def _demag_tensor_field_based(mesh):
    """Fourier transform of the demag tensor.

    Computes the demag tensor in Fourier space. Only the six different
    components Nxx, Nyy, Nzz, Nxy, Nxz, Nyz are returned.

    This version is using discretisedfield which makes it easy to understand
    but slow compared to the numpy version. (The reason is the array
    initialisation which is basically a large for-loop.) For actual use the
    numpy version should be used. This version is kept as a reference.

    Parameters
    ----------
    mesh : discretisedfield.Mesh
        Mesh to compute the demag tensor on.

    Returns
    -------
    discretisedfield.Field
        Demag tensor in Fourier space.

    """
    p1 = [(-i + 1) * j - j / 2 for i, j in zip(mesh.n, mesh.cell)]
    p2 = [(i - 1) * j + j / 2 for i, j in zip(mesh.n, mesh.cell)]
    n = [2 * i - 1 for i in mesh.n]
    mesh_new = df.Mesh(p1=p1, p2=p2, n=n)

    return df.Field(
        mesh_new,
        dim=6,
        value=_N(mesh_new),
        components=["xx", "yy", "zz", "xy", "xz", "yz"],
    ).fftn


def demag_tensor(mesh):
    """Fourier transform of the demag tensor.

    Computes the demag tensor in Fourier space. Only the six different
    components Nxx, Nyy, Nzz, Nxy, Nxz, Nyz are returned.

    The implementation is based on Albert et al. JMMM 387 (2015)
    https://doi.org/10.1016/j.jmmm.2015.03.081

    Parameters
    ----------
    mesh : discretisedfield.Mesh
        Mesh to compute the demag tensor on.

    Returns
    -------
    discretisedfield.Field
        Demag tensor in Fourier space.
    """
    warnings.warn(
        "This method is still experimental. Users are strongly encouraged to use oommfc"
        " for the calculation of the demag field."
    )
    x = np.linspace(
        (-mesh.n[0] + 1) * mesh.cell[0],
        (mesh.n[0] - 1) * mesh.cell[0],
        mesh.n[0] * 2 - 1,
    )
    y = np.linspace(
        (-mesh.n[1] + 1) * mesh.cell[1],
        (mesh.n[1] - 1) * mesh.cell[1],
        mesh.n[1] * 2 - 1,
    )
    z = np.linspace(
        (-mesh.n[2] + 1) * mesh.cell[2],
        (mesh.n[2] - 1) * mesh.cell[2],
        mesh.n[2] * 2 - 1,
    )
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    values = np.stack(_N(mesh)((xx, yy, zz)), axis=3)

    p1 = [(-i + 1) * j - j / 2 for i, j in zip(mesh.n, mesh.cell)]
    p2 = [(i - 1) * j + j / 2 for i, j in zip(mesh.n, mesh.cell)]
    n = [2 * i - 1 for i in mesh.n]
    mesh_new = df.Mesh(p1=p1, p2=p2, n=n)

    return df.Field(
        mesh_new, dim=6, value=values, components=["xx", "yy", "zz", "xy", "xz", "yz"]
    ).fftn


def demag_field(m, tensor):
    """Calculate the demagnetisation field.

    The calculation of the demag field is based on Albert et al. JMMM 387
    (2015) https://doi.org/10.1016/j.jmmm.2015.03.081

    Parameters
    ----------
    m : discretisedfield.Field
        Magnetisation field

    tensor : discretisedfield.field
        Demagnetisation tensor obatained with ``dft.demag_tensor``

    Returns
    -------
    discretisedfield.Field
        Demagnetisation field
    """
    warnings.warn(
        "This method is still experimental. Users are strongly encouraged to use oommfc"
        " for the calculation of the demag field."
    )
    m_pad = m.pad(
        {d: (0, m.mesh.n[i] - 1) for d, i in zip(["x", "y", "z"], range(3))},
        mode="constant",
    )
    m_fft = m_pad.fftn

    hx_fft = tensor.xx * m_fft.x + tensor.xy * m_fft.y + tensor.xz * m_fft.z
    hy_fft = tensor.xy * m_fft.x + tensor.yy * m_fft.y + tensor.yz * m_fft.z
    hz_fft = tensor.xz * m_fft.x + tensor.yz * m_fft.y + tensor.zz * m_fft.z

    H = (hx_fft << hy_fft << hz_fft).ifftn
    return df.Field(
        m.mesh,
        dim=3,
        value=H.array[m.mesh.n[0] - 1 :, m.mesh.n[1] - 1 :, m.mesh.n[2] - 1 :, :],
    ).real


def _f(x, y, z):
    """Helper function to compute the demag tensor.

    This method implements function f from Albert et al. JMMM 387 (2015)
    https://doi.org/10.1016/j.jmmm.2015.03.081 which is required for the demag
    tensor.

    x, y, and z are mesh midpoints (either single points or numpy arrays).
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    # the total fraction goes to zero when the denominator is zero
    return (
        abs(y)
        / 2
        * (z2 - x2)
        * np.arcsinh(
            np.divide(
                abs(y), np.sqrt(x2 + z2), out=np.zeros_like(x), where=(x2 + z2) != 0
            )
        )
        + abs(z)
        / 2
        * (y2 - x2)
        * np.arcsinh(
            np.divide(
                abs(z), np.sqrt(x2 + y2), out=np.zeros_like(x), where=(x2 + y2) != 0
            )
        )
        - abs(x * y * z)
        * np.arctan(
            np.divide(
                abs(y * z),
                abs(x) * np.sqrt(x2 + y2 + z2),
                out=np.zeros_like(x),
                where=x != 0,
            )
        )
        + 1 / 6 * (2 * x2 - y2 - z2) * np.sqrt(x2 + y2 + z2)
    )


def _g(x, y, z):
    """Helper function to compute the demag tensor.

    This method implements function g from Albert et al. JMMM 387 (2015)
    https://doi.org/10.1016/j.jmmm.2015.03.081 which is required for the demag
    tensor.

    x, y, and z are mesh midpoints (either single points or numpy arrays).
    """
    x2 = x**2
    y2 = y**2
    z2 = z**2
    # the total fraction goes to zero when the denominator is zero
    return (
        x
        * y
        * z
        * np.arcsinh(
            np.divide(z, np.sqrt(x2 + y2), out=np.zeros_like(x), where=(x2 + y2) != 0)
        )
        + y
        / 6
        * (3 * z2 - y2)
        * np.arcsinh(
            np.divide(x, np.sqrt(y2 + z2), out=np.zeros_like(x), where=(y2 + z2) != 0)
        )
        + x
        / 6
        * (3 * z2 - x2)
        * np.arcsinh(
            np.divide(y, np.sqrt(x2 + z2), out=np.zeros_like(x), where=(x2 + z2) != 0)
        )
        - z**3
        / 6
        * np.arctan(
            np.divide(
                x * y, z * np.sqrt(x2 + y2 + z2), out=np.zeros_like(x), where=z != 0
            )
        )
        - z
        * y**2
        / 2
        * np.arctan(
            np.divide(
                x * z, y * np.sqrt(x2 + y2 + z2), out=np.zeros_like(x), where=y != 0
            )
        )
        - z
        * x**2
        / 2
        * np.arctan(
            np.divide(
                y * z, x * np.sqrt(x2 + y2 + z2), out=np.zeros_like(x), where=x != 0
            )
        )
        - x * y * np.sqrt(x2 + y2 + z2) / 3
    )


def _N_element(x, y, z, mesh, function):
    """Helper function to compute the demag tensor."""
    dx, dy, dz = mesh.cell
    value = 0.0
    for i in itertools.product([0, 1], repeat=6):
        value += (-1) ** np.sum(i) * function(
            x + (i[0] - i[3]) * dx, y + (i[1] - i[4]) * dy, z + (i[2] - i[5]) * dz
        )
    return -value / (4 * np.pi * np.prod(mesh.cell))


def _N(mesh):
    """Helper function to compute the demag tensor."""

    def _inner(p):
        x, y, z = p
        return (
            _N_element(x, y, z, mesh, _f),  # Nxx
            _N_element(y, z, x, mesh, _f),  # Nyy
            _N_element(z, x, y, mesh, _f),  # Nzz
            _N_element(x, y, z, mesh, _g),  # Nxy
            _N_element(x, z, y, mesh, _g),  # Nxz
            _N_element(y, z, x, mesh, _g),  # Nyz
        )

    return _inner
