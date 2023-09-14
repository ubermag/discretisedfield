import itertools
import warnings

import numpy as np
from scipy import ndimage

import discretisedfield as df
import discretisedfield.util as dfu


def topological_charge_density(field, /, method="continuous"):
    r"""Topological charge density.

    This method computes the topological charge density for a vector field having three
    value components (i.e. ``nvdim=3``). Two different methods are available and can be
    selected using ``method``:

    1. Continuous method for calculation of the topological charge density in xy-plane:

        .. math::

            q = \frac{1}{4\pi} \mathbf{n} \cdot \left(\frac{\partial
            \mathbf{n}}{\partial x} \times \frac{\partial
            \mathbf{n}}{\partial y} \right),

        where :math:`\mathbf{n}` is the orientation field.

    2. Berg-Luescher method. Details can be found in:

        1. B. Berg and M. Luescher. Definition and statistical distributions of
        a topological number in the lattice O(3) sigma-model. Nuclear Physics B
        190 (2), 412-424 (1981).

        2. J.-V. Kim and J. Mulkers. On quantifying the topological charge in
        micromagnetics using a lattice-based approach. IOP SciNotes 1, 025211
        (2020).

    Topological charge is defined on two-dimensional geometries only. Therefore,
    the field must be "sliced" using the ``discretisedfield.Field.sel``
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
    >>> f = df.Field(mesh, nvdim=3, value=(1, 1, -1))
    ...
    >>> dft.topological_charge_density(f.sel('z'))
    Field(...)
    >>> dft.topological_charge_density(f.sel('z'), method='berg-luescher')
    Field(...)

    2. An attempt to compute the topological charge density of a scalar field.

    >>> f = df.Field(mesh, nvdim=1, value=12)
    >>> dft.topological_charge_density(f.sel('z'))
    Traceback (most recent call last):
    ...
    ValueError: ...

    3. Attempt to compute the topological charge density of a vector field,
    which is not sliced.

    >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
    >>> dft.topological_charge_density(f)
    Traceback (most recent call last):
    ...
    ValueError: ...

    .. seealso:: :py:func:`~discretisedfield.tools.topological_charge`

    """
    if field.nvdim != 3:
        raise ValueError(
            f"Cannot compute topological charge density for {field.nvdim=} field."
        )

    if field.mesh.region.ndim != 2:
        raise ValueError(
            "The topological charge density can only be computed on fields with 2"
            f" spatial dimensions, not {field.mesh.region.ndim=}."
        )

    of = field.orientation  # unit field - orientation field

    if method == "continuous":
        axis1 = field.mesh.region.dims[0]
        axis2 = field.mesh.region.dims[1]
        return 1 / (4 * np.pi) * of.dot(of.diff(axis1).cross(of.diff(axis2)))

    elif method == "berg-luescher":
        q = df.Field(field.mesh, nvdim=1, valid=of.valid)

        # Area of a single triangle
        area = 0.5 * field.mesh.cell[0] * field.mesh.cell[1]

        for i, j in itertools.product(range(of.mesh.n[0]), range(of.mesh.n[1])):
            if of.valid[i, j]:
                v0 = of.array[i, j]
                # Extract 4 neighbouring vectors (if they exist)
                v1 = (
                    of.array[i + 1, j]
                    if i + 1 < of.mesh.n[0] and of.valid[i + 1, j]
                    else None
                )
                v2 = (
                    of.array[i, j + 1]
                    if j + 1 < of.mesh.n[1] and of.valid[i, j + 1]
                    else None
                )
                v3 = of.array[i - 1, j] if i - 1 >= 0 and of.valid[i - 1, j] else None
                v4 = of.array[i, j - 1] if j - 1 >= 0 and of.valid[i, j - 1] else None

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
                    q.array[i, j] = charge / (area * triangle_count)

        return q

    else:
        raise ValueError(
            f"'method' can be either 'continuous' or 'berg-luescher', not '{method}'."
        )


def topological_charge(field, /, method="continuous", absolute=False):
    """Topological charge.

    This function computes topological charge for a vector field of three dimensions
    (i.e. ``nvdim=3``). There are two possible methods, which can be chosen using
    ``method`` parameter. For details on method, please refer to
    :py:func:`~discretisedfield.tools.topological_charge_density`. Absolute
    topological charge given as integral over the absolute values of the
    topological charge density can be computed by passing ``absolute=True``.

    Topological charge is defined on two-dimensional samples. Therefore,
    the field must be "sliced" using ``discretisedfield.Field.sel``
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
    >>> f = df.Field(mesh, nvdim=3, value=(1, 1, -1))
    >>> dft.topological_charge(f.sel('z'), method='continuous')
    0.0
    >>> dft.topological_charge(f.sel('z'), method='continuous',
    ...                                      absolute=True)
    0.0
    >>> dft.topological_charge(f.sel('z'), method='berg-luescher')
    0.0
    >>> dft.topological_charge(f.sel('z'), method='berg-luescher',
    ...                                      absolute=True)
    0.0

    2. Attempt to compute the topological charge of a scalar field.

    >>> f = df.Field(mesh, nvdim=1, value=12)
    >>> dft.topological_charge(f.sel('z'))
    Traceback (most recent call last):
    ...
    ValueError: ...

    3. Attempt to compute the topological charge of a vector field, which
    is not sliced.

    >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
    >>> dft.topological_charge(f)
    Traceback (most recent call last):
    ...
    ValueError: ...

    .. seealso::

        :py:func:`~discretisedfield.tools.topological_charge_density`

    """
    if field.nvdim != 3:
        raise ValueError(f"Cannot compute topological charge for {field.nvdim=} field.")

    if field.mesh.region.ndim != 2:
        raise ValueError(
            "The topological charge can only be computed on fields with 2"
            f" spatial dimensions, not {field.mesh.region.ndim=}."
        )

    q = topological_charge_density(field, method=method)
    if absolute:
        return float(abs(q).integrate())
    else:
        return float(q.integrate())


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
    >>> f = df.Field(mesh, nvdim=3, value=(1, 1, -1))
    ...
    >>> dft.emergent_magnetic_field(f)
    Field(...)

    """
    if field.nvdim != 3:
        raise ValueError(
            f"Cannot compute emergent magnetic field for {field.nvdim=}"
            " field. It must be three-dimensional vector field."
        )
    elif field.mesh.region.ndim != 3:
        raise ValueError(
            f"Cannot compute emergent magnetic field for {field.mesh.region.ndim=}"
            " region. It must be three-dimensional region."
        )

    geo_dims = field.mesh.region.dims

    F1 = field.dot(field.diff(geo_dims[1]).cross(field.diff(geo_dims[2])))
    F2 = field.dot(field.diff(geo_dims[2]).cross(field.diff(geo_dims[0])))
    F3 = field.dot(field.diff(geo_dims[0]).cross(field.diff(geo_dims[1])))

    return F1 << F2 << F3


def neighbouring_cell_angle(field, /, direction, units="rad"):
    """Calculate angles between neighbouring cells.

    This method calculates the angle between value vectors in all
    neighbouring cells. The calculation is only possible for vector fields of three
    dimensions (i.e. ``nvdim=3``). Angles are computed in degrees if ``units='deg'`` and
    in radians if ``units='rad'``.

    The resulting field has one discretisation cell less in the specified
    direction.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    direction : str

        The spatial direction in which the angles are calculated.

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
    >>> field = df.Field(mesh, nvdim=3, value=(0, 1, 0))
    ...
    >>> dft.neighbouring_cell_angle(field, direction='z')
    Field(...)

    """
    if not field.nvdim == 3:
        raise ValueError(
            f"Cannot compute value angles for a field with {field.nvdim=}."
            " Field must be three-dimensional."
        )

    if direction not in field.mesh.region.dims:
        raise ValueError(f"Cannot compute value angles for {direction=}.")

    if units not in ["rad", "deg"]:
        raise ValueError(f"Units {units=} not supported.")

    # Orientation field
    fo = field.orientation

    sclices_one = list()
    sclices_two = list()
    delta_p = list()
    for dim in fo.mesh.region.dims:
        if dim == direction:
            sclices_one.append(slice(-1))
            sclices_two.append(slice(1, None))
            delta_p.append(getattr(fo.mesh, f"d{dim}") / 2.0)
        else:
            sclices_one.append(slice(None))
            sclices_two.append(slice(None))
            delta_p.append(0)
    dot_product = np.einsum(
        "...j,...j->...", fo.array[(*sclices_one,)], fo.array[(*sclices_two,)]
    )

    # Define new mesh.
    p1 = np.add(field.mesh.region.pmin, delta_p)
    p2 = np.subtract(field.mesh.region.pmax, delta_p)
    mesh = df.Mesh(p1=p1, p2=p2, cell=field.mesh.cell)

    angles = np.arccos(np.clip(dot_product, -1.0, 1.0))
    if units == "deg":
        angles = np.degrees(angles)

    return df.Field(mesh, nvdim=1, value=angles.reshape(*angles.shape, 1))


def max_neighbouring_cell_angle(field, /, units="rad"):
    """Calculate maximum angle between neighbouring cells in all directions.

    This function computes an angle between a cell and all its six neighbouring
    cells and assigns the maximum to that cell. The calculation is only
    possible for vector fields of three dimensions (i.e. ``nvdim=3``). Angles are
    computed in degrees if ``units='deg'`` and in radians if ``units='rad'``.

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
    >>> field = df.Field(mesh, nvdim=3, value=(0, 1, 0))
    ...
    >>> dft.max_neighbouring_cell_angle(field)
    Field(...)

    """
    max_angles = np.zeros((*field.mesh.n, 2 * field.mesh.region.ndim))
    for i, dim in enumerate(field.mesh.region.dims):
        slices_one = [
            slice(1, None) if i == j else slice(None)
            for j in range(field.mesh.region.ndim)
        ]
        slices_two = [
            slice(-1) if i == j else slice(None) for j in range(field.mesh.region.ndim)
        ]
        max_angles[(*slices_one, 2 * i)] = neighbouring_cell_angle(
            field, dim, units=units
        ).array.squeeze()
        max_angles[(*slices_two, (2 * i) + 1)] = neighbouring_cell_angle(
            field, dim, units=units
        ).array.squeeze()

    max_angles = max_angles.max(axis=-1, keepdims=True)

    return df.Field(field.mesh, nvdim=1, value=max_angles)


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

        Direction of neighbouring cells. Can be ``None`` or one of the geometric
        dimensions. If ``None``, all directions are taken into account.
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
    >>> field = df.Field(mesh, nvdim=3, \
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
    >>> field = df.Field(mesh, nvdim=3, \
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
        cell_angles = max_neighbouring_cell_angle(field, units=units).array
    else:
        cell_angles = neighbouring_cell_angle(
            field, direction=direction, units=units
        ).array
    _, num_features = ndimage.label(cell_angles > min_angle)
    return num_features


def count_bps(field, /, direction):
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

        Geometric direction in which to compute arrangement.

    Returns
    -------
    dict

        Dictionary containing information about BPs.

    Examples
    --------
    """
    if field.mesh.region.ndim != 3:
        raise ValueError(f"The region must be 3D, not {field.mesh.region.ndim}D.")
    elif field.nvdim != 3:
        raise ValueError(f"The field must be 3D vector, not {field.nvdim}D.")
    elif direction not in field.mesh.region.dims:
        raise ValueError(
            f"The specified direction ({direction}) must be one of the"
            f" geometric dimensions {field.mesh.region.dims}."
        )

    F_div = emergent_magnetic_field(field.orientation).div

    averaged = [dim for dim in field.mesh.region.dims if dim != direction]

    F_red = F_div.integrate(direction=averaged[0]).integrate(direction=averaged[1])
    F_int = F_red.integrate(direction=direction, cumulative=True)
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
        nvdim=6,
        value=_N(mesh_new),
        vdims=["xx", "yy", "zz", "xy", "xz", "yz"],
    ).fftn()


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
        mesh_new, nvdim=6, value=values, vdims=["xx", "yy", "zz", "xy", "xz", "yz"]
    ).fftn()


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
    m_fft = m_pad.fftn()

    hx_fft = (
        tensor.ft_xx * m_fft.ft_x
        + tensor.ft_xy * m_fft.ft_y
        + tensor.ft_xz * m_fft.ft_z
    )
    hy_fft = (
        tensor.ft_xy * m_fft.ft_x
        + tensor.ft_yy * m_fft.ft_y
        + tensor.ft_yz * m_fft.ft_z
    )
    hz_fft = (
        tensor.ft_xz * m_fft.ft_x
        + tensor.ft_yz * m_fft.ft_y
        + tensor.ft_zz * m_fft.ft_z
    )

    H = hx_fft << hy_fft << hz_fft
    H.vdims = ["ft_x", "ft_y", "ft_z"]
    H = H.ifftn()
    return df.Field(
        m.mesh,
        nvdim=3,
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
