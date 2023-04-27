import numpy as np

import discretisedfield.tools as dft


def hopf_index(field):
    r"""Hopf index.

    Hopf index for a (magnetic) unit vector field :math:`\boldsymbol{m}`
    is defined as:

    .. math::

        H = -\frac{1}{(8\pi)^2} \int \mathrm{d}^3 r \, \boldsymbol{F} \cdot
        \boldsymbol{A},

    where :math:`\boldsymbol{F}` is the emergent magnetic field
    :math:`F_i = \varepsilon_{ijk} \boldsymbol{m} \cdot (\partial_j \boldsymbol{m}
    \times \partial_k \boldsymbol{m})`
    and :math:`\boldsymbol{A}` is defined implicitly through :math:`\nabla \times
    \boldsymbol{A} = \boldsymbol{F}`.

    Parameters
    ----------
    field : discretisedfield.Field

        Vector field.

    Returns
    -------
    float

        Hopf index.

    Raises
    ------
    ValueError

        If the field is not three-dimensional.

    Example
    -------
    1. Compute Hopf index of a spatially constant vector field.

    >>> import discretisedfield as df
    >>> import discretisedfield.tools as dft
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (10, 10, 10)
    >>> cell = (2, 2, 2)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    >>> f = df.Field(mesh, nvdim=3, value=(1, 1, -1))
    ...
    >>> dft.hopf_index(f)
    Field(...)

    """

    if field.nvdim != 3:
        raise ValueError(f"Cannot compute Hopf index for {field.nvdim=} field.")

    if field.mesh.region.ndim != 3:
        raise ValueError(
            "The Hopf index can only be computed on fields with 3"
            f" spatial dimensions, not {field.mesh.region.ndim=}."
        )

    of = field.orientation  # Unit vector field
    emergent_magnetic_field = dft.emergent_magnetic_field(of)

    axis2 = field.mesh.region.dims[1]  # y-axis

    # Vector potential
    Ax = -emergent_magnetic_field.z.integrate(axis2, cumulative=True)
    Az = emergent_magnetic_field.x.integrate(axis2, cumulative=True)

    integrand = Ax * emergent_magnetic_field.x + Az * emergent_magnetic_field.z
    H = -float(integrand.integrate() / (4 * np.pi) ** 2)

    return H
