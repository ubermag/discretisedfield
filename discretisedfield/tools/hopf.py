import numpy as np

import discretisedfield.tools as dft


def hopf_index(field):
    """TODO Docstring"""

    if field.nvdim != 3:
        raise ValueError(f"Cannot compute Hopf index for {field.nvdim=} field.")

    if field.mesh.region.ndim != 3:
        raise ValueError(
            "The Hopf indexcan only be computed on fields with 3"
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
