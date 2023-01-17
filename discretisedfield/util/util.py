import cmath

import numpy as np
import ubermagutil.units as uu


def array2tuple(array):
    return array.item() if array.size == 1 else tuple(array.tolist())


def bergluescher_angle(v1, v2, v3):
    if np.dot(v1, np.cross(v2, v3)) == 0:
        # If the triple product is zero, then rho=0 and division by zero is
        # encountered. In this case, all three vectors are in-plane and the
        # space angle is zero.
        return 0.0
    else:
        rho = (
            2 * (1 + np.dot(v1, v2)) * (1 + np.dot(v2, v3)) * (1 + np.dot(v3, v1))
        ) ** 0.5

        numerator = (
            1
            + np.dot(v1, v2)
            + np.dot(v2, v3)
            + np.dot(v3, v1)
            + 1j * (np.dot(v1, np.cross(v2, v3)))
        )

        exp_omega = numerator / rho

        return 2 * cmath.log(exp_omega).imag / (4 * np.pi)


def assemble_index(value, n, dictionary):
    index = [value] * n
    for key, value in dictionary.items():
        index[key] = value

    return tuple(index)


def rescale_xarray(array, multiplier):
    """Rescale xarray dimensions."""
    if multiplier == 1:
        return array
    prefix = uu.rsi_prefixes[multiplier]
    try:
        units = [array[i].units for i in "xyz"]
    except AttributeError:
        units = None
    array = array.assign_coords({i: array[i] / multiplier for i in "xyz"})
    if units:
        for i, unit in zip("xyz", units):
            array[i].attrs["units"] = prefix + unit
    return array
