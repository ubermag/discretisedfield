import cmath
import collections

import numpy as np
import ubermagutil.units as uu

axesdict = collections.OrderedDict(x=0, y=1, z=2)
raxesdict = {value: key for key, value in axesdict.items()}

# Color pallete as hex and int.
cp_hex = [
    "#4c72b0",
    "#dd8452",
    "#55a868",
    "#c44e52",
    "#8172b3",
    "#937860",
    "#da8bc3",
    "#8c8c8c",
    "#ccb974",
    "#64b5cd",
]
cp_int = [int(color[1:], 16) for color in cp_hex]


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
    index = [
        value,
    ] * n
    for key, value in dictionary.items():
        index[key] = value

    return tuple(index)


def plot_line(ax, p1, p2, *args, **kwargs):
    ax.plot(*zip(p1, p2), *args, **kwargs)


def plot_box(ax, pmin, pmax, *args, **kwargs):
    x1, y1, z1 = pmin
    x2, y2, z2 = pmax

    plot_line(ax, (x1, y1, z1), (x2, y1, z1), *args, **kwargs)
    plot_line(ax, (x1, y2, z1), (x2, y2, z1), *args, **kwargs)
    plot_line(ax, (x1, y1, z2), (x2, y1, z2), *args, **kwargs)
    plot_line(ax, (x1, y2, z2), (x2, y2, z2), *args, **kwargs)

    plot_line(ax, (x1, y1, z1), (x1, y2, z1), *args, **kwargs)
    plot_line(ax, (x2, y1, z1), (x2, y2, z1), *args, **kwargs)
    plot_line(ax, (x1, y1, z2), (x1, y2, z2), *args, **kwargs)
    plot_line(ax, (x2, y1, z2), (x2, y2, z2), *args, **kwargs)

    plot_line(ax, (x1, y1, z1), (x1, y1, z2), *args, **kwargs)
    plot_line(ax, (x2, y1, z1), (x2, y1, z2), *args, **kwargs)
    plot_line(ax, (x1, y2, z1), (x1, y2, z2), *args, **kwargs)
    plot_line(ax, (x2, y2, z1), (x2, y2, z2), *args, **kwargs)


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
