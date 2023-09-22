import collections
import colorsys

import numpy as np

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
hv_key_dim = collections.namedtuple("hv_key_dim", ["data", "unit"])


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


def inplane_angle(field, x, y):
    """Compute the angle to the x axis of the in-plane part of the vector field.

    The in-plane part is defined with two vdims x and y.
    """
    if field.nvdim == 1:
        raise ValueError("This method can only be used for vector fields.")
    if x is None and y is None:
        raise ValueError("At least one of x and y most not be None.")
    if x is not None and x not in field.vdims:
        raise ValueError(f"{x} component is not part of {field.vdims=}")
    if y is not None and y not in field.vdims:
        raise ValueError(f"{y} component is not part of {field.vdims=}")

    # TODO should we restrict the calculation to field.valid?
    angle_array = np.arctan2(
        getattr(field, y).array if x is not None else 0,
        getattr(field, x).array if y is not None else 0,
    )
    angle_array[angle_array < 0] += 2 * np.pi
    return field.__class__(
        field.mesh, nvdim=1, value=angle_array, unit="rad", valid=field.valid
    )


def normalise_to_range(values, to_range, from_range=None, int_round=True):
    """Normalise values.

    If from_range is not specified, min and max of values are mapped to min and max of
    to_range otherwise min and max of from_range are mapped to min and max of to_range.

    """
    values = np.asarray(values)

    values -= from_range[0] if from_range else values.min()  # min value is 0
    # For uniform fields, avoid division by zero.
    if from_range or values.max() != 0:
        values /= (
            (from_range[1] - from_range[0]) if from_range else values.max()
        )  # all values in (0, 1)
    values *= to_range[1] - to_range[0]  # all values in (0, r[1]-r[0])
    values += to_range[0]  # all values is range (r[0], r[1])
    if int_round:
        values = values.round()
        values = values.astype(int)

    return values


def hls2rgb(hue, lightness=None, saturation=None, lightness_clim=None):
    """Convert hsl to rgb."""
    hue = normalise_to_range(hue, (0, 1), (0, 2 * np.pi), int_round=False)
    if lightness is not None:
        if lightness_clim is None:
            lightness_clim = (0, 1)
        lightness = normalise_to_range(lightness, lightness_clim, int_round=False)
    else:
        lightness = np.ones_like(hue)
    if saturation is not None:
        saturation = normalise_to_range(saturation, (0, 1), int_round=False)
    else:
        saturation = np.ones_like(hue)

    rgb = np.apply_along_axis(
        lambda x: colorsys.hls_to_rgb(*x), -1, np.dstack((hue, lightness, saturation))
    )

    return rgb.squeeze()
