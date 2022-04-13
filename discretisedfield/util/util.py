import cmath
import collections
import colorsys

import numpy as np

import discretisedfield as df

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


def normalise_to_range(values, value_range, int_round=True):
    values = np.asarray(values)

    values -= values.min()  # min value is 0
    # For uniform fields, avoid division by zero.
    if values.max() != 0:
        values /= values.max()  # all values in (0, 1)
    values *= value_range[1] - value_range[0]  # all values in (0, r[1]-r[0])
    values += value_range[0]  # all values is range (r[0], r[1])
    if int_round:
        values = values.round()
        values = values.astype(int)

    return values


def hls2rgb(hue, lightness=None, saturation=None, lightness_clim=None):
    hue = normalise_to_range(hue, (0, 1), int_round=False)
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


def fromvtk_legacy(filename):
    """Read the field from a VTK file (legacy).

    This method reads vtk files written with discretisedfield <= 0.61.0
    in which the data is stored as point data instead of cell data.
    """
    with open(filename, "r") as f:
        content = f.read()
    lines = content.split("\n")

    # Determine the dimension of the field.
    if "VECTORS" in content:
        dim = 3
        data_marker = "VECTORS"
        skip = 0  # after how many lines data starts after marker
    else:
        dim = 1
        data_marker = "SCALARS"
        skip = 1

    # Extract the metadata
    mdatalist = ["X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"]
    n = []
    cell = []
    origin = []
    for i, line in enumerate(lines):
        for mdatum in mdatalist:
            if mdatum in line:
                n.append(int(line.split()[1]))
                coordinates = list(map(float, lines[i + 1].split()))
                origin.append(coordinates[0])
                if len(coordinates) > 1:
                    cell.append(coordinates[1] - coordinates[0])
                else:
                    # If only one cell exists, 1nm cell is used by default.
                    cell.append(1e-9)

    # Create objects from metadata info
    p1 = np.subtract(origin, np.multiply(cell, 0.5))
    p2 = np.add(p1, np.multiply(n, cell))
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
    field = df.Field(mesh, dim=dim)

    # Find where data starts.
    for i, line in enumerate(lines):
        if line.startswith(data_marker):
            start_index = i
            break

    # Extract data.
    for i, line in zip(mesh.indices, lines[start_index + skip + 1 :]):
        if not line[0].isalpha():
            field.array[i] = list(map(float, line.split()))

    return field
