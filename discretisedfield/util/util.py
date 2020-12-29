import cmath
import numbers
import colorsys
import collections
import numpy as np

axesdict = collections.OrderedDict(x=0, y=1, z=2)
raxesdict = {value: key for key, value in axesdict.items()}

# Color pallete as hex and int.
cp_hex = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
          '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
cp_int = [int(color[1:], 16) for color in cp_hex]


def array2tuple(array):
    if array.size == 1:
        return array.item()
    else:
        return tuple(array.tolist())


def as_array(mesh, dim, val):
    array = np.empty((*mesh.n, dim))
    if isinstance(val, numbers.Real) and (dim == 1 or val == 0):
        # The array for a scalar field with numbers.Real value or any
        # field with zero value.
        array.fill(val)
    elif isinstance(val, (tuple, list, np.ndarray)) and len(val) == dim:
        array[..., :] = val
    elif isinstance(val, np.ndarray) and val.shape == array.shape:
        array = val
    elif callable(val):
        for index, point in zip(mesh.indices, mesh):
            array[index] = val(point)
    elif isinstance(val, dict) and mesh.subregions:
        for index, point in zip(mesh.indices, mesh):
            for region in mesh.subregions.keys():
                if point in mesh.subregions[region]:
                    array[index] = val[region]
                    break
    else:
        msg = f'Unsupported {type(val)} or invalid value dimensions.'
        raise ValueError(msg)
    return array


def bergluescher_angle(v1, v2, v3):
    if np.dot(v1, np.cross(v2, v3)) == 0:
        # If the triple product is zero, then rho=0 and division by zero is
        # encountered. In this case, all three vectors are in-plane and the
        # space angle is zero.
        return 0.0
    else:
        rho = (2 *
               (1 + np.dot(v1, v2)) *
               (1 + np.dot(v2, v3)) *
               (1 + np.dot(v3, v1)))**0.5

        numerator = (1 +
                     np.dot(v1, v2) +
                     np.dot(v2, v3) +
                     np.dot(v3, v1) +
                     1j*(np.dot(v1, np.cross(v2, v3))))

        exp_omega = numerator/rho

        return 2 * cmath.log(exp_omega).imag / (4*np.pi)


def assemble_index(value, n, dictionary):
    index = [value, ] * n
    for key, value in dictionary.items():
        index[key] = value

    return tuple(index)


def vtk_scalar_data(field, name):
    header = [f'SCALARS {name} double',
              'LOOKUP_TABLE default']
    data = [str(value) for point, value in field]

    return header + data


def vtk_vector_data(field, name):
    header = [f'VECTORS {name} double']
    data = ['{} {} {}'.format(*value) for point, value in field]

    return header + data


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
    values = np.array(values)

    values -= values.min()  # min value is 0
    # For uniform fields, avoid division by zero.
    if values.max() != 0:
        values /= values.max()  # all values in (0, 1)
    values *= (value_range[1] - value_range[0])  # all values in (0, r[1]-r[0])
    values += value_range[0]  # all values is range (r[0], r[1])
    if int_round:
        values = values.round()
        values = values.astype(int)

    return values


def hls2rgb(hue, lightness=None, saturation=None):
    hue = normalise_to_range(hue, (0, 1), int_round=False)
    if lightness is not None:
        lightness = normalise_to_range(lightness, (0, 1), int_round=False)
    else:
        lightness = np.ones_like(hue)
    if saturation is not None:
        saturation = normalise_to_range(saturation, (0, 1), int_round=False)
    else:
        saturation = np.ones_like(hue)

    return np.apply_along_axis(lambda x: colorsys.hls_to_rgb(*x),
                               -1,
                               np.dstack((hue, lightness, saturation)))
