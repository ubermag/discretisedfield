import cmath
import numbers
import colorsys
import collections
import functools
import numpy as np

axesdict = collections.OrderedDict(x=0, y=1, z=2)
raxesdict = {value: key for key, value in axesdict.items()}

# Color pallete as hex and int.
cp_hex = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
          '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']
cp_int = [int(color[1:], 16) for color in cp_hex]


def array2tuple(array):
    return array.item() if array.size == 1 else tuple(array.tolist())


@functools.singledispatch
def as_array(val, mesh, dim, dtype):
    raise TypeError('Unsupported type {type(val)}.')


# to avoid str being interpreted as iterable
@as_array.register(str)
def _(val, mesh, dim, dtype):
    raise TypeError('Unsupported type {type(val)}.')


@as_array.register(numbers.Complex)
@as_array.register(collections.abc.Iterable)
def _(val, mesh, dim, dtype):
    if isinstance(val, numbers.Complex) and dim > 1 and val != 0:
        raise ValueError('Wrong dimension 1 provided for value;'
                         f' expected dimension is {dim}')
    if dtype is None:
        dtype = max(np.asarray(val).dtype, np.float64)
    return np.full((*mesh.n, dim), val, dtype=dtype)


@as_array.register(collections.abc.Callable)
def _(val, mesh, dim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    array = np.empty((*mesh.n, dim), dtype=dtype)
    for index, point in zip(mesh.indices, mesh):
        res = val(point)
        array[index] = res
    return array


@as_array.register(dict)
def _(val, mesh, dim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    if dtype is None:
        dtype = np.float64
    if 'default' in val and not callable(val['default']):
        fill_value = val['default']
    else:
        fill_value = np.nan
    array = np.full((*mesh.n, dim), fill_value, dtype=dtype)

    for subregion in reversed(mesh.subregions.keys()):
        # subregions can overlap, first subregion takes precedence
        try:
            submesh = mesh[subregion]
            subval = val[subregion]
        except KeyError:
            continue
        else:
            slices = mesh.region2slices(submesh.region)
            array[slices] = as_array(subval, submesh, dim, dtype)

    if np.any(np.isnan(array)):
        # not all subregion keys specified and 'default' is missing or callable
        if 'default' not in val:
            raise KeyError("Key 'default' required if not all subregion keys"
                           " are specified.")
        subval = val['default']
        for ix, iy, iz in np.argwhere(np.isnan(array[..., 0])):
            # only spatial indices required -> array[..., 0]
            array[ix, iy, iz] = subval(mesh.index2point((ix, iy, iz)))

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


def hls2rgb(hue, lightness=None, saturation=None, lightness_clim=None):
    hue = normalise_to_range(hue, (0, 1), int_round=False)
    if lightness is not None:
        if lightness_clim is None:
            lightness_clim = (0, 1)
        lightness = normalise_to_range(lightness, lightness_clim,
                                       int_round=False)
    else:
        lightness = np.ones_like(hue)
    if saturation is not None:
        saturation = normalise_to_range(saturation, (0, 1), int_round=False)
    else:
        saturation = np.ones_like(hue)

    rgb = np.apply_along_axis(lambda x: colorsys.hls_to_rgb(*x),
                              -1,
                              np.dstack((hue, lightness, saturation)))

    return rgb.squeeze()
