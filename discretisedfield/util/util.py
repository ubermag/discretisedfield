import k3d
import cmath
import random
import numbers
import collections
import matplotlib
import numpy as np
import itertools as it
import discretisedfield as df
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


axesdict = collections.OrderedDict([('x', 0), ('y', 1), ('z', 2)])
raxesdict = {value: key for key, value in axesdict.items()}
colormap = [0x3498db, 0xe74c3c, 0x27ae60, 0xf1c40f, 0x8e44ad, 0xecf0f1]


def array2tuple(array):
    return tuple(array.tolist())


def compatible(field1, field2):
    if not isinstance(field1, df.Field) or \
       not isinstance(field2, df.Field):
        msg = ('Binary operator can be applied only on '
               'discretisedfield.Field objects.')
        raise TypeError(msg)
    elif field1.mesh != field2.mesh:
        msg = 'Fields must be defined on the same mesh.'
        raise ValueError(msg)
    elif field1.dim != field2.dim:
        msg = 'Fields must have the same dimension.'
        raise ValueError(msg)
    else:
        # A binary operator can be applied between two fields.
        return True


def bergluescher_angle(v1, v2, v3):
    rho = (2 *
           (1 + np.dot(v1, v2)) *
           (1 + np.dot(v2, v3)) *
           (1 + np.dot(v3, v1)))**0.5

    numerator = (1 + \
                 np.dot(v1, v2) + \
                 np.dot(v2, v3) + \
                 np.dot(v3, v1) + \
                 1j*(np.dot(v1, np.cross(v2, v3))))

    exp_omega = numerator/rho
    
    return 2 * cmath.log(exp_omega).imag / (4*np.pi)


def assemble_index(index_dict):
    index = [0, 0, 0]
    for key, value in index_dict.items():
        index[key] = value

    return tuple(index)

    
def plane_info(*args, **kwargs):
    info = dict()
    # The plane is defined with: planeaxis and point. They are
    # extracted from *args and *kwargs.
    if args and not kwargs:
        if len(args) != 1:
            msg = 'Only one arg can be passed.'
            raise ValueError(msg)

        # Only planeaxis is provided via args and the point will be
        # defined later as a centre of the sample.
        planeaxis = args[0]
        point = None
    elif kwargs and not args:
        if len(kwargs.keys()) != 1:
            msg = 'Only one kwarg can be passed.'
            raise ValueError(msg)

        # Both planeaxis and point are provided via kwargs.
        planeaxis = list(kwargs.keys())[0]
        point = list(kwargs.values())[0]
    else:
        msg = 'Either one arg or one kwarg can be passed.'
        raise ValueError(msg)

    if planeaxis not in axesdict.keys():
        msg = f'Plane axis must be one of {axesdict.keys()}.'
        raise ValueError(msg)

    info['planeaxis'] = axesdict[planeaxis]
    info['point'] = point

    # Get indices of in-plane axes.
    axes = tuple(filter(lambda val: val != info['planeaxis'],
                        axesdict.values()))
    info['axis1'], info['axis2'] = axes

    return info


def add_random_colors(colormap, regions):
    """Generate random colours if necessary and add them to colormap
    list."""
    if len(regions) > 6:
        for i in range(len(regions)-6):
            found = False
            while not found:
                color = random.randint(0, 16777215)
                found = True
            colormap.append(color)

    return colormap


def plot_line(ax, p1, p2, *args, **kwargs):
    """Plot a line between points p1 and p2 on axis ax."""
    ax.plot(*zip(p1, p2), *args, **kwargs)


def plot_box(ax, p1, p2, *args, **kwargs):
    """Plots a cube that spans between p1 and p2 on the given axis.

    Args:
      ax (matplolib axis): matplolib axis object
      p1 (tuple, list, np.ndarray): First cube point
        p1 is of length 3 (xmin, ymin, zmax).
      p2 (tuple, list, np.ndarray): Second cube point
        p2 is of length 3 (xmin, ymin, zmax).
      color (str): matplotlib color string
      linewidth (Real): matplotlib linewidth parameter

    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2

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


def as_array(mesh, dim, val):
    array = np.empty(mesh.n + (dim,))
    if isinstance(val, numbers.Real) and (dim == 1 or val == 0):
        # The array for a scalar field with numbers.Real value or any
        # field with zero value.
        array.fill(val)
    elif isinstance(val, (tuple, list, np.ndarray)) and len(val) == dim:
        array[..., :] = val
    elif isinstance(val, np.ndarray) and val.shape == array.shape:
        array = val
    elif callable(val):
        for index, point in zip(mesh.indices, mesh.coordinates):
            array[index] = val(point)
    elif isinstance(val, dict) and mesh.regions:
        for index, point in zip(mesh.indices, mesh.coordinates):
            for region in mesh.regions.keys():
                if point in mesh.regions[region]:
                    array[index] = val[region]
                    break
    else:
        msg = (f'Unsupported type(val)={type(val)} '
               'or invalid value dimensions.')
        raise ValueError(msg)
    return array


def voxels(plot_array, pmin, pmax, colormap, outlines=False,
           plot=None, **kwargs):
    plot_array = plot_array.astype(np.uint8)  # to avoid k3d warning
    if plot is None:
        plot = k3d.plot()
        plot.display()

    xmin, ymin, zmin = pmin
    xmax, ymax, zmax = pmax
    bounds = [xmin, xmax, ymin, ymax, zmin, zmax]

    plot += k3d.voxels(plot_array,
                       color_map=colormap,
                       bounds=bounds,
                       outlines=outlines,
                       **kwargs)

    return plot


def points(plot_array, point_size=0.1, color=0x99bbff, plot=None, **kwargs):
    plot_array = plot_array.astype(np.float32)  # to avoid k3d warning

    if plot is None:
        plot = k3d.plot()
        plot.display()
    plot += k3d.points(plot_array, point_size=point_size, color=color,
                       **kwargs)

    return plot


def vectors(coordinates, vectors, colors=[], plot=None, **kwargs):
    coordinates = coordinates.astype(np.float32)  # to avoid k3d warning
    vectors = vectors.astype(np.float32)  # to avoid k3d warning

    if plot is None:
        plot = k3d.plot()
        plot.display()
    plot += k3d.vectors(coordinates, vectors, colors=colors, **kwargs)

    return plot


def num2hexcolor(n, cmap):
    return int(matplotlib.colors.rgb2hex(cmap(n)[:3])[1:], 16)
