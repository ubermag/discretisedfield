import k3d
import cmath
import numbers
import collections
import numpy as np
import seaborn as sns
import ubermagutil.units as uu
import matplotlib.pyplot as plt

axesdict = collections.OrderedDict(x=0, y=1, z=2)
raxesdict = {value: key for key, value in axesdict.items()}


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


def assemble_index(index_dict):
    index = [0, 0, 0]
    for key, value in index_dict.items():
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


def color_palette(cmap, n, value_type):
    cp = sns.color_palette(palette=cmap, n_colors=n)
    if value_type == 'rgb':
        return cp
    else:
        return list(map(lambda c: int(c[1:], 16), cp.as_hex()))


def normalise_to_range(values, value_range):
    values = np.array(values)

    values -= values.min()  # min value is 0
    # For uniform fields, avoid division by zero.
    if values.max() != 0:
        values /= values.max()  # all values in (0, 1)
    values *= (value_range[1] - value_range[0])  # all values in (0, r[1]-r[0])
    values += value_range[0]  # all values is range (r[0], r[1])
    values = values.round()
    values = values.astype(int)

    return values


def k3d_parameters(plot, multiplier, value):
    if plot is None:
        plot = k3d.plot()
        plot.display()

    if multiplier is None:
        multiplier = uu.si_max_multiplier(value)

    unit = f' ({uu.rsi_prefixes[multiplier]}m)'
    plot.axes = [i + unit for i in 'xyz']

    return plot, multiplier


def k3d_plot_region(plot, region, multiplier):
    if not plot.objects:  # if plot was not displayed (interactive plotting)
        plot += voxels(np.ones((1, 1, 1)),
                       pmin=region.pmin,
                       pmax=region.pmax,
                       color_palette=color_palette('deep', 1, 'int')[0],
                       multiplier=multiplier,
                       opacity=0.025)


def k3d_setup_interactive_plot(plot):
    # Delete all objects except the field region.
    for object in plot.objects[1:]:
        plot -= object

    # Make sure the camera and grid do not move in interactive plots. If
    # the plot is not displayed (plot.outputs==[]), it will be displayed.
    if not plot.outputs:
        plot.display()
        plot.camera_auto_fit = False
        plot.grid_auto_fit = False


def voxels(plot_array, pmin, pmax, color_palette, multiplier=1, outlines=False,
           **kwargs):
    plot_array = plot_array.astype(np.uint8)  # to avoid k3d warning

    xmin, ymin, zmin = np.divide(pmin, multiplier)
    xmax, ymax, zmax = np.divide(pmax, multiplier)
    bounds = [xmin, xmax, ymin, ymax, zmin, zmax]

    return k3d.voxels(plot_array, color_map=color_palette, bounds=bounds,
                      outlines=outlines, **kwargs)


def points(coordinates, color, point_size, multiplier=1, **kwargs):
    coordinates = coordinates.astype(np.float32)  # to avoid k3d warning

    coordinates = np.divide(coordinates, multiplier)

    return k3d.points(coordinates, point_size=point_size,
                      color=color, **kwargs)


def vectors(coordinates, vectors, colors=[], multiplier=1, vector_multiplier=1,
            **kwargs):
    coordinates = coordinates.astype(np.float32)  # to avoid k3d warning
    vectors = vectors.astype(np.float32)  # to avoid k3d warning

    coordinates = np.divide(coordinates, multiplier)
    vectors = np.divide(vectors, vector_multiplier)

    # Plot middle of the arrow is at the cell centre.
    coordinates = coordinates - 0.5*vectors

    return k3d.vectors(coordinates, vectors, colors=colors, **kwargs)
