import numpy as np
import itertools as it


def as_array(mesh, dim, val):
    val_array = np.empty(mesh.n + (dim,))
    if isinstance(val, (int, float)) and (dim == 1 or val == 0):
        val_array.fill(val)
    elif isinstance(val, (tuple, list, np.ndarray)) and len(val) == dim:
        val_array[..., :] = val
    elif isinstance(val, np.ndarray) and val.shape == val_array.shape:
        val_array = val
    elif callable(val):
        for index, point in zip(mesh.indices, mesh.coordinates):
            val_array[index] = val(point)
    else:
        raise TypeError("Unsupported type(val)={}.".format(type(val)))
    return val_array


def plot_line(ax, p1, p2, *args, **kwargs):
    """
    Plot a line between points p1 and p2 on axis ax.

    """
    ax.plot(*zip(p1, p2), *args, **kwargs)
    return ax


def plot_box(ax, p1, p2, *args, **kwargs):
    """
    Plots a cube that spans between p1 and p2 on the given axis.

    Args:
      ax (matplolib axis): matplolib axis object
      p1 (tuple, list, np.ndarray): First cube point
        p1 is of length 3 (xmin, ymin, zmax).
      p2 (tuple, list, np.ndarray): Second cube point
        p2 is of length 3 (xmin, ymin, zmax).
      color (str): matplotlib color string
      linewidth (Real): matplotlib linewidth parameter

    """
    # for a, b in it.combinations(it.product(*zip(p1, p2)), 2):
    #    plot_line(ax, a, b, *args, **kwargs)

    x1, y1, z1 = p1
    x2, y2, z2 = p2

    # Plot individual lines (edges) of a cube.
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

    return ax
