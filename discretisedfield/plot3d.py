import k3d
import numpy as np
import matplotlib


def voxels(plot_array, pmin, pmax, colormap, outlines=False,
           plot=None, **kwargs):
    plot_array = plot_array.astype(np.uint8)  # to avoid k3d warning
    xmin, ymin, zmin = pmin
    xmax, ymax, zmax = pmax

    if plot is None:
        plot = k3d.plot()
        plot.display()
    plot += k3d.voxels(plot_array,
                       color_map=colormap,
                       bounds=[xmin, xmax, ymin, ymax, zmin, zmax],
                       outlines=outlines,
                       **kwargs)

    
def get_colors(vectors, colormap='viridis'):
    """Return list of color for each verctor in 0xaabbcc format."""
    cmap = matplotlib.cm.get_cmap(colormap, 256)

    vc = vectors[..., 0]
    vc = np.interp(vc, (vc.min(), vc.max()), (0, 1))
    colors = cmap(vc)
    colors = [int('0x{}'.format(matplotlib.colors.to_hex(rgb)[1:]), 16)
              for rgb in colors]
    colors = list(zip(colors, colors))

    return colors


def k3d_points(plot_array, k3d_plot=None, point_size=0.15,
               color=0x99bbff, **kwargs):
    if k3d_plot is None:
        k3d_plot = k3d.plot()
        k3d_plot.display()
    k3d_plot += k3d.points(plot_array, point_size=point_size, color=color,
                           **kwargs)


def k3d_vectors(coordinates, vectors, k3d_plot=None, points=False,
                **kwargs):
    colors = get_colors(vectors, **kwargs)

    if k3d_plot is None:
        k3d_plot = k3d.plot()
        k3d_plot.display()
    k3d_plot += k3d.vectors(coordinates, vectors, colors=colors)

    if points:
            k3d_points(coordinates + 0.5 * vectors, k3d_plot=k3d_plot)


def get_int_component2(field_component):
    max_value = np.nanmax(field_component)
    min_value = np.nanmin(field_component)
    value_range = max_value - min_value

    nx, ny, nz = field_component.shape

    # Put values in 0-255 range
    if value_range != 0:
        int_component = (field_component + abs(min_value)) / value_range * 254
        int_component += 1
    else:
        int_component = 128 * np.ones(field_component.shape)  # place in the middle of colormap

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if np.isnan(field_component[i, j, k]):
                    int_component[i, j, k] = int(0)
                else:
                    int_component[i, j, k] = int(int_component[i, j, k])

    return int_component.astype(np.int)


def k3d_scalar(field_component, mesh, k3d_plot=None, colormap='viridis',
               **kwargs):
    int_component = get_int_component2(field_component)
    int_component = int_component.astype(np.uint8)  # to avoid the warning

    cmap = matplotlib.cm.get_cmap(colormap, 256)

    vc = int_component[..., 0]
    vc = np.interp(vc, (vc.min(), vc.max()), (0, 1))
    colors = cmap(vc)
    colors = [int('0x{}'.format(matplotlib.colors.to_hex(rgb)[1:]), 16)
              for rgb in colors.reshape((int(960 / 4), 4))]
    colors = list(zip(colors, colors))

    xmin, ymin, zmin = mesh.pmin
    xmax, ymax, zmax = mesh.pmax

    if k3d_plot is None:
        k3d_plot = k3d.plot()
        k3d_plot.display()

    k3d_plot += k3d.voxels(int_component, color_map=colors,
                           xmin=xmin, xmax=xmax,
                           ymin=ymin, ymax=ymax,
                           zmin=zmin, zmax=zmax,
                           outlines=False,
                           **kwargs)


def k3d_isosurface(field, level, mesh, k3d_plot=None, **kwargs):
    xmin, ymin, zmin = mesh.pmin
    xmax, ymax, zmax = mesh.pmax

    plot_array = np.sum(field**2, axis=-1)
    plot_array = plot_array.astype(np.float32)  # to avoid the warning

    if k3d_plot is None:
        k3d_plot = k3d.plot()
        k3d_plot.display()
    k3d_plot += k3d.marching_cubes(plot_array,
                                   level=level,
                                   xmin=xmin, xmax=xmax,
                                   ymin=ymin, ymax=ymax,
                                   zmin=zmin, zmax=zmax,
                                   **kwargs)
