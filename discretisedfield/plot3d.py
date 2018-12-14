import numpy as np
import matplotlib

def check_k3d_install():
    try:
        import k3d
        return True
    except ImportError:
        msg = '''
        Missing module k3d.
        Please install and activate k3d via pip using:

        $ pip install k3d
        $ jupyter nbextension install --py --sys-prefix k3d
        $ jupyter nbextension enable --py --sys-prefix k3d

        More information on the project Github`s page:
        https://github.com/K3D-tools/K3D-jupyter
        '''
        raise ImportError(msg)


def k3d_points(plot_array, k3d_plot=None, point_size=0.15,
               color=0x99bbff, **kwargs):

    if check_k3d_install():
        import k3d

    plot_array = plot_array.astype(np.float32)  # to avoid the warning

    if k3d_plot is None:
        k3d_plot = k3d.plot()
        k3d_plot.display()
    k3d_plot += k3d.points(plot_array, point_size=point_size, color=color,
                           **kwargs)


def k3d_vectors(coordinates, vectors, k3d_plot=None, points=False,
                colormap='viridis'):

    if check_k3d_install():
        import k3d

    # To avoid the warning
    coordinates = coordinates.astype(np.float32)
    vectors = vectors.astype(np.float32)

    cmap = matplotlib.cm.get_cmap(colormap, 256)

    vc = vectors[..., 0]
    vc = np.interp(vc, (vc.min(), vc.max()), (0, 1))
    colors = cmap(vc)
    colors = [int('0x{}'.format(matplotlib.colors.to_hex(rgb)[1:]), 16)
              for rgb in colors]
    colors = list(zip(colors, colors))

    if k3d_plot is None:
        k3d_plot = k3d.plot()
        k3d_plot.display()
    k3d_plot += k3d.vectors(coordinates, vectors, colors=colors)

    if points:
            k3d_points(coordinates + 0.5 * vectors, k3d_plot=k3d_plot)


def k3d_vox(plot_array, mesh, k3d_plot=None,
            colormap=[0x99bbff, 0xff4d4d], **kwargs):

    if check_k3d_install():
        import k3d

    plot_array = plot_array.astype(np.uint8)  # to avoid the warning

    xmin, ymin, zmin = mesh.pmin
    xmax, ymax, zmax = mesh.pmax

    if k3d_plot is None:
        k3d_plot = k3d.plot()
        k3d_plot.display()
    k3d_plot += k3d.voxels(plot_array,
                           color_map=colormap,
                           xmin=xmin, xmax=xmax,
                           ymin=ymin, ymax=ymax,
                           zmin=zmin, zmax=zmax,
                           outlines=False,
                           **kwargs)
