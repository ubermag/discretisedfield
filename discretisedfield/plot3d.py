import numpy as np


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
