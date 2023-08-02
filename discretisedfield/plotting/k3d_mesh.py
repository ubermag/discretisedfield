import k3d
import numpy as np
import ubermagutil.units as uu

import discretisedfield.plotting.util as plot_util


class K3dMesh:
    def __init__(self, mesh):
        if mesh.region.ndim != 3:
            raise RuntimeError("Only 3d meshes can be plotted.")
        self.mesh = mesh

    def __call__(
        self, *, plot=None, color=plot_util.cp_int[:2], multiplier=None, **kwargs
    ):
        """``k3d`` plot.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The color of the region and the discretisation cell can
        be specified using ``color`` length-2 tuple, where the first element is
        the colour of the region and the second element is the colour of the
        discretisation cell.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally.

        color : (2,) array_like

            Colour of the region and the discretisation cell. Defaults to the
            default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        Examples
        --------
        1. Visualising the mesh using ``k3d``.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> mesh.k3d()
        Plot(...)

        """
        if self.mesh.region.ndim != 3:
            raise ValueError(
                "Only meshes with 3 spatial dimensions can be plotted not"
                f" {self.data.mesh.region.ndim=}."
            )

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        plot_array = np.ones(tuple(reversed(self.mesh.n))).astype(np.uint8)
        plot_array[0, 0, -1] = 2  # mark the discretisation cell

        bounds = [
            i
            for sublist in zip(
                np.divide(self.mesh.region.pmin, multiplier),
                np.divide(self.mesh.region.pmax, multiplier),
            )
            for i in sublist
        ]

        plot += k3d.voxels(
            plot_array, color_map=color, bounds=bounds, outlines=False, **kwargs
        )

        plot.axes = [
            rf"{dim}\,(\text{{{uu.rsi_prefixes[multiplier]}{unit}}})"
            for dim, unit in zip(self.mesh.region.dims, self.mesh.region.units)
        ]

    def subregions(
        self, *, plot=None, color=plot_util.cp_int, multiplier=None, **kwargs
    ):
        """``k3d`` subregions plot.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The color of the subregions can be specified using
        ``color``.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally.

        color : array_like

            Colour of the subregions. Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        Examples
        --------
        1. Visualising subregions using ``k3d``.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
        ...
        >>> mesh.k3d.subregions()
        Plot(...)

        """
        if self.mesh.region.ndim != 3:
            raise ValueError(
                "Only meshes with 3 spatial dimensions can be plotted not"
                f" {self.data.mesh.region.ndim=}."
            )

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        plot_array = np.zeros(self.mesh.n)
        for index in self.mesh.indices:
            # colour all voxels in the same subregion with the same colour
            # to make it easier to identify subregions
            for i, subregion in enumerate(self.mesh.subregions.values()):
                if self.mesh.index2point(index) in subregion:
                    # +1 to avoid 0 value - invisible voxel
                    plot_array[index] = (i % len(color)) + 1
                    break
        # swap axes for k3d.voxels and astypr to avoid k3d warning
        plot_array = np.swapaxes(plot_array, 0, 2).astype(np.uint8)

        bounds = [
            i
            for sublist in zip(
                np.divide(self.mesh.region.pmin, multiplier),
                np.divide(self.mesh.region.pmax, multiplier),
            )
            for i in sublist
        ]

        plot += k3d.voxels(
            plot_array, color_map=color, bounds=bounds, outlines=False, **kwargs
        )

        plot.axes = [
            rf"{dim}\,(\text{{{uu.rsi_prefixes[multiplier]}{unit}}})"
            for dim, unit in zip(self.mesh.region.dims, self.mesh.region.units)
        ]
