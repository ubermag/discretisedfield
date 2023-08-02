import k3d
import numpy as np
import ubermagutil.units as uu

import discretisedfield.plotting.util as plot_util


class K3dRegion:
    def __init__(self, region):
        if region.ndim != 3:
            raise RuntimeError("Only 3d regions can be plotted.")
        self.region = region

    def __call__(
        self, *, plot=None, color=plot_util.cp_int[0], multiplier=None, **kwargs
    ):
        """``k3d`` plot.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The colour of the region can be specified using
        ``color`` argument.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally.

        color : int, optional

            Colour of the region. Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        Examples
        --------
        1. Visualising the region using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.k3d()
        Plot(...)

        """
        if self.region.ndim != 3:
            raise RuntimeError("Only 3-dimensional regions can be plotted.")

        if plot is None:
            plot = k3d.plot()
            plot.display()

        multiplier = self._setup_multiplier(multiplier)

        plot_array = np.ones((1, 1, 1)).astype(np.uint8)  # avoid k3d warning

        rescaled_region = self.region.scale(1 / multiplier)
        bounds = [
            i
            for sublist in zip(rescaled_region.pmin, rescaled_region.pmax)
            for i in sublist
        ]

        plot += k3d.voxels(
            plot_array, color_map=color, bounds=bounds, outlines=False, **kwargs
        )

        self._axis_labels(plot, multiplier)

    def _setup_multiplier(self, multiplier):
        return self.region.multiplier if multiplier is None else multiplier

    def _axis_labels(self, plot, multiplier):
        plot.axes = [
            rf"{dim}\,(\text{{{uu.rsi_prefixes[multiplier]}{unit}}})"
            for dim, unit in zip(self.region.dims, self.region.units)
        ]
