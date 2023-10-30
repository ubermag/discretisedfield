import pyvista as pv
import ubermagutil.units as uu

import discretisedfield.plotting.util as plot_util


class PyVistaRegion:
    def __init__(self, region):
        if region.ndim != 3:
            raise RuntimeError("Only 3d regions can be plotted.")
        self.region = region

    def __call__(
        self,
        *,
        plotter=None,
        color=plot_util.cp_hex[0],
        multiplier=None,
        filename=None,
        **kwargs,
    ):
        """``pyvista`` plot.

        If ``plot`` is not passed, a new `pyvista` plotter object is created
        automatically. The colour of the region can be specified using
        ``color`` argument.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        Parameters
        ----------
        plot : pyvista.Plotter, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally.

        color : tuple, optional

            Colour of the region. Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        Examples
        --------
        1. Visualising the region using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.pyvista()

        """
        if self.region.ndim != 3:
            raise RuntimeError("Only 3-dimensional regions can be plotted.")

        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        multiplier = self._setup_multiplier(multiplier)

        rescaled_region = self.region.scale(1 / multiplier, reference_point=(0, 0, 0))

        bounds = tuple(
            val
            for pair in zip(rescaled_region.pmin, rescaled_region.pmax)
            for val in pair
        )

        # Create a box (cube) mesh using pyvista
        box = pv.Box(bounds)

        # Add the box to the plotter
        plot.add_mesh(box, color=color, **kwargs)
        # plot.show_bounds(axes_ranges=bounds)
        label = self._axis_labels(multiplier)
        plot.show_grid(xtitle=label[0], ytitle=label[1], ztitle=label[2])

        if plotter is None:
            plot.show()

        if filename is not None:
            plot.screenshot(filename=filename)

    def _setup_multiplier(self, multiplier):
        return self.region.multiplier if multiplier is None else multiplier

    def _axis_labels(self, multiplier):
        return [
            rf"{dim} ({uu.rsi_prefixes[multiplier]}{unit})"
            for dim, unit in zip(self.region.dims, self.region.units)
        ]
