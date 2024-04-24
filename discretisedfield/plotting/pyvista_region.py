import copy

import pyvista as pv
import ubermagutil.units as uu

import discretisedfield.plotting.util as plot_util


class PyVistaRegion:
    def __init__(self, region):
        if region.ndim != 3:
            raise RuntimeError("Only 3d regions can be plotted.")
        self.region = copy.deepcopy(region)

    def __call__(
        self,
        *,
        plotter=None,
        color=plot_util.cp_hex[0],
        multiplier=None,
        filename=None,
        **kwargs,
    ):
        """Generates a ``pyvista`` plot of a 3-dimensional region.

        This method utilises ``pyvista`` to visualise the region.

        If a ``plotter`` is not supplied, it initialises and uses its
        own ``pyvista.Plotter``.

        Additional keyword arguments are forwarded to the ``pyvista.add_mesh``
        method to allow further customisation.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        color : tuple, optional

            Colour of the region in hexadecimal. Defaults to the default color palette.

        multiplier : numbers.Real, optional

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.

        **kwargs

            Arbitrary keyword arguments that are passed directly to the
            `pyvista.add_mesh` method for additional customisation of the plot.

        Raises
        ------
        RuntimeError
            If the region is not 3-dimensional.

        Examples
        --------
        1. Visualising a region using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.pyvista() # doctest: +SKIP

        """
        if self.region.ndim != 3:
            raise RuntimeError("Only 3-dimensional regions can be plotted.")

        plot = pv.Plotter() if plotter is None else plotter

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
            plot_util._pyvista_save_to_file(filename, plot)

    def _setup_multiplier(self, multiplier):
        return self.region.multiplier if multiplier is None else multiplier

    def _axis_labels(self, multiplier):
        return [
            rf"{dim} ({uu.rsi_prefixes[multiplier]}{unit})"
            for dim, unit in zip(self.region.dims, self.region.units)
        ]
