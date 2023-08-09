import ubermagutil.units as uu

import discretisedfield as df
import discretisedfield.plotting.util as plot_util
from discretisedfield.plotting.mpl import Mpl


class MplMesh(Mpl):
    def __init__(self, mesh):
        if mesh.region.ndim != 3:
            raise RuntimeError("Only 3d meshes can be plotted.")
        self.mesh = mesh

    def __call__(
        self,
        *,
        ax=None,
        figsize=None,
        color=plot_util.cp_hex[:2],
        multiplier=None,
        box_aspect="auto",
        filename=None,
        **kwargs,
    ):
        """``matplotlib`` plot.

        If ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. The color of lines depicting the region and the
        discretisation cell can be specified using ``color`` length-2 tuple,
        where the first element is the colour of the region and the second
        element is the colour of the discretisation cell. The plot is saved in
        PDF-format if ``filename`` is passed.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        This method is based on ``matplotlib.pyplot.plot``, so any keyword
        arguments accepted by it can be passed (for instance, ``linewidth``,
        ``linestyle``, etc.).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the plot is added. Defaults to ``None`` - axes are
            created internally.

        figsize : (2,) tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        color : (2,) array_like

            A valid ``matplotlib`` color for lines depicting the region.
            Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        box_aspect : str, array_like (3), optional

            Set the aspect-ratio of the plot. If set to `'auto'` the aspect
            ratio is determined from the edge lengths of the region on which
            the mesh is defined. To set different aspect ratios a tuple can be
            passed. Defaults to ``'auto'``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Examples
        --------
        1. Visualising the mesh using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, n=(50, 50, 5))
        ...
        >>> mesh.mpl()

        """
        ax = self._setup_axes(ax, figsize, projection="3d")

        multiplier = self._setup_multiplier(multiplier)

        rescaled_mesh = self.mesh.scale(1 / multiplier, reference_point=(0, 0, 0))
        rescaled_mesh.region.units = [
            f"{uu.rsi_prefixes[multiplier]}{unit}" for unit in self.mesh.region.units
        ]
        cell_region = df.Region(
            p1=rescaled_mesh.region.pmin,
            p2=rescaled_mesh.region.pmin + rescaled_mesh.cell,
            units=rescaled_mesh.region.units,
        )
        rescaled_mesh.region.mpl(
            ax=ax, color=color[0], box_aspect=box_aspect, multiplier=1, **kwargs
        )
        cell_region.mpl(ax=ax, color=color[1], box_aspect=None, multiplier=1, **kwargs)

        self._savefig(filename)

    def subregions(
        self,
        *,
        ax=None,
        figsize=None,
        color=plot_util.cp_hex,
        multiplier=None,
        show_region=False,
        box_aspect="auto",
        filename=None,
        **kwargs,
    ):
        """``matplotlib`` subregions plot.

        If ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. The color of lines depicting subregions and can be
        specified using ``color`` list. The plot is saved in PDF-format if
        ``filename`` is passed. The whole region is only shown if
        ``show_region=True``.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        This method is based on ``matplotlib.pyplot.plot``, so any keyword
        arguments accepted by it can be passed (for instance, ``linewidth``,
        ``linestyle``, etc.).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the plot is added. Defaults to ``None`` - axes are
            created internally.

        figsize : (2,) tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        color : array_like

            Subregion colours. Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        show_region : bool, optional

            If ``True`` also plot the whole region. Defaults to ``False``.

        box_aspect : str, array_like (3), optional

            Set the aspect-ratio of the plot. If set to `'auto'` the aspect
            ratio is determined from the edge lengths of the region on which
            the mesh is defined. To set different aspect ratios a tuple can be
            passed. Defaults to ``'auto'``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Examples
        --------
        1. Visualising subregions using ``matplotlib``.

        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
        ...
        >>> mesh.mpl.subregions()

        """
        ax = self._setup_axes(ax, figsize, projection="3d")

        multiplier = self._setup_multiplier(multiplier)

        if box_aspect == "auto":
            ax.set_box_aspect(self.mesh.region.edges)
        elif box_aspect is not None:
            ax.set_box_aspect(box_aspect)

        if show_region:
            self.mesh.region.mpl(
                ax=ax, multiplier=multiplier, color="grey", box_aspect=None
            )

        for i, subregion in enumerate(self.mesh.subregions.values()):
            subregion.mpl(
                ax=ax,
                multiplier=multiplier,
                color=color[i % len(color)],
                box_aspect=None,
                **kwargs,
            )

        self._savefig(filename)

    def _setup_multiplier(self, multiplier):
        return (
            uu.si_max_multiplier(self.mesh.region.edges)
            if multiplier is None
            else multiplier
        )

    def _axis_labels(self, ax, multiplier):
        raise NotImplementedError
