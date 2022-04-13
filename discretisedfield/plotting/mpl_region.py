import ubermagutil.units as uu

import discretisedfield.util as dfu
from discretisedfield.plotting.mpl import Mpl


class MplRegion(Mpl):
    def __init__(self, region):
        self.region = region

    def __call__(
        self,
        *,
        ax=None,
        figsize=None,
        multiplier=None,
        color=dfu.cp_hex[0],
        box_aspect="auto",
        filename=None,
        **kwargs,
    ):
        r"""``matplotlib`` plot.

        If ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. The colour of lines depicting the region can be specified
        using ``color`` argument, which must be a valid ``matplotlib`` color.
        The plot is saved in PDF-format if ``filename`` is passed.

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

        color : int, str, tuple, optional

            A valid ``matplotlib`` color for lines depicting the region.
            Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        box_aspect : str, array_like (3), optional

            Set the aspect-ratio of the plot. If set to `'auto'` the aspect
            ratio is determined from the edge lengths of the region. To set
            different aspect ratios a tuple can be passed. Defaults to
            ``'auto'``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Examples
        --------
        1. Visualising the region using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.mpl()

        """
        ax = self._setup_axes(ax, figsize, projection="3d")

        multiplier = self._setup_multiplier(multiplier)

        kwargs.setdefault("color", color)

        rescaled_region = self.region / multiplier

        if box_aspect == "auto":
            ax.set_box_aspect(rescaled_region.edges)
        elif box_aspect is not None:
            ax.set_box_aspect(box_aspect)

        dfu.plot_box(
            ax=ax, pmin=rescaled_region.pmin, pmax=rescaled_region.pmax, **kwargs
        )

        self._axis_labels(ax, multiplier)

        # Overwrite default plotting parameters.
        ax.set_facecolor("#ffffff")  # white face color
        ax.tick_params(axis="both", which="major", pad=0)  # no pad for ticks

        self._savefig(filename)

    def _setup_multiplier(self, multiplier):
        return self.region.multiplier if multiplier is None else multiplier

    def _axis_labels(self, ax, multiplier):
        unit = rf" ({uu.rsi_prefixes[multiplier]}{self.region.unit})"
        ax.set(xlabel=f"x {unit}", ylabel=f"y {unit}", zlabel=f"z {unit}")
