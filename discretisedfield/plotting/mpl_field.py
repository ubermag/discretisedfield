"""Matplotlib-based plotting."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ubermagutil.units as uu

import discretisedfield as df
import discretisedfield.util as dfu
from discretisedfield.plotting.mpl import Mpl, add_colorwheel


class MplField(Mpl):
    """Matplotlib-based plotting methods.

    Before the field can be plotted, it must be sliced with a plane (e.g.
    ``field.plane('z')``). This class should not be accessed directly. Use
    ``field.mpl`` to use the different plotting methods.

    Parameters
    ----------
    field : df.Field

        Field sliced with a plane, e.g. field.plane('x').

    Raises
    ------
    ValueError

        If the field has not been sliced.

    .. seealso::

        py:func:`~discretisedfield.Field.mpl`

    """

    def __init__(self, field):
        if not field.mesh.attributes["isplane"]:
            msg = "The field must be sliced before it can be plotted."
            raise ValueError(msg)

        self.field = field

        self.planeaxis = dfu.raxesdict[field.mesh.attributes["planeaxis"]]
        self.planeaxis_point = {
            dfu.raxesdict[
                self.field.mesh.attributes["planeaxis"]
            ]: self.field.mesh.attributes["point"]
        }
        self.axis1 = self.field.mesh.attributes["axis1"]
        self.axis2 = self.field.mesh.attributes["axis2"]
        # TODO: After refactoring code, maybe n can become
        # part of PlaneMesh.
        self.n = (self.field.mesh.n[self.axis1], self.field.mesh.n[self.axis2])

    def __call__(
        self,
        ax=None,
        figsize=None,
        multiplier=None,
        scalar_kw=None,
        vector_kw=None,
        filename=None,
    ):
        """Plot the field on a plane.

        This is a convenience method used for quick plotting, and it combines
        ``discretisedfield.plotting.Mpl.scalar`` and
        ``discretisedfield.plotting.Mpl.vector`` methods. Depending on the
        dimensionality of the field's value, it automatically determines what
        plot is going to be shown. For a scalar field, only
        ``discretisedfield.plotting.Mpl.scalar`` is used, whereas for a vector
        field, both ``discretisedfield.plotting.Mpl.scalar`` and
        ``discretisedfield.plotting.Mpl.vector`` plots are shown so that vector
        plot visualises the in-plane components of the vector and scalar plot
        encodes the out-of-plane component.

        All the default values can be changed by passing dictionaries to
        ``scalar_kw`` and ``vector_kw``, which are then used in subplots. The
        way parameters of this function are used to create plots can be
        understood with the following code snippet. ``scalar_field`` and
        ``vector_field`` are computed internally (based on the dimension of the
        field).

        .. code-block::

            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)

            if scalar_field is not None:
                scalar_field.mpl.scalar(ax=ax, multiplier=multiplier,
                                        **scalar_kw)
            if vector_field is not None:
                vector_field.mpl.vector(ax=ax, multiplier=multiplier,
                                        **vector_kw)

            if filename is not None:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
            ```

            Therefore, to understand the meaning of the keyword arguments which
            can be passed to this method, please refer to
            ``discretisedfield.plotting.Mpl.scalar`` and
            ``discretisedfield.plotting.Mpl.vector`` documentation. Filtering
            of the scalar component is applied by default (using the norm for
            vector fields, absolute values for scalar fields). To turn of
            filtering add ``{'filter_field': None}`` to ``scalar_kw``.

        Example
        -------
        .. plot:: :context: close-figs

            1. Visualising the field using ``matplotlib``.

            >>> import discretisedfield as df
            ...
            >>> p1 = (0, 0, 0)
            >>> p2 = (100, 100, 100)
            >>> n = (10, 10, 10)
            >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
            >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
            >>> field.plane(z=50, n=(5, 5)).mpl()

        .. seealso::

            :py:func:`~discretisedfield.plotting.Mpl.scalar`
            :py:func:`~discretisedfield.plotting.Mpl.vector`

        """
        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        scalar_kw = {} if scalar_kw is None else scalar_kw.copy()
        vector_kw = {} if vector_kw is None else vector_kw.copy()
        vector_kw.setdefault("use_color", False)
        vector_kw.setdefault("colorbar", False)

        # Set up default scalar and vector fields.
        if self.field.dim == 1:
            scalar_field = self.field
            vector_field = None

        elif self.field.dim == 2:
            scalar_field = None
            vector_field = self.field

        else:
            vector_field = self.field
            scalar_field = getattr(self.field, self.planeaxis)
            scalar_kw.setdefault("colorbar_label", f"{self.planeaxis}-component")

        scalar_kw.setdefault("filter_field", self.field.norm)

        if scalar_field is not None:
            scalar_field.mpl.scalar(ax=ax, multiplier=multiplier, **scalar_kw)
        if vector_field is not None:
            vector_field.mpl.vector(ax=ax, multiplier=multiplier, **vector_kw)

        self._axis_labels(ax, multiplier)

        self._savefig(filename)

    def scalar(
        self,
        ax=None,
        figsize=None,
        multiplier=None,
        filter_field=None,
        colorbar=True,
        colorbar_label="",
        filename=None,
        symmetric_clim=False,
        **kwargs,
    ):
        r"""Plot the scalar field on a plane.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be a scalar field
        (``dim=1``). Otherwise, ``ValueError`` is raised. ``mpl.scalar`` adds
        the plot to ``matplotlib.axes.Axes`` passed via ``ax`` argument. If
        ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. By passing ``filter_field`` the points at which the pixels
        are not coloured can be determined. More precisely, only those
        discretisation cells where ``filter_field != 0`` are plotted. Colorbar
        is shown by default and it can be removed from the plot by passing
        ``colorbar=False``. The label for the colorbar can be defined by
        passing ``colorbar_label`` as a string. If ``symmetric_clim=True`` is
        passed colorbar limits are internally computed to be symmetric around
        zero. This is most useful in combination with a diverging colormap.
        ``clim`` takes precedence over ``symmetric_clim`` if both are
        specified.

        It is often the case that the region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3  (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all mesh points will be divided by
        :math:`1\,\text{nm}` and :math:`\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally. The plot can be saved as a PDF when ``filename``
        is passed.

        This method plots the field using ``matplotlib.pyplot.imshow``
        function, so any keyword arguments accepted by it can be passed (for
        instance, ``cmap`` - colormap, ``clim`` - colorbar limits, etc.).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the field plot is added. Defaults to ``None`` - axes
            are created internally.

        figsize : (2,) tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        filter_field : discretisedfield.Field, optional

            A scalar field used for determining whether certain discretisation
            cells are coloured. More precisely, only those discretisation cells
            where ``filter_field != 0`` are plotted. Defaults to ``None``.

        colorbar : bool, optional

            If ``True``, colorbar is shown and it is hidden when ``False``.
            Defaults to ``True``.

        colorbar_label : str, optional

            Colorbar label. Defaults to ``None``.

        symmetric_clim : bool, optional

            Automatic ``clim`` symmetric around 0. Defaults to False.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the field has not been sliced, its dimension is not 1, or the
            dimension of ``filter_field`` is not 1.

        Example
        -------
        .. plot::
            :context: close-figs

            1. Visualising the scalar field using ``matplotlib``.

            >>> import discretisedfield as df
            ...
            >>> p1 = (0, 0, 0)
            >>> p2 = (100, 100, 100)
            >>> n = (10, 10, 10)
            >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
            >>> field = df.Field(mesh, dim=1, value=2)
            ...
            >>> field.plane('y').mpl.scalar()

        .. seealso:: :py:func:`~discretisedfield.plotting.Mpl.vector`

        """
        if self.field.dim > 1:
            msg = f"Cannot plot {self.field.dim=} field."
            raise ValueError(msg)

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)
        extent = self._extent(multiplier)

        values = self.field.array.copy().reshape(self.n)
        self._filter_values(filter_field, values)

        if symmetric_clim and "clim" not in kwargs.keys():
            vmin = np.min(values, where=~np.isnan(values), initial=0)
            vmax = np.max(values, where=~np.isnan(values), initial=0)
            vmax_abs = max(abs(vmin), abs(vmax))
            kwargs["clim"] = (-vmax_abs, vmax_abs)

        cp = ax.imshow(np.transpose(values), origin="lower", extent=extent, **kwargs)

        if colorbar:
            cbar = plt.colorbar(cp, ax=ax)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        self._axis_labels(ax, multiplier)

        self._savefig(filename)

    def lightness(
        self,
        ax=None,
        figsize=None,
        multiplier=None,
        filter_field=None,
        lightness_field=None,
        clim=None,
        colorwheel=True,
        colorwheel_xlabel=None,
        colorwheel_ylabel=None,
        colorwheel_args=None,
        filename=None,
        **kwargs,
    ):
        """Lightness plots.

        Uses HSV to show in-plane angle and lightness for out-of-plane (3d) or
        norm (1d/2d) of the field. By passing a scalar field as
        ``lightness_field``, lightness component is can be specified
        independently of the field dimension. Most parameters are the same as
        for ``discretisedfield.plotting.Mpl.scalar``. Colormap cannot be passed
        using ``kwargs``. Instead of having a colorbar a ``colorwheel`` is
        displayed by default.

        lightness_field : discretisedfield.Field, optional

            A scalar field used for adding lightness to the color. Field values
            are hue. Defaults to ``None``.

        colorwheel : bool, optional

            To control if a colorwheel is shown, defaults to ``True``.

        colorwheel_xlabel : str, optional

            If specified, the string and an arrow are plotted onto the
            colorwheel (in x-direction).

        colorwheel_ylabel : str, optional

            If specified, the string and an arrow are plotted onto the
            colorwheel (in y-direction).

        colorwheel_args : dict, optional

            Additional keyword arguments to pass to the colorwheel function.
            For details see ``discretisedfield.plotting.Mpl.colorwheel`` and
            ``mpl_toolkits.axes_grid1.inset_locator.inset_axes``.

        Example
        -------
        .. plot::
            :context: close-figs

            1. Visualising the scalar field using ``matplotlib``.

            >>> import discretisedfield as df
            ...
            >>> p1 = (0, 0, 0)
            >>> p2 = (100, 100, 100)
            >>> n = (10, 10, 10)
            >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
            >>> field = df.Field(mesh, dim=3, value=(1, 2, 3))
            ...
            >>> field.plane('z').mpl.lightness()

        """
        if self.field.dim == 2:
            if lightness_field is None:
                lightness_field = self.field.norm
            if filter_field is None:
                filter_field = self.field.norm
            return self.field.angle.mpl.lightness(
                ax=ax,
                figsize=figsize,
                multiplier=multiplier,
                filter_field=filter_field,
                lightness_field=lightness_field,
                clim=clim,
                colorwheel=colorwheel,
                colorwheel_xlabel=colorwheel_xlabel,
                colorwheel_ylabel=colorwheel_ylabel,
                colorwheel_args=colorwheel_args,
                filename=filename,
                **kwargs,
            )
        elif self.field.dim == 3:
            if lightness_field is None:
                lightness_field = getattr(self.field, self.planeaxis)
            if filter_field is None:
                filter_field = self.field.norm
            return self.field.angle.mpl.lightness(
                ax=ax,
                figsize=figsize,
                multiplier=multiplier,
                filter_field=filter_field,
                lightness_field=lightness_field,
                clim=clim,
                colorwheel=colorwheel,
                colorwheel_xlabel=colorwheel_xlabel,
                colorwheel_ylabel=colorwheel_ylabel,
                colorwheel_args=colorwheel_args,
                filename=filename,
                **kwargs,
            )

        ax = self._setup_axes(ax, figsize)

        if filter_field is None:
            filter_field = self.field.norm

        multiplier = self._setup_multiplier(multiplier)
        extent = self._extent(multiplier)

        if lightness_field is None:
            lightness_field = self.field.norm
        else:
            if lightness_field.dim != 1:
                msg = f"Cannot use {lightness_field.dim=} lightness_field."
                raise ValueError(msg)

        values = self.field.array.copy().reshape(self.n)

        lightness_plane = lightness_field.plane(**self.planeaxis_point)
        if lightness_plane.mesh != self.field.mesh:
            lightness_plane = df.Field(self.field.mesh, dim=1, value=lightness_plane)
        lightness = lightness_plane.array.reshape(self.n)

        rgb = dfu.hls2rgb(
            hue=values, lightness=lightness, saturation=None, lightness_clim=clim
        ).squeeze()
        self._filter_values(filter_field, rgb)

        # alpha channel to hide points with nan values (filter field)
        # all three rgb values are set to nan
        rgba = np.empty((*rgb.shape[:-1], 4))
        rgba[..., :3] = rgb
        rgba[..., 3] = 1.0
        rgba[..., 3][np.isnan(rgb[..., 0])] = 0

        kwargs["cmap"] = "hsv"  # only hsv cmap allowed
        ax.imshow(
            np.transpose(rgba, (1, 0, 2)), origin="lower", extent=extent, **kwargs
        )

        if colorwheel:
            if colorwheel_args is None:
                colorwheel_args = {}
            cw_ax = add_colorwheel(ax, **colorwheel_args)
            if colorwheel_xlabel is not None:
                cw_ax.arrow(100, 100, 60, 0, width=5, fc="w", ec="w")
                cw_ax.annotate(colorwheel_xlabel, (115, 140), c="w")
            if colorwheel_ylabel is not None:
                cw_ax.arrow(100, 100, 0, -60, width=5, fc="w", ec="w")
                cw_ax.annotate(colorwheel_ylabel, (40, 80), c="w")

        self._axis_labels(ax, multiplier)

        self._savefig(filename)

    def vector(
        self,
        ax=None,
        figsize=None,
        multiplier=None,
        use_color=True,
        color_field=None,
        colorbar=True,
        colorbar_label="",
        filename=None,
        **kwargs,
    ):
        r"""Plot the vector field on a plane.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be a vector field
        (``dim=2`` or ``dim=3``). Otherwise, ``ValueError`` is raised.
        ``mpl.vector`` adds the plot to ``matplotlib.axes.axes`` passed via
        ``ax`` argument. If ``ax`` is not passed, ``matplotlib.axes.axes``
        object is created automatically and the size of a figure can be
        specified using ``figsize``. By default, plotted vectors are coloured
        according to the out-of-plane component of the vectors if the field has
        ``dim=3``. This can be changed by passing ``color_field`` with
        ``dim=1``. To disable colouring of the plot, ``use_color=False`` can be
        passed. A uniform vector colour can be obtained by specifying
        ``use_color=false`` and ``color=color`` which is passed to matplotlib.
        Colorbar is shown by default and it can be removed from the plot by
        passing ``colorbar=False``. The label for the colorbar can be defined
        by passing ``colorbar_label`` as a string.

        It is often the case that the region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3  (..., -6, -3, 0, 3, 6,...). according to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all mesh points will be divided by
        :math:`1\,\text{nm}` and :math:`\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally. The plot can be saved as a pdf when ``filename``
        is passed.

        This method plots the field using ``matplotlib.pyplot.quiver``
        function, so any keyword arguments accepted by it can be passed (for
        instance, ``cmap`` - colormap, ``clim`` - colorbar limits, etc.). In
        particular, there are cases when ``matplotlib`` fails to find optimal
        scale for plotting vectors. More precisely, sometimes vectors appear
        too large in the plot. This can be resolved by passing ``scale``
        argument, which scales all vectors in the plot. In other words, larger
        ``scale``, smaller the vectors and vice versa. Please note that scale
        can be in a very large range (e.g. 1e20).

        Parameters
        ----------
        ax : matplotlib.axes.axes, optional

            Axes to which the field plot is added. defaults to ``None`` - axes
            are created internally.

        figsize : tuple, optional

            The size of a created figure if ``ax`` is not passed. defaults to
            ``None``.

        color_field : discretisedfield.field, optional

            A scalar field used for colouring the vectors. defaults to ``None``
            and vectors are coloured according to their out-of-plane
            components.

        colorbar : bool, optional

            If ``true``, colorbar is shown and it is hidden when ``false``.
            defaults to ``true``.

        colorbar_label : str, optional

            Colorbar label. Defaults to ``None``.

        multiplier : numbers.real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\,\text{nm}` and :math:`\text{nm}` units will
            be used as axis labels. defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. defaults to ``None``.

        Raises
        ------
        ValueError

            If the field has not been sliced, its dimension is not 3, or the
            dimension of ``color_field`` is not 1.

        Example
        -------
        .. plot::
            :context: close-figs

            1. Visualising the vector field using ``matplotlib``.

            >>> import discretisedfield as df
            ...
            >>> p1 = (0, 0, 0)
            >>> p2 = (100, 100, 100)
            >>> n = (10, 10, 10)
            >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
            >>> field = df.Field(mesh, dim=3, value=(1.1, 2.1, 3.1))
            ...
            >>> field.plane('y').mpl.vector()

        .. seealso:: :py:func:`~discretisedfield.field.mpl_scalar`

        """
        if self.field.dim not in [2, 3]:
            msg = f"cannot plot dim={self.field.dim} field."
            raise ValueError(msg)

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points1 = self.field.mesh.midpoints[self.axis1] / multiplier
        points2 = self.field.mesh.midpoints[self.axis2] / multiplier

        values = self.field.array.copy().reshape(self.n + (self.field.dim,))
        self._filter_values(self.field.norm, values)

        quiver_args = [
            points1,
            points2,
            np.transpose(values[..., self.axis1]),
            np.transpose(values[..., self.axis2]),
        ]

        if use_color:
            if color_field is None:
                if self.field.dim == 2:
                    warnings.warn(
                        "Automatic coloring is only supported for 3d"
                        f' fields. Ignoring "{use_color=}".'
                    )
                    use_color = False
                else:
                    color_field = getattr(self.field, self.planeaxis)
            if use_color:
                color_plane = color_field.plane(**self.planeaxis_point)
                if color_plane.mesh != self.field.mesh:
                    color_plane = df.Field(self.field.mesh, dim=1, value=color_plane)
                quiver_args.append(color_plane.array.reshape(self.n).transpose())

        cp = ax.quiver(*quiver_args, pivot="mid", **kwargs)

        ax.set_aspect("equal")
        if colorbar and use_color:
            cbar = plt.colorbar(cp, ax=ax)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        self._axis_labels(ax, multiplier)

        self._savefig(filename)

    def contour(
        self,
        ax=None,
        figsize=None,
        multiplier=None,
        filter_field=None,
        colorbar=True,
        colorbar_label=None,
        filename=None,
        **kwargs,
    ):
        r"""Contour line plot.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be a scalar field
        (``dim=1``). Otherwise, ``ValueError`` is raised. ``mpl.contour`` adds
        the plot to ``matplotlib.axes.Axes`` passed via ``ax`` argument. If
        ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. By passing ``filter_field`` the points at which the pixels
        are not coloured can be determined. More precisely, only those
        discretisation cells where ``filter_field != 0`` are plotted. Colorbar
        is shown by default and it can be removed from the plot by passing
        ``colorbar=False``. The label for the colorbar can be defined by
        passing ``colorbar_label`` as a string.

        It is often the case that the region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3  (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all mesh points will be divided by
        :math:`1,\text{nm}` and :math:`\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally. The plot can be saved as a PDF when ``filename``
        is passed.

        This method plots the field using ``matplotlib.pyplot.contour``
        function, so any keyword arguments accepted by it can be passed (for
        instance, ``levels`` - number of levels, etc.).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the field plot is added. Defaults to ``None`` - axes
            are created internally.

        figsize : (2,) tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        filter_field : discretisedfield.Field, optional

            A scalar field used for determining whether certain discretisation
            cells are coloured. More precisely, only those discretisation cells
            where ``filter_field != 0`` are plotted. Defaults to ``None``.

        colorbar : bool, optional

            If ``True``, colorbar is shown and it is hidden when ``False``.
            Defaults to ``True``.

        colorbar_label : str, optional

            Colorbar label. Defaults to ``None``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\,\text{nm}` and :math:`\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the field has not been sliced, its dimension is not 1, or the
            dimension of ``filter_field`` is not 1.

        Example
        -------
        .. plot::
            :context: close-figs

            1. Visualising the scalar field using ``matplotlib``.

            >>> import discretisedfield as df
            >>> import math
            ...
            >>> p1 = (0, 0, 0)
            >>> p2 = (100, 100, 100)
            >>> n = (10, 10, 10)
            >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
            >>> def init_value(point):
            ...     x, y, z = point
            ...     return math.sin(x) + math.sin(y)
            >>> field = df.Field(mesh, dim=1, value=init_value)
            >>> field.plane('z').mpl.contour()

        """
        if self.field.dim != 1:
            msg = f"Cannot plot dim={self.field.dim} field."
            raise ValueError(msg)

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points1 = self.field.mesh.midpoints[self.axis1] / multiplier
        points2 = self.field.mesh.midpoints[self.axis2] / multiplier

        values = self.field.array.copy().reshape(self.n)
        self._filter_values(filter_field, values)

        cp = ax.contour(points1, points2, np.transpose(values), **kwargs)
        ax.set_aspect("equal")

        if colorbar:
            cbar = plt.colorbar(cp, ax=ax)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        self._axis_labels(ax, multiplier)

        self._savefig(filename)

    def _setup_multiplier(self, multiplier):
        return self.field.mesh.region.multiplier if multiplier is None else multiplier

    def _filter_values(self, filter_field, values):
        if filter_field is None:
            return values

        if filter_field.dim != 1:
            msg = f"Cannot use {filter_field.dim=}."
            raise ValueError(msg)

        filter_plane = filter_field.plane(**self.planeaxis_point)
        if filter_plane.mesh != self.field.mesh:
            filter_plane = df.Field(self.field.mesh, dim=1, value=filter_plane)

        values[filter_plane.array.reshape(self.n) == 0] = np.nan

    def _axis_labels(self, ax, multiplier):
        unit = (
            rf" ({uu.rsi_prefixes[multiplier]}"
            rf'{self.field.mesh.attributes["unit"]})'
        )
        ax.set_xlabel(dfu.raxesdict[self.axis1] + unit)
        ax.set_ylabel(dfu.raxesdict[self.axis2] + unit)

    def _extent(self, multiplier):
        # TODO Requires refactoring of df.Mesh
        # rescaled_region = self.field.mesh.region.rescale(multiplier)
        # pmin = rescaled_region.pmin
        # pmax = rescaled_region.pmax
        # TODO: After refactoring code, maybe extent can become
        # part of PlaneMesh.
        pmin = np.divide(self.field.mesh.region.pmin, multiplier)
        pmax = np.divide(self.field.mesh.region.pmax, multiplier)

        return [pmin[self.axis1], pmax[self.axis1], pmin[self.axis2], pmax[self.axis2]]

    def __dir__(self):
        dirlist = dir(self.__class__)

        for attr in ["vector"] if self.field.dim == 1 else ["scalar", "contour"]:
            dirlist.remove(attr)

        return dirlist
