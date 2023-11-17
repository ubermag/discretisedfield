"""Matplotlib-based plotting."""
import warnings

import matplotlib.pyplot as plt
import numpy as np
import ubermagutil.units as uu
from mpl_toolkits.axes_grid1 import Size, make_axes_locatable

import discretisedfield.plotting.util as plot_util
from discretisedfield.plotting.mpl import Mpl, add_colorwheel


class MplField(Mpl):
    """Matplotlib-based plotting methods.

    Before the field can be plotted, it must be ensured that it is defined on two
    dimensional geometry. This class should not be accessed directly. Use
    ``field.mpl`` to use the different plotting methods.

    Parameters
    ----------
    field : df.Field

        Field defined on a two-dimensional plane.

    Raises
    ------
    ValueError

        If the field has not a two-dimensional plane.

    .. seealso::

        py:func:`~discretisedfield.Field.mpl`

    """

    def __init__(self, field):
        if field.mesh.region.ndim != 2:
            raise RuntimeError(
                "Only fields on 2d meshes can be plotted with matplotlib, not"
                f" {field.mesh.region.ndim=}."
            )

        self.field = field

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

        Therefore, to understand the meaning of the keyword arguments which can be
        passed to this method, please refer to ``discretisedfield.plotting.Mpl.scalar``
        and ``discretisedfield.plotting.Mpl.vector`` documentation. Filtering of the
        scalar component is applied by default (using the norm for vector fields,
        absolute values for scalar fields). To turn of filtering add ``{'filter_field':
        None}`` to ``scalar_kw``.

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
            >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 0))
            >>> field.sel(z=50).resample(n=(5, 5)).mpl()

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
        if self.field.nvdim == 1:
            scalar_field = self.field
            vector_field = None

        elif self.field.nvdim == 2:
            scalar_field = None
            vector_field = self.field

        elif self.field.nvdim == 3:
            vector_field = self.field
            # find vector components pointing along the two axes 0 and 1
            vdims = [
                self.field._r_dim_mapping[self.field.mesh.region.dims[0]],
                self.field._r_dim_mapping[self.field.mesh.region.dims[1]],
            ]
            # find the third vector component for the scalar plot
            scalar_vdim = (set(self.field.vdims) - set(vdims)).pop()
            scalar_field = getattr(self.field, scalar_vdim)
            scalar_kw.setdefault(
                "colorbar_label",
                f"{scalar_vdim}-component",
            )
        else:
            raise RuntimeError(
                "The `mpl()` function cannot determine unique "
                f"directions to plot for {self.field.nvdim=}."
            )

        scalar_kw.setdefault("filter_field", self.field._valid_as_field)

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
        ``field.sel('z')``, assuming the geometry has three dimensions). In addition,
        field must be a scalar field (``nvdim=1``). Otherwise, ``ValueError`` is raised.
        ``mpl.scalar`` adds the plot to ``matplotlib.axes.Axes`` passed via ``ax``
        argument. If ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
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
            >>> field = df.Field(mesh, nvdim=1, value=2)
            ...
            >>> field.sel('y').mpl.scalar()

        .. seealso:: :py:func:`~discretisedfield.plotting.Mpl.vector`

        """
        if self.field.nvdim > 1:
            raise RuntimeError(f"Cannot plot {self.field.nvdim=} field.")

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)
        extent = self._extent(multiplier)

        values = self.field.array.copy().reshape(self.field.mesh.n)

        if filter_field is None:
            filter_field = self.field._valid_as_field

        self._filter_values(filter_field, values)

        if symmetric_clim and "clim" not in kwargs.keys():
            vmin = np.min(values, where=~np.isnan(values), initial=0)
            vmax = np.max(values, where=~np.isnan(values), initial=0)
            vmax_abs = max(abs(vmin), abs(vmax))
            kwargs["clim"] = (-vmax_abs, vmax_abs)

        cp = ax.imshow(np.transpose(values), origin="lower", extent=extent, **kwargs)

        if colorbar:
            self._add_colorbar(ax, cp, colorbar_label)

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

        Parameters
        ----------
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

        Examples
        --------
        .. plot::
            :context: close-figs

            1. Visualising the scalar field using ``matplotlib``.

            >>> import discretisedfield as df
            ...
            >>> p1 = (0, 0, 0)
            >>> p2 = (100, 100, 100)
            >>> n = (10, 10, 10)
            >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
            >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
            ...
            >>> field.sel('z').mpl.lightness()

        """
        if self.field.nvdim == 2:
            if lightness_field is None:
                lightness_field = self.field.norm
            if filter_field is None:
                filter_field = self.field._valid_as_field
            x = self.field._r_dim_mapping[self.field.mesh.region.dims[0]]
            y = self.field._r_dim_mapping[self.field.mesh.region.dims[1]]
            return plot_util.inplane_angle(self.field, x, y).mpl.lightness(
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
        elif self.field.nvdim == 3:
            if lightness_field is None:
                if not self.field.vdim_mapping:
                    raise ValueError("'vdim_mapping' is required for lightness plots.")
                # find vector components pointing along the two axes 0 and 1
                vdims = [
                    self.field._r_dim_mapping[self.field.mesh.region.dims[0]],
                    self.field._r_dim_mapping[self.field.mesh.region.dims[1]],
                ]
                # find the third vector component for lightness
                lightness_vdim = (set(self.field.vdims) - set(vdims)).pop()
                lightness_field = getattr(self.field, lightness_vdim)
            if filter_field is None:
                filter_field = self.field._valid_as_field
            x = self.field._r_dim_mapping[self.field.mesh.region.dims[0]]
            y = self.field._r_dim_mapping[self.field.mesh.region.dims[1]]
            return plot_util.inplane_angle(self.field, x, y).mpl.lightness(
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
        elif self.field.nvdim > 3:
            raise RuntimeError(
                f"Only fields with `nvdim<=3` can be plotted. Not {self.field.nvdim=}"
            )

        ax = self._setup_axes(ax, figsize)

        if filter_field is None:
            filter_field = self.field._valid_as_field

        multiplier = self._setup_multiplier(multiplier)
        extent = self._extent(multiplier)

        if lightness_field is None:
            lightness_field = self.field.norm
        elif lightness_field.nvdim != 1:
            raise ValueError(f"Cannot use {lightness_field.nvdim=} lightness_field.")
        elif lightness_field.mesh.region.ndim != 2:
            raise ValueError(
                "'lightness_field' must be defined on a 2d mesh, not"
                f" {lightness_field.mesh.region.ndim=}."
            )

        values = self.field.array.copy().reshape(self.field.mesh.n)

        if not np.array_equal(lightness_field.mesh.n, self.field.mesh.n):
            lightness_field = lightness_field.resample(self.field.mesh.n)
        lightness = lightness_field.array.reshape(self.field.mesh.n)

        rgb = plot_util.hls2rgb(
            hue=values, lightness=lightness, saturation=None, lightness_clim=clim
        ).squeeze()
        self._filter_values(filter_field, rgb)

        # alpha channel to hide points with nan values (filter field)
        # all three rgb values are set to nan
        rgba = np.empty((*rgb.shape[:-1], 4))
        rgba[..., :3] = rgb
        rgba[..., 3] = 1.0
        # nan -> zero with alpha=0 to avoid cast warning
        rgba[np.isnan(rgb[..., 0])] = 0

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
        vdims=None,
        use_color=True,
        color_field=None,
        colorbar=True,
        colorbar_label="",
        filename=None,
        **kwargs,
    ):
        r"""Plot the vector field on a plane.

        Before the field can be plotted, it must be sliced to a plane (e.g.
        ``field.sel('z')``, assuming the geometry has 3 dimensions). In addition, field
        must be a vector field of dimensionality two or three (i.e. ``nvdim=2`` or
        ``nvdim=3``). Otherwise, ``ValueError`` is raised. ``mpl.vector`` adds the plot
        to ``matplotlib.axes.axes`` passed via ``ax`` argument. If ``ax`` is not passed,
        ``matplotlib.axes.axes`` object is created automatically and the size of a
        figure can be specified using ``figsize``. By default, plotted vectors are
        coloured according to the out-of-plane component of the vectors if the field has
        ``nvdim=3``. This can be changed by passing ``color_field`` with
        ``nvdim=1``. To disable colouring of the plot, ``use_color=False`` can be
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

        vdims : List[str], optional

            Names of the components to be used for the x and y component of the plotted
            arrows. This information is used to associate field components and spatial
            directions. Optionally, one of the list elements can be ``None`` if the
            field has no component in that direction. ``vdims`` is required for 2d
            vector fields.

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
            >>> field = df.Field(mesh, nvdim=3, value=(1.1, 2.1, 3.1))
            ...
            >>> field.sel('y').mpl.vector()

        .. seealso:: :py:func:`~discretisedfield.field.mpl_scalar`

        """
        if vdims is None and not self.field.vdim_mapping:
            raise ValueError("'vdims' is required for a field without 'vdim_mapping'.")
        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points1 = self.field.mesh.cells[0] / multiplier
        points2 = self.field.mesh.cells[1] / multiplier

        values = self.field.array.copy()
        self._filter_values(self.field._valid_as_field, values)

        if vdims is None:
            # find vector components pointing along the two axes 0 and 1
            vdims = [
                self.field._r_dim_mapping[self.field.mesh.region.dims[0]],
                self.field._r_dim_mapping[self.field.mesh.region.dims[1]],
            ]
        elif len(vdims) != 2:
            raise ValueError(f"{vdims=} must contain two elements.")

        arrow_x = self.field.vdims.index(vdims[0]) if vdims[0] else None
        arrow_y = self.field.vdims.index(vdims[1]) if vdims[1] else None
        if arrow_x is None and arrow_y is None:
            raise ValueError(f"At least one element of {vdims=} must be not None.")

        quiver_args = [
            points1,
            points2,
            np.transpose(
                values[..., arrow_x]
                if arrow_x is not None
                else np.zeros(self.field.mesh.n)
            ),
            np.transpose(
                values[..., arrow_y]
                if arrow_y is not None
                else np.zeros(self.field.mesh.n)
            ),
        ]

        if use_color and color_field is None:
            if self.field.nvdim != 3:
                warnings.warn(
                    "Automatic coloring is only supported for 3d"
                    f' fields. Ignoring "{use_color=}".'
                )
                use_color = False
            else:
                # find the third vector component for colouring
                color_vdim = (set(self.field.vdims) - set(vdims)).pop()
                color_field = getattr(self.field, color_vdim)

        if use_color:
            if color_field.nvdim != 1:
                raise ValueError(f"Cannot use {color_field.nvdim=}.")
            if color_field.mesh.region.ndim != 2:
                raise ValueError(
                    "'color_field' must be defined on a 2d mesh, not"
                    f" {color_field.mesh.region.ndim=}."
                )
            if not np.array_equal(color_field.mesh.n, self.field.mesh.n):
                color_field = color_field.resample(self.field.mesh.n)
            quiver_args.append(color_field.array.reshape(self.field.mesh.n).transpose())

        cp = ax.quiver(*quiver_args, pivot="mid", **kwargs)

        ax.set_aspect("equal")
        if colorbar and use_color:
            self._add_colorbar(ax, cp, colorbar_label)

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

        Before the field can be plotted, it must be sliced to give a plane (e.g.
        ``field.sel('z')``, assuming ``field`` has three geometrical dimensions). In
        addition, field must be a scalar field (``nvdim=1``). Otherwise, ``ValueError``
        is raised. ``mpl.contour`` adds the plot to ``matplotlib.axes.Axes`` passed via
        ``ax`` argument. If ``ax`` is not passed, ``matplotlib.axes.Axes`` object is
        created automatically and the size of a figure can be specified using
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

            If the field has not 2D, its dimension is not 1, or the dimension of
            ``filter_field`` is not 1.

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
            >>> field = df.Field(mesh, nvdim=1, value=init_value)
            >>> field.sel('z').mpl.contour()

        """
        if self.field.nvdim != 1:
            raise RuntimeError(f"Cannot plot nvdim={self.field.nvdim} field.")

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points1 = self.field.mesh.cells[0] / multiplier
        points2 = self.field.mesh.cells[1] / multiplier

        values = self.field.array.copy().reshape(self.field.mesh.n)

        if filter_field is None:
            filter_field = self.field._valid_as_field

        self._filter_values(filter_field, values)

        cp = ax.contour(points1, points2, np.transpose(values), **kwargs)
        ax.set_aspect("equal")

        if colorbar:
            self._add_colorbar(ax, cp, colorbar_label)

        self._axis_labels(ax, multiplier)

        self._savefig(filename)

    def _setup_multiplier(self, multiplier):
        return self.field.mesh.region.multiplier if multiplier is None else multiplier

    def _filter_values(self, filter_field, values):
        if filter_field is None:
            return values

        if filter_field.nvdim != 1:
            raise ValueError(f"Cannot use {filter_field.nvdim=}.")
        if filter_field.mesh.region.ndim != 2:
            raise ValueError(
                "'filter_field' must be defined on a 2d mesh, not"
                f" {filter_field.mesh.region.ndim=}."
            )

        if not np.array_equal(filter_field.mesh.n, self.field.mesh.n):
            filter_field = filter_field.resample(self.field.mesh.n)

        values[filter_field.array.reshape(self.field.mesh.n) == 0] = np.nan

    def _axis_labels(self, ax, multiplier):
        ax.set_xlabel(
            rf"{self.field.mesh.region.dims[0]}"
            rf" ({uu.rsi_prefixes[multiplier]}{self.field.mesh.region.units[0]})"
        )
        ax.set_ylabel(
            rf"{self.field.mesh.region.dims[1]}"
            rf" ({uu.rsi_prefixes[multiplier]}{self.field.mesh.region.units[1]})"
        )

    def _extent(self, multiplier):
        # Rescale about the origin to not keep the old centre point.
        # Example: Region(0, 50e-9) should become Region(0, 50) in nm and not being
        # centred around 25e-9.
        reference_point = (0, 0)  # 2d point because plotting requires 2d meshes
        rescaled_region = self.field.mesh.region.scale(1 / multiplier, reference_point)
        pmin = rescaled_region.pmin
        pmax = rescaled_region.pmax

        return [pmin[0], pmax[0], pmin[1], pmax[1]]

    def __dir__(self):
        dirlist = dir(self.__class__)

        for attr in ["vector"] if self.field.nvdim == 1 else ["scalar", "contour"]:
            dirlist.remove(attr)

        return dirlist

    def _add_colorbar(
        self,
        ax,
        cp,
        colorbar_label,
        min_height_inches=2.0,
        min_width_inches=0.35,
        min_pad_inches=0.1,
    ):
        """
        Adds a colorbar to the current plot with specified dimensions and padding.

        Parameters
        ----------
        ax : matplotlib.axes.Axes

            The parent axes to which the colorbar is added.

        cp : ScalarMappable

            The plot to which the colorbar is associated.

        colorbar_label : str

            Label for the colorbar. If None, no label is added.

        min_height_inches : float, optional

            Minimum height for the colorbar in inches. Default is 2.0 inches.

        min_width_inches : float, optional

            Minimum width for the colorbar in inches. Default is 0.35 inches.

        min_pad_inches : float, optional

            Minimum padding from the parent axes in inches. Default is 0.1 inches.

        """

        # Retrieve the figure associated with the axis
        fig = ax.figure
        # Extract figure dimensions in inches
        fig_width_inches, fig_height_inches = fig.get_size_inches()
        # Get position of the current axes (normalized to figure size)
        pos = ax.get_position()

        # Normalize height relative to ax
        min_height_norm = min_height_inches / (fig_height_inches * (pos.y1 - pos.y0))
        # Normalize width and padding relative to figure size
        min_width_norm = min_width_inches / fig_width_inches
        min_pad_norm = min_pad_inches / fig_width_inches

        # Decide on pad and width based on a threshold of 5% of the figure width
        if min_pad_norm > 0.05:
            pad_h = Size.Fixed(min_pad_inches)
        else:
            pad_h = Size.AxesX(ax, aspect=0.05)

        if min_width_norm > 0.05:
            width_h = Size.Fixed(min_width_inches)
        else:
            width_h = Size.AxesX(ax, aspect=0.05)

        # Determine the vertical aspect ratio for the colorbar
        if min_height_norm > 1:
            v_aspect = min_height_norm
        else:
            v_aspect = 1

        # Check for any existing colorbars associated with the current axis
        existing_colorbars = [
            a for a in fig.get_axes() if f"cb_{id(ax)}" in a.get_label()
        ]

        # Create a divider for the axes to place the colorbar
        divider = make_axes_locatable(ax)

        # Define a unique label for the colorbar axis to prevent any overlaps
        cax = fig.add_axes(
            divider.get_position(), label=f"cb_{id(ax)}_{len(existing_colorbars)}"
        )

        # Update divider settings based on existing colorbars
        if len(existing_colorbars) == 0:
            h = [Size.AxesX(ax), pad_h, width_h]
            divider.set_horizontal(h)
        else:
            divider.new_horizontal(pad_h, pack_start=False)
            divider.new_horizontal(width_h, pack_start=False)

        # Set the vertical aspect for the divider
        v = [Size.AxesY(ax, aspect=v_aspect)]
        divider.set_vertical(v)

        # Set the location of the main plot's axes to its original position.
        # The main plot's axes will be placed at the starting (0,0) position
        # on the grid defined by the divider.
        ax.set_axes_locator(divider.new_locator(nx=0, ny=0))

        # Loop through any existing colorbars associated with the main axes.
        # Adjust their positions to accommodate the new colorbar.
        # Each colorbar is positioned two steps away on the x-axis
        # (e.g., main plot at 0, first colorbar at 2, second at 4, and so on).
        for i, cb in enumerate(existing_colorbars, start=1):
            cb.set_axes_locator(divider.new_locator(nx=2 * i, ny=0))

        # Set the position of the new colorbar (`cax`).
        # It will be placed next to the last existing colorbar or next
        # to the main plot if it's the first colorbar.
        cax.set_axes_locator(
            divider.new_locator(nx=2 * (len(existing_colorbars) + 1), ny=0)
        )

        # Create the colorbar with the provided ScalarMappable object
        cbar = plt.colorbar(cp, cax=cax)

        # Add a label to the colorbar if provided
        if colorbar_label is not None:
            cbar.ax.set_ylabel(colorbar_label)
