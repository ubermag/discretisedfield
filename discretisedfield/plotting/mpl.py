"""Matplotlib-based plotting."""
import warnings

import mpl_toolkits.axes_grid1.inset_locator
import matplotlib.pyplot as plt
import numpy as np

import discretisedfield.util as dfu
import ubermagutil.units as uu


class Mpl:
    """Matplotlib-based plotting convenience methods.

    Parameters
    ----------
    data : df.Field

        Field sliced with a plane, e.g. field.plane('x').

    """

    def __init__(self, field):
        # We never allow plotting fields that are not sliced.
        if not field.mesh.attributes['isplane']:
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        self.field = field  # TODO: Consider renaming data to field.
        self.planeaxis = dfu.raxesdict[field.mesh.attributes['planeaxis']]

    def __call__(self,
                 ax=None,
                 figsize=None,
                 multiplier=None,
                 scalar_kwargs=None,
                 vector_kwargs=None,
                 filename=None):
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

        All the default values can be changed by passing arguments, which are
        then used in subplots. The way parameters of this function are used to
        create plots can be understood with the following code snippet.

        # TODO: Review example after API stabilises.

        .. code-block::

            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)

            scalar_field.mpl_scalar(ax=ax, filter_field=scalar_filter_field,
                                    lightness_field=scalar_lightness_field,
                                    colorbar=scalar_colorbar,
                                    colorbar_label=scalar_colorbar_label,
                                    multiplier=multiplier, cmap=scalar_cmap,
                                    clim=scalar_clim,)

            vector_field.mpl_vector(ax=ax, use_color=use_vector_color,
                                    color_field=vector_color_field,
                                    colorbar=vector_colorbar,
                                    colorbar_label=vector_colorbar_label,
                                    multiplier=multiplier, scale=vector_scale,
                                    cmap=vector_cmap, clim=vector_clim,)

            if filename is not None:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
            ```

            Therefore, to understand the meaning of the arguments which can be
            passed to this method, please refer to
            ``discretisedfield.plotting.Mpl.scalar`` and
            ``discretisedfield.plotting.Mpl.vector`` documentation.

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

        scalar_kwargs = {} if scalar_kwargs is None else scalar_kwargs
        vector_kwargs = {} if vector_kwargs is None else vector_kwargs
        vector_kwargs.setdefault('use_color', False)
        vector_kwargs.setdefault('colorbar', False)

        # Set up default scalar and vector fields.
        # TODO: Do we allow user to specify what scalar and vector fields are?
        # Did we have that before?
        if self.field.dim == 1:
            scalar_field = self.field
            vector_field = None

        elif self.field.dim == 2:
            scalar_field = None
            vector_field = self.field

        else:
            vector_field = self.field
            scalar_field = getattr(self.field, self.planeaxis)
            scalar_kwargs['colorbar_label'] = scalar_kwargs.get(
                'colorbar_label', f'{self.planeaxis}-component')

        # TODO user should specify filter_field=None to avoid filtering
        # TODO what is the norm if dim=1
        scalar_kwargs.setdefault('filter_field', self.field.norm)

        if scalar_field is not None:
            scalar_field.mpl.scalar(ax=ax, multiplier=multiplier,
                                    **scalar_kwargs)
        if vector_field is not None:
            vector_field.mpl.vector(ax=ax, multiplier=multiplier,
                                    **vector_kwargs)

        self._axis_labels(ax, multiplier)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)

    def scalar(self,
               ax=None,
               figsize=None,
               multiplier=None,
               filter_field=None,
               colorbar=True,
               colorbar_label=None,  # TODO: Can we maybe use an empty string?
               filename=None,
               symmetric_clim=False,
               **kwargs):
        r"""Plot the scalar field on a plane.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be a scalar field
        (``dim=1``). Otherwise, ``ValueError`` is raised. ``mpl_scalar`` adds
        the plot to ``matplotlib.axes.Axes`` passed via ``ax`` argument. If
        ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. By passing ``filter_field`` the points at which the pixels
        are not coloured can be determined. More precisely, only those
        discretisation cells where ``filter_field != 0`` are plotted. By
        passing a scalar field as ``lightness_field``, lightness component is
        added to HSL colormap. In this case, colormap cannot be passed using
        ``kwargs``. Colorbar is shown by default and it can be removed from the
        plot by passing ``colorbar=False``. The label for the colorbar can be
        defined by passing ``colorbar_label`` as a string.

        It is often the case that the region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3  (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
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

        lightness_field : discretisedfield.Field, optional

            A scalar field used for adding lightness to the color. Field values
            are hue. Defaults to ``None``.

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
            msg = f'Cannot plot {self.field.dim=} field.'
            raise ValueError(msg)

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points, values = map(list, zip(*list(self.field)))

        pmin = np.divide(self.field.mesh.region.pmin, multiplier)
        pmax = np.divide(self.field.mesh.region.pmax, multiplier)

        # TODO: After refactoring code, maybe extent and n can become
        # part of PlaneMesh.
        extent = [pmin[self.field.mesh.attributes['axis1']],
                  pmax[self.field.mesh.attributes['axis1']],
                  pmin[self.field.mesh.attributes['axis2']],
                  pmax[self.field.mesh.attributes['axis2']]]
        n = (self.field.mesh.n[self.field.mesh.attributes['axis2']],
             self.field.mesh.n[self.field.mesh.attributes['axis1']])

        values = self._filter_values(filter_field, points, values)

        if symmetric_clim and 'clim' not in kwargs.keys():
            vmin = np.min(values)
            vmax = np.max(values)
            if np.sign(vmin) != np.sign(vmax):
                vmax_abs = max(abs(vmin), vmax)
                kwargs['clim'] = (-vmax_abs, vmax_abs)
            else:
                warnings.warn('Symmetric clim only possible if the field has '
                              'positive and negative values.')
        cp = ax.imshow(np.array(values).reshape(n),
                       origin='lower', extent=extent, **kwargs)

        if colorbar:
            cbar = plt.colorbar(cp, ax=ax)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        self._axis_labels(ax, multiplier)

        if filename is not None:
            # TODO: We use pad inches 0 and 0.02.
            # We should figure out which one is the best.
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def lightness(self,
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
                  **kwargs):
        """Lightness plots.

        Uses HSV to show inplane angle and lightness for out-of-plane (3d) or
        norm (1d/2d) of the field.

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
                **kwargs)
        elif self.field.dim == 3:
            if lightness_field is None:
                lightness_field = getattr(self.field, self.planeaxis)
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
                **kwargs)

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points, values = map(list, zip(*list(self.field)))

        # TODO Requires refactoring of df.Mesh
        # rescaled_region = self.field.mesh.region.rescale(multiplier)
        # pmin = rescaled_region.pmin
        # pmax = rescaled_region.pmax

        pmin = np.divide(self.field.mesh.region.pmin, multiplier)
        pmax = np.divide(self.field.mesh.region.pmax, multiplier)

        # This depends on plane mesh orientation and should be in PlaneMesh.
        extent = [pmin[self.field.mesh.attributes['axis1']],
                  pmax[self.field.mesh.attributes['axis1']],
                  pmin[self.field.mesh.attributes['axis2']],
                  pmax[self.field.mesh.attributes['axis2']]]
        n = (self.field.mesh.n[self.field.mesh.attributes['axis2']],
             self.field.mesh.n[self.field.mesh.attributes['axis1']])

        if lightness_field is None:
            lightness_field = self.field.norm
        else:
            if lightness_field.dim != 1:
                msg = f'Cannot use {lightness_field.dim=} lightness_field.'
                raise ValueError(msg)

        lightness = [lightness_field(i) for i in self.field.mesh]

        # values = self._filter_values(filter_field, points, values)
        rgb = dfu.hls2rgb(hue=values,
                          lightness=lightness,
                          saturation=None,
                          lightness_clim=clim).squeeze()

        # TODO: filter field affects color range
        # rgb = self._filter_values(filter_field, points, rgb)
        # rgba = np.ones((len(rgb), 4))
        # rgba[..., :3] = rgb
        # rgba[..., 3][np.isnan(rgb[:, 0])] = 0

        kwargs['cmap'] = 'hsv'  # only hsv cmap allowed
        # TODO filtering
        # ax.imshow(rgba.reshape((*n, 4)), origin='lower',
        #           extent=extent, **kwargs)
        ax.imshow(rgb.reshape((*n, 3)), origin='lower',
                  extent=extent, **kwargs)
        if colorwheel:
            if colorwheel_args is None:
                colorwheel_args = {}
            cw_ax = add_colorwheel(ax, **colorwheel_args)
            if colorwheel_xlabel is not None:
                cw_ax.arrow(100, 100, 60, 0, width=5, fc='w', ec='w')
                cw_ax.annotate(colorwheel_xlabel, (115, 140), c='w')
            if colorwheel_ylabel is not None:
                cw_ax.arrow(100, 100, 0, -60, width=5, fc='w', ec='w')
                cw_ax.annotate(colorwheel_ylabel, (40, 80), c='w')

        self._axis_labels(ax, multiplier)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def vector(self,
               ax=None,
               figsize=None,
               multiplier=None,
               use_color=True,
               color_field=None,
               colorbar=True,
               colorbar_label=None,
               filename=None,
               **kwargs):
        r"""Plot the vector field on a plane.

        before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). in addition, field must be a vector field
        (``dim=3``). otherwise, ``valueerror`` is raised. ``mpl_vector`` adds
        the plot to ``matplotlib.axes.axes`` passed via ``ax`` argument. if
        ``ax`` is not passed, ``matplotlib.axes.axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. by default, plotted vectors are coloured according to the
        out-of-plane component of the vectors. this can be changed by passing
        ``color_field`` with ``dim=1``. to disable colouring of the plot,
        ``use_color=false`` can be passed. a uniform vector colour can be
        obtained by specifying ``color=color`` which is passed to matplotlib
        and ``use_color=false``. colorbar is shown by default and it can
        be removed from the plot by passing ``colorbar=false``. the label for
        the colorbar can be defined by passing ``colorbar_label`` as a string.
        it is often the case that the region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3  (..., -6, -3, 0, 3, 6,...). according to that value, the
        axes will be scaled and appropriate units shown. for instance, if
        ``multiplier=1e-9`` is passed, all mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. if ``multiplier`` is not passed, the best one is
        calculated internally. the plot can be saved as a pdf when ``filename``
        is passed.

        this method plots the field using ``matplotlib.pyplot.quiver``
        function, so any keyword arguments accepted by it can be passed (for
        instance, ``cmap`` - colormap, ``clim`` - colorbar limits, etc.). in
        particular, there are cases when ``matplotlib`` fails to find optimal
        scale for plotting vectors. more precisely, sometimes vectors appear
        too large in the plot. this can be resolved by passing ``scale``
        argument, which scales all vectors in the plot. in other words, larger
        ``scale``, smaller the vectors and vice versa. please note that scale
        can be in a very large range (e.g. 1e20).

        parameters
        ----------
        ax : matplotlib.axes.axes, optional

            axes to which the field plot is added. defaults to ``None`` - axes
            are created internally.

        figsize : tuple, optional

            the size of a created figure if ``ax`` is not passed. defaults to
            ``None``.

        color_field : discretisedfield.field, optional

            a scalar field used for colouring the vectors. defaults to ``None``
            and vectors are coloured according to their out-of-plane
            components.

        colorbar : bool, optional

            if ``true``, colorbar is shown and it is hidden when ``false``.
            defaults to ``true``.

        colorbar_label : str, optional

            colorbar label. defaults to ``None``.

        multiplier : numbers.real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). according to that
            value, the axes will be scaled and appropriate units shown. for
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. defaults to ``None``.

        filename : str, optional

            if filename is passed, the plot is saved. defaults to ``None``.

        raises
        ------
        valueerror

            if the field has not been sliced, its dimension is not 3, or the
            dimension of ``color_field`` is not 1.

        example
        -------
        .. plot::
            :context: close-figs

            1. visualising the vector field using ``matplotlib``.

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
            msg = f'cannot plot dim={self.field.dim} field.'
            raise ValueError(msg)

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points, values = map(list, zip(*list(self.field)))

        # remove points and values where norm is 0.
        points = [p for p, v in zip(points, values)
                  if not np.equal(v, 0).all()]
        values = [v for v in values if not np.equal(v, 0).all()]

        if use_color:
            if color_field is None:
                # todo raises an exception by default; options:
                # - warning + automatically specify use_color=false
                # - use_color=false as default
                if self.field.dim == 2:
                    msg = 'automatic coloring is only supported for 3d fields.'
                    raise ValueError(msg)
                color_field = getattr(self.field, self.planeaxis)

            colors = [color_field(p) for p in points]

        # "unpack" values inside arrays and convert to np.ndarray.
        points = np.array(list(zip(*points)))
        values = np.array(list(zip(*values)))

        points = np.divide(points, multiplier)

        if use_color:
            cp = ax.quiver(points[self.field.mesh.attributes['axis1']],
                           points[self.field.mesh.attributes['axis2']],
                           values[self.field.mesh.attributes['axis1']],
                           values[self.field.mesh.attributes['axis2']],
                           colors, pivot='mid', **kwargs)
        else:
            ax.quiver(points[self.field.mesh.attributes['axis1']],
                      points[self.field.mesh.attributes['axis2']],
                      values[self.field.mesh.attributes['axis1']],
                      values[self.field.mesh.attributes['axis2']],
                      pivot='mid', **kwargs)

        ax.set_aspect('equal')
        if colorbar and use_color:
            cbar = plt.colorbar(cp, ax=ax)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        self._axis_labels(ax, multiplier)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def contour(self,
                ax=None,
                figsize=None,
                multiplier=None,
                filter_field=None,
                colorbar=True,
                colorbar_label=None,
                filename=None,
                **kwargs):
        """Contour line plot.

        Parameters
        ----------
        ...

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
            ...
            >>> field.plane('z').mpl.contour()

        """
        if self.field.dim != 1:
            msg = f'Cannot plot dim={self.field.dim} field.'
            raise ValueError(msg)

        ax = self._setup_axes(ax, figsize)

        multiplier = self._setup_multiplier(multiplier)

        points, values = map(list, zip(*list(self.field)))

        values = self._filter_values(filter_field, points, values)

        n = (self.field.mesh.n[self.field.mesh.attributes['axis2']],
             self.field.mesh.n[self.field.mesh.attributes['axis1']])

        points = np.array(list(zip(*points)))
        points = np.divide(points, multiplier)

        values = np.array(values).reshape(n)

        cp = ax.contour(points[self.field.mesh.attributes['axis1']].reshape(n),
                        points[self.field.mesh.attributes['axis2']].reshape(n),
                        values, **kwargs)
        ax.set_aspect('equal')

        if colorbar:
            cbar = plt.colorbar(cp, ax=ax)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        self._axis_labels(ax, multiplier)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def _setup_axes(self, ax, figsize, **kwargs):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, **kwargs)

        return ax

    def _setup_multiplier(self, multiplier):
        return (self.field.mesh.region.multiplier
                if multiplier is None else multiplier)

    def _filter_values(self, filter_field, points, values):
        if filter_field is None:
            return values

        if filter_field.dim != 1:
            msg = f'Cannot use {filter_field.dim=}.'
            raise ValueError(msg)

        return [values[i] if filter_field(point) != 0 else np.nan
                for i, point in enumerate(points)]

    def _axis_labels(self, ax, multiplier):
        unit = (rf' ({uu.rsi_prefixes[multiplier]}'
                rf'{self.field.mesh.attributes["unit"]})')
        ax.set_xlabel(dfu.raxesdict[self.field.mesh.attributes['axis1']]
                      + unit)
        ax.set_ylabel(dfu.raxesdict[self.field.mesh.attributes['axis2']]
                      + unit)

    def __dir__(self):
        dirlist = dir(self.__class__)

        for attr in ['vector'] if self.field.dim == 1 else ['scalar',
                                                            'contour']:
            dirlist.remove(attr)

        return dirlist


def add_colorwheel(ax, width=1, height=1, loc='lower right', **kwargs):
    """Colorwheel for hsv plots.

    Creates colorwheel on new inset axis.

    Example
    -------

    .. plot::
        :context: close-figs

        1. Adding a colorwheel to an empty axis
        >>> import discretisedfield.plotting as dfp
        >>> import matplotlib.pyplot as plt
        ...
        >>> fig, ax = plt.subplots()
        >>> ins_ax = dfp.add_colorwheel(ax)

    """
    n = 200
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)

    rgb = dfu.hls2rgb(hue=theta, lightness=r,
                      lightness_clim=[0, 1 / np.sqrt(2)])

    theta = theta.reshape((n, n, 1))

    rgba = np.zeros((n, n, 4))
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            if xi**2 + yi**2 <= 1:
                rgba[i, j, :3] = rgb[i, j, :]
                rgba[i, j, 3] = 1

    ax_ins = mpl_toolkits.axes_grid1.inset_locator.inset_axes(
        ax, width=width, height=height, loc=loc, **kwargs)
    ax_ins.imshow(rgba[:, ::-1, :])
    ax_ins.axis('off')
    return ax_ins
