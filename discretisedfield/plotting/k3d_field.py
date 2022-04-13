"""K3d based plotting."""
import k3d
import matplotlib
import numpy as np
import ubermagutil.units as uu

import discretisedfield.util as dfu


class K3dField:
    """K3d plotting."""

    def __init__(self, data):
        self.data = data

    def nonzero(
        self,
        plot=None,
        color=dfu.cp_int[0],
        multiplier=None,
        interactive_field=None,
        **kwargs,
    ):
        r"""``k3d`` plot of non-zero discretisation cells.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The colour of the non-zero discretisation cells can be
        specified using ``color`` argument.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        For interactive plots the field itself, before being sliced with the
        field, must be passed as ``interactive_field``. For example, if
        ``field.x.plane('z')`` is plotted, ``interactive_field=field`` must be
        passed. In addition, ``k3d.plot`` object cannot be created internally
        and it must be passed and displayed by the user.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally. This is not true in the case of an interactive
            plot, when ``plot`` must be created externally.

        color : int, optional

            Colour of the non-zero discretisation cells. Defaults to the
            default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        interactive_field : discretisedfield.Field, optional

            The whole field object (before any slices) used for interactive
            plots. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the dimension of the field is not 1.

        Examples
        --------
        1. Visualising non-zero discretisation cells using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, -50e-9)
        >>> p2 = (50e-9, 50e-9, 50e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> def normfun(point):
        ...     x, y, z = point
        ...     if x**2 + y**2 < 30**2:
        ...         return 1
        ...     else:
        ...         return 0
        >>> field.norm = normfun
        ...
        >>> field.norm.k3d.nonzero()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.plotting.K3d.voxels`

        """
        if self.data.dim != 1:
            msg = f"Cannot plot dim={self.data.dim} field."
            raise ValueError(msg)

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.data.mesh.region.edges)

        unit = (
            rf" ({uu.rsi_prefixes[multiplier]}" rf'{self.data.mesh.attributes["unit"]})'
        )

        if interactive_field is not None:
            plot.camera_auto_fit = False

            objects_to_be_removed = []
            for i in plot.objects:
                if i.name != "total_region":
                    objects_to_be_removed.append(i)
            for i in objects_to_be_removed:
                plot -= i

            if not any([o.name == "total_region" for o in plot.objects]):
                interactive_field.mesh.region.k3d(
                    plot=plot, multiplier=multiplier, name="total_region", opacity=0.025
                )

        # all voxels have the same color
        plot_array = np.ones_like(self.data.array)
        # remove voxels where field is zero
        plot_array[self.data.array == 0] = 0
        plot_array = plot_array[..., 0]  # remove an empty dimension
        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)
        plot_array = plot_array.astype(np.uint8)  # to avoid k3d warning

        bounds = [
            i
            for sublist in zip(
                np.divide(self.data.mesh.region.pmin, multiplier),
                np.divide(self.data.mesh.region.pmax, multiplier),
            )
            for i in sublist
        ]

        plot += k3d.voxels(
            plot_array, color_map=color, bounds=bounds, outlines=False, **kwargs
        )

        plot.axes = [i + r"\,\text{{{}}}".format(unit) for i in dfu.axesdict.keys()]

    def scalar(
        self,
        plot=None,
        filter_field=None,
        cmap="cividis",
        multiplier=None,
        interactive_field=None,
        **kwargs,
    ):
        """``k3d`` plot of a scalar field.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The colormap can be specified using ``cmap`` argument.
        By passing ``filter_field`` the points at which the voxels are not
        shown can be determined. More precisely, only those discretisation
        cells where ``filter_field != 0`` are plotted.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        For interactive plots the field itself, before being sliced with the
        field, must be passed as ``interactive_field``. For example, if
        ``field.x.plane('z')`` is plotted, ``interactive_field=field`` must be
        passed. In addition, ``k3d.plot`` object cannot be created internally
        and it must be passed and displayed by the user.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally. This is not true in the case of an interactive
            plot, when ``plot`` must be created externally.

        filter_field : discretisedfield.Field, optional

            Scalar field. Only discretisation cells where ``filter_field != 0``
            are shown. Defaults to ``None``.

        cmap : str, optional

            Colormap.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        interactive_field : discretisedfield.Field, optional

            The whole field object (before any slices) used for interactive
            plots. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the dimension of the field is not 1.

        Example
        -------
        1. Plot the scalar field using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50, -50, -50)
        >>> p2 = (50, 50, 50)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> field = df.Field(mesh, dim=1, value=5)
        >>> field.k3d.scalar()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.plotting.K3d.vector`

        """
        if self.data.dim != 1:
            msg = f"Cannot plot dim={self.data.dim} field."
            raise ValueError(msg)

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if filter_field is not None:
            if filter_field.dim != 1:
                msg = f"Cannot use dim={self.data.dim} filter_field."
                raise ValueError(msg)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.data.mesh.region.edges)

        unit = (
            rf" ({uu.rsi_prefixes[multiplier]}" rf'{self.data.mesh.attributes["unit"]})'
        )

        if interactive_field is not None:
            plot.camera_auto_fit = False

            objects_to_be_removed = []
            for i in plot.objects:
                if i.name != "total_region":
                    objects_to_be_removed.append(i)
            for i in objects_to_be_removed:
                plot -= i

            if not any(o.name == "total_region" for o in plot.objects):
                interactive_field.mesh.region.k3d(
                    plot=plot, multiplier=multiplier, name="total_region", opacity=0.025
                )

        plot_array = np.copy(self.data.array)  # make a deep copy
        plot_array = plot_array[..., 0]  # remove an empty dimension

        # All values must be in (1, 255) -> (1, n-1), for n=256 range, with
        # maximum n=256. This is the limitation of k3d.voxels(). Voxels where
        # values are zero, are invisible.
        plot_array = dfu.normalise_to_range(plot_array, (1, 255))
        # Remove voxels where filter_field = 0.
        if filter_field is not None:
            for i in self.data.mesh.indices:
                if filter_field(self.data.mesh.index2point(i)) == 0:
                    plot_array[i] = 0
        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)
        plot_array = plot_array.astype(np.uint8)  # to avoid k3d warning

        cmap = matplotlib.cm.get_cmap(cmap, 256)
        cmap_int = []
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            cmap_int.append(int(matplotlib.colors.rgb2hex(rgb)[1:], 16))

        bounds = [
            i
            for sublist in zip(
                np.divide(self.data.mesh.region.pmin, multiplier),
                np.divide(self.data.mesh.region.pmax, multiplier),
            )
            for i in sublist
        ]

        plot += k3d.voxels(
            plot_array, color_map=cmap_int, bounds=bounds, outlines=False, **kwargs
        )

        plot.axes = [i + r"\,\text{{{}}}".format(unit) for i in dfu.axesdict.keys()]

    def vector(
        self,
        plot=None,
        color_field=None,
        cmap="cividis",
        head_size=1,
        points=True,
        point_size=None,
        vector_multiplier=None,
        multiplier=None,
        interactive_field=None,
        **kwargs,
    ):
        """``k3d`` plot of a vector field.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. By passing ``color_field`` vectors are coloured
        according to the values of that field. The colormap can be specified
        using ``cmap`` argument. The head size of vectors can be changed using
        ``head_size``. The size of the plotted vectors is computed
        automatically in order to fit the plot. However, it can be adjusted
        using ``vector_multiplier``.

        By default both vectors and points, corresponding to discretisation
        cell coordinates, are plotted. They can be removed from the plot by
        passing ``points=False``. The size of the points are calculated
        automatically, but it can be adjusted with ``point_size``.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        For interactive plots the field itself, before being sliced with the
        field, must be passed as ``interactive_field``. For example, if
        ``field.plane('z')`` is plotted, ``interactive_field=field`` must be
        passed. In addition, ``k3d.plot`` object cannot be created internally
        and it must be passed and displayed by the user.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally. This is not true in the case of an interactive
            plot, when ``plot`` must be created externally.

        color_field : discretisedfield.Field, optional

            Scalar field. Vectors are coloured according to the values of
            ``color_field``. Defaults to ``None``.

        cmap : str, optional

            Colormap.

        head_size : int, optional

            The size of vector heads. Defaults to ``None``.

        points : bool, optional

            If ``True``, points are shown together with vectors. Defaults to
            ``True``.

        point_size : int, optional

            The size of the points if shown in the plot. Defaults to ``None``.

        vector_multiplier : numbers.Real, optional

            All vectors are divided by this value before being plotted.
            Defaults to ``None``.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        interactive_field : discretisedfield.Field, optional

            The whole field object (before any slices) used for interactive
            plots. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the dimension of the field is not 3.

        Examples
        --------
        1. Visualising the vector field using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(0, 0, 1))
        ...
        >>> field.k3d.vector()
        Plot(...)

        """
        if self.data.dim != 3:
            msg = f"Cannot plot dim={self.data.dim} field."
            raise ValueError(msg)

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if color_field is not None:
            if color_field.dim != 1:
                msg = f"Cannot use dim={self.data.dim} color_field."
                raise ValueError(msg)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.data.mesh.region.edges)

        unit = (
            rf" ({uu.rsi_prefixes[multiplier]}" rf'{self.data.mesh.attributes["unit"]})'
        )

        if interactive_field is not None:
            plot.camera_auto_fit = False

            objects_to_be_removed = []
            for i in plot.objects:
                if i.name != "total_region":
                    objects_to_be_removed.append(i)
            for i in objects_to_be_removed:
                plot -= i

            if not any(o.name == "total_region" for o in plot.objects):
                interactive_field.mesh.region.k3d(
                    plot=plot, multiplier=multiplier, name="total_region", opacity=0.025
                )

        coordinates, vectors, color_values = [], [], []
        norm_field = self.data.norm  # assigned to be computed only once
        for point, value in self.data:
            if norm_field(point) != 0:
                coordinates.append(point)
                vectors.append(value)
                if color_field is not None:
                    color_values.append(color_field(point))

        if color_field is not None:
            color_values = dfu.normalise_to_range(color_values, (0, 255))

            # Generate double pairs (body, head) for colouring vectors.
            cmap = matplotlib.cm.get_cmap(cmap, 256)
            cmap_int = []
            for i in range(cmap.N):
                rgb = cmap(i)[:3]
                cmap_int.append(int(matplotlib.colors.rgb2hex(rgb)[1:], 16))

            colors = []
            for cval in color_values:
                colors.append(2 * (cmap_int[cval],))
        else:
            # Uniform colour.
            colors = len(vectors) * ([2 * (dfu.cp_int[1],)])

        coordinates = np.array(coordinates)
        vectors = np.array(vectors)

        if vector_multiplier is None:
            vector_multiplier = (
                vectors.max() / np.divide(self.data.mesh.cell, multiplier).min()
            )

        coordinates = np.divide(coordinates, multiplier)
        vectors = np.divide(vectors, vector_multiplier)

        coordinates = coordinates.astype(np.float32)
        vectors = vectors.astype(np.float32)

        plot += k3d.vectors(
            coordinates - 0.5 * vectors,
            vectors,
            colors=colors,
            head_size=head_size,
            **kwargs,
        )

        if points:
            if point_size is None:
                # If undefined, the size of the point is 1/4 of the smallest
                # cell dimension.
                point_size = np.divide(self.data.mesh.cell, multiplier).min() / 4

            plot += k3d.points(coordinates, color=dfu.cp_int[0], point_size=point_size)

        plot.axes = [i + r"\,\text{{{}}}".format(unit) for i in dfu.axesdict.keys()]

    def __dir__(self):
        dirlist = dir(self.__class__)
        if self.data.dim == 1:
            need_removing = ["k3d_vector"]
        else:
            need_removing = ["k3d_scalar", "k3d_nonzero"]

        for attr in need_removing:
            dirlist.remove(attr)

        return dirlist
