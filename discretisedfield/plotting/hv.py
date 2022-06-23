"""Holoviews-based plotting."""
import contextlib
import warnings

import holoviews as hv
import numpy as np
import xarray as xr

import discretisedfield.util as dfu


class Hv:
    """Holoviews-based plotting methods.

    Plots based on Holoviews can be created without prior slicing. Instead, a slider is
    created for the directions not shown in the plot. This class should not be accessed
    directly. Use ``field.hv`` to use the different plotting methods.

    Hv has a class property ``norm_filter`` that controls the default behaviour of
    ``Hv.__call__``, the convenience plotting method that is typically available as
    ``field.hv()``. By default ``norm_filter=True`` and plots created with ``hv()`` use
    automatic filtering based on the norm of the field. To disable automatic filtering
    globally use ``discretisedfield.plotting.defaults.norm_filter = False``.

    Parameters
    ----------
    array : xarray.DataArray

        DataArray to plot.

    """

    _norm_filter = True

    def __init__(self, array):
        import hvplot.xarray  # noqa, delayed import because it creates (empty) output

        self.array = array

    def __call__(
        self,
        kdims,
        vdims=None,
        roi=None,
        norm_filter=None,
        scalar_kw=None,
        vector_kw=None,
    ):
        """Create an optimal plot combining ``hv.scalar`` and ``hv.vector``.

        This is a convenience method for quick plotting. It combines ``hv.scalar`` and
         ``hv.vector``. Depending on the dimensionality of the object, it automatically
         determines the typo of plot in the following order:

        1. For scalar objects (no dimension with name ``'comp'``) only
           ``discretisedfield.plotting.Hv.scalar`` is used.

        2. When ``vdims`` is specified a vector plot (without coloring) is created for
           the "in-plane" part of the vector field (defined via ``vdims``). If the field
           has vector dimensionality larger than two an additional scalar plot is
           created for all remaining vector components (with a drop-down selection).

        3. If the field is a 3d vector field and defined in 3d space the in-plane and
           out-of-plane vector components are determined (guessed) automatically. A
           scalar plot is created for the out-of-plane component and overlayed with a
           vector plot for the in-plane components (without coloring).

        4. For all other vector fields a scalar plot with a drop-down selection for the
           individual vector components is created.

        Based on the norm of the object (absolute values for scalar fields) automatic
        filtering is applied, i.e. all cells where the norm is zero are excluded from
        the plot. To manually filter out parts of the plot (e.g. areas where the norm of
        the field is zero) an additional ``roi`` can be passed. It can take an
        ``xarray.DataArray`` or a ``discretisedfield.Field`` and hides all points where
        ``roi`` is 0. It relies on ``xarray``s broadcasting and the object passed to
        ``roi`` must only have the same dimensions as the ones specified as ``kdims``.

        To disable filtering pass ``norm_filter=False``. To disable filtering for all
        plots globally set ``discretisedfield.plotting.defaults.norm_filter = False``.
        If norm filtering has been disabled globally use ``norm_filter=True`` to enable
        it for a single plot.

        All default values of ``hv.scalar`` and ``hv.vector`` can be changed by passing
        dictionaries to ``scalar_kw`` and ``vector_kw``, which are then used in
        subplots.

        To reduce the number of points in the plot a simple re-sampling is available.
        The parameter ``n`` in ``scalar_kw`` or ``vector_kw``. can be used to specify
        the number of points in different directions. A tuple of length 2 can be used to
        specify the number of points in the two ``kdims``. A dictionary can be used to
        specify the number of points in arbitrary dimensions. Keys of the dictionary
        must be dimensions of the array. Dimensions that are not specified are not
        modified. Note, that the re-sampling method is very basic and does not do any
        sort of interpolation (it just picks the nearest point). The extreme points in
        each direction are always kept. Equidistant points are picked in between. The
        same resampling has to be used for all slider dimensions.

        Therefore, to understand the meaning of the keyword arguments which can be
        passed to this method, please refer to
        ``discretisedfield.plotting.Hv.scalar`` and
        ``discretisedfield.plotting.Hv.vector`` documentation.

        Parameters
        ----------
        kdims : List[str]

            Array coordinates plotted in plot x and plot y directon.

        vdims : List[str], optional

            Names of the components to be used for the x and y component of the plotted
            arrows. This information is used to associate field components and spatial
            directions. Optionally, one of the list elements can be ``None`` if the
            field has no component in that direction. ``vdims`` is required for non-3d
            vector fields and for fields that do not have 3 spatial coordinates.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            roi is non-zero are included in the output.

        norm_filtering : bool, optional

            If ``True`` use a default roi based on the norm of the field, if ``False``
            do not filter automatically. If not specified the value of
            ``discretisedfield.plotting.defaults.norm_filter`` is used. This allows
            globally disabling the filtering.

        scalar_kw : dict

            Additional keyword arguments that are
            ``discretisedfield.plotting.Hv.scalar``

        vector_kw : dict

            Additional keyword arguments that are
            ``discretisedfield.plotting.Hv.vector``

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a combined plot for each slider
            value.

        Examples
        --------
        1. Simple combined scalar and vector plot with ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
        ...
        >>> field.hv(kdims=['x', 'y'])
        :DynamicMap...

        """
        scalar_kw = {} if scalar_kw is None else scalar_kw.copy()
        vector_kw = {} if vector_kw is None else vector_kw.copy()

        if norm_filter or (norm_filter is None and self._norm_filter):
            if roi is None and "comp" not in self.array.dims:
                roi = np.abs(self.array)
            elif roi is None:
                roi = xr.apply_ufunc(
                    np.linalg.norm,
                    self.array,
                    input_core_dims=[["comp"]],
                    kwargs={"axis": -1},
                )
            scalar_kw.setdefault("roi", roi)
            vector_kw.setdefault("roi", roi)

        vector_kw.setdefault("use_color", False)

        if "comp" not in self.array.dims:
            return self.scalar(kdims=kdims, **scalar_kw)
        elif vdims:
            scalar_dims = list(set(self.array.comp.to_numpy()) - set(vdims))
            with contextlib.suppress(KeyError):
                vector_kw.pop("vdims")
            vector = self.vector(kdims=kdims, vdims=vdims, **vector_kw)
            if len(scalar_dims) == 0:
                return vector
            else:
                scalar = self.__class__(self.array.sel(comp=scalar_dims)).scalar(
                    kdims=kdims, **scalar_kw
                )
                return scalar * vector
        elif len(self.array.comp) == 3 and self.array.ndim == 4:
            # map spatial coordinates x, y, z and vector comp names
            mapping = dict(zip(self.array.dims, range(4)))
            mapping.pop("comp")
            vdims = [
                self.array.comp.to_numpy()[mapping.pop(kdims[i])] for i in range(2)
            ]
            scalar_dim = self.array.comp.to_numpy()[mapping.popitem()[1]]
            scalar = self.__class__(self.array.sel(comp=scalar_dim)).scalar(
                kdims=kdims, **scalar_kw
            )
            vector_kw.setdefault("vdims", vdims)
            vector = self.__class__(self.array).vector(kdims=kdims, **vector_kw)
            return scalar * vector
        else:
            warnings.warn(
                "Automatic detection of vector components not possible for array with"
                f" ndim={self.array.ndim} and comp={list(self.array.comp.to_numpy())}."
                " To get a combined scalar and vector plot pass `vdims`."
            )
            return self.scalar(kdims=kdims, **scalar_kw)

    def scalar(self, kdims, roi=None, n=None, **kwargs):
        """Create an image plot for scalar data or individual components.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.Image`` objects. The plot shows the plane defined with the two
        spatial directions passed to ``kdims``. If a vector field is passed (that means
        the field dimension is greater than 1) an additional ``panel.Select`` widget for
        the field components is created automatically. It is not necessary to create a
        cut-plane first.

        To filter out parts of the plot (e.g. areas where the norm of the field is zero)
        an additional ``roi`` can be passed. It can take an ``xarray.DataArray`` or a
        ``discretisedfield.Field`` and hides all points where ``roi`` is 0. It relies on
        ``xarray``s broadcasting and the object passed to ``roi`` must only have the
        same dimensions as the ones specified as ``kdims``. No automatic filtering is
        applied.

        To reduce the number of points in the plot a simple re-sampling is available.
        The parameter ``n`` can be used to specify the number of points in different
        directions. A tuple of length 2 can be used to specify the number of points in
        the two ``kdims``. A dictionary can be used to specify the number of points in
        arbitrary dimensions. Keys of the dictionary must be dimensions of the array.
        Dimensions that are not specified are not modified. Note, that the re-sampling
        method is very basic and does not do any sort of interpolation (it just picks
        the nearest point). The extreme points in each direction are always kept.
        Equidistant points are picked in between.

        Additional keyword arguments are directly forwarded to the ``xarray.hvplot``
        method. Please refer to the documentation of ``hvplot`` (and ``holoviews``) for
        available options and additional documentation on how to modify the plot after
        creation.

        Parameters
        ----------
        kdims : List[str]

            Array coordinates plotted in plot x and plot y directon.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            roi is non-zero are included in the output.

        n : array_like, dict, optional

            Re-sampling of the array with the given number of points. If an array-like
            is passed it must have length 2 and the values are used for the two kdims.
            If a dictionary is passed its keys must correspond to (some of) the
            dimensions of the array. If not specified no re-sampling is done.

        kwargs

            Additional keyword arguments that are forwarded to ``xarray.hvplot``.

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a ``holoviews.Image`` for each
            slider value.

        Raises
        ------
        ValueError

            If ``kdims`` has not length 2 or contains strings that are not part of the
            objects dimensions (``'x'``, ``'y'``, or ``'z'`` for standard
            discretisedfield.Field objects).

        Examples
        --------
        1. Simple scalar plot with ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
        ...
        >>> field.hv.scalar(kdims=['x', 'z'])
        :DynamicMap...

        """
        x, y, kwargs = self._prepare_scalar_plot(kdims, roi, n, kwargs)
        return self.array.hvplot(x=x, y=y, **kwargs)

    def vector(
        self,
        kdims,
        vdims=None,
        cdim=None,
        roi=None,
        n=None,
        use_color=True,
        colorbar_label=None,
        **kwargs,
    ):
        """Create a vector plot.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.VectorField`` objects. The plot shows the plane defined with the two
        spatial directions passed to ``kdims``. The length of the arrows corresponds to
        the norm of the in-plane component of the field. It is not necessary to create a
        cut-plane first.

        ``vdims`` defines the components of the plotted object shown in the plot x and
        plot y direction (defined with ``kdims``). If the object has three vector
        components and three spatial dimensions ``vdims`` can be omitted and the vector
        components and spatial directions are combined based on their order.

        For 3d vector fields the color by default encodes the out-of-plane component.
        Other fields cannot be colored automatically. To assign a non-uniform color
        manually use ``cdim``. It either accepts a string to select one of the field
        vector components or a scalar ``discretisedfield.Field`` or
        ``xarray.DataArray``. Coloring can be disabled with ``use_color``.

        To filter out parts of the plot (e.g. areas where the norm of the field is zero)
        an additional ``roi`` can be passed. It can take an ``xarray.DataArray`` or a
        ``discretisedfield.Field`` and hides all points where ``roi`` is 0. It relies on
        ``xarray``s broadcasting and the object passed to ``roi`` must only have the
        same dimensions as the ones specified as ``kdims``. No automatic filtering is
        applied.

        To reduce the number of points in the plot a simple re-sampling is available.
        The parameter ``n`` can be used to specify the number of points in different
        directions. A tuple of length 2 can be used to specify the number of points in
        the two ``kdims``. A dictionary can be used to specify the number of points in
        arbitrary dimensions. Keys of the dictionary must be dimensions of the array.
        Dimensions that are not specified are not modified. Note, that the re-sampling
        method is very basic and does not do any sort of interpolation (it just picks
        the nearest point). The extreme points in each direction are always kept.
        Equidistant points are picked in between.

        This method is based on ``holoviews.VectorPlot``. Additional keyword arguments
        are directly forwarded to the ``.opts()`` method of the resulting object. Please
        refer to the documentation of ``holoviews`` for available options and additional
        documentation on how to modify the plot after creation.

        Parameters
        ----------
        kdims : List[str]

            Array coordinates plotted in plot x and plot y directon.

        vdims : List[str], optional

            Names of the components to be used for the x and y component of the plotted
            arrows. This information is used to associate field components and spatial
            directions. Optionally, one of the list elements can be ``None`` if the
            field has no component in that direction. ``vdims`` is required for non-3d
            vector fields and for fields that do not have 3 spatial coordinates.

        cdim : str, xarray.DataArray, discretisedfield.Field, optional

            A string can be used to select one of the vector components of the field. To
            color according to different data scalar ``discretisedfield.Field`` or
            ``xarray.DataArray`` can be used to color the arrows. This option has no
            effect when ``use_color=False``. If not passed the out-of-plane component is
            used for 3d vector fields defined in 3d space. Otherwise, a warning is show
            and automatic coloring is disabled.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            roi is non-zero are included in the output.

        n : array_like, dict, optional

            Re-sampling of the array with the given number of points. If an array-like
            is passed it must have length 2 and the values are used for the two kdims.
            If a dictionary is passed its keys must correspond to (some of) the
            dimensions of the array. If not specified no re-sampling is done.

        use_color : bool, optional

            If ``True`` the field is colored according to the out-of-plane component. If
            ``False`` all arrows have a uniform color, by default black. To change the
            uniform color pass e.g.``color= 'blue'``. Defaults to ``True``.

        colorbar_label : str, optional

            Label to show on the colorbar.

        kwargs

            Additional keyword arguments that are forwarded to
            ``holoviews.VectorField.opts()``.

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a ``holoviews.Image`` for each
            slider value.

        Raises
        ------
        ValueError

            If ``kdims`` has not length 2 or contains strings that are not part of the
            objects dimensions (``'x'``, ``'y'``, or ``'z'`` for standard
            discretisedfield.Field objects).

            If the object has no dimension ``comp`` that defines the vector components.

            If a plot cannot be created without specifying ``vdims`` or if the ``vdims``
            are not correct.

        Examples
        --------
        1. Simple vector plot with ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 3))
        ...
        >>> field.hv.vector(kdims=['x', 'y'])
        :DynamicMap...

        """
        self._check_kdims(kdims)
        x, y = kdims
        if "comp" not in self.array.dims:
            raise ValueError(
                "The vector plot method can only operate on DataArrays which have a"
                " vector component called 'comp'."
            )
        if (self.array.ndim != 4 or len(self.array.comp) != 3) and vdims is None:
            raise ValueError(
                f"`vdims` are required for arrays with {self.array.ndim - 1} spatial"
                " dimensions and {len(self.array.comp)} components."
            )

        if vdims is None:
            arrow_x = self.array.coords["comp"].values[dfu.axesdict[x]]
            arrow_y = self.array.coords["comp"].values[dfu.axesdict[y]]
            vdims = [arrow_x, arrow_y]
        else:
            if len(vdims) != 2:
                raise ValueError(f"{vdims=} must contain two elements.")
            arrow_x, arrow_y = vdims
            if arrow_x is None and arrow_y is None:
                raise ValueError(f"At least one element of {vdims=} must be not None.")

        # vector field norm
        filter_values = xr.apply_ufunc(
            np.linalg.norm, self.array, input_core_dims=[["comp"]], kwargs={"axis": -1}
        )
        filter_values = self._filter_values(filter_values, roi, kdims)
        mag = np.sqrt(
            (self.array.sel(comp=arrow_x) ** 2 if arrow_x else 0)
            + (self.array.sel(comp=arrow_y) ** 2 if arrow_y else 0)
        )
        ip_vector = xr.Dataset(
            {
                "angle": np.arctan2(
                    self.array.sel(comp=arrow_y) if arrow_y else 0,
                    self.array.sel(comp=arrow_x) if arrow_x else 0,
                    where=np.logical_and(filter_values != 0, ~np.isnan(filter_values)),
                    out=np.full(
                        np.array(self.array.shape)[
                            np.array(self.array.dims) != "comp"  # comp at any position
                        ],
                        np.nan,
                    ),
                ),
                "mag": mag / np.max(np.abs(mag)),
            }
        )

        if use_color:
            if cdim is not None:
                if isinstance(cdim, str):
                    color_comp = self.array.sel(comp=cdim)
                    if colorbar_label is None:
                        colorbar_label = cdim
                elif isinstance(cdim, xr.DataArray):
                    color_comp = cdim
                else:
                    color_comp = cdim.to_xarray()

                if colorbar_label is None:
                    with contextlib.suppress(AttributeError):
                        colorbar_label = color_comp.name
                ip_vector["color_comp"] = color_comp
            else:
                if len(self.array.comp) != 3:  # 3 spatial components + vector 'comp'
                    warnings.warn(
                        "Automatic coloring is only supported for 3d"
                        f' vector arrays. Ignoring "{use_color=}".'
                    )
                    use_color = False
                else:
                    c_comp = (set(self.array.comp.to_numpy()) - set(vdims)).pop()
                    if colorbar_label is None:
                        colorbar_label = c_comp
                    ip_vector["color_comp"] = self.array.sel(comp=c_comp)

        vdims = ["angle", "mag"]
        kwargs.setdefault("data_aspect", 1)
        if use_color:  # can be disabled at this point for 2d arrays
            vdims.append("color_comp")
            kwargs.setdefault("colorbar", True)

        ip_vector = self._resample(ip_vector, kdims, n)

        def _vectorplot(*values):
            plot = hv.VectorField(
                data=ip_vector.sel(
                    **dict(zip(dyn_kdims, values)), method="nearest"
                ).squeeze(),
                kdims=kdims,
                vdims=vdims,
            )
            plot.opts(magnitude="mag", **kwargs)
            if use_color:
                plot.opts(color="color_comp")
                if colorbar_label:
                    plot.opts(clabel=colorbar_label)
            for dim in plot.dimensions():
                if dim.name in "xyz":
                    with contextlib.suppress(AttributeError):
                        dim.unit = ip_vector[dim.name].units
            return plot

        dyn_kdims = [
            dim
            for dim in self.array.dims
            if dim not in kdims + ["comp"] and len(ip_vector[dim]) > 1
        ]
        dyn_map = hv.DynamicMap(_vectorplot, kdims=dyn_kdims).redim.values(
            **{dim: ip_vector[dim].data for dim in dyn_kdims}
        )
        # redim does not work with xarray DataArrays
        for dim in dyn_map.dimensions():
            with contextlib.suppress(AttributeError):
                dim.unit = self.array[dim.name].units
        return dyn_map

    def contour(self, kdims, roi=None, n=None, **kwargs):
        """Plot contour lines of scalar fields or vector components.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.Contours``. The plot shows the plane defined with the two spatial
        directions passed to ``kdims``. If a vector field is passed (that means the
        field dimension is greater than 1) an additional ``panel.Select`` widget for the
        field components is created automatically. It is not necessary to create a
        cut-plane first.

        To filter out parts of the plot (e.g. areas where the norm of the field is zero)
        an additional ``roi`` can be passed. It can take an ``xarray.DataArray`` or a
        ``discretisedfield.Field`` and hides all points where ``roi`` is 0. It relies on
        ``xarray``s broadcasting and the object passed to ``roi`` must only have the
        same dimensions as the ones specified as ``kdims``. No automatic filtering is
        applied.

        To reduce the number of points in the plot a simple re-sampling is available.
        The parameter ``n`` can be used to specify the number of points in different
        directions. A tuple of length 2 can be used to specify the number of points in
        the two ``kdims``. A dictionary can be used to specify the number of points in
        arbitrary dimensions. Keys of the dictionary must be dimensions of the array.
        Dimensions that are not specified are not modified. Note, that the re-sampling
        method is very basic and does not do any sort of interpolation (it just picks
        the nearest point). The extreme points in each direction are always kept.
        Equidistant points are picked in between.

        Additional keyword arguments are directly forwarded to
         ``harray.hvplot.contour``. Please refer to the documentation of ``hvplot`` (and
         ``holoviews``) for available options and additional documentation on how to
         modify the plot after creation.

        Parameters
        ----------
        kdims : List[str]

            Array coordinates plotted in plot x and plot y directon.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            roi is non-zero are included in the output.

        n : array_like, dict, optional

            Re-sampling of the array with the given number of points. If an array-like
            is passed it must have length 2 and the values are used for the two kdims.
            If a dictionary is passed its keys must correspond to (some of) the
            dimensions of the array. If not specified no re-sampling is done.

        kwargs

            Additional keyword arguments that are forwarded to
            ``xarray.hvplot.contour``.

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a ``holoviews.Image`` for each
            slider value.

        Raises
        ------
        ValueError

            If ``kdims`` has not length 2 or contains strings that are not part of the
            objects dimensions (``'x'``, ``'y'``, or ``'z'`` for standard
            discretisedfield.Field objects).

        Examples
        --------
        1. Simple contour-line plot with ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
        ...
        >>> field.hv.contour(kdims=['y', 'z'])
        :DynamicMap...

        """
        x, y, kwargs = self._prepare_scalar_plot(kdims, roi, n, kwargs)
        return self.array.hvplot.contour(x=x, y=y, **kwargs)

    def _filter_values(self, values, roi, kdims):
        if roi is None:
            return values

        if not isinstance(roi, xr.DataArray):
            roi = roi.to_xarray()

        for kdim in kdims:
            if kdim not in roi.dims:
                raise KeyError(f"Missing dim {kdim} in the filter.")
            if len(self.array[kdim].data) != len(roi[kdim].data) or not np.allclose(
                self.array[kdim].data, roi[kdim].data
            ):
                raise ValueError(f"Coordinates for dim {kdim} do not match.")

        return values.where(roi != 0)

    def _check_kdims(self, kdims):
        if len(kdims) != 2:
            raise ValueError(f"{kdims=} must have length 2.")
        for dim in kdims:
            if dim not in self.array.dims:
                raise ValueError(
                    f"Unknown dimension {dim=} in kdims; must be in {self.array.dims}."
                )

    def _prepare_scalar_plot(self, kdims, roi, n, kwargs):
        self._check_kdims(kdims)
        x, y = kdims
        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)
        self.array = self._filter_values(self.array, roi, kdims)
        self.array = self._resample(self.array, kdims, n).squeeze()
        return x, y, kwargs

    def _resample(self, array, kdims, n):
        if n is None:
            return array
        elif isinstance(n, tuple):
            if len(n) != 2:
                raise ValueError(f"{len(n)=} must be 2 if a tuple is passed.")
            vals = {
                dim: np.linspace(array[dim].min(), array[dim].max(), ni)
                for dim, ni in zip(kdims, n)
            }
        elif isinstance(n, dict):
            vals = {
                dim: np.linspace(array[dim].min(), array[dim].max(), ni)
                for dim, ni in n.items()
            }
        else:
            raise TypeError(
                f"Invalid type {type(n)} for parameter n. Must be tuple or dict."
            )
        resampled = array.sel(**vals, method="nearest")
        resampled = resampled.assign_coords(vals)
        for dim in vals.keys():
            with contextlib.suppress(AttributeError):
                resampled[dim]["units"] = array[dim].units
        return resampled
