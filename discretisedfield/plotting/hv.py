"""Holoviews-based plotting."""
import warnings

import holoviews as hv
import numpy as np
import xarray as xr

import discretisedfield as df
import discretisedfield.util as dfu


class Hv:
    """Holoviews-based plotting methods.

    Plotting with holoviews can be created without prior slicing. Instead, a slider
    is created for the out-of-plane direction. This class should not be accessed
    directly. Use ``field.hv`` to use the different plotting methods.

    Parameters
    ----------
    array : xarray.DataArray

        DataArray to plot.

    """

    def __init__(self, array):
        import hvplot.xarray  # noqa, delayed import because it creates (empty) output

        self.array = array

    def __call__(self, slider, scalar_kw=None, vector_kw=None):
        """Plot scalar and vector components on a plane.

        This is a convenience method for quick plotting. It combines
         ``Field.hv.scalar`` and ``Field.hv.vector``. Depending on the
         dimensionality of the field's value, it automatically determines what plot is
         going to be shown. For a scalar field, only
         ``discretisedfield.plotting.Hv.scalar`` is used, whereas for a vector
         field, both ``discretisedfield.plotting.Hv.scalar`` and
         ``discretisedfield.plotting.Hv.vector`` plots are shown so that vector
         plot visualises the in-plane components of the vector and scalar plot encodes
         the out-of-plane component. The field is shown on a plane normal to the
         ``slider`` direction.

        All the default values can be changed by passing dictionaries to
        ``scalar_kw`` and ``vector_kw``, which are then used in subplots.

        Therefore, to understand the meaning of the keyword arguments which can be
        passed to this method, please refer to
        ``discretisedfield.plotting.Hv.scalar`` and
        ``discretisedfield.plotting.Hv.vector`` documentation. Filtering of the
        scalar component is applied by default (using the norm for vector fields,
        absolute values for scalar fields). To turn off filtering add ``{'filter_field':
        None}`` to ``scalar_kw``.


        Parameters
        ----------
        slider : str

            Spatial direction for which a slider is created. Can be one of ``'x'``,
            ``'y'``, or ``'z'``.

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
        >>> field.hv(slider='z')
        :DynamicMap...

        """
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")

        scalar_kw = {} if scalar_kw is None else scalar_kw.copy()
        vector_kw = {} if vector_kw is None else vector_kw.copy()
        scalar_kw.setdefault("filter_field", self.field.norm)

        vector_kw.setdefault("use_color", False)

        if self.field.dim == 1:
            return self.field.hv.scalar(slider, **scalar_kw)
        elif self.field.dim == 2:
            return self.field.hv.vector(slider, **vector_kw)
        elif self.field.dim == 3:
            scalar_comp = self.field.components[dfu.axesdict[slider]]
            scalar = getattr(self.field, scalar_comp).hv.scalar(
                slider, clabel=f"{scalar_comp}-component", **scalar_kw
            )
            vector = self.field.hv.vector(slider, **vector_kw)
            return scalar * vector

    def scalar(self, kdims, roi=None, **kwargs):
        """Plot the scalar field on a plane.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.Image`` objects. The field is shown on a plane normal to the
        ``slider`` direction. If the field dimension is greater than 1 an additional
        ``panel.Select`` widget for the field components is created automatically.
        It is not necessary to create a cut-plane first.

        To filter out parts of the field (e.g. areas where the norm of the field is
        zero) an additional ``filter_field`` can be passed. No automatic filtering is
        applied.

        The method internally creates an ``xarray`` of the field and uses
        ``xarray.hvplot``. Additional keyword arguments are directly forwarded to this
        method. Please refer to the documentation of ``hvplot`` (and ``holoviews``) for
        available options and additional documentation on how to modify the plot after
        creation.

        Parameters
        ----------
        kdims : List[str]

            Array coordinates plotted in plot x and plot y directon.

        filter_field : df.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            filter_field is non-zero are included in the output.

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

            If ``slider`` is not ``'x'``, ``'y'``, or ``'z'``.

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
        >>> field.hv.scalar(slider='z')
        :DynamicMap...

        """
        x, y, kwargs = self._prepare_scalar_plot(kdims, roi, kwargs)
        return self.array.hvplot(x=x, y=y, **kwargs)

    def vector(
        self,
        kdims,
        vdims=None,
        cdim=None,
        roi=None,
        use_color=True,
        colorbar_label=None,
        **kwargs,
    ):
        """Plot the vector field on a plane.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.VectorField`` objects. The field is shown on a plane normal to the
        ``slider`` direction. The length of the arrows corresponds to the norm of the
        in-plane component of the field, the color to the out-of-plane component. It is
        not necessary to create a cut-plane first.

        To filter out parts of the field (e.g. areas where the norm of the field is
        zero) an additional ``filter_field`` can be passed. Parts of the field where the
        norm is zero are not shown by default.

        The method internally creates an ``xarray`` of the field and uses
        ``holoviews.VectorPlot``. Additional keyword arguments are directly forwarded to
        the ``.opts()`` method of this object. Please refer to the documentation of
        ``holoviews`` for available options and additional documentation on how to
        modify the plot after creation.

        Parameters
        ----------
        slider : str

            Spatial direction for which a slider is created. Can be one of ``'x'``,
            ``'y'``, or ``'z'``.

        vdims : List[str], optional

            Names of the components to be used for the x and y component of the plotted
            arrows. This information is used to associate field components and spatial
            directions. Optionally, one of the list elements can be ``None`` if the
            field has no component in that direction. ``vdims`` is required for 2d
            vector fields.

        filter_field : df.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            filter_field is non-zero are included in the output.

        use_color : bool, optional

            If ``True`` the field is colored according to the out-of-plane component. If
            ``False`` all arrows have a uniform color, by default black. To change the
            uniform color pass e.g.``color= 'blue'``. Defaults to ``True``.

        color_field : df.Field, optional

            Scalar field that is used to color the arrows when ``use_color=True``. If
            not passed the out-of-plane component is used for coloring.

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

            If ``slider`` is not ``'x'``, ``'y'``, or ``'z'`` or if the field is scalar
            or has dim > 3.

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
        >>> field.hv.vector(slider='z')
        :DynamicMap...

        """
        self._check_kdims(kdims)
        x, y = kdims
        if "comp" not in self.array.dims:
            raise ValueError(
                "The vector plot method can only operate on DataArrays which have a"
                " vector component called 'comp'."
            )
        if self.array.ndim != 4 and vdims is None:
            raise ValueError(f'`vdims` are required for arrays with {self.array.ndim - 1} spatial dimensions.')

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
        self._filter_values(roi, filter_values)
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
        vdims = ["angle", "mag"]
        kwargs.setdefault("data_aspect", 1)

        if use_color:
            if cdim:
                if isinstance(cdim, str):
                    color_field = self.array.sel(comp=cdim)
                    if colorbar_label is None:
                        colorbar_label = cdim
                elif isinstance(color_field, xr.DataArray):
                    color_field = cdim
                else:
                    color_field = color_field.to_xarray()

                if colorbar_label is None:
                    try:
                        colorbar_label = color_field.name
                    except AttributeError:
                        pass
                ip_vector["color_comp"] = color_field
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
        if use_color:  # can be disabled at this point for 2d arrays
            vdims.append("color_comp")
            kwargs.setdefault("colorbar", True)

        def _vectorplot(*values):
            plot = hv.VectorField(
                data=ip_vector.sel(**dict(zip(dyn_kdims, values)), method="nearest"),
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
                    try:
                        dim.unit = ip_vector[dim.name].units
                    except AttributeError:
                        pass
            return plot

        dyn_kdims = [dim for dim in self.array.dims if dim not in kdims + ["comp"]]
        dyn_map = hv.DynamicMap(_vectorplot, kdims=dyn_kdims).redim.values(
            **{dim: self.array[dim].data for dim in dyn_kdims}  # data / multiplier
        )
        # redim does not work with xarrays
        for dim in dyn_map.dimensions():
            try:
                dim.unit = self.array[dim.name].units
            except AttributeError:
                pass
        return dyn_map

    def contour(self, kdims, filter_field=None, **kwargs):
        """Plot the scalar field on a plane.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.Contours`` objects. The field is shown on a plane normal to the
        ``slider`` direction. If the field dimension is greater than 1 an additional
        ``panel.Select`` widget for the field components is created automatically.
        It is not necessary to create a cut-plane first.

        To filter out parts of the field (e.g. areas where the norm of the field is
        zero) an additional ``filter_field`` can be passed. No automatic filtering is
        applied.

        The method internally creates an ``xarray`` of the field and uses
        ``xarray.hvplot.contour``. Additional keyword arguments are directly forwarded
        to this method. Please refer to the documentation of ``hvplot`` (and
        ``holoviews``) for available options and additional documentation on how to
        modify the plot after creation.

        Parameters
        ----------
        slider : str

            Spatial direction for which a slider is created. Can be one of ``'x'``,
            ``'y'``, or ``'z'``.

        filter_field : df.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            filter_field is non-zero are included in the output.

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

            If ``slider`` is not ``'x'``, ``'y'``, or ``'z'``.

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
        >>> field.hv.contour(slider='z')
        :DynamicMap...

        """
        x, y, kwargs = self._prepare_scalar_plot(kdims, filter_field, kwargs)
        return self.array.hvplot.contour(x=x, y=y, **kwargs)

    def _filter_values(self, values, roi, kdims):
        if roi is None:
            return values

        if not isinstance(roi, xr.DataArray):
            roi = roi.to_xarray()

        for kdim in kdims:
            if kdim not in roi.dims:
                raise KeyError(f'Missing dim {kdim} in the filter.')
        #roi = roi.sel(**{dim: self.array[dim].data for dim in kdims}, method="nearest")
        #roi.reset_dimensions
        #return roi
        # TODO this needs to be fixed
        # if self.field.mesh.region not in filter_field.mesh.region:
        #     raise ValueError(
        #         "The filter_field region does not contain the field;"
        #         f" {filter_field.mesh.region=}, {self.field.mesh.region=}."
        #     )

        # if not filter_field.mesh | self.field.mesh:
        #     filter_field = df.Field(self.field.mesh, dim=1, value=filter_field)
        # values.data[filter_field.to_xarray().data == 0] = np.nan
        return values.where(roi != 0)


    def _check_kdims(self, kdims):
        if len(kdims) != 2:
            raise ValueError(f"{kdims=} must have length 2.")
        for dim in kdims:
            if dim not in self.array.dims:
                raise ValueError(
                    f"Unknown dimension {dim=} in kdims; must be in {self.array.dims}."
                )

    def _prepare_scalar_plot(self, kdims, filter_field, kwargs):
        self._check_kdims(kdims)
        x, y = kdims
        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)
        self.array = self._filter_values(self.array, filter_field, kdims)
        return x, y, kwargs
