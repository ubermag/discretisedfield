"""Holoviews-based plotting."""
import warnings

import holoviews as hv
import numpy as np
import xarray as xr

import discretisedfield as df
import discretisedfield.util as dfu


class HvplotField:
    """Holoviews-based plotting methods.

    Plotting with holoviews can be created without prior slicing. Instead, a slider
    is created for the out-of-plane direction. This class should not be accessed
    directly. Use ``field.hvplot`` to use the different plotting methods.

    Parameters
    ----------
    field : df.Field

        Field to plot.

    .. seealso::

        py:func:`~discretisedfield.Field.hvplot`

    """

    def __init__(self, field):
        import hvplot.xarray  # noqa, delayed import because it creates (empty) output

        self.field = field
        self.xrfield = field.to_xarray()

    def __call__(self, slider, multiplier=None, scalar_kw=None, vector_kw=None):
        """Plot scalar and vector components on a plane.

        This is a convenience method for quick plotting. It combines
         ``Field.hvplot.scalar`` and ``Field.hvplot.vector``. Depending on the
         dimensionality of the field's value, it automatically determines what plot is
         going to be shown. For a scalar field, only
         ``discretisedfield.plotting.HvplotField.scalar`` is used, whereas for a vector
         field, both ``discretisedfield.plotting.HvplotField.scalar`` and
         ``discretisedfield.plotting.HvplotField.vector`` plots are shown so that vector
         plot visualises the in-plane components of the vector and scalar plot encodes
         the out-of-plane component. The field is shown on a plane normal to the
         ``slider`` direction.

        All the default values can be changed by passing dictionaries to
        ``scalar_kw`` and ``vector_kw``, which are then used in subplots.

        Therefore, to understand the meaning of the keyword arguments which can be
        passed to this method, please refer to
        ``discretisedfield.plotting.HvplotField.scalar`` and
        ``discretisedfield.plotting.HvplotField.vector`` documentation. Filtering of the
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
            ``discretisedfield.plotting.HvplotField.scalar``

        vector_kw : dict

            Additional keyword arguments that are
            ``discretisedfield.plotting.HvplotField.vector``

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a combined plot for each slider
            value.

        Examples
        --------
        1. Simple combined scalar and vector plot with ``hvplot``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
        ...
        >>> field.hvplot(slider='z')
        :DynamicMap...

        """
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")

        scalar_kw = {} if scalar_kw is None else scalar_kw.copy()
        vector_kw = {} if vector_kw is None else vector_kw.copy()
        scalar_kw.setdefault("filter_field", self.field.norm)

        scalar_kw["multiplier"] = multiplier
        vector_kw["multiplier"] = multiplier
        vector_kw.setdefault("use_color", False)

        if self.field.dim == 1:
            return self.field.hvplot.scalar(slider, **scalar_kw)
        elif self.field.dim == 2:
            return self.field.hvplot.vector(slider, **vector_kw)
        elif self.field.dim == 3:
            scalar = getattr(
                self.field, self.field.components[dfu.axesdict[slider]]
            ).hvplot.scalar(slider, **scalar_kw)
            vector = self.field.hvplot.vector(slider, **vector_kw)
            return scalar * vector

    def scalar(self, slider, multiplier=None, filter_field=None, **kwargs):
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
        slider : str

            Spatial direction for which a slider is created. Can be one of ``'x'``,
            ``'y'``, or ``'z'``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

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
        1. Simple scalar plot with ``hvplot``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
        ...
        >>> field.hvplot.scalar(slider='z')
        :DynamicMap...

        """
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))
        groupby = [slider] if self.field.dim == 1 else [slider, "comp"]
        multiplier = self._setup_multiplier(multiplier)

        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)
        self._filter_values(filter_field, self.xrfield)
        self.xrfield = dfu.rescale_xarray(self.xrfield, multiplier)
        return self.xrfield.hvplot(x=x, y=y, groupby=groupby, **kwargs)

    def vector(
        self,
        slider,
        vdims=None,
        multiplier=None,
        filter_field=None,
        use_color=True,
        color_field=None,
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

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

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
        1. Simple vector plot with ``hvplot``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 3))
        ...
        >>> field.hvplot.vector(slider='z')
        :DynamicMap...

        """
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        if self.field.dim != 3 and vdims is None:
            raise ValueError(f"vdims are required for {self.field.dim=} field.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))
        multiplier = self._setup_multiplier(multiplier)

        if vdims is None:
            arrow_x = self.field.components[dfu.axesdict[x]]
            arrow_y = self.field.components[dfu.axesdict[y]]
        else:
            if len(vdims) != 2:
                raise ValueError(f"{vdims=} must contain two elements.")
            arrow_x, arrow_y = vdims
            if arrow_x is None and arrow_y is None:
                raise ValueError(f"At least one element of {vdims=} must be not None.")

        filter_values = self.field.norm.to_xarray()
        self._filter_values(filter_field, filter_values)
        self.xrfield = dfu.rescale_xarray(self.xrfield, multiplier)
        ip_vector = xr.Dataset(
            {
                "angle": np.arctan2(
                    self.xrfield.sel(comp=arrow_y) if arrow_y else 0,
                    self.xrfield.sel(comp=arrow_x) if arrow_x else 0,
                    where=np.logical_and(filter_values != 0, ~np.isnan(filter_values)),
                    out=np.full(self.field.mesh.n, np.nan),
                ),
                "mag": np.sqrt(
                    self.xrfield.sel(comp=arrow_x) ** 2
                    if arrow_x
                    else 0 + self.xrfield.sel(comp=arrow_y) ** 2
                    if arrow_y
                    else 0
                ),
            }
        )
        vdims = ["angle", "mag"]
        kwargs.setdefault("data_aspect", 1)

        if use_color:
            if color_field:
                ip_vector["color_comp"] = dfu.rescale_xarray(
                    color_field.to_xarray(), multiplier
                )
            else:
                if self.field.dim != 3:
                    warnings.warn(
                        "Automatic coloring is only supported for 3d"
                        f' fields. Ignoring "{use_color=}".'
                    )
                    use_color = False
                else:
                    ip_vector["color_comp"] = self.xrfield.isel(
                        comp=dfu.axesdict[slider]
                    )
        if use_color:  # can be disabled at this point for 2d fields
            vdims.append("color_comp")
            kwargs.setdefault("colorbar", True)

        def _vectorplot(val):
            plot = hv.VectorField(
                data=ip_vector.sel(**{slider: val, "method": "nearest"}),
                kdims=[x, y],
                vdims=vdims,
            )
            plot.opts(magnitude="mag", **kwargs)
            if use_color:
                plot.opts(color="color_comp")
            for dim in plot.dimensions():
                if dim.name in "xyz":
                    try:
                        dim.unit = ip_vector[dim.name].units
                    except AttributeError:
                        pass
            return plot

        dyn_map = hv.DynamicMap(_vectorplot, kdims=slider).redim.values(
            **{slider: getattr(self.field.mesh.midpoints, slider) / multiplier}
        )
        for dim in dyn_map.dimensions():
            if dim.name == slider:
                try:
                    dim.unit = ip_vector[dim.name].units
                except AttributeError:
                    pass
        return dyn_map

    def contour(self, slider, multiplier=None, filter_field=None, **kwargs):
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

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

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
        1. Simple contour-line plot with ``hvplot``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
        ...
        >>> field.hvplot.contour(slider='z')
        :DynamicMap...

        """
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))
        groupby = [slider] if self.field.dim == 1 else [slider, "comp"]
        multiplier = self._setup_multiplier(multiplier)

        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)
        self._filter_values(filter_field, self.xrfield)
        self.xrfield = dfu.rescale_xarray(self.xrfield, multiplier)
        return self.xrfield.hvplot.contour(x=x, y=y, groupby=groupby, **kwargs)

    def _filter_values(self, filter_field, values):
        if filter_field is None:
            return values

        if filter_field.dim != 1:
            raise ValueError(f"Cannot use {filter_field.dim=}.")

        if self.field.mesh.region not in filter_field.mesh.region:
            raise ValueError(
                "The filter_field region does not contain the field;"
                f" {filter_field.mesh.region=}, {self.field.mesh.region=}."
            )

        if not filter_field.mesh | self.field.mesh:
            filter_field = df.Field(self.field.mesh, dim=1, value=filter_field)
        values.data[filter_field.to_xarray().data == 0] = np.nan

    def _setup_multiplier(self, multiplier):
        return self.field.mesh.region.multiplier if multiplier is None else multiplier
