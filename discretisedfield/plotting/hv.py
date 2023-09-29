"""Holoviews-based plotting."""
import contextlib
import copy
import functools
import warnings

import holoviews as hv
import numpy as np
import xarray as xr

import discretisedfield as df

from .util import hv_key_dim

# HoloViews shows a warning about a deprecated call to unique when creating
# the dynamic map.
# The developers have confirmed, that this warning can be ignored
# and the problem will be fixed in the next HoloViews release.
# https://discourse.holoviz.org/t/futurewarning-when-creating-a-dynamicmap/6108
# The warnings filtering can be removed once this is fixed.
warnings.filterwarnings(
    "ignore",
    message="unique with argument",
    category=FutureWarning,
    module="holoviews.core.util",
)


class Hv:
    """Holoviews-based plotting methods.

    Plots based on Holoviews can be created without prior slicing. Instead, a slider is
    created for the directions not shown in the plot. This class should not be accessed
    directly. Use ``field.hv`` to use the different plotting methods.

    Data in the plots is filtered based on ``field.valid``. Only cells where
    ``valid==True`` are visible in the plots.

    Parameters
    ----------
    key_dims : dict[df.plotting.util.hv_key_dim]

        Key dimensions of the plot (kdims and dynamic kdims [for which holoviews will
        create widgets]) in a dictionary. The keys are the names of the dimensions,
        values namedtuples containing the data and unit (can be an empty string) of the
        dimensions.

    callback : callable

        Callback function to provide data. It must accept arbitrary keyword arguments
        and will be called with all dynamic kdims and their current values.

    vdims_guess_callback : callable, optional

       Callback function to provide (guess) vdims for the __call__ method if no vdims
       are passed in. This method is required because the order of key_dims is not
       defined.

    """

    def __init__(self, key_dims, callback, vdim_guess_callback=None):
        # no tests for key_dims and callback as the class is not directly used by users
        if not hv.extension._loaded:
            hv.extension("bokeh", logo=False)
        self.key_dims = key_dims
        self.callback = callback
        self.vdim_guess_callback = vdim_guess_callback

    def __call__(
        self,
        kdims,
        vdims=None,
        roi=None,
        scalar_kw=None,
        vector_kw=None,
    ):
        """Create an optimal plot combining ``hv.scalar`` and ``hv.vector``.

        This is a convenience method for quick plotting. It combines ``hv.scalar`` and
         ``hv.vector``. Depending on the dimensionality of the object, it automatically
         determines the type of plot in the following order:

        1. For scalar objects (no dimension with name ``'vdims'``) only
           ``discretisedfield.plotting.Hv.scalar`` is used. The parameter ``vdims`` is
           ignored.

        2. When ``vdims`` is specified a vector plot (without coloring) is created for
           the "in-plane" part of the vector field (defined via ``vdims``). If the field
           has vector dimensionality larger than two an additional scalar plot is
           created for all remaining vector components (with a drop-down selection).

        3. If ``vdims`` is not specified the method tries to guess the correct ``vdims``
           from the ``kdims`` by matching spatial coordinates and vector components
           based on the order they are defined in. This only works if both have the same
           number of elements, e.g. a 3d vector field in 3d space.

        4. For all other vector fields a scalar plot with a drop-down selection for the
           individual vector components is created.

        Based on the norm of the object (absolute values for scalar fields) automatic
        filtering is applied, i.e. all cells where the norm is zero are excluded from
        the plot. To manually filter out parts of the plot (e.g. areas where the norm of
        the field is zero) an additional ``roi`` can be passed. It can take an
        ``xarray.DataArray`` or a ``discretisedfield.Field`` and hides all points where
        ``roi`` is 0. It relies on ``xarray``s broadcasting and the object passed to
        ``roi`` must only have the same dimensions as the ones specified as ``kdims``.

        All default values of ``hv.scalar`` and ``hv.vector`` can be changed by passing
        dictionaries to ``scalar_kw`` and ``vector_kw``, which are then used in
        subplots.
        To understand the meaning of the keyword arguments which can be passed to this
        method, please refer to ``discretisedfield.plotting.Hv.scalar`` and
        ``discretisedfield.plotting.Hv.vector`` documentation.

        To reduce the number of points in the plot a simple re-sampling is available.
        The parameter ``n`` in ``scalar_kw`` or ``vector_kw`` can be used to specify the
        number of points in different directions. A tuple of length 2 can be used to
        specify the number of points in the two ``kdims``. Note, that the re-sampling
        method is very basic and does not do any sort of interpolation (it just picks
        the nearest point). The extreme points in each direction are always kept.
        Equidistant points are picked in between.

        Parameters
        ----------
        kdims : List[str]

            Names of the two geometrical directions forming the plane to be used for
            plotting the data.

        vdims : List[str], optional

            Names of the components to be used for plotting the arrows. This information
            is used to associate field components and spatial directions. Optionally,
            one of the list elements can be ``None`` if the field has no component in
            that direction.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Filter out certain areas in the plot. Only cells where the roi is non-zero
            are included in the output.

        scalar_kw : dict

            Additional keyword arguments that are passed to
            ``discretisedfield.plotting.Hv.scalar``

        vector_kw : dict

            Additional keyword arguments that are passed to
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
        >>> field = df.Field(mesh, nvdim=1, value=2)
        ...
        >>> field.hv(kdims=['x', 'y'])
        :DynamicMap...

        """
        scalar_kw = {} if scalar_kw is None else scalar_kw.copy()
        vector_kw = {} if vector_kw is None else vector_kw.copy()

        vector_kw.setdefault("use_color", False)

        if "vdims" not in self.key_dims:
            return self.scalar(kdims=kdims, **scalar_kw)

        # try to guess vdims if not passed
        if vdims is None and self.vdim_guess_callback is not None:
            vdims = self.vdim_guess_callback(kdims)

        if vdims:
            scalar_comps = list(set(self.key_dims["vdims"].data) - set(vdims))
            with contextlib.suppress(KeyError):
                vector_kw.pop("vdims")
            vector = self.vector(kdims=kdims, vdims=vdims, **vector_kw)
            if len(scalar_comps) == 0:
                return vector
            else:
                key_dims = copy.deepcopy(self.key_dims)
                if len(scalar_comps) > 1:
                    key_dims["vdims"] = hv_key_dim(scalar_comps, key_dims["vdims"].unit)
                    callback = self.callback
                else:
                    # manually remove component 'vdims' from returned xarray to avoid
                    # a drop-down with one element (the out-of-plane component)
                    def callback(*args, **kwargs):
                        res = self.callback(*args, **kwargs)
                        return res.sel(vdims=scalar_comps[0]).drop_vars(
                            "vdims", errors="ignore"
                        )

                    key_dims.pop("vdims")

                scalar = self.__class__(key_dims, callback).scalar(
                    kdims=kdims, **scalar_kw
                )
                return scalar * vector
        else:
            return self.scalar(kdims=kdims, **scalar_kw)

    def scalar(self, kdims, roi=None, n=None, **kwargs):
        """Create an image plot for scalar data or individual components.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.Image`` objects. The plot shows the plane defined with the two
        spatial directions passed to ``kdims``. If a vector field is passed (that means
        the field dimension 'vdims' is greater than 1) an additional ``panel.Select``
        widget for the field components is created automatically. It is not necessary to
        create a cut-plane first.

        To filter out parts of the plot (e.g. areas where the norm of the field is zero)
        an additional ``roi`` can be passed. It can take an ``xarray.DataArray`` or a
        ``discretisedfield.Field`` and hides all points where ``roi`` is 0. It relies on
        ``xarray``s broadcasting and the object passed to ``roi`` must only have the
        same dimensions as the ones specified as ``kdims``. No automatic filtering is
        applied.

        To reduce the number of points in the plot a simple re-sampling is available.
        The parameter ``n`` can be used to specify the number of points in different
        directions. A tuple of length 2 can be used to specify the number of points in
        the two ``kdims``. Note, that the re-sampling method is very basic and does not
        do any sort of interpolation (it just picks the nearest point). The extreme
        points in each direction are always kept. Equidistant points are picked in
        between.

        Additional keyword arguments are directly forwarded to the ``.opts`` method of
        the resulting ``holoviews.Image``. Please refer to the documentation of
        ```holoviews`` (in particular ``holoviews.Image``) for available options and
        additional documentation on how to modify the plot after creation.

        Parameters
        ----------
        kdims : List[str]

            Names of the two geometrical directions forming the plane to be used for
            plotting the data.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            roi is non-zero are included in the output.

        n : array_like(2), optional

            Re-sampling of the array with the given number of points. If not specified
            no re-sampling is done.

        kwargs

            Additional keyword arguments that are forwarded to ``.opts`` of the
            ``holoviews.Image`` object.

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a ``holoviews.Image`` for each
            slider value.

        Raises
        ------
        ValueError

            If ``kdims`` does not have length 2 or contains strings that are not part of
            the geometrical directions of the field.

        Examples
        --------
        1. Simple scalar plot with ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=1, value=2)
        ...
        >>> field.hv.scalar(kdims=['x', 'z'])
        :DynamicMap...

        """
        self._check_kdims(kdims)
        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)

        dyn_kdims = [dim for dim in self.key_dims if dim not in kdims]

        roi = self._setup_roi(roi, kdims)
        self._check_n(n)

        def _plot(*values):
            data = self.callback(**dict(zip(dyn_kdims, values)))
            data = self._filter_values(
                data, roi, kdims, dyn_kdims=dict(zip(dyn_kdims, values))
            )
            data = self._resample(data, kdims, n)
            plot = hv.Image(data=data, kdims=kdims).opts(**kwargs)

            for dim in plot.kdims:
                dim.unit = self.key_dims[dim.name].unit

            return plot

        return hv.DynamicMap(_plot, kdims=dyn_kdims).redim.values(
            **{dim: self.key_dims[dim].data for dim in dyn_kdims}
        )

    def vector(
        self,
        kdims,
        vdims=None,
        cdim=None,
        roi=None,
        n=None,
        use_color=True,
        **kwargs,
    ):
        """Create a vector plot.

        This method creates a dynamic holoviews plot (``holoviews.DynamicMap``) based on
        ``holoviews.VectorField`` objects. The plot shows the plane defined with the two
        spatial directions passed to ``kdims``. The length of the arrows corresponds to
        the norm of the in-plane component of the field. It is not necessary to create a
        cut-plane first.

        ``vdims`` co-relates the vector directions with the geometrical direction
        defined with ``kdims``. If not specified, ``kdims`` is used to guess the values.

        For 3d vector fields the color by default encodes the out-of-plane component.
        Other fields cannot be colored automatically. To assign a non-uniform color
        manually use ``cdim``. It either accepts a string to select one of the field
        vector components or a scalar ``discretisedfield.Field`` or
        ``xarray.DataArray``. Coloring can be disabled with ``use_color=False``.

        To filter out parts of the plot (e.g. areas where the norm of the field is zero)
        an additional ``roi`` can be passed. It can take an ``xarray.DataArray`` or a
        ``discretisedfield.Field`` and hides all points where ``roi`` is zero. It relies
        on ``xarray``s broadcasting and the object passed to ``roi`` must only have the
        same dimensions as the ones specified as ``kdims``. No automatic filtering is
        applied.

        To reduce the number of points in the plot a simple re-sampling is available.
        The parameter ``n`` can be used to specify the number of points in different
        directions. A tuple of length 2 can be used to specify the number of points in
        the two ``kdims``. Note, that the re-sampling method is very basic and does not
        do any sort of interpolation (it just picks the nearest point). The extreme
        points in each direction are always kept. Equidistant points are picked in
        between.

        This method is based on ``holoviews.VectorPlot``. Additional keyword arguments
        are directly forwarded to the ``.opts()`` method of the ``holoviews.Image``
        object. Please refer to the documentation of ``holoviews`` (in particular
        ``holoviews.VectorField``) for available options and additional documentation on
        how to modify the plot after creation.

        Parameters
        ----------
        kdims : List[str]

            Names of the two geometrical directions forming the plane to be used for
            plotting the data.

        vdims : List[str], optional

            Names of the components to be used for plotting the arrows. This information
            is used to associate field components and spatial directions. Optionally,
            one of the list elements can be ``None`` if the field has no component in
            that direction. If ``vdims`` is not specified the method tries to guess the
            correct ``vdims`` from the ``kdims`` by matching spatial coordinates and
            vector components based on the order they are defined in. This only works if
            both have the same number of elements, e.g. a 3d vector field in 3d space.

        cdim : str, xarray.DataArray, discretisedfield.Field, optional

            A string can be used to select one of the vector components of the field. To
            color according to different data scalar ``discretisedfield.Field`` or
            ``xarray.DataArray`` can be used to color the arrows. This option has no
            effect when ``use_color=False``. If not passed the out-of-plane component is
            used for 3d vector fields. Otherwise, a warning is show and automatic
            coloring is disabled.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            roi is non-zero are included in the output.

        n : array_like, optional

            Re-sampling of the array with the given number of points. If not specified
            no re-sampling is done.

        use_color : bool, optional

            If ``True`` the field is colored according to the out-of-plane component. If
            ``False`` all arrows have a uniform color, by default black. To change the
            uniform color pass e.g.``color= 'blue'``. Defaults to ``True``.

        kwargs

            Additional keyword arguments that are forwarded to
            ``holoviews.VectorField.opts()``.

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a ``holoviews.VectorField`` for
            each slider value.

        Raises
        ------
        ValueError

            If ``kdims`` does not have length 2 or contains strings that are not part of
            the geometrical directions of the field.

            If the object has no dimension ``vdims`` that defines the vector components.

        Examples
        --------
        1. Simple vector plot with ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        ...
        >>> field.hv.vector(kdims=['x', 'y'], vdims=['x', 'y'])
        :DynamicMap...

        """
        self._check_kdims(kdims)
        if "vdims" not in self.key_dims:
            raise ValueError(
                "The vector plot method can only operate on data with a"
                " vector component called 'vdims'."
            )
        if cdim is not None:
            if not isinstance(cdim, str):
                raise TypeError("cdim must be of type str")
            elif cdim not in self.key_dims["vdims"].data:
                raise ValueError(f"The vector dimension {cdim} does not exist.")

        # try to guess vdims if not passed
        if vdims is None and self.vdim_guess_callback is not None:
            vdims = self.vdim_guess_callback(kdims)
        if vdims is None or len(vdims) != 2:
            raise ValueError(f"{vdims=} must contain two elements.")

        arrow_x, arrow_y = vdims
        if arrow_x is None and arrow_y is None:
            raise ValueError(f"At least one element of {vdims=} must be not None.")

        roi = self._setup_roi(roi, kdims)
        self._check_n(n)

        dyn_kdims = [dim for dim in self.key_dims if dim not in kdims + ["vdims"]]

        kwargs.setdefault("data_aspect", 1)

        def _plot(use_color, cdim, *values):
            # use_color and cdim have to be defined in here; otherwise an
            # UnboundLocalError is raised
            # roi, n, kdims, dyn_kdims, arrow_x, arrow_y, and kwargs work fine
            data = self.callback(**dict(zip(dyn_kdims, values)))
            data = self._filter_values(
                data, roi, kdims, dyn_kdims=dict(zip(dyn_kdims, values))
            )
            data = self._resample(data, kdims, n)

            vector_norm = xr.apply_ufunc(
                np.linalg.norm, data, input_core_dims=[["vdims"]], kwargs={"axis": -1}
            )
            vector_vdims = ["angle", "mag"]
            vector_data = {}
            vector_data["mag"] = np.sqrt(
                (data.sel(vdims=arrow_x) ** 2 if arrow_x else 0)
                + (data.sel(vdims=arrow_y) ** 2 if arrow_y else 0)
            )
            vector_data["angle"] = np.arctan2(
                data.sel(vdims=arrow_y) if arrow_y else 0,
                data.sel(vdims=arrow_x) if arrow_x else 0,
                where=np.logical_and(vector_norm != 0, ~np.isnan(vector_norm)).data,
                out=np.full(vector_norm.shape, np.nan),
            )

            if use_color and cdim is None:
                if len(data.vdims) == 3:
                    cdim = (set(data.vdims.to_numpy()) - set(vdims)).pop()
                else:
                    warnings.warn(
                        "Automatic coloring is only supported for 3d"
                        f' vector fields. Ignoring "{use_color=}".'
                    )
                    use_color = False

            if use_color:
                vector_vdims.append("color_comp")
                kwargs["color"] = "color_comp"
                vector_data["color_comp"] = data.sel(vdims=cdim).drop_vars(
                    "vdims", errors="ignore"
                )
                kwargs.setdefault("clabel", cdim)
                kwargs.setdefault("colorbar", True)

            plot = hv.VectorField(
                data=xr.Dataset(vector_data),
                kdims=kdims,
                vdims=vector_vdims,
            )
            plot.opts(magnitude="mag", **kwargs)

            for dim in plot.kdims:
                dim.unit = self.key_dims[dim.name].unit

            return plot

        return hv.DynamicMap(
            functools.partial(_plot, use_color, cdim), kdims=dyn_kdims
        ).redim.values(**{dim: self.key_dims[dim].data for dim in dyn_kdims})

    def contour(self, kdims, roi=None, n=None, levels=10, **kwargs):
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
        the two ``kdims``. Note, that the re-sampling method is very basic and does not
        do any sort of interpolation (it just picks the nearest point). The extreme
        points in each direction are always kept. Equidistant points are picked in
        between.

        Additional keyword arguments are directly forwarded to the ``.opts`` method of
         the ``holoviews.DynamicMap``. Please refer to the documentation of
         ``holoviews`` (in particular `holoviews.Contours``) for available options and
         additional documentation on how to modify the plot after creation.

        Parameters
        ----------
        kdims : List[str]

            Names of the two geometrical directions forming the plane to be used for
            plotting the data.

        roi : xarray.DataArray, discretisedfield.Field, optional

            Field to filter out certain areas in the plot. Only cells where the
            roi is non-zero are included in the output.

        n : array_like(2), optional

            Re-sampling of the array with the given number of points. If an array-like
            is passed it must have length 2 and the values are used for the two kdims.
            If not specified no re-sampling is done.

        levels : int, optional

            The number of contour lines, defaults to 10.

        kwargs

            Additional keyword arguments that are forwarded to ``.opts`` of the
            ``holoviews.DynamicMap`` object.

        Returns
        -------
        holoviews.DynamicMap

            A ``holoviews.DynamicMap`` that "creates" a ``holoviews.Contours`` object
            for each slider value.

        Raises
        ------
        ValueError

            If ``kdims`` has not length 2 or contains strings that are not part of the
            geometrical directions of the field.

        Examples
        --------
        1. Simple contour-line plot with ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=1, value=2)
        ...
        >>> field.hv.contour(kdims=['y', 'z'])
        :DynamicMap...

        """
        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)
        return hv.operation.contours(
            self.scalar(kdims, roi, n, colorbar=False), levels=levels
        ).opts(**kwargs)

    def _check_kdims(self, kdims):
        if len(kdims) != 2:
            raise ValueError(f"{kdims=} must have length 2.")
        for dim in kdims:
            if dim not in self.key_dims:
                raise ValueError(
                    f"Unknown dimension {dim=} in kdims; must be in"
                    f" {self.key_dims.keys()}."
                )

    def _setup_roi(self, roi, kdims):
        if roi is None:
            return None
        elif isinstance(roi, df.Field):
            roi = roi.to_xarray()
        elif callable(roi):
            # this has to come after the Field check because Field is callable
            return roi  # no checks can be performed without knowing slider values
        elif not isinstance(roi, xr.DataArray):
            raise TypeError(f"Unsupported type {type(roi)} for 'roi'.")

        if "vdims" in roi.dims:
            raise ValueError("Only scalar roi is supported.")

        for kdim in kdims:
            if kdim not in roi.dims:
                raise KeyError(f"Missing dim {kdim} in the filter.")
            if len(self.key_dims[kdim].data) != len(roi[kdim].data) or not np.allclose(
                self.key_dims[kdim].data, roi[kdim].data
            ):
                raise ValueError(f"Coordinates for dim {kdim} do not match.")

        extra_roi_dims = set(roi.dims) - set(self.key_dims)
        if len(extra_roi_dims) > 0:
            raise ValueError(
                f"Additional dimension(s) {extra_roi_dims} in roi are not supported."
            )

        for kdim in roi.dims:
            if kdim in kdims:
                continue
            if len(self.key_dims[kdim].data) != len(roi[kdim].data) or not np.allclose(
                self.key_dims[kdim].data, roi[kdim].data
            ):
                raise ValueError(f"Coordinates for dim {kdim} do not match.")

        return roi

    @staticmethod
    def _filter_values(values, roi, kdims, dyn_kdims):
        if roi is None:
            return values

        if callable(roi):
            roi_selection = copy.deepcopy(dyn_kdims)
            with contextlib.suppress(KeyError):
                roi_selection.pop("vdims")
            roi = roi(**roi_selection)
            if "vdims" in roi.dims:
                roi = xr.apply_ufunc(
                    np.linalg.norm,
                    roi,
                    input_core_dims=[["vdims"]],
                    kwargs={"axis": -1},
                ).drop_vars("vdims", errors="ignore")
            else:
                roi = np.abs(roi)

        for dyn_kdim, dyn_val in dyn_kdims.items():
            if dyn_kdim in roi.dims:
                # TODO add cell accuracy to method: nearest
                method = {} if isinstance(dyn_val, str) else {"method": "nearest"}
                roi = roi.sel(**{dyn_kdim: dyn_val}, **method).drop_vars(dyn_kdim)
        for dim in roi.dims:
            if dim not in dyn_kdims and len(roi[dim]) == 1:
                roi = roi.squeeze(dim=dim)

        assert len(roi.dims) == 2, (
            f"Additional dimension(s) {set(roi.dims) - set(kdims)} in roi are not"
            " supported."
        )

        return values.where(roi != 0)

    @staticmethod
    def _check_n(n):
        if n is None:
            return
        elif not isinstance(n, (tuple, list, np.ndarray)):
            raise TypeError(
                f"Invalid type {type(n)} for parameter n. Must be array-like."
            )
        elif len(n) != 2:
            raise ValueError(f"{len(n)=} must be 2.")

    @staticmethod
    def _resample(array, kdims, n):
        if n is None:
            return array

        vals = {
            dim: np.linspace(array[dim].min(), array[dim].max(), ni)
            for dim, ni in zip(kdims, n)
        }
        resampled = array.sel(**vals, method="nearest")
        resampled = resampled.assign_coords(vals)
        for dim in vals.keys():
            with contextlib.suppress(AttributeError):
                resampled[dim]["units"] = array[dim].units
        return resampled
