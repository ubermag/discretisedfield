import collections
import functools
import numbers

import numpy as np
import xarray as xr

import discretisedfield as df
import discretisedfield.plotting as dfp

from .field import Field


class CellField(Field):
    def __call__(self, point):
        return self.array[self.mesh.point2index(point)]

    # diff, integrate depending on how we calculate those for the VertexField

    @property
    def _hv_key_dims(self):
        """Dict of key dimensions of the field.

        Keys are the field dimensions (domain and vector space, e.g. x, y, z, vdims)
        that have length > 1. Values are named_tuples ``hv_key_dim(data, unit)`` that
        contain the data (which has to fulfil len(data) > 1, typically as a numpy array
        or list) and the unit of a string (empty string if there is no unit).

        """
        key_dims = {
            dim: dfp.util.hv_key_dim(coords, unit)
            for dim, unit in zip(self.mesh.region.dims, self.mesh.region.units)
            if len(coords := getattr(self.mesh.cells, dim)) > 1
        }
        if self.nvdim > 1:
            key_dims["vdims"] = dfp.util.hv_key_dim(self.vdims, "")
        return key_dims

    def line(self, p1, p2, n=100):
        points = list(self.mesh.line(p1=p1, p2=p2, n=n))
        values = [self(p) for p in points]

        return df.Line(
            points=points,
            values=values,
            point_columns=self.mesh.region.dims,
            value_columns=[f"v{dim}" for dim in self.vdims]
            if self.vdims is not None
            else "v",
        )  # TODO scalar fields have no vdim

    def __getitem__(self, item):
        submesh = self.mesh[item]

        index_min = self.mesh.point2index(
            submesh.index2point((0,) * submesh.region.ndim)
        )
        index_max = np.add(index_min, submesh.n)
        slices = [slice(i, j) for i, j in zip(index_min, index_max)]
        return self.__class__(
            submesh,
            nvdim=self.nvdim,
            value=self.array[tuple(slices)],
            vdims=self.vdims,
            unit=self.unit,
            valid=self.valid[tuple(slices)],
            vdim_mapping=self.vdim_mapping,
        )

    @property
    def mpl(self):
        """Plot interface, matplotlib based.

        This property provides access to the different plotting methods. It is
        also callable to quickly generate plots. For more details and the
        available methods refer to the documentation linked below.

        .. seealso::

            :py:func:`~discretisedfield.plotting.Mpl.__call__`
            :py:func:`~discretisedfield.plotting.Mpl.scalar`
            :py:func:`~discretisedfield.plotting.Mpl.vector`
            :py:func:`~discretisedfield.plotting.Mpl.lightness`
            :py:func:`~discretisedfield.plotting.Mpl.contour`

        Examples
        --------
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

        """
        return dfp.MplField(self)

    def to_xarray(self, name="field", unit=None):
        """Field value as ``xarray.DataArray``.

        The function returns an ``xarray.DataArray`` with the dimensions
        ``self.mesh.region.dims`` and ``vdims`` (only if ``field.nvdim > 1``). The
        coordinates of the geometric dimensions are derived from ``self.mesh.points``,
        and for vector field components from ``self.vdims``. Addtionally,
        the values of ``self.mesh.cell``, ``self.mesh.region.pmin``, and
        ``self.mesh.region.pmax`` are stored as ``cell``, ``pmin``, and ``pmax``
        attributes of the DataArray. The ``unit`` attribute of geometric
        dimensions is set to the respective strings in ``self.mesh.region.units``.

        The name and unit of the field ``DataArray`` can be set by passing
        ``name`` and ``unit``. If the type of value passed to any of the two
        arguments is not ``str``, then a ``TypeError`` is raised.

        Parameters
        ----------
        name : str, optional

            String to set name of the field ``DataArray``.

        unit : str, optional

            String to set units of the field ``DataArray``.

        Returns
        -------
        xarray.DataArray

            Field values DataArray.

        Raises
        ------
        TypeError

            If either ``name`` or ``unit`` argument is not a string.

        Examples
        --------
        1. Create a field

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(1, 0, 0), norm=1.)
        ...
        >>> field
        Field(...)

        2. Create `xarray.DataArray` from field

        >>> xa = field.to_xarray()
        >>> xa
        <xarray.DataArray 'field' (x: 10, y: 10, z: 10, vdims: 3)>
        ...

        3. Select values of `x` component

        >>> xa.sel(vdims='x')
        <xarray.DataArray 'field' (x: 10, y: 10, z: 10)>
        ...

        """
        if not isinstance(name, str):
            msg = "Name argument must be a string."
            raise TypeError(msg)

        if unit is not None and not isinstance(unit, str):
            msg = "Unit argument must be a string."
            raise TypeError(msg)

        axes = self.mesh.region.dims

        data_array_coords = {axis: getattr(self.mesh.cells, axis) for axis in axes}

        geo_units_dict = dict(zip(axes, self.mesh.region.units))

        if self.nvdim > 1:
            data_array_dims = axes + ("vdims",)
            if self.vdims is not None:
                data_array_coords["vdims"] = self.vdims
            field_array = self.array
        else:
            data_array_dims = axes
            field_array = np.squeeze(self.array, axis=-1)

        data_array = xr.DataArray(
            field_array,
            dims=data_array_dims,
            coords=data_array_coords,
            name=name,
            attrs=dict(
                units=unit or self.unit,
                cell=self.mesh.cell,
                pmin=self.mesh.region.pmin,
                pmax=self.mesh.region.pmax,
                nvdim=self.nvdim,
                tolerance_factor=self.mesh.region.tolerance_factor,
                data_location="cell",
            ),
        )

        # TODO save vdim_mapping

        for dim in geo_units_dict:
            data_array[dim].attrs["units"] = geo_units_dict[dim]

        return data_array

    @classmethod
    def from_xarray(cls, xa):
        """Create ``discretisedfield.Field`` from ``xarray.DataArray``

        The class method accepts an ``xarray.DataArray`` as an argument to
        return a ``discretisedfield.Field`` object. The first n (or n-1) dimensions of
        the DataArray are considered geometric dimensions of a scalar (or vector) field.
        In case of a vector field, the last dimension must be named ``vdims``. The
        DataArray attribute ``nvdim`` determines whether it is a scalar or a vector
        field (i.e. ``nvdim = 1`` is a scalar field and ``nvdim >= 1`` is a vector
        field). Hence, ``nvdim`` attribute must be present, greater than or equal to
        one, and of an integer type.

        The DataArray coordinates corresponding to the geometric dimensions represent
        the discretisation along the respective dimension and must have equally spaced
        values. The coordinates of ``vdims`` represent the name of field components
        (e.g. ['x', 'y', 'z'] for a 3D vector field).

        Additionally, it is expected to have ``cell``, ``p1``, and ``p2`` attributes for
        creating the right mesh for the field; however, in the absence of these, the
        coordinates of the geometric axes dimensions are utilized. It should be noted
        that ``cell`` attribute is required if any of the geometric directions has only
        a single cell.

        Parameters
        ----------
        xa : xarray.DataArray

            DataArray to create Field.

        Returns
        -------
        discretisedfield.Field

            Field created from DataArray.

        Raises
        ------
        TypeError

            - If argument is not ``xarray.DataArray``.
            - If ``nvdim`` attribute in not an integer.

        KeyError

            - If at least one of the geometric dimension coordinates has a single
              value and ``cell`` attribute is missing.
            - If ``nvdim`` attribute is absent.

        ValueError

            - If DataArray does not have a dimension ``vdims`` when attribute ``nvdim``
              is grater than one.
            - If coordinates of geometrical dimensions are not equally spaced.

        Examples
        --------
        1. Create a DataArray

        >>> import xarray as xr
        >>> import numpy as np
        ...
        >>> xa = xr.DataArray(np.ones((20, 20, 20, 3), dtype=float),
        ...                   dims = ['x', 'y', 'z', 'vdims'],
        ...                   coords = dict(x=np.arange(0, 20),
        ...                                 y=np.arange(0, 20),
        ...                                 z=np.arange(0, 20),
        ...                                 vdims=['x', 'y', 'z']),
        ...                   name = 'mag',
        ...                   attrs = dict(cell=[1., 1., 1.],
        ...                                p1=[1., 1., 1.],
        ...                                p2=[21., 21., 21.],
        ...                                nvdim=3),)
        >>> xa
        <xarray.DataArray 'mag' (x: 20, y: 20, z: 20, vdims: 3)>
        ...

        2. Create Field from DataArray

        >>> import discretisedfield as df
        ...
        >>> field = df.Field.from_xarray(xa)
        >>> field
        Field(...)
        >>> field.mean()
        array([1., 1., 1.])

        """
        if not isinstance(xa, xr.DataArray):
            raise TypeError("Argument must be a xarray.DataArray.")

        if "nvdim" not in xa.attrs:
            raise KeyError(
                'The DataArray must have an attribute "nvdim" to identify a scalar or'
                " a vector field."
            )

        if xa.attrs["nvdim"] < 1:
            raise ValueError('"nvdim" attribute must be greater or equal to 1.')
        elif not isinstance(xa.attrs["nvdim"], int):
            raise TypeError("The value of nvdim must be an integer.")

        if xa.attrs["nvdim"] > 1 and "vdims" not in xa.dims:
            raise ValueError(
                'The DataArray must have a dimension "vdims" when "nvdim" attribute is'
                " greater than 1."
            )

        dims_list = [dim for dim in xa.dims if dim != "vdims"]

        for i in dims_list:
            if xa[i].values.size > 1 and not np.allclose(
                np.diff(xa[i].values), np.diff(xa[i].values).mean()
            ):
                raise ValueError(f"Coordinates of {i} must be equally spaced.")

        try:
            cell = xa.attrs["cell"]
        except KeyError:
            if any(len_ == 1 for len_ in xa.values.shape[:-1]):
                raise KeyError(
                    "DataArray must have a 'cell' attribute if any "
                    "of the geometric directions has a single cell."
                ) from None
            cell = [np.diff(xa[i].values).mean() for i in dims_list]

        p1 = (
            xa.attrs["pmin"]
            if "pmin" in xa.attrs
            else [xa[i].values[0] - c / 2 for i, c in zip(dims_list, cell)]
        )
        p2 = (
            xa.attrs["pmax"]
            if "pmax" in xa.attrs
            else [xa[i].values[-1] + c / 2 for i, c in zip(dims_list, cell)]
        )

        if any("units" not in xa[i].attrs for i in dims_list):
            region = df.Region(p1=p1, p2=p2, dims=dims_list)
            mesh = df.Mesh(region=region, cell=cell)
        else:
            region = df.Region(
                p1=p1, p2=p2, dims=dims_list, units=[xa[i].units for i in dims_list]
            )
            mesh = df.Mesh(region=region, cell=cell)

        if "tolerance_factor" in xa.attrs:
            mesh.region.tolerance_factor = xa.attrs["tolerance_factor"]

        vdims = xa.vdims.values if "vdims" in xa.coords else None
        nvdim = xa.attrs["nvdim"]
        val = np.expand_dims(xa.values, axis=-1) if nvdim == 1 else xa.values
        # print(val.shape)
        # TODO load vdim_mapping
        return cls(
            mesh=mesh, nvdim=nvdim, value=val, vdims=vdims, dtype=xa.values.dtype
        )

    @functools.singledispatchmethod
    def _as_array(self, val, mesh, nvdim, dtype):
        raise TypeError(f"Unsupported type {type(val)}.")

    # to avoid str being interpreted as iterable
    @_as_array.register(str)
    def _(self, val, mesh, nvdim, dtype):
        raise TypeError(f"Unsupported type {type(val)}.")

    @_as_array.register(numbers.Complex)
    @_as_array.register(collections.abc.Iterable)
    def _(self, val, mesh, nvdim, dtype):
        if isinstance(val, numbers.Complex) and nvdim > 1 and val != 0:
            raise ValueError(
                f"Wrong dimension 1 provided for value; expected dimension is {nvdim}"
            )

        if isinstance(val, collections.abc.Iterable):
            if nvdim == 1 and np.array_equal(np.shape(val), mesh.n):
                return np.expand_dims(val, axis=-1)
            elif np.shape(val)[-1] != nvdim:
                raise ValueError(
                    f"Wrong dimension {len(val)} provided for value; expected dimension"
                    f" is {nvdim}."
                )
        dtype = dtype or max(np.asarray(val).dtype, np.float64)
        return np.full((*mesh.n, nvdim), val, dtype=dtype)

    @_as_array.register(collections.abc.Callable)
    def _(self, val, mesh, nvdim, dtype):
        # will only be called on user input
        # dtype must be specified by the user for complex values
        array = np.empty((*mesh.n, nvdim), dtype=dtype)
        for index, point in zip(mesh.indices, mesh):
            # Conversion to array and reshaping is required for numpy >= 1.24
            # and for certain inputs, e.g. a tuple of numpy arrays which can e.g. occur
            # for 1d vector fields.
            array[index] = np.asarray(val(point)).reshape(nvdim)
        return array

    @_as_array.register(dict)
    def _(self, val, mesh, nvdim, dtype):
        # will only be called on user input
        # dtype must be specified by the user for complex values
        dtype = dtype or np.float64
        fill_value = (
            val["default"]
            if "default" in val and not callable(val["default"])
            else np.nan
        )
        array = np.full((*mesh.n, nvdim), fill_value, dtype=dtype)

        for subregion in reversed(mesh.subregions.keys()):
            # subregions can overlap, first subregion takes precedence
            try:
                submesh = mesh[subregion]
                subval = val[subregion]
            except KeyError:
                continue  # subregion not in val when implicitly set via "default"
            else:
                slices = mesh.region2slices(submesh.region)
                array[slices] = self._as_array(subval, submesh, nvdim, dtype)

        if np.any(np.isnan(array)):
            # not all subregion keys specified and 'default' is missing or callable
            if "default" not in val:
                raise KeyError(
                    "Key 'default' required if not all subregion keys are specified."
                )
            subval = val["default"]
            for idx in np.argwhere(np.isnan(array[..., 0])):
                # only spatial indices required -> array[..., 0]
                # conversion to array and reshaping similar to "callable" implementation
                array[idx] = np.asarray(subval(mesh.index2point(idx))).reshape(nvdim)

        return array


# We cannot register to self inside the class
@CellField._as_array.register(CellField)
def _(self, val, mesh, nvdim, dtype):
    if mesh.region not in val.mesh.region:
        raise ValueError(
            f"{val.mesh.region} of the provided field does not "
            f"contain {mesh.region} of the field that is being "
            "created."
        )
    value = (
        val.to_xarray()
        .sel(
            **{dim: getattr(mesh.cells, dim) for dim in mesh.region.dims},
            method="nearest",
        )
        .data
    )
    if nvdim == 1:
        # xarray dataarrays for scalar data are three dimensional
        return value.reshape(*mesh.n, -1)
    return value
