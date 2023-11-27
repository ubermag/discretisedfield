import collections
import functools
import numbers

import numpy as np
import xarray as xr

import discretisedfield as df
import discretisedfield.util as dfu
from discretisedfield.plotting.util import hv_key_dim

from .field import Field


class VertexField(Field):
    def __call__(self, point):
        """TODO Returns nearest node for now."""
        if point not in self.mesh.region:
            raise ValueError(f"{point=} not in '{self.mesh.region}'.")

        vertices = self.mesh.vertices
        index = tuple(np.argmin(point[i] - vertices[i]) for i in range(self.nvdim))

        return self.array[index]

    def diff(self, direction, order=1, restrict2valid=True):
        """Maybe this is slighly wrong and we should ask Claas about this."""
        super().diff(direction, order=order, restrict2valid=restrict2valid)

    def integrate(self, direction=None, cumulative=False):
        """Maybe this is slighly wrong and we should ask Claas about this."""
        super().integrate(direction=direction, cumulative=cumulative)

    def line(self, p1, p2, n):
        def mesh_cell_line(p1, p2, n):
            if p1 not in self.mesh.region or p2 not in self.mesh.region:
                msg = f"Point {p1=} or point {p2=} is outside the mesh region."
                raise ValueError(msg)

            dl = np.subtract(p2, p1) / n
            for i in range(n):
                yield dfu.array2tuple(np.add(p1, i * dl))

        points = list(mesh_cell_line(p1=p1, p2=p2, n=n))
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
        raise NotImplementedError

    def mpl(self):
        pass  # @Swapneel

    @property
    def _hv_key_dims(self):
        """Dict of key dimensions of the field.

        Keys are the field dimensions (domain and vector space, e.g. x, y, z, vdims)
        that have length > 1. Values are named_tuples ``hv_key_dim(data, unit)`` that
        contain the data (which has to fulfil len(data) > 1, typically as a numpy array
        or list) and the unit of a string (empty string if there is no unit).

        """
        key_dims = {
            dim: hv_key_dim(coords, unit)
            for dim, unit in zip(self.mesh.region.dims, self.mesh.region.units)
            if len(coords := getattr(self.mesh.vertices, dim)) > 1
        }
        if self.nvdim > 1:
            key_dims["vdims"] = hv_key_dim(self.vdims, "")
        return key_dims

    # def hv(self):
    #     pass  # @Swapneel
    #
    # NOTE: We are ignoring all the FFTs for now.

    def to_xarray(self, name="field", unit=None):
        """VertexField value as ``xarray.DataArray``.

        The method returns an ``xarray.DataArray`` with the dimensions
        ``self.mesh.region.dims`` and ``vdims`` (only if ``field.nvdim > 1``). The
        coordinates of the geometric dimensions are derived from ``self.mesh.vertices``
        and for vector field components from ``self.vdims``. Additionally,
        the values of ``self.mesh.cell``, ``self.mesh.region.pmin``, and
        ``self.mesh.region.pmax`` are stored as ``cell``, ``pmin``, and ``pmax``
        attributes of the DataArray. The ``unit`` attribute of geometric
        dimensions is set to the respective strings in ``self.mesh.region.units``.

        The name and unit of the ``DataArray`` can be set by passing ``name`` and
        ``unit`` respectively. If the type of value passed to any of the two
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

            VertexField values DataArray.

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
        >>> field = df.VertexField(mesh=mesh, nvdim=3, value=(1, 0, 0), norm=1.)
        ...
        >>> field
        Field(...)

        2. Create `xarray.DataArray` from field

        >>> xa = field.to_xarray()
        >>> xa
        <xarray.DataArray 'field' (x: 11, y: 11, z: 11, vdims: 3)>
        ...

        3. Select values of `x` component

        >>> xa.sel(vdims='x')
        <xarray.DataArray 'field' (x: 11, y: 11, z: 11)>
        ...

        """
        if not isinstance(name, str):
            msg = "Name argument must be a string."
            raise TypeError(msg)

        if unit is not None and not isinstance(unit, str):
            msg = "Unit argument must be a string."
            raise TypeError(msg)

        axes = self.mesh.region.dims

        data_array_coords = {axis: getattr(self.mesh.vertices, axis) for axis in axes}

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
                data_location="vertex",
            ),
        )

        # TODO save vdim_mapping

        for dim in geo_units_dict:
            data_array[dim].attrs["units"] = geo_units_dict[dim]

        return data_array

    @classmethod
    def from_xarray(cls, xa):
        raise NotImplementedError

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
            if nvdim == 1 and np.array_equal(np.shape(val), mesh.n + 1):
                return np.expand_dims(val, axis=-1)
            elif np.shape(val)[-1] != nvdim:
                raise ValueError(
                    f"Wrong dimension {len(val)} provided for value; expected dimension"
                    f" is {nvdim}."
                )
        dtype = dtype or max(np.asarray(val).dtype, np.float64)
        return np.full((*(mesh.n + 1), nvdim), val, dtype=dtype)

    # TODO: reimplement the remaining _as_array functions. @Swapneel
