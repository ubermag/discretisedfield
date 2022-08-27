import collections
import copy
import functools
import numbers
import pathlib

import numpy as np
import xarray as xr

import discretisedfield as df

from . import io


class Field:
    def __init__(
        self,
        mesh=None,
        dim=None,
        value=0.0,
        norm=None,
        components=None,
        dtype=None,
        units=None,
        dims=None,
    ):
        if isinstance(value, xr.DataArray):
            if any(
                arg is not None for arg in (mesh, dim, components, dtype, units, dims)
            ):
                raise ValueError(
                    "No extra arguments are allowed if value is xr.DataArray."
                )
            else:
                self._init_from_xarray(value)
        else:
            self._init_from_others(mesh, dim, value, components, dtype, units, dims)

        self.units = units
        self.norm = norm

    def _init_from_others(self, mesh, dim, value, components, dtype, units, dims):
        if any(arg is None for arg in (mesh, dim)):
            raise TypeError("Missing arguments mesh or dim.")

        if not isinstance(mesh, df.Mesh):
            raise TypeError(f"Wrong type for mesh: {type(mesh)} not supported.")

        self._mesh = mesh
        pmin = np.array(mesh.region.pmin)
        pmax = np.array(mesh.region.pmax)
        n = np.array(mesh.n)

        if dims is not None:
            assert len(pmin) == len(dims)
        elif len(pmin) == 3:  # TODO remove this
            dims = ["x", "y", "z"]
        else:
            dims = [f"x{i}" for i in range(len(pmin))]

        # TODO fix this
        # vdim = 1 if len(data.shape) == len(pmin) else data.shape[-1]
        vdim = dim

        vdims = components  # TODO fix this
        if vdims is not None:
            assert len(vdims) == vdim
            if isinstance(vdims, tuple):
                vdims = np.ndarray(vdims)
        else:
            vdims = [f"v{i}" for i in range(vdim)]

        cell = (pmax - pmin) / n
        coords = {
            dim: np.linspace(pmin_i + cell_i / 2, pmax_i - cell_i / 2, n_i)
            for dim, pmin_i, pmax_i, n_i, cell_i in zip(dims, pmin, pmax, n, cell)
        }

        if vdim > 1:
            coords["vdims"] = vdims
            dims += ["vdims"]

        self._data = xr.DataArray(
            _as_array(value, mesh, vdim, dtype),
            dims=dims,
            coords=coords,
            name="field",
        )

        for dim in dims:
            if dim != "vdims":
                self._data[dim].attrs["units"] = "m"
        self._data.attrs["cell"] = cell
        self._cell = cell

    def _init_from_xarray(self, value):

        if len(value.coords) != value.data.ndim:
            raise AttributeError(
                "The xr.DataArray must have the same number of coordinates as"
                " the axes of underlying data array."
            )

        # We are assuming that the last dimension label of the DataArray
        # is "vdims" if it is a vector field. Otherwise it is considered
        # a scalar field.
        if value.coords.dims[-1] == "vdims":
            dims = value.coords.dims[:-1]
        else:
            dims = value.coords.dims

        for i in dims:
            if value[i].data.size > 1 and not np.allclose(
                np.diff(value[i].data), np.diff(value[i].data).mean()
            ):
                raise ValueError(f"Coordinates of {i} must be equally spaced.")

        try:
            cell = value.attrs["cell"]
        except KeyError:
            if any(len_ == 1 for len_ in value.data.shape[:-1]):
                raise KeyError(
                    "DataArray must have a 'cell' attribute if any "
                    "of the spatial directions has a single cell."
                ) from None
            cell = [np.diff(value[i].data).mean() for i in dims]

        p1 = (
            value.attrs["p1"]
            if "p1" in value.attrs
            else [value[i].data[0] - c / 2 for i, c in zip(dims, cell)]
        )
        p2 = (
            value.attrs["p2"]
            if "p2" in value.attrs
            else [value[i].data[-1] + c / 2 for i, c in zip(dims, cell)]
        )

        # if any("units" not in xa[i].attrs for i in xa.coords.dims[:-1]):
        self._mesh = df.Mesh(p1=p1, p2=p2, cell=cell)  # TODO: Check for units!
        # else:
        #     mesh = df.Mesh(
        #         p1=p1, p2=p2, cell=cell, attributes={"unit": xa["z"].attrs["units"]}
        #     )

        self._cell = cell
        self._data = value

    @classmethod
    def coordinate_field(cls, mesh):
        field = cls(mesh, dim=mesh.dim)
        dim_midpoints_list = [
            field.data[coord].data for coord in field.data.coords[:-1]
        ]
        dim_points_list = np.meshgrid(*dim_midpoints_list, indexing="ij")
        for i in range(mesh.dim):
            field.data.data[..., i] = dim_points_list[i]

        return field

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        raise RuntimeError(
            "The `.data` attribute is read-only. To update the data use the"
            " `update_field_values` method."
        )

    @property
    def value(self):
        raise AttributeError(
            " Attribute `value` has been removed. Use `data` instead to read data."
        )

    @value.setter
    def value(self, data):
        raise AttributeError(
            "The `value` attribute has been removed. Use the "
            " `update_field_values([])` method to update data."
        )

    def update_field_values(self, values, dtype=None):
        print(f"Updating with {values=}")
        self._data.data = _as_array(
            values, self.mesh, self.nvdims, dtype or self.data.data.dtype
        )

    @property
    def pmin(self):
        pmin = []
        for dim in self.data.dims:
            if dim == "vdims":
                continue
            center_min = self.data[dim][0]
            pmin.append(center_min - self._cell[self.dims.index(dim)] / 2)
        return np.array(pmin)

    @property
    def pmax(self):
        pmax = []
        for dim in self.data.dims:
            if dim == "vdims":
                continue
            center_max = self.data[dim][-1]
            pmax.append(center_max - self._cell[self.dims.index(dim)] / 2)
        return np.array(pmax)

    @property
    def n(self):
        return self.data.shape[:-1] if self.is_vectorfield else self.data.shape

    @property
    def is_vectorfield(self):
        return "vdims" in self.data.dims

    @property
    def cell(self):
        return self._cell

    @property
    def ndims(self):
        """Number of spatial dimensions."""
        return len(self.dims)

    @property
    def dims(self):
        """Labels of the spatial dimensions."""
        xr_dims = list(self.data.dims)
        return xr_dims[:-1] if "vdims" in xr_dims else xr_dims

    @property
    def nvdims(self):
        """Number of value dimensions."""
        return len(self.vdims) if self.vdims else 1

    @property
    def vdims(self):
        """Labels of the value dimensions."""
        return list(self.data.vdims.data) if "vdims" in self.data.dims else None

    @property
    def mesh(self):
        return self._mesh

    @property
    def norm(self):
        """Norm of the field.

        Computes the norm of the field and returns ``discretisedfield.Field``
        with ``dim=1``. Norm of a scalar field is interpreted as an absolute
        value of the field. Alternatively, ``discretisedfield.Field.__abs__``
        can be called for obtaining the norm of the field. TODO

        The field norm can be set by passing ``numbers.Real``,
        ``numpy.ndarray``, or callable. If the field has ``dim=1`` or it
        contains zero values, norm cannot be set and ``ValueError`` is raised.

        Parameters
        ----------
        numbers.Real, numpy.ndarray, callable

            Norm value.

        Returns
        -------
        discretisedfield.Field

            Norm of the field if ``dim>1`` or absolute value for ``dim=1``.

        Raises
        ------
        ValueError

            If the norm is set with wrong type, shape, or value. In addition,
            if the field is scalar (``dim=1``) or the field contains zero
            values.

        Examples
        --------
        1. Manipulating the field norm.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (1, 1, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field.norm
        Field(...)
        >>> field.norm.average
        1.0
        >>> field.norm = 2
        >>> field.average
        (0.0, 0.0, 2.0)
        >>> field.value = (1, 0, 0)
        >>> field.norm.average
        1.0

        Set the norm for a zero field.
        >>> field.value = 0
        >>> field.average
        (0.0, 0.0, 0.0)
        >>> field.norm = 1
        >>> field.average
        (0.0, 0.0, 0.0)

        .. seealso:: :py:func:`~discretisedfield.Field.__abs__`

        """
        if self.nvdims == 1:
            res = abs(self.data)
        else:
            res = np.linalg.norm(self.data, axis=-1)

        return self.__class__(self.mesh, dim=1, value=res, units=self.units)

    @norm.setter
    def norm(self, val):
        if val is not None:
            if self.nvdims == 1:
                raise ValueError(f"Cannot set norm for scalar field. ({self.nvdims=})")

            # normalise field to 1.0
            self.data.data = np.divide(  # should we use self.__truediv__ instead?
                self.data.data,
                self.norm.data.data[..., np.newaxis],
                out=np.zeros_like(self.data.data),
                where=self.norm.data.data[..., np.newaxis] != 0.0,
            )

            # multiply field with val
            val_array = _as_array(val, self.mesh, dim=1, dtype=None)
            self.data.data *= val_array[..., np.newaxis]

    @property
    def orientation(self):
        """Orientation field.

        This method computes the orientation (direction) of a vector field and
        returns ``discretisedfield.Field`` with the same dimension. More
        precisely, at every mesh discretisation cell, the vector is divided by
        its norm, so that a unit vector is obtained. However, if the vector at
        a discretisation cell is a zero-vector, it remains unchanged. In the
        case of a scalar (``dim=1``) field, ``ValueError`` is raised.

        Returns
        -------
        discretisedfield.Field

            Orientation field.

        Raises
        ------
        ValueError

            If the field is has ``dim=1``.
        """
        if self.nvdims == 1:
            raise ValueError(
                f"Cannot compute orientation for scalar field. ({self.nvdims=})"
            )
        fieldCopy = self.__class__(value=copy.deepcopy(self.data))
        fieldCopy.update_field_values(0)
        norm = self.norm.data.data[..., np.newaxis]
        np.divide(self.data.data, norm, out=fieldCopy.data.data, where=norm != 0.0)
        return fieldCopy

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, units):
        if units is not None and not isinstance(units, str):
            raise TypeError(f"Wrong type for {units=}; must be of type str.")
        self._units = units

    def is_same_mesh(self, other, tolerance_factor=1e-5):
        """Check if two Field objects are defined on the same mesh."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"Object of type {type(other)} not supported.")
        if self.dims != other.dims:
            return False
        for dim in self.dims:
            if len(self.data[dim]) != len(other.data[dim]):
                return False
            else:
                # change absolute tolerance to a tolerance relative to the cell size
                # to take into account the overall scale
                if not np.allclose(
                    self.data[dim],
                    other.data[dim],
                    atol=self.cell[self.dims.index(dim)] * tolerance_factor,
                ):
                    return False
        return True

    def is_same_vectorspace(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Object of type {type(other)} not supported.")
        return self.vdims == other.vdims

    def __eq__(self, other):
        """Check equality based on all attributes.

        Takes into account

        - vector units
        - number and names of spatial dimensions
        - number and names of value dimensions
        - coordinates of the spatial dimensions
        - data
        """
        if not isinstance(other, self.__class__):
            return False
        if self.units != other.units:
            return False
        if self.dims != other.dims:
            return False
        if self.vdims != other.vdims:
            return False
        for dim in self.dims:
            if np.any(self.data[dim].data != other.data[dim].data):
                return False
        return np.all(self.data.data == other.data.data)

    def translate(self, point):
        # TODO write tests
        if self.mesh.subregions:
            raise NotImplementedError("Translate does not yet support subregions.")
        return self.__class__(
            mesh=df.Mesh(p1=self.pmin, p2=self.pmax, n=self.n, bc=self.mesh.bc),
            dim=self.nvdims,
            value=self.data,
            components=self.vdims,
        )

    def scale(self, factor):
        # TODO write tests
        if self.mesh.subregions:
            raise NotImplementedError("Scale does not yet support subregions.")
        return self.__class__(
            mesh=df.Mesh(
                p1=self.pmin * factor, p2=self.pmax * factor, n=self.n, bc=self.mesh.bc
            ),
            dim=self.vdims,
            value=self.data,
            components=self.vdims,
        )

    @property
    def edges(self):
        return self.pmax - self.pmin

    # TODO proper testing requires n-dimension mesh and region
    def sel(self, *args, **kwargs):
        """Select a submesh."""
        if self.mesh.subregions:
            raise ValueError(
                "Selections are not yet supported for fields with subregions."
            )
        kwargs = kwargs or {}
        if args:
            for arg in args:
                if not isinstance(arg, str):
                    raise TypeError("Positional arguments must be strings")
                if arg in kwargs:
                    raise ValueError(f"Dimension {arg} is specified twice.")
                else:
                    kwargs[arg] = self.edges[self.dims.index(arg)] / 2

        pmin = []
        pmax = []
        dims = []
        n = []
        for dim in self.dims:
            if dim in kwargs:
                sel = kwargs[dim]
                if isinstance(sel, collections.abc.Iterable):
                    sel = slice(*sel)
                if isinstance(sel, slice):
                    cell_i = self._cell[self.dims.index(dim)]
                    center_min = self.data[dim].sel(
                        **{dim: sel.start}, method="nearest", tolerance=cell_i / 2
                    )
                    center_max = self.data[dim].sel(
                        **{dim: sel.stop}, method="nearest", tolerance=cell_i / 2
                    )
                    pmin.append(center_min - cell_i / 2)
                    pmax.append(center_max + cell_i / 2)
                    dims.append(dim)
                    kwargs[dim] = slice(pmin[-1], pmax[-1])
                    # n.append(self.n[self.dims.index(dim)])
                    n.append(int(np.round((pmax[-1] - pmin[-1]) / cell_i)))
                else:
                    cell_i = self._cell[self.dims.index(dim)]
                    kwargs[dim] = (
                        self.data[dim]
                        .sel(**{dim: sel}, method="nearest", tolerance=cell_i / 2)
                        .data.tolist()
                    )
            else:
                pmin.append(self.pmin[self.dims.index(dim)])
                pmax.append(self.pmax[self.dims.index(dim)])
                n.append(self.n[self.dims.index(dim)])
                dims.append(dim)

        # TODO use the next line and remove the remaining lines of this method
        # return self.__class__.from_xarray(self.data.sel(**kwargs))

        class Region:
            def __init__(self, p1, p2):
                self.pmin = p1
                self.pmax = p2

        class Mesh:
            def __init__(self, p1, p2, n, bc):
                self.region = Region(p1, p2)
                self.n = n
                self.bc = bc

        return self.__class__(
            Mesh(p1=pmin, p2=pmax, n=n, bc=self.mesh.bc),
            dim=self.nvdims,
            value=self.data.sel(**kwargs),
            units=self.units,
            dims=dims,
        )

    # mathematical operations

    def __abs__(self):
        return self.__class__(
            self.mesh, dim=self.nvdims, value=np.abs(self.data.data), units=self.units
        )

    def __pos__(self):
        return self

    def __neg__(self):
        return -1 * self

    def __add__(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
            other = other.data

        return self.__class__(
            self.mesh, dim=self.nvdims, value=self.data + other, units=self.units
        )

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
            other = other.data

        return self.__class__(
            self.mesh, dim=self.nvdims, value=self.data - other, units=self.units
        )

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
            other = other.data

        return self.__class__(
            self.mesh, dim=self.nvdims, value=self.data * other, units=self.units
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
            other = other.data

        return self.__class__(
            self.mesh, dim=self.nvdims, value=self.data / other, units=self.units
        )

    def __rtruediv__(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
            other = other.data

        return self.__class__(
            self.mesh, dim=self.nvdims, value=other / self.data, units=self.units
        )

    def __pow__(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
            other = other.data

        return self.__class__(
            self.mesh, dim=self.nvdims, value=self.data**other, units=self.units
        )

    def dot(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
            other = other.data
        elif isinstance(other, xr.DataArray):
            other = other
        elif isinstance(other, (tuple, list, np.ndarray)):
            other = self.__class__(self.mesh, dim=self.nvdims, value=other).data
        else:
            msg = (
                f"Unsupported operand type(s) for the cross product: {type(self)=} and"
                f" {type(other)=}."
            )
            raise TypeError(msg)

        if self.nvdims == 1:
            return self.__class__(
                self.mesh,
                dim=1,
                value=self.data * other.data,
            )
        else:
            return self.__class__(
                self.mesh,
                dim=1,
                value=np.einsum("...l,...l->...", self.data, other.data),
            )

    def cross(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)

            if self.nvdims != 3 or other.nvdims != 3:
                msg = (
                    f"Cannot apply the cross product on {self.nvdims=} and"
                    f" {other.nvdims=} fields. The cross product is only supported on"
                    " field with 3 vector dimentions."
                )
                raise ValueError(msg)
            other = other.data
        elif isinstance(other, xr.DataArray):
            other = other
        elif isinstance(other, (tuple, list, np.ndarray)):
            other = self.__class__(self.mesh, dim=self.nvdims, value=other).data
        else:
            msg = (
                f"Unsupported operand type(s) for the cross product: {type(self)=} and"
                f" {type(other)=}."
            )
            raise TypeError(msg)

        return self.__class__(
            self.mesh,
            dim=self.nvdims,
            value=np.cross(self.data.data, other.data),
            units=self.units,
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Field class support for numpy ``ufuncs``."""
        # See reference implementation at:
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
        for x in inputs:
            if not isinstance(x, (self.__class__, np.ndarray, numbers.Number)):
                return NotImplemented
        out = kwargs.get("out", ())
        if out:
            for x in out:
                if not isinstance(x, self.__class__):
                    return NotImplemented

        mesh = [x.mesh for x in inputs if isinstance(x, self.__class__)]
        inputs = tuple(x.data if isinstance(x, self.__class__) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.data for x in out)

        result = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(result, tuple):
            if len(result) != len(mesh):
                raise ValueError("wrong number of Field objects")
            return tuple(
                self.__class__(
                    m,
                    dim=x.shape[-1],
                    value=x,
                    components=self.components,
                )
                for x, m in zip(result, mesh)
            )
        elif method == "at":
            return None
        else:
            return self.__class__(
                mesh[0],
                dim=result.shape[-1],
                value=result,
                components=self.components,
            )

    # derivative-related

    def derivative(self):
        raise NotImplementedError()

    @property
    def grad(self):
        raise NotImplementedError()

    @property
    def div(self):
        raise NotImplementedError()

    @property
    def curl(self):
        raise NotImplementedError()

    @property
    def laplace(self):
        raise NotImplementedError()

    # FFT

    @property
    def fftn(self):
        raise NotImplementedError()

    @property
    def ifftn(self):
        raise NotImplementedError()

    @property
    def rfftn(self):
        raise NotImplementedError()

    @property
    def irfftn(self):
        raise NotImplementedError()

    # complex values

    @property
    def real(self):
        """Real part of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.nvdims,
            value=self.data.real,
            units=self.units,
        )

    @property
    def imag(self):
        """Imaginary part of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.nvdims,
            value=self.data.imag,
            units=self.units,
        )

    @property
    def phase(self):
        """Phase of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.nvdims,
            value=np.angle(self.data),
            units=self.units,
        )

    @property
    def conjugate(self):
        """Complex conjugate of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.nvdims,
            value=self.data.conjugate(),
            units=self.units,
        )

    # other mathematical operations

    def integral(self):
        raise NotImplementedError()

    def angle(self, other):
        if isinstance(other, self.__class__):
            self.is_same_mesh_field(other)
        elif self.nvdims == 1 and isinstance(other, numbers.Complex):
            other = self.__class__(self.mesh, dim=self.nvdims, value=other)
        elif self.nvdims != 1 and isinstance(other, (tuple, list, np.ndarray)):
            other = self.__class__(self.mesh, dim=self.nvdims, value=other)
        else:
            msg = (
                f"Unsupported operand type(s) for angle: {type(self)=} and"
                f" {type(other)=}."
            )
            raise TypeError(msg)

        angle_array = np.arccos((self.dot(other) / (self.norm * other.norm)).data)
        return self.__class__(
            self.mesh,
            dim=1,
            value=angle_array,
            units=self.units,
        )

    @property
    def average(self):  # -> mean
        return self.data.mean(dim=self.dims).data

    # other methods

    def __call__(self, point):
        data = self.data
        for dim, p, cell_i in zip(self.dims, point, self.cell):
            data = data.sel(**{dim: p}, method="nearest", tolerance=cell_i / 2)
        return data.data

    def __getattr__(self, attr):
        if self.nvdims > 1 and attr in self.vdims:
            attr_array = self.data[..., self.vdims.index(attr)]
            return self.__class__(
                mesh=self.mesh, dim=1, value=attr_array, units=self.units
            )
        msg = f"Object has no attribute {attr}."
        raise AttributeError(msg)

    def __getitem__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def concat(self, other, vdims=None):
        if isinstance(other, self.__class__):
            if not self.is_same_mesh(other):
                raise ValueError(
                    "Cannot concatenate fields defined on different meshes."
                )
        elif isinstance(other, numbers.Complex):
            return self.concat(self.__class__(self.mesh, dim=1, value=other))
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self.concat(self.__class__(self.mesh, dim=len(other), value=other))
        else:
            raise TypeError(f"Unsupported type {type(other)} passed.")

        if self.nvdims == 1 and other.nvdims == 1:
            data = np.stack([self.to_numpy(), other.to_numpy()], axis=-1)
        elif self.nvdims == 1:
            data = np.concatenate(
                [self.to_numpy()[..., np.newaxis], other.to_numpy()], axis=-1
            )
        elif other.nvdims == 1:
            data = np.concatenate(
                [self.to_numpy(), other.to_numpy()[..., np.newaxis]], axis=-1
            )
        else:
            data = np.concatenate([self.to_numpy(), other.to_numpy()], axis=-1)

        dims = [*self.dims, "vdims"]
        coords = {dim: self.data[dim] for dim in self.dims}
        if vdims is not None and len(vdims) != data.shape[-1]:
            raise ValueError("Wrong number of vdims provided.")
        coords["vdims"] = vdims or [f"v{i}" for i in range(data.shape[-1])]
        xr_data = xr.DataArray(data, dims=dims, coords=coords)
        xr_data.attrs["cell"] = self.cell

        for dim in self.dims:
            xr_data[dim].attrs["units"] = self.data[dim].units
        return self.__class__(value=xr_data)

    def allclose(self, other, rtol=1e-5, atol=1e-8):
        if not isinstance(other, self.__class__):
            raise TypeError(
                "Cannot apply allclose method between "
                f"{type(self)=} and {type(other)=} objects."
            )
        if self.is_same_mesh(other) and self.is_same_vectorspace(other):
            return np.allclose(self.data.data, other.data.data, rtol=rtol, atol=atol)
        else:
            return False

    def line(self):
        raise NotImplementedError()

    def pad(self):
        raise NotImplementedError()

    def to_xarray(self):
        return self.data

    def plane(self):
        raise NotImplementedError()

    # io and external data structures

    def to_vtk(self):
        raise NotImplementedError()

    def to_file(
        self, filename, representation="bin8", extend_scalar=False, save_subregions=True
    ):
        """Write the field to OVF, HDF5, or VTK file.

        If the extension of ``filename`` is ``.vtk``, a VTK file is written
        (:py:func:`~discretisedfield.io.field_to_vtk`). The representation of the data
        (``'bin'`` [``'bin8'`` as equivalent], ``'txt'``, or ``'xml'``) is passed as
        ``'representation'``.

        For ``.ovf``, ``.omf``, or ``.ohf`` extensions, the field is saved to
        OVF file (:py:func:`~discretisedfield.io.field_to_ovf`). In that case,
        the representation of data (``'bin4'``, ``'bin8'``, or ``'txt'``) is
        passed as ``representation`` and if ``extend_scalar=True``, a scalar
        field will be saved as a vector field. More precisely, if the value at
        a cell is X, that cell will be saved as (X, 0, 0).

        Finally, if the extension of ``filename`` is ``.hdf5``, HDF5 file will
        be written (:py:func:`~discretisedfield.io.field_to_hdf5`).

        Parameters
        ----------
        filename : str

            Name of the file written.

        representation : str, optional

            Only supported for OVF and VTK files. In the case of OVF files
            (``.ovf``, ``.omf``, or ``.ohf``) the representation can be
            ``'bin4'``, ``'bin8'``, or ``'txt'``. For VTK files (``.vtk``) the
            representation can be ``bin``, ``xml``, or ``txt``. Defaults to
            ``'bin8'`` (interpreted as ``bin`` for VTK files).

        extend_scalar : bool, optional

            If ``True``, a scalar field will be saved as a vector field. More
            precisely, if the value at a cell is 3, that cell will be saved as
            (3, 0, 0). This is valid only for the OVF file formats. Defaults to
            ``False``.

        save_subregions : bool, optional

            If ``True`` and subregions are defined for the mesh the subregions will be
            saved to a json file. Defaults to ``True``.

        Example
        -------
        1. Write field to the OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5e-9)
        >>> p2 = (5e-9, 15e-9, 15e-9)
        >>> n = (5, 15, 20)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(5, 6, 7))
        ...
        >>> filename = 'mytestfile.omf'
        >>> field.write(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.fromfile(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        2. Write field to the VTK file.

        >>> filename = 'mytestfile.vtk'
        >>> field.write(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> os.remove(filename)  # delete the file

        3. Write field to the HDF5 file.

        >>> filename = 'mytestfile.hdf5'
        >>> field.write(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.fromfile(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        See also
        --------
        ~discretisedfield.Field.fromfile
        ~discretisedfield.io.field_to_ovf
        ~discretisedfield.io.field_to_vtk
        ~discretisedfield.io.field_to_hdf5

        """
        filename = pathlib.Path(filename)
        if filename.suffix in [".omf", ".ovf", ".ohf"]:
            io.field_to_ovf(
                self,
                filename,
                representation=representation,
                extend_scalar=extend_scalar,
                save_subregions=save_subregions,
            )
        elif filename.suffix in [".hdf5", ".h5"]:
            io.field_to_hdf5(self, filename, save_subregions=save_subregions)
        elif filename.suffix == ".vtk":
            io.field_to_vtk(
                self,
                filename,
                representation=representation,
                save_subregions=save_subregions,
            )
        else:
            raise ValueError(
                f"Writing file with extension {filename.suffix} not supported."
            )

    # keep `write` as the legacy name for now
    write = to_file

    @classmethod
    def from_file(cls, filename):
        """Read the field from an OVF (1.0 or 2.0), VTK, or HDF5 file.

        The extension of the ``filename`` should correspond to either:
            - OVF (``.ovf``, ``.omf``, ``.ohf``, ``.oef``)
            - VTK (``.vtk``), or
            - HDF5 (``.hdf5`` or ``.h5``).

        This method automatically determines the file type based on the file name
        extension.

        Parameters
        ----------
        filename : str

            Name of the file to be read.

        Returns
        -------
        discretisedfield.Field

            Field read from the file.

        Example
        -------
        1. Read the field from an OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> dirname = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'oommf-ovf2-bin4.omf')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(...)

        2. Read a field from the VTK file.

        >>> filename = os.path.join(dirname, 'vtk-file.vtk')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(...)

        3. Read a field from the HDF5 file.

        >>> filename = os.path.join(dirname, 'hdf5-file.hdf5')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(...)

        See also
        --------
        ~discretisedfield.Field.write
        ~discretisedfield.io.field_from_ovf
        ~discretisedfield.io.field_from_vtk
        ~discretisedfield.io.field_from_hdf5

        """
        filename = pathlib.Path(filename)
        if filename.suffix in [".omf", ".ovf", ".ohf", ".oef"]:
            return io.field_from_ovf(filename)
        elif filename.suffix == ".vtk":
            return io.field_from_vtk(filename)
        elif filename.suffix in [".hdf5", ".h5"]:
            return io.field_from_hdf5(filename)
        else:
            raise ValueError(
                f"Reading file with extension {filename.suffix} not supported."
            )

    # keep `write` as the legacy name for now
    fromfile = from_file

    # plotting

    @property
    def mpl(self):
        raise NotImplementedError()

    @property
    def hv(self):  # -> should work already
        raise NotImplementedError()

    @property
    def k3d(self):
        raise NotImplementedError()

    def to_numpy(self):
        return self.data.data

    def __repr__(self):
        return repr(self.data)

    def _repr_html_(self):
        return self.data._repr_html_()

    def is_same_mesh_field(self, other):  # TODO move to utils
        if not self.is_same_mesh(other):
            raise ValueError(
                "To perform this operation both fields must have the same mesh."
            )

        if self.is_vectorfield and other.is_vectorfield:
            if not self.is_same_vectorspace(other):
                raise ValueError(
                    "To perform this operation both fields must have the same"
                    " vector components."
                )


@functools.singledispatch
def _as_array(val, mesh, dim, dtype):
    raise TypeError("Unsupported type {type(val)}.")


# to avoid str being interpreted as iterable
@_as_array.register(str)
def _(val, mesh, dim, dtype):
    raise TypeError("Unsupported type {type(val)}.")


@_as_array.register(numbers.Complex)
@_as_array.register(collections.abc.Iterable)
def _(val, mesh, dim, dtype):
    if isinstance(val, numbers.Complex) and dim > 1 and val != 0:
        raise ValueError(
            f"Wrong dimension 1 provided for value; expected dimension is {dim}"
        )
    dtype = dtype or max(np.asarray(val).dtype, np.float64)
    return np.full(mesh.n if dim == 1 else (*mesh.n, dim), val, dtype=dtype)


@_as_array.register(collections.abc.Callable)
def _(val, mesh, dim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    array = np.empty(mesh.n if dim == 1 else (*mesh.n, dim), dtype=dtype)
    for index, point in zip(mesh.indices, mesh):
        array[index] = val(point)
    return array


@_as_array.register(Field)
def _(val, mesh, dim, dtype):
    if mesh.region not in val.mesh.region:
        raise ValueError(
            f"{val.mesh.region} of the provided field does not "
            f"contain {mesh.region} of the field that is being "
            "created."
        )
    value = val.data.sel(
        x=mesh.midpoints.x, y=mesh.midpoints.y, z=mesh.midpoints.z, method="nearest"
    ).data
    return value


@_as_array.register(dict)
def _(val, mesh, dim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    dtype = dtype or np.float64
    fill_value = (
        val["default"] if "default" in val and not callable(val["default"]) else np.nan
    )
    array = np.full(mesh.n if dim == 1 else (*mesh.n, dim), fill_value, dtype=dtype)

    for subregion in reversed(mesh.subregions.keys()):
        # subregions can overlap, first subregion takes precedence
        try:
            submesh = mesh[subregion]
            subval = val[subregion]
        except KeyError:
            continue
        else:
            slices = mesh.region2slices(submesh.region)
            array[slices] = _as_array(subval, submesh, dim, dtype)

    if np.any(np.isnan(array)):
        # not all subregion keys specified and 'default' is missing or callable
        if "default" not in val:
            raise KeyError(
                "Key 'default' required if not all subregion keys are specified."
            )
        subval = val["default"]
        for ix, iy, iz in np.argwhere(np.isnan(array[..., 0])):
            # only spatial indices required -> array[..., 0]
            array[ix, iy, iz] = subval(mesh.index2point((ix, iy, iz)))

    return array
