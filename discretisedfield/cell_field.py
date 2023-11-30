import collections
import functools
import numbers

import numpy as np

import discretisedfield as df
import discretisedfield.plotting as dfp

from .field import Field


class CellField(Field):
    def __call__(self, point):
        return self.array[self.mesh.point2index(point)]

    # diff, integrate depending on how we calculate those for the VertexField

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
