import collections
import functools
import numbers

import numpy as np

import discretisedfield as df
import discretisedfield.util as dfu

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

    def hv(self):
        pass  # @Swapneel

    # NOTE: We are ignoring all the FFTs for now.

    def to_xarray(self, name="field", unit=None):
        pass  # @Swapneel

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
