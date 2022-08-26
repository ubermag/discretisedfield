import numpy as np
import xarray as xr

import discretisedfield as df


class Field:
    def __init__(
        self, mesh, dims, value=0.0, norm=None, components=None, dtype=None, units=None
    ):
        pmin = np.array(mesh.region.pmin)
        pmax = np.array(mesh.region.pmax)
        n = np.array(mesh.n)
        assert len(pmin) == len(pmax)
        assert len(pmin) == len(n)

        dims = None  # TODO remove this

        if dims:
            assert len(pmin) == len(dims)
        elif len(pmin) == 3:  # TODO remove this
            dims = ["x", "y", "z"]
        else:
            dims = [f"x{i}" for i in range(len(pmin))]

        data = value  # TODO fix this

        vdim = 1 if len(data.shape) == len(pmin) else data.shape[-1]
        vdims = components  # TODO fix this
        if vdims:
            assert len(vdims) == vdim
        else:
            vdims = [f"v{i}" for i in range(vdim)]

        cell = (pmax - pmin) / n
        coords = {
            dim: np.linspace(pmin_i + cell_i / 2, pmax_i - cell_i / 2, n_i)
            for dim, pmin_i, pmax_i, n_i, cell_i in zip(dims, pmin, pmax, n, cell)
        }

        if vdim > 1:
            coords["vdims"] = vdims

        self.data = xr.DataArray(
            data, dims=dims + ["vdims"], coords=coords, name="field"
        )
        for dim in dims:
            if dim != "vdims":
                self.data[dim].attrs["units"] = "m"
        self.data.attrs["cell"] = cell
        self._cell = cell
        # self._subregions = subregions or {}  # TODO fix this
        # self._bc = bc  # TODO fix this

    # @property
    # def subregions(self):
    #    return self._subregions

    # @subregions.setter
    # def subregions(self, subregions):
    #    # checks
    #    self._subregions = subregions

    @classmethod
    def from_xarray(cls):  # -> still required (?)
        raise NotImplementedError()

    @classmethod
    def coordinate_field(cls):
        raise NotImplementedError()

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
        xr_dims = self.data.dims
        return xr_dims[:-1] if "vdims" in xr_dims else xr_dims

    @property
    def nvdims(self):
        """Number of value dimensions."""
        return len(self.vdims) if self.vdims else 1

    @property
    def vdims(self):
        """Labels of the value dimensions."""
        return self.data.vdims if "vdims" in self.data.dims else None

    @property
    def mesh(self):
        self.mesh = df.Mesh(
            p1=self.pmin,
            p2=self.pmax,
            n=self.n,
            subregions=self._subregions,
            bc=self._bc,
        )

    @property
    def norm(self):
        raise NotImplementedError()

    @norm.setter
    def norm(self, norm):
        raise NotImplementedError()

    @property
    def orientation(self):
        raise NotImplementedError()

    @property
    def units(self):
        raise NotImplementedError()

    def check_same_mesh(self, other):
        """Check if two Field objects are defined on the same mesh."""
        if not isinstance(other, self.__class__):
            raise TypeError("Object of type {type(other)} not supported.")
        if self.ndims != other.ndims or self.nvdims != other.nvdims:
            return False
        if self.dims != other.dims or self.vdims != other.vdims:
            return False
        # for dim is self.dims:
        #    pass
        raise NotImplementedError()

    def translate(self, point):
        # TODO test
        return self.__class__(
            pmin=self.pmin + point,
            pmax=self.pmax + point,
            n=self.n,
            data=self.data,
            dims=self.dims,
            vdims=self.vdims,
        )

    def scale(self, factor):
        # TODO test
        return self.__class__(
            pmin=self.pmin * factor,
            pmax=self.pmax * factor,
            n=self.n,
            data=self.data,
            dims=self.dims,
            vdims=self.vdims,
        )

    @property
    def edges(self):
        return self.pmax - self.pmin

    def sel(self, *args, **kwargs):
        """Select a submesh."""
        kwargs = kwargs or {}
        if args:
            for arg in args:
                if not isinstance(arg, str):
                    raise TypeError("Positional arguments must be strings")
                if arg in kwargs:
                    raise ValueError("Dimension {arg} is specified twice.")
                else:
                    kwargs[arg] = self.edges[self._dims.index(arg)] / 2

        self.data.sel(**kwargs)
        pmin = []
        pmax = []
        dims = []
        for dim, sel in kwargs:
            if isinstance(sel, tuple) or isinstance(sel, list):
                sel = slice(*sel)
            if isinstance(sel, slice):
                center_min = self.data[dim].sel(**{dim: sel.start}, method="nearest")
                center_max = self.data[dim].sel(**{dim: sel.stop}, method="nearest")

                pmin.append(center_min - self._cell[self.data.dims.index(dim)] / 2)
                pmax.append(center_max + self._cell[self.data.dims.index(dim)] / 2)
                dims.append(dim)
            else:
                # cutplane
                pass
        return self.__class__(
            pmin=self.pmin,
            pmax=self.pmax,
            n=self.n,
            data=self.data,
            dims=self.dims,
            vdims=self.vdims,
        )

    # mathematical operations

    def __abs__(self):
        raise NotImplementedError()

    def __pos__(self):
        raise NotImplementedError()

    def __neg__(self):
        raise NotImplementedError()

    def __add__(self, other):
        if isinstance(other, self.__class__):
            other = other.data
        return self.__class__(
            pmin=self.pmin,
            pmax=self.pmax,
            n=self.n,
            data=self.data + other,
            dims=self.dims,
            vdims=self.vdims,
        )

    def __sub__(self):
        raise NotImplementedError()

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            other = other.data
        return self.__class__(
            pmin=self.pmin,
            pmax=self.pmax,
            n=self.n,
            data=self.data * other,
            dims=self.dims,
            vdims=self.vdims,
        )

    def __truediv__(self):
        raise NotImplementedError()

    def __pow__(self):
        raise NotImplementedError()

    def __matmul__(self):  # -> dot
        raise NotImplementedError()

    def __and__(self):  # -> cross
        raise NotImplementedError()

    def __array_ufunc__(self):
        raise NotImplementedError()

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
        raise NotImplementedError()

    @property
    def imag(self):
        raise NotImplementedError()

    @property
    def phase(self):
        raise NotImplementedError()

    @property
    def conjugate(self):
        raise NotImplementedError()

    # other mathematical operations

    def integral(self):
        raise NotImplementedError()

    @property
    def angle(self):  # -> method angle(vector)
        raise NotImplementedError()

    def average(self):  # -> mean
        raise NotImplementedError()

    # other methods

    def __call__(self):
        raise NotImplementedError()

    def __getattr__(self):
        raise NotImplementedError()

    def __getitem__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __lshift__(self):  # -> concat
        raise NotImplementedError()

    def allclose(self):
        raise NotImplementedError()

    def line(self):
        raise NotImplementedError()

    def pad(self):
        raise NotImplementedError()

    def to_xarray(self):  # -> still required (?)
        raise NotImplementedError()

    def plane(self):
        raise NotImplementedError()

    # io and external data structures

    def to_vtk(self):
        raise NotImplementedError()

    def from_file(self):
        raise NotImplementedError()

    def to_file(self):  # the old write method
        raise NotImplementedError()

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
        raise NotImplementedError()

    def __repr__(self):
        return repr(self.data)

    def _repr_html_(self):
        return self.data._repr_html_()
