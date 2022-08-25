import numpy as np
import xarray as xr

import discretisedfield as df


class Field:
    def __init__(
        self, pmin, pmax, n, data, dims=None, vdims=None, subregions=None, bc=""
    ):
        pmin = np.array(pmin)
        pmax = np.array(pmax)
        n = np.array(n)
        assert len(pmin) == len(pmax)
        assert len(pmin) == len(n)
        if dims:
            assert len(pmin) == len(dims)
        else:
            dims = [f"x{i}" for i in range(len(pmin))]

        vdim = 1 if len(data.shape) == len(pmin) else data.shape[-1]
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
        self._vdims = vdims
        self._dims = dims
        self.vdims = vdims
        self.dims = dims
        self.subregions = subregions or {}
        self.mesh = df.Mesh(
            p1=self.pmin, p2=self.pmax, n=self.n, subregions=subregions, bc=bc
        )

    # @property
    # def subregions(self):
    #    return self._subregions

    # @subregions.setter
    # def subregions(self, subregions):
    #    # checks
    #    self._subregions = subregions

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

    def __add__(self, other):
        if isinstance(other, self.__class__):
            other = other.data
        return self.__class__(
            pmin=self.pmin,
            pmax=self.pmax,
            n=self.n,
            data=self.data + other,
            dims=self._dims,
            vdims=self._vdims,
        )

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            other = other.data
        return self.__class__(
            pmin=self.pmin,
            pmax=self.pmax,
            n=self.n,
            data=self.data * other,
            dims=self._dims,
            vdims=self._vdims,
        )

    def translate(self, point):
        return self.__class__(
            pmin=self.pmin + point,
            pmax=self.pmax + point,
            n=self.n,
            data=self.data,
            dims=self._dims,
            vdims=self._vdims,
        )

    def scale(self, factor):
        return self.__class__(
            pmin=self.pmin * factor,
            pmax=self.pmax * factor,
            n=self.n,
            data=self.data,
            dims=self._dims,
            vdims=self._vdims,
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
            dims=self._dims,
            vdims=self._vdims,
        )

    def __abs__(self):
        raise NotImplementedError()

    def __and__(self):
        raise NotImplementedError()

    def __array_ufunc__(self):
        raise NotImplementedError()

    def __call__(self):
        raise NotImplementedError()

    def __eq__(self):
        raise NotImplementedError()

    def __getattr__(self):
        raise NotImplementedError()

    def __getitem__(self):
        raise NotImplementedError()

    def __iter__(self):
        raise NotImplementedError()

    def __lshift__(self):  # -> concat
        raise NotImplementedError()

    def __matmul__(self):  # -> dot
        raise NotImplementedError()

    def __neg__(self):
        raise NotImplementedError()

    def __pos__(self):
        raise NotImplementedError()

    def __pow__(self):
        raise NotImplementedError()

    def __sub__(self):
        raise NotImplementedError()

    def __truediv__(self):
        raise NotImplementedError()

    def allclose(self):
        raise NotImplementedError()

    def coordinate_field(self):
        raise NotImplementedError()

    def derivative(self):
        raise NotImplementedError()

    def from_xarray(self):
        raise NotImplementedError()

    def fromfile(self):  # -> from_file
        raise NotImplementedError()

    def integral(self):
        raise NotImplementedError()

    def line(self):
        raise NotImplementedError()

    def plat(self):
        raise NotImplementedError()

    def plane(self):
        raise NotImplementedError()

    def project(self):
        raise NotImplementedError()

    def to_vtk(self):
        raise NotImplementedError()

    def to_xarray(self):
        raise NotImplementedError()

    def write(self):  # -> to_file
        raise NotImplementedError()

    @property
    def angle(self):  # -> method angle(vector)
        raise NotImplementedError()

    @property
    def average(self):  # -> mean
        raise NotImplementedError()

    @property
    def components(self):
        raise NotImplementedError()

    @property
    def conjugate(self):
        raise NotImplementedError()

    @property
    def curl(self):
        raise NotImplementedError()

    @property
    def dim(self):
        raise NotImplementedError()

    @property
    def div(self):
        raise NotImplementedError()

    @property
    def fftn(self):
        raise NotImplementedError()

    @property
    def grad(self):
        raise NotImplementedError()

    @property
    def hv(self):  # -> should work already
        raise NotImplementedError()

    @property
    def ifftn(self):
        raise NotImplementedError()

    @property
    def imag(self):
        raise NotImplementedError()

    @property
    def irfftn(self):
        raise NotImplementedError()

    @property
    def k3d(self):
        raise NotImplementedError()

    @property
    def laplace(self):
        raise NotImplementedError()

    @property
    def mesh(self):
        raise NotImplementedError()

    @property
    def mpl(self):
        raise NotImplementedError()

    @property
    def norm(self):
        raise NotImplementedError()

    @property
    def orientation(self):
        raise NotImplementedError()

    @property
    def phase(self):
        raise NotImplementedError()

    @property
    def real(self):
        raise NotImplementedError()

    @property
    def rfftn(self):
        raise NotImplementedError()

    @property
    def units(self):
        raise NotImplementedError()

    @property
    def value(self):
        raise NotImplementedError()

    @property
    def zero(self):  # -> delete this method
        raise NotImplementedError()

    def __repr__(self):
        return repr(self.data) + "\ncell:" + str(self.cell)

    def _repr_html_(self):
        return self.data._repr_html_()
