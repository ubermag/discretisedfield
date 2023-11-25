import itertools
import os
import random
import re
import tempfile
import types

import holoviews as hv
import k3d
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.fft as spfft
import xarray as xr

import discretisedfield as df

from .test_mesh import html_re as mesh_html_re

html_re = (
    r"<strong>Field</strong>\s*<ul>\s*"
    rf"<li>{mesh_html_re}</li>\s*"
    r"<li>nvdim = \d+</li>\s*"
    r"(<li>vdims:\s*<ul>(<li>.*</li>\s*)+</ul>\s*</li>)?\s*"
    r"(<li>unit = .+</li>)?\s*"
    r"</ul>"
)

# Test inputs for initialising fields in the form [value, dtype]

# scalar functions: take a length-n tuple and return a scalar
sfuncs = [
    [lambda c: 1, np.float64],
    [lambda c: -2.4, np.float64],
    [lambda c: -6.4e-15, np.float64],
    [lambda c: 1 + 2j, np.complex128],
    [lambda c: sum(c) + 1, np.float64],
]
# vector functions: take a length-n tuple and return a length-3 vector
vfuncs = [
    [lambda c: (1, 2, 0), np.float64],
    [lambda c: (-2.4, 1e-3, 9), np.float64],
    [lambda c: (1 + 1j, 2 + 2j, 3 + 3j), np.complex128],
    [lambda c: (0, 1j, 1), np.complex128],
    [lambda c: (sum(c), np.min(c), np.max(c)), np.float64],
]
# scalar constants
consts = [
    [0, None],
    [-5.0, None],
    [np.pi, None],
    [1e-15, None],
    [1.2e12, None],
    [random.random(), None],
    [1 + 1j, None],
]
# vector constants
iters = [
    [(0, 0, 1), None],
    [(0, -5.1, np.pi), None],
    [[70, 1e15, 2 * np.pi], None],
    [[5, random.random(), np.pi], None],
    [np.array([4, -1, 3.7]), None],
    [np.array([2.1, 0.0, -5 * random.random()]), None],
    [(1 + 1j, 1 + 1j, 1 + 1j), None],
    [(0, 0, 1j), None],
    [np.random.random(3) + np.random.random(3) * 1j, None],
]


def check_hv(plot, types):
    # generate the first plot output to have enough data in plot.info
    hv.renderer("bokeh").get_plot(plot)
    # find strings like "    :DynamicMap [comp,z]" or "    :Image    [x,y]"
    # the number of spaces can vary
    assert sorted(
        re.findall(r"(?<=:)\w+ \[[^]]+\]", re.sub(r"\s+", " ", str(plot)))
    ) == sorted(types)


@pytest.fixture
def test_field():
    mesh = df.Mesh(p1=(-5e-9, -5e-9, -5e-9), p2=(5e-9, 5e-9, 5e-9), n=(5, 5, 5))
    c_array = mesh.coordinate_field().array

    # The norm is defined via numpy for performance reasons;
    # In the simple loop form it would be:
    # x, y, _ = point
    # if x**2 + y**2 <= 5e-9**2:
    #     return 1e5
    # else:
    #     return 0
    def norm(points):
        return np.where(
            (points[..., 0] ** 2 + points[..., 1] ** 2) <= 5e-9**2, 1e5, 0
        )[..., np.newaxis]

    # Values are defined in numpy for performance reasons
    # We define vector fields with vx=0, vy=0, vz=+/-1 for x<0 / x>0
    def value(points):
        res = np.zeros((*mesh.n, 3))
        res[..., 2] = np.where(points[..., 0] <= 0, 1, -1)
        return res

    return df.Field(
        mesh,
        nvdim=3,
        value=value(c_array),
        norm=norm(c_array),
        vdims=["a", "b", "c"],
        valid="norm",
    )


@pytest.mark.parametrize("value, dtype", consts + sfuncs)
def test_init_scalar_valid_args(valid_mesh, value, dtype):
    f = df.Field(valid_mesh, nvdim=1, value=value, dtype=dtype)
    assert isinstance(f, df.Field)
    assert f.array.shape == (*valid_mesh.n, 1)

    assert isinstance(f.mesh, df.Mesh)
    assert isinstance(f.nvdim, int)
    assert f.nvdim == 1
    assert isinstance(f.array, np.ndarray)
    assert f.array.shape == (*f.mesh.n, f.nvdim)
    rstr = repr(f)
    assert isinstance(rstr, str)
    pattern = (
        r"^Field\(Mesh\(Region\(pmin=\[.+\], pmax=\[.+\], .+\), .+\)," r" nvdim=\d+\)$"
    )
    if f.vdims:
        pattern = pattern[:-3] + r", vdims: \(.+\)\)$"
    if f.unit is not None:
        pattern = pattern[:-3] + r", unit=.+\)$"
    assert re.search(pattern, rstr)

    assert isinstance(f._repr_html_(), str)
    assert re.search(html_re, f._repr_html_(), re.DOTALL)

    assert isinstance(f.__iter__(), types.GeneratorType)
    assert len(list(f)) == len(f.mesh)


@pytest.mark.parametrize("value, dtype", iters + vfuncs)
def test_init_vector_valid_args(valid_mesh, value, dtype):
    f = df.Field(valid_mesh, nvdim=3, value=value, dtype=dtype)
    assert isinstance(f, df.Field)

    assert isinstance(f.mesh, df.Mesh)
    assert isinstance(f.nvdim, int)
    assert f.nvdim == 3
    assert isinstance(f.array, np.ndarray)
    assert f.array.shape == (*f.mesh.n, f.nvdim)
    rstr = repr(f)
    assert isinstance(rstr, str)
    pattern = (
        r"^Field\(Mesh\(Region\(pmin=\[.+\], pmax=\[.+\], .+\), .+\)," r" nvdim=\d+\)$"
    )
    if f.vdims:
        pattern = pattern[:-3] + r", vdims: \(.+\)\)$"
    if f.unit is not None:
        pattern = pattern[:-3] + r", unit=.+\)$"
    assert re.search(pattern, rstr)

    assert isinstance(f._repr_html_(), str)
    assert re.search(html_re, f._repr_html_(), re.DOTALL)
    assert isinstance(f.__iter__(), types.GeneratorType)
    assert len(list(f)) == len(f.mesh)


@pytest.mark.parametrize("unit", [None, "T", "A/m"])
@pytest.mark.parametrize(
    "mesh, nvdim, value, dtype",
    [
        [df.Mesh(p1=0, p2=10, n=10), 1, lambda c: c + c**2 * 1j, np.complex128],
        [
            df.Mesh(p1=-5e-9, p2=10e-9, n=11),
            4,
            lambda c: (c, c**2, c**3 + 100, c - 1),
            np.float64,
        ],
        [
            df.Mesh(p1=(0, 0), p2=(1, 1), n=[1, 1]),
            2,
            lambda c: (c[0] + 10, c[1]),
            np.float64,
        ],
        [
            df.Mesh(p1=(0, 0, 0, 0), p2=(1, 2, 3, 4), cell=(1, 1, 1, 1)),
            3,
            lambda c: (c[0] - 1, c[1] + 70, c[2] * 0.1 + c[3]),
            np.float64,
        ],
    ],
)
def test_init_special_combinations(mesh, nvdim, value, dtype, unit):
    f = df.Field(mesh, nvdim=nvdim, value=value, dtype=dtype, unit=unit)
    assert isinstance(f, df.Field)
    assert f.array.shape == (*mesh.n, nvdim)

    assert isinstance(f.mesh, df.Mesh)
    assert isinstance(f.nvdim, int)
    assert f.nvdim == nvdim
    assert isinstance(f.array, np.ndarray)
    assert f.array.shape == (*f.mesh.n, f.nvdim)
    rstr = repr(f)
    assert isinstance(rstr, str)
    pattern = (
        r"^Field\(Mesh\(Region\(pmin=\[.+\], pmax=\[.+\], .+\), .+\)," r" nvdim=\d+\)$"
    )
    if f.vdims:
        pattern = pattern[:-3] + r", vdims: \(.+\)\)$"
    if f.unit is not None:
        pattern = pattern[:-3] + r", unit=.+\)$"
    assert re.search(pattern, rstr)

    assert isinstance(f._repr_html_(), str)
    assert re.search(html_re, f._repr_html_(), re.DOTALL)

    assert isinstance(f.__iter__(), types.GeneratorType)
    assert len(list(f)) == len(f.mesh)


def test_init_invalid_arguments():
    p1 = (0, 0, 0)
    p2 = (10e-9, 10e-9, 10e-9)
    n = (5, 5, 5)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    with pytest.raises(TypeError):
        df.Field("meaningless_mesh_string", nvdim=1)

    # wrong abc.Iterable
    with pytest.raises(TypeError):
        df.Field(mesh, nvdim=1, value="string")

    # all builtin types are numeric types or Iterable
    class WrongType:
        pass

    with pytest.raises(TypeError):
        df.Field(mesh, nvdim=1, value=WrongType())


@pytest.mark.parametrize(
    "nvdim, error",
    [(0, ValueError), (-1, ValueError), ("dim", TypeError), ((2, 3), TypeError)],
)
def test_init_invalid_nvdims(mesh_3d, nvdim, error):
    with pytest.raises(error):
        df.Field(mesh_3d, nvdim=nvdim)


def test_set_with_ndarray(valid_mesh):
    f = df.Field(valid_mesh, nvdim=3, value=np.ones((*valid_mesh.n, 3)))

    assert isinstance(f, df.Field)
    assert np.allclose(f.mean(), (1, 1, 1))

    with pytest.raises(ValueError):
        f.update_field_values(np.ones((2, 2)))


@pytest.mark.parametrize("func, dtype", sfuncs)
def test_set_with_callable_scalar(valid_mesh, func, dtype):
    f = df.Field(valid_mesh, nvdim=1, value=func, dtype=dtype)
    assert isinstance(f, df.Field)

    def random_point(f):
        return (
            np.random.random(valid_mesh.region.ndim) * f.mesh.region.edges
            + f.mesh.region.pmin
        )

    rp = random_point(f)
    # Make sure to be at the centre of the cell
    rp = f.mesh.index2point(f.mesh.point2index(rp))
    assert f(rp) == func(rp)


@pytest.mark.parametrize("func, dtype", vfuncs)
def test_set_with_callable_vector(valid_mesh, func, dtype):
    f = df.Field(valid_mesh, nvdim=3, value=func, dtype=dtype)
    assert isinstance(f, df.Field)

    def random_point(f):
        return (
            np.random.random(valid_mesh.region.ndim) * f.mesh.region.edges
            + f.mesh.region.pmin
        )

    rp = random_point(f)
    rp = f.mesh.index2point(f.mesh.point2index(rp))
    assert np.all(f(rp) == func(rp))


def test_set_with_dict():
    # 3d space with two subregions; one constant and one callable value
    p1 = (0, 0, 0)
    p2 = (10e-9, 10e-9, 10e-9)
    n = (5, 5, 5)
    subregions = {
        "r1": df.Region(p1=(0, 0, 0), p2=(4e-9, 10e-9, 10e-9)),
        "r2": df.Region(p1=(4e-9, 0, 0), p2=(10e-9, 10e-9, 10e-9)),
    }
    mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)

    field = df.Field(mesh, nvdim=3, value={"r1": (0, 0, 1), "r2": lambda c: c})
    assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1))
    assert np.allclose(field((8e-9, 2.5e-9, 9e-9)), (9e-9, 3e-9, 9e-9), atol=0)

    # subregions do not span the entire space
    subregions = {"r1": df.Region(p1=(0, 0, 0), p2=(4e-9, 10e-9, 10e-9))}
    mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
    with pytest.raises(KeyError):
        field = df.Field(mesh, nvdim=3, value={"r1": (0, 0, 1)})

    # subregions do not span the entire space but there is a "default"
    field = df.Field(mesh, nvdim=3, value={"r1": (0, 0, 1), "default": (1, 1, 1)})
    assert np.all(field((3e-9, 7e-9, 9e-9)) == (0, 0, 1))
    assert np.all(field((8e-9, 2e-9, 9e-9)) == (1, 1, 1))

    # no values for subregions, only "default"
    field = df.Field(mesh, nvdim=3, value={"default": (1, 1, 1)})
    assert np.all(field.array == (1, 1, 1))

    # 1d space with one subregion and callable default
    p1 = 0
    p2 = 10e-9
    n = 5
    subregions = {
        "r1": df.Region(p1=0, p2=4e-9),
    }
    mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
    # dtype has to be specified for isinstance(value, dict)
    field = df.Field(
        mesh,
        nvdim=3,
        value={"r1": (0, 0, 1 + 2j), "default": lambda c: (c[0], 0, 0)},
        dtype=np.complex128,
    )
    assert np.all(field(3e-9) == (0, 0, 1 + 2j))
    assert np.allclose(field(8e-9), (9e-9, 0, 0), atol=0)


def test_set_exception(valid_mesh):
    with pytest.raises(TypeError):
        df.Field(valid_mesh, nvdim=3, value="meaningless_string")

    with pytest.raises(ValueError):
        df.Field(valid_mesh, nvdim=3, value=5 + 5j)


def test_vdims(valid_mesh):
    valid_components = ["a", "b", "c", "d", "e", "f"]
    invalid_components = ["a", "grad", "b", "div", "array", "c"]
    for nvdim in range(2, 7):
        f = df.Field(
            valid_mesh,
            nvdim=nvdim,
            value=list(range(nvdim)),
            vdims=valid_components[:nvdim],
        )
        assert f.vdims == valid_components[:nvdim]
        assert isinstance(f, df.Field)

        with pytest.raises(ValueError):
            df.Field(
                valid_mesh,
                nvdim=nvdim,
                value=list(range(nvdim)),
                vdims=invalid_components[:nvdim],
            )

    # wrong number of components
    with pytest.raises(ValueError):
        df.Field(valid_mesh, nvdim=3, value=(1, 1, 1), vdims=valid_components)
    with pytest.raises(ValueError):
        df.Field(valid_mesh, nvdim=3, value=(1, 1, 1), vdims=["x", "y"])

    # components not unique
    with pytest.raises(ValueError):
        df.Field(valid_mesh, nvdim=3, value=(1, 1, 1), vdims=["x", "y", "x"])

    # test lshift
    f1 = df.Field(valid_mesh, nvdim=1, value=1)
    f2 = df.Field(valid_mesh, nvdim=1, value=2)
    f3 = df.Field(valid_mesh, nvdim=1, value=3)

    f12 = f1 << f2
    assert isinstance(f12, df.Field)
    assert np.allclose(f12.array[(0,) * valid_mesh.region.ndim, :], [1, 2])
    assert f12.x == f1
    assert f12.y == f2

    f123 = f1 << f2 << f3
    # check value at one point
    assert np.allclose(f123.array[(0,) * valid_mesh.region.ndim, :], [1, 2, 3])
    assert f123.x == f1
    assert f123.y == f2
    assert f123.z == f3

    fa = df.Field(valid_mesh, nvdim=1, value=10, vdims=["a"])
    fb = df.Field(valid_mesh, nvdim=1, value=20, vdims=["b"])

    # default components if not all fields have component labels
    f1a = f1 << fa
    assert isinstance(f1a, df.Field)
    assert f1a.vdims == ["x", "y"]

    # custom components if all fields have custom components
    fab = fa << fb
    assert isinstance(fab, df.Field)
    assert fab.vdims == ["a", "b"]


def test_unit(test_field):
    assert test_field.unit is None
    mesh = test_field.mesh
    field = df.Field(mesh, nvdim=3, value=(1, 2, 3), unit="A/m")
    assert isinstance(field, df.Field)
    assert field.unit == "A/m"
    field.unit = "mT"
    assert field.unit == "mT"
    with pytest.raises(TypeError):
        field.unit = 3
    assert field.unit == "mT"
    field.unit = None
    assert field.unit is None

    with pytest.raises(TypeError):
        df.Field(mesh, nvdim=1, unit=1)


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_valid_single_value(valid_mesh, nvdim):
    # Default
    f = df.Field(
        valid_mesh,
        nvdim=nvdim,
    )
    assert np.array_equal(f.valid.shape, valid_mesh.n)
    assert f.valid.dtype == bool
    assert np.all(f.valid)
    assert f.mesh == f._valid_as_field.mesh
    assert f.valid.dtype == f._valid_as_field.array.dtype
    assert np.array_equal(f.valid, f._valid_as_field.array.squeeze(axis=-1))
    # Constant
    f = df.Field(valid_mesh, nvdim=nvdim, valid=True)
    assert np.array_equal(f.valid.shape, valid_mesh.n)
    assert np.all(f.valid)
    assert f.mesh == f._valid_as_field.mesh
    assert np.array_equal(f.valid, f._valid_as_field.array.squeeze(axis=-1))
    f = df.Field(valid_mesh, nvdim=nvdim, valid=False)
    assert np.array_equal(f.valid.shape, valid_mesh.n)
    assert f.valid.dtype == bool
    assert np.all(~f.valid)
    assert f.mesh == f._valid_as_field.mesh
    assert np.array_equal(f.valid, f._valid_as_field.array.squeeze(axis=-1))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_valid_set_on_norm(ndim, nvdim):
    mesh = df.Mesh(p1=(0,) * ndim, p2=(10,) * ndim, cell=(1,) * ndim)

    def norm_func(point):
        if np.all(point < 5):
            return 5.0
        else:
            return 0

    f = df.Field(mesh, nvdim=nvdim, value=(1,) * nvdim, norm=norm_func, valid="norm")
    assert np.array_equal(f.valid.shape, mesh.n)
    assert f.valid.dtype == bool
    assert f.mesh == f._valid_as_field.mesh
    assert np.array_equal(f.valid, f._valid_as_field.array.squeeze(axis=-1))
    for idx in f.mesh.indices:
        if all(f.mesh.index2point(idx) < 5):
            # Use [0] to examine single element numpy array
            assert f.valid[tuple(idx)]
        else:
            assert not f.valid[tuple(idx)]


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_valid_set_call(ndim, nvdim):
    mesh = df.Mesh(p1=(0,) * ndim, p2=(10,) * ndim, cell=(1,) * ndim)

    def valid_func(point):
        return all(point < 5)

    # Default
    f = df.Field(mesh, nvdim=nvdim, valid=valid_func)
    assert np.array_equal(f.valid.shape, mesh.n)
    assert f.valid.dtype == bool
    assert f.mesh == f._valid_as_field.mesh
    assert np.array_equal(f.valid, f._valid_as_field.array.squeeze(axis=-1))
    for idx in f.mesh.indices:
        if all(f.mesh.index2point(idx) < 5):
            assert f.valid[tuple(idx)]
        else:
            assert not f.valid[tuple(idx)]

    def valid_func(point):
        if all(point < 5):
            return 5.0
        else:
            return 0

    f = df.Field(mesh, nvdim=nvdim, valid=valid_func)
    assert np.array_equal(f.valid.shape, mesh.n)
    assert f.valid.dtype == bool
    assert f.mesh == f._valid_as_field.mesh
    assert np.array_equal(f.valid, f._valid_as_field.array.squeeze(axis=-1))
    for idx in f.mesh.indices:
        if all(f.mesh.index2point(idx) < 5):
            assert f.valid[tuple(idx)]
        else:
            assert not f.valid[tuple(idx)]


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_valid_array(ndim, nvdim):
    mesh = df.Mesh(p1=(0,) * ndim, p2=(10,) * ndim, cell=(1,) * ndim)

    def val_func(point):
        return point[0]

    f = df.Field(mesh, nvdim=1, value=val_func)
    expected_valid = f.array[..., 0] < 5

    f = df.Field(mesh, nvdim=nvdim, valid=expected_valid)
    assert np.all(expected_valid == f.valid)
    assert f.mesh == f._valid_as_field.mesh
    assert np.array_equal(f.valid, f._valid_as_field.array.squeeze(axis=-1))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_valid_operators(ndim, nvdim):
    mesh = df.Mesh(p1=(0,) * ndim, p2=(10,) * ndim, cell=(1,) * ndim)

    def val_func(point):
        return point[0]

    f1 = df.Field(mesh, nvdim=1, value=val_func)
    expected_valid = f1.array[..., 0] < 5
    f2 = df.Field(mesh, nvdim=nvdim, value=(1,) * nvdim, valid=expected_valid)

    f3 = f1 + f2
    assert np.array_equal(f3.valid, np.logical_and(f1.valid, f2.valid))

    f3 = f1 - f2
    assert np.array_equal(f3.valid, np.logical_and(f1.valid, f2.valid))

    f3 = f1 * f2
    assert np.array_equal(f3.valid, np.logical_and(f1.valid, f2.valid))

    f3 = f1 / f2
    assert np.array_equal(f3.valid, np.logical_and(f1.valid, f2.valid))


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_value(valid_mesh, nvdim):
    f = df.Field(valid_mesh, nvdim=nvdim)
    f.update_field_values(np.arange(nvdim) + 1)

    # Set with array
    assert np.allclose(f.mean(), np.arange(nvdim) + 1)

    # Set with scalar
    if nvdim == 1:
        f.update_field_values(3.0)
        assert np.allclose(f.mean(), 3.0)

        f.update_field_values(np.array([2]))
        assert np.allclose(f.mean(), np.array([2]))
    else:
        with pytest.raises(ValueError):
            f.update_field_values(3.0)
        with pytest.raises(ValueError):
            f.update_field_values(np.array([2]))

    # Array with wrong shape
    with pytest.raises(ValueError):
        f.update_field_values(np.arange(nvdim + 1))

    if nvdim > 2:
        with pytest.raises(ValueError):
            f.update_field_values(np.arange(nvdim - 1))

    # Set with wrong type
    with pytest.raises(TypeError):
        f.update_field_values("string")


def test_average():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
    f = df.Field(mesh, nvdim=3, value=(2, 2, 2))
    with pytest.raises(AttributeError):
        f.average


@pytest.mark.parametrize("norm_value", [1, 2.1, 1e-3])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_norm(valid_mesh, nvdim, norm_value):
    f = df.Field(valid_mesh, nvdim=nvdim, value=(2,) * nvdim)

    assert np.allclose(f.norm.array, 2 * np.sqrt(nvdim))
    assert np.allclose(f.array, 2)

    f.norm = 1
    assert np.allclose(f.norm.array, 1)
    assert np.allclose(f.array, 1 / np.sqrt(nvdim))

    f = df.Field(valid_mesh, nvdim=nvdim, value=(3.0,) * nvdim, norm=norm_value)

    assert np.all(valid_mesh.n == f.norm.mesh.n)
    assert f.norm.array.shape == (*tuple(f.mesh.n), 1)
    assert np.allclose(f.norm.array, norm_value)


def test_norm_is_not_preserved():
    p1 = (0, 0, 0)
    p2 = (10e-9, 10e-9, 10e-9)
    n = (5, 5, 5)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    f = df.Field(mesh, nvdim=3)

    f.update_field_values((0, 3, 0))
    f.norm = 1
    assert np.all(f.norm.array == 1)

    f.update_field_values((0, 2, 0))
    assert np.all(f.norm.array == 2)


def test_norm_zero_field():
    p1 = (0, 0, 0)
    p2 = (10e-9, 10e-9, 10e-9)
    n = (5, 5, 5)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    f = df.Field(mesh, nvdim=3, value=(0, 0, 0))
    f.norm = 1  # Does not change the norm of zero field
    assert np.all(f.norm.array == 0)


@pytest.mark.parametrize(
    "p1, p2, n, nvdim, vdim_mapping, vdim_mapping_check",
    [
        # no mapping for scalar fields
        [0, 1, 5, 1, None, {}],
        [(0, 0), (1, 1), (5, 5), 1, None, {}],
        # default mapping for vector fields
        [(0, 0), (1, 1), (5, 5), 2, None, {d: d for d in "xy"}],
        [(0, 0, 0), (1, 1, 1), (5, 5, 5), 3, None, {d: d for d in "xyz"}],
        [(0,) * 4, (1,) * 4, (5,) * 4, 4, None, {f"v{i}": f"x{i}" for i in range(4)}],
        # manual mapping for dim - vdim mismatch
        [0, 1, 5, 2, {"x": "x", "y": None}, {"x": "x", "y": None}],
        [
            (0, 0),
            (1, 1),
            (5, 5),
            3,
            {"x": "x", "y": "y", "z": "z"},  # simulates sel 3d -> 2d
            {"x": "x", "y": "y", "z": "z"},
        ],
    ],
)
def test_vdim_mapping(p1, p2, n, nvdim, vdim_mapping, vdim_mapping_check):
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    field = df.Field(mesh, nvdim=nvdim, vdim_mapping=vdim_mapping)
    assert field.vdim_mapping == vdim_mapping_check


def test_r_dim_mapping():
    mesh = df.Mesh(p1=(0,) * 3, p2=(1,) * 3, n=(10,) * 3)
    field = df.Field(mesh, nvdim=3)

    assert field.vdim_mapping == {d: d for d in "xyz"}
    assert field._r_dim_mapping == {d: d for d in "xyz"}

    field.vdim_mapping = {"x": "a", "y": "b", "z": "c"}
    # values do not match region dims -> no vector components along spatial x, y, z
    assert field._r_dim_mapping == {"x": None, "y": None, "z": None}
    # change region dims -> vector components along spatial a, b, c
    field.mesh.region.dims = ["a", "b", "c"]
    assert field._r_dim_mapping == {"a": "x", "b": "y", "c": "z"}

    field.mesh.region.dims = ["x", "y", "z"]
    field.vdim_mapping = {"x": "x", "y": None, "z": None}
    assert field._r_dim_mapping == {"x": "x", "y": None, "z": None}


@pytest.mark.parametrize(
    "nvdim, vdim_mapping, error",
    [
        [2, {"a": "x", "b": "y"}, ValueError],  # invalid vdim
        [2, {"x": "x"}, ValueError],  # missing vdim
        [2, ("x", "y"), TypeError],  # invalid mapping type
    ],
)
def test_vdim_mapping_error(nvdim, vdim_mapping, error):
    mesh = df.Mesh(p1=(0, 0), p2=(1, 1), n=(5, 5))
    with pytest.raises(error):
        df.Field(mesh, nvdim=nvdim, vdim_mapping=vdim_mapping)


def test_vdims_vdim_mapping():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(1, 1, 1), n=(10, 10, 10))

    # 3 vdims -> automatic mapping
    f = df.Field(mesh, nvdim=3)
    assert f.vdim_mapping == {d: d for d in "xyz"}
    f.vdims = list("abc")
    assert f.vdim_mapping == dict(zip("abc", "xyz"))

    # 2 vdims with manual mapping
    f = df.Field(mesh, nvdim=2, vdim_mapping={"x": "y", "y": "z"})
    f.vdims = ["a", "b"]
    assert f.vdim_mapping == {"a": "y", "b": "z"}

    # 2 vdims -> no automatic mapping -> no default mapping
    f = df.Field(mesh, nvdim=2)
    assert f.vdim_mapping == {}
    f.vdims = ["a", "b"]
    assert f.vdim_mapping == {}
    f.vdim_mapping = {"a": "x", "b": "y"}
    f.vdims = ["v1", "v2"]
    assert f.vdim_mapping == {"v1": "x", "v2": "y"}


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_orientation(valid_mesh, nvdim):
    # No zero-norm cells
    inital_value = np.zeros(nvdim)
    inital_value[-1] = 2
    f = df.Field(valid_mesh, nvdim=nvdim, value=inital_value)
    assert isinstance(f.orientation, df.Field)
    assert np.allclose(f.orientation.mean(), inital_value / 2)


def test_orientation_func():
    # Test with zero-norm cells
    p1 = (-5e-9, -5e-9, -5e-9)
    p2 = (5e-9, 5e-9, 5e-9)
    cell = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def value_fun(point):
        if point[0] <= mesh.region.center[0]:
            return (0, 0, 0)
        else:
            return (3, 0, 4)

    f = df.Field(mesh, nvdim=3, value=value_fun)
    assert np.allclose(f.orientation(mesh.region.center - mesh.cell), (0, 0, 0))
    assert np.allclose(f.orientation(mesh.region.center + mesh.cell), (0.6, 0, 0.8))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_call(ndim, nvdim):
    mesh = df.Mesh(p1=(0.0,) * ndim, p2=(10.0,) * ndim, cell=(2.0,) * ndim)

    def val_func(point):
        return (point[0],) * nvdim

    f = df.Field(mesh, nvdim=nvdim, value=val_func)

    # test center of the cells
    assert np.allclose(f((1.0,) * ndim), (1.0,) * nvdim)
    assert np.allclose(f((5.0,) * ndim), (5.0,) * nvdim)
    assert np.allclose(f((9.0,) * ndim), (9.0,) * nvdim)

    # Regions are inclusive: [ ]
    # test with points exactly on the boundary of the mesh
    assert np.allclose(f((0.0,) * ndim), (1.0,) * nvdim)
    assert np.allclose(f((10.0,) * ndim), (9.0,) * nvdim)

    # cells are half-open: [ )
    # test with points exactly on the boundary of cells
    assert np.allclose(f((1.9,) * ndim), (1.0,) * nvdim)
    assert np.allclose(f((2,) * ndim), (3.0,) * nvdim)
    assert np.allclose(f((2.1,) * ndim), (3.0,) * nvdim)

    with pytest.raises(ValueError):
        f((5.0,) * (ndim + 1))

    with pytest.raises(ValueError):
        f((5.0,) * (ndim - 1))

    with pytest.raises(ValueError):
        f((-1.0,) * ndim)

    with pytest.raises(TypeError):
        f("invalid_input")

    with pytest.raises(TypeError):
        f(None)


def test_mean():
    tol = 1e-12

    p1 = (-5e-9, -5e-9, -5e-9)
    p2 = (5e-9, 4e-9, 3e-9)
    cell = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    f = df.Field(mesh, nvdim=1, value=2)
    assert abs(f.mean() - 2) < tol

    f = df.Field(mesh, nvdim=3, value=(0, 1, 2))
    assert np.allclose(f.mean(), (0, 1, 2))

    # Test with direction
    out = f.mean(direction="x")
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("y", "z")
    assert np.array_equal(out.mesh.n, (9, 8))
    out = f.mean(direction="y")
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("x", "z")
    assert np.array_equal(out.mesh.n, (10, 8))
    out = f.mean(direction="z")
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("x", "y")
    assert np.array_equal(out.mesh.n, (10, 9))

    with pytest.raises(ValueError):
        f.mean(direction="a")

    out = f.mean(direction=["x", "y"])
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("z",)
    assert np.array_equal(out.mesh.n, [8])
    out = f.mean(direction=["y", "z"])
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("x",)
    assert np.array_equal(out.mesh.n, [10])
    out = f.mean(direction=["x", "z"])
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("y",)
    assert np.array_equal(out.mesh.n, [9])
    out = f.mean(direction=["z", "y"])
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("x",)
    assert np.array_equal(out.mesh.n, [10])
    out = f.mean(direction=("x", "y"))
    assert np.allclose(out.array, (0, 1, 2))
    assert out.mesh.region.dims == ("z",)
    assert np.array_equal(out.mesh.n, [8])

    with pytest.raises(ValueError):
        f.mean(direction=["x", "a"])

    with pytest.raises(ValueError):
        f.mean(direction=["a", "x"])

    with pytest.raises(ValueError):
        f.mean(direction=["a", "b"])

    with pytest.raises(ValueError):
        f.mean(direction=["x", "x"])

    assert np.allclose(f.mean(direction=["x", "y", "z"]), f.mean())
    assert np.allclose(f.mean(direction=("x", "y", "z")), f.mean())

    with pytest.raises(ValueError):
        f.mean(direction=["x", "y", "z", "a"])

    with pytest.raises(ValueError):
        f.mean(direction=["x", "y", "z", "z"])


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_field_component(valid_mesh, nvdim):
    valid_components = ["a", "b", "c", "d", "e", "f"]
    f = df.Field(valid_mesh, nvdim=nvdim, vdims=valid_components[:nvdim])
    assert all(isinstance(getattr(f, i), df.Field) for i in valid_components[:nvdim])
    assert all(getattr(f, i).nvdim == 1 for i in valid_components[:nvdim])

    # Default
    f = df.Field(valid_mesh, nvdim=nvdim)
    if nvdim in [2, 3]:
        valid_components = ["x", "y", "z"]
    elif nvdim > 3:
        valid_components = [f"v{i}" for i in range(nvdim)]
    else:
        # nvdim = 1 exception
        with pytest.raises(AttributeError):
            f.x.nvdim
        return

    assert all(isinstance(getattr(f, i), df.Field) for i in valid_components[:nvdim])
    assert all(getattr(f, i).nvdim == 1 for i in valid_components[:nvdim])


def test_get_attribute_exception(mesh_3d):
    f = df.Field(mesh_3d, nvdim=3)
    with pytest.raises(AttributeError) as excinfo:
        f.__getattr__("nonexisting_attribute")
        assert "has no attribute" in str(excinfo.value)


def test_dir(valid_mesh):
    # Not testing component labels as this is already tested for in
    # test_field_component

    f = df.Field(valid_mesh, nvdim=3, value=(5, 6, -9))
    assert all(attr in dir(f) for attr in ["x", "y", "z"])

    f = df.Field(valid_mesh, nvdim=1, value=1)
    assert all(attr not in dir(f) for attr in ["x", "y", "z"])


def test_eq():
    p1 = (-5e-9, -5e-9, -5e-9)
    p2 = (15e-9, 5e-9, 5e-9)
    cell = (5e-9, 1e-9, 2.5e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    f1 = df.Field(mesh, nvdim=1, value=0.2)
    f2 = df.Field(mesh, nvdim=1, value=0.2)
    f3 = df.Field(mesh, nvdim=1, value=3.1)
    f4 = df.Field(mesh, nvdim=3, value=(1, -6, 0))
    f5 = df.Field(mesh, nvdim=3, value=(1, -6, 0))
    f6 = df.Field(mesh, nvdim=3, value=(1, -6, 0), vdims=list("abc"))
    f7 = df.Field(mesh, nvdim=3, value=(1, -6, 0), unit="A/m")

    assert f1 == f2
    assert not f1 != f2
    assert not f1 == f3
    assert f1 != f3
    assert not f2 == f4
    assert f2 != f4
    assert f4 == f5
    assert not f4 != f5
    assert not f1 == 0.2
    assert f1 != 0.2
    assert f5 == f6
    assert f5 == f7


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
@pytest.mark.parametrize("tol_value", [1e-10, 1e-5, 0.5, 2])
@pytest.mark.parametrize("base_value", [0.5, 1, -1, 1e-9])
def test_allclose_rtol(valid_mesh, nvdim, tol_value, base_value):
    base_value = np.full(nvdim, base_value)
    f1 = df.Field(valid_mesh, nvdim=nvdim, value=base_value)
    f2 = df.Field(valid_mesh, nvdim=nvdim, value=base_value * (1 + tol_value * 0.9))
    f3 = df.Field(valid_mesh, nvdim=nvdim, value=base_value * (1 + tol_value * 1.1))

    assert f1.allclose(f2, rtol=tol_value, atol=0)
    assert not f1.allclose(f3, rtol=tol_value, atol=0)


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
@pytest.mark.parametrize("tol_value", [1e-7, 1e-5, 0.5, 2])
@pytest.mark.parametrize("base_value", [0, 0.5, 1, -1, 1e-9])
def test_allclose_atol(valid_mesh, nvdim, tol_value, base_value):
    base_value = np.full(nvdim, base_value)
    f1 = df.Field(valid_mesh, nvdim=nvdim, value=base_value)
    f2 = df.Field(valid_mesh, nvdim=nvdim, value=base_value + tol_value * 0.9)
    f3 = df.Field(valid_mesh, nvdim=nvdim, value=base_value + tol_value * 1.1)

    # Need an rtol to deal with floating point accuracy
    rtol = (base_value - tol_value) * 1e-10

    assert f1.allclose(f2, atol=tol_value, rtol=rtol)
    assert not f1.allclose(f3, atol=tol_value, rtol=rtol)


@pytest.mark.parametrize("invalid_input", [2, "string", None, []])
def test_allclose_invalid_type(valid_mesh, invalid_input):
    f1 = df.Field(valid_mesh, nvdim=1, value=0)

    with pytest.raises(TypeError):
        f1.allclose(invalid_input)


def test_point_neg():
    p1 = (-5e-9, -5e-9, -5e-9)
    p2 = (5e-9, 5e-9, 5e-9)
    cell = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # Scalar field
    f = df.Field(mesh, nvdim=1, value=3)
    res = -f
    assert isinstance(res, df.Field)
    assert res.mean() == -3
    assert f == +f
    assert f == -(-f)
    assert f == +(-(-f))

    # Vector field
    f = df.Field(mesh, nvdim=3, value=(1, 2, -3))
    res = -f
    assert isinstance(res, df.Field)
    assert np.allclose(res.mean(), (-1, -2, 3))
    assert f == +f
    assert f == -(-f)
    assert f == +(-(-f))


def test_pow(mesh_3d):
    # Scalar field
    f = df.Field(mesh_3d, nvdim=1, value=2)
    res = f**2
    assert res.mean() == 4
    res = f ** (-1)
    assert res.mean() == 0.5

    # Vector field
    f = df.Field(mesh_3d, nvdim=3, value=(1, 2, -2))
    res = f**2
    assert np.allclose(res.mean(), (1, 4, 4))

    # 4D field
    f = df.Field(mesh_3d, nvdim=4, value=(1, 2, -2, 3))
    res = f**2
    assert np.allclose(res.mean(), (1, 4, 4, 9))

    # Attempt to raise to non numbers.Real
    with pytest.raises(TypeError):
        f ** "a"
    res = f**f
    assert np.allclose(res.mean(), (1, 4, 0.25, 27))

    with pytest.raises(TypeError):
        f ** ((1,) * 5)


def test_add_subtract(mesh_3d):
    # Scalar fields
    f1 = df.Field(mesh_3d, nvdim=1, value=1.2)
    f2 = df.Field(mesh_3d, nvdim=1, value=-0.2)
    res = f1 + f2
    assert np.allclose(res.mean(), 1)
    res = f1 - f2
    assert np.allclose(res.mean(), 1.4)
    f1 += f2
    assert np.allclose(f1.mean(), 1)
    f1 -= f2
    assert np.allclose(f1.mean(), 1.2)

    # Vector fields
    f1 = df.Field(mesh_3d, nvdim=3, value=(1, 2, 3))
    f2 = df.Field(mesh_3d, nvdim=3, value=(-1, -3, -5))
    res = f1 + f2
    assert np.allclose(res.mean(), (0, -1, -2))
    res = f1 - f2
    assert np.allclose(res.mean(), (2, 5, 8))
    f1 += f2
    assert np.allclose(f1.mean(), (0, -1, -2))
    f1 -= f2
    assert np.allclose(f1.mean(), (1, 2, 3))

    # Artithmetic checks
    assert f1 + f2 + (1, 1, 1) == (1, 1, 1) + f2 + f1
    assert f1 - f2 - (0, 0, 0) == (0, 0, 0) - (f2 - f1)
    assert f1 + (f1 + f2) == (f1 + f1) + f2
    assert f1 - (f1 + f2) == f1 - f1 - f2
    assert f1 + f2 - f1 == f2 + (0, 0, 0)

    # Constants
    f1 = df.Field(mesh_3d, nvdim=1, value=1.2)
    f2 = df.Field(mesh_3d, nvdim=3, value=(-1, -3, -5))
    res = f1 + 2
    assert np.allclose(res.mean(), 3.2)
    res = f1 - 1.2
    assert np.allclose(res.mean(), 0)
    f1 += 2.5
    assert np.allclose(f1.mean(), 3.7)
    f1 -= 3.7
    assert np.allclose(f1.mean(), 0)
    res = f2 + (1, 3, 5)
    assert np.allclose(res.mean(), (0, 0, 0))
    res = f2 - (1, 2, 3)
    assert np.allclose(res.mean(), (-2, -5, -8))
    f2 += (1, 1, 1)
    assert np.allclose(f2.mean(), (0, -2, -4))
    f2 -= (-1, -2, 3)
    assert np.allclose(f2.mean(), (1, 0, -7))

    # Exceptions
    with pytest.raises(TypeError):
        res = f1 + "2"

    # Fields with different dimensions
    res = f1 + f2
    assert np.allclose(res.mean(), (1, 0, -7))

    # Fields defined on different meshes
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
    f1 = df.Field(mesh1, nvdim=1, value=1.2)
    f2 = df.Field(mesh2, nvdim=1, value=1)
    with pytest.raises(ValueError):
        res = f1 + f2
    with pytest.raises(ValueError):
        f1 += f2
    with pytest.raises(ValueError):
        f1 -= f2


def test_mul_truediv(mesh_3d):
    # Scalar fields
    f1 = df.Field(mesh_3d, nvdim=1, value=1.2)
    f2 = df.Field(mesh_3d, nvdim=1, value=-2)
    res = f1 * f2
    assert np.allclose(res.mean(), -2.4)
    res = f1 / f2
    assert np.allclose(res.mean(), -0.6)
    f1 *= f2
    assert np.allclose(f1.mean(), -2.4)
    f1 /= f2
    assert np.allclose(f1.mean(), 1.2)

    # Scalar field with a constant
    f = df.Field(mesh_3d, nvdim=1, value=5)
    res = f * 2
    assert np.allclose(res.mean(), 10)
    res = 3 * f
    assert np.allclose(res.mean(), 15)
    res = f * (1, 2, 3)
    assert np.allclose(res.mean(), (5, 10, 15))
    res = (1, 2, 3) * f
    assert np.allclose(res.mean(), (5, 10, 15))
    res = f / 2
    assert np.allclose(res.mean(), 2.5)
    res = 10 / f
    assert np.allclose(res.mean(), 2)
    res = (5, 10, 15) / f
    assert np.allclose(res.mean(), (1, 2, 3))
    f *= 10
    assert np.allclose(f.mean(), 50)
    f /= 10
    assert np.allclose(f.mean(), 5)

    # Scalar field with a vector field
    f1 = df.Field(mesh_3d, nvdim=1, value=2)
    f2 = df.Field(mesh_3d, nvdim=3, value=(-1, -3, 5))
    res = f1 * f2  # __mul__
    assert np.allclose(res.mean(), (-2, -6, 10))
    res = f2 * f1  # __rmul__
    assert np.allclose(res.mean(), (-2, -6, 10))
    res = f2 / f1  # __truediv__
    assert np.allclose(res.mean(), (-0.5, -1.5, 2.5))
    f2 *= f1  # __imul__
    assert np.allclose(f2.mean(), (-2, -6, 10))
    f2 /= f1  # __truediv__
    assert np.allclose(f2.mean(), (-1, -3, 5))
    res = f1 / f2  # __rtruediv__
    assert np.allclose(res.mean(), (-2, -2 / 3, 2 / 5))

    # Vector field with a scalar
    f = df.Field(mesh_3d, nvdim=3, value=(1, 2, 0))
    res = f * 2
    assert np.allclose(res.mean(), (2, 4, 0))
    res = 5 * f
    assert np.allclose(res.mean(), (5, 10, 0))
    res = f / 2
    assert np.allclose(res.mean(), (0.5, 1, 0))
    f *= 2
    assert np.allclose(f.mean(), (2, 4, 0))
    f /= 2
    assert np.allclose(f.mean(), (1, 2, 0))
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        res = 10 / f
    assert np.allclose(res.mean(), (10, 5, np.inf))

    # Further checks
    f1 = df.Field(mesh_3d, nvdim=1, value=2)
    f2 = df.Field(mesh_3d, nvdim=3, value=(-1, -3, -5))
    assert f1 * f2 == f2 * f1
    assert 1.3 * f2 == f2 * 1.3
    assert -5 * f2 == f2 * (-5)
    assert (1, 2.2, -1) * f1 == f1 * (1, 2.2, -1)
    assert f1 * (f1 * f2) == (f1 * f1) * f2
    assert f1 * f2 / f1 == f2
    assert np.allclose((f2 * f2).mean(), (1, 9, 25))
    assert np.allclose((f2 / f2).mean(), (1, 1, 1))

    # Exceptions
    f1 = df.Field(mesh_3d, nvdim=1, value=1.2)
    f2 = df.Field(mesh_3d, nvdim=3, value=(-1, -3, -5))
    with pytest.raises(TypeError):
        res = f2 * "a"
    with pytest.raises(TypeError):
        res = "a" / f1
    with pytest.raises(TypeError):
        f2 *= "a"
    with pytest.raises(TypeError):
        f2 /= "a"

    # Fields defined on different meshes
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
    f1 = df.Field(mesh1, nvdim=1, value=1.2)
    f2 = df.Field(mesh2, nvdim=1, value=1)
    with pytest.raises(ValueError):
        res = f1 * f2
    with pytest.raises(ValueError):
        res = f1 / f2
    with pytest.raises(ValueError):
        f1 *= f2
    with pytest.raises(ValueError):
        f1 /= f2


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_dot(mesh_3d, nvdim):
    # Zero vectors
    f1 = df.Field(mesh_3d, nvdim=nvdim, value=(0,) * nvdim)
    res = f1.dot(f1)
    assert res.nvdim == 1
    assert np.allclose(res.mean(), 0)

    # Check norm computed using dot product
    assert f1.norm.allclose((f1.dot(f1)) ** (0.5))

    # Create a list of othogonal fields
    fields = []
    for i in range(nvdim):
        v = np.zeros(nvdim)
        v[i] = 1
        temp = df.Field(mesh_3d, nvdim=nvdim, value=v)
        fields.append(temp)

    # Check if orthogonal and commutative

    for i, j in itertools.product(range(nvdim), range(nvdim)):
        assert fields[i].dot(fields[j]).mean() == 0 if i != j else 1
        assert np.allclose(
            fields[i].dot(fields[j]).mean(), fields[j].dot(fields[i]).mean()
        )

    # Vector field with a constant
    f = df.Field(mesh_3d, nvdim=nvdim, value=np.arange(nvdim))
    res = f.dot(np.ones(nvdim))
    assert np.allclose(res.mean(), np.sum(np.arange(nvdim)))


def test_dot_3d(mesh_3d):
    # Spatially varying vectors
    def value_fun1(point):
        x, y, z = point
        return (x, y, z)

    def value_fun2(point):
        x, y, z = point
        return (z, x, y)

    f1 = df.Field(mesh_3d, nvdim=3, value=value_fun1)
    f2 = df.Field(mesh_3d, nvdim=3, value=value_fun2)

    # Check if commutative
    assert f1.dot(f2) == f2.dot(f1)

    # The dot product should be x*z + y*x + z*y
    assert np.allclose((f1.dot(f2))((15e-9, 5e-9, 12.5e-9)), 3.25e-16, atol=0)
    assert np.allclose((f1.dot(f2))((15e-9, 5e-9, 7.5e-9)), 2.25e-16, atol=0)
    assert np.allclose((f1.dot(f2))((5e-9, 5e-9, 12.5e-9)), 1.5e-16, atol=0)

    # Exceptions
    f1 = df.Field(mesh_3d, nvdim=1, value=1.2)
    f2 = df.Field(mesh_3d, nvdim=3, value=(-1, -3, -5))
    with pytest.raises(ValueError):
        f1.dot(f2)
    with pytest.raises(ValueError):
        f1.dot(f2)
    with pytest.raises(TypeError):
        f1.dot(3)

    # Fields defined on different meshes
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
    f1 = df.Field(mesh1, nvdim=3, value=(1, 2, 3))
    f2 = df.Field(mesh2, nvdim=3, value=(3, 2, 1))
    with pytest.raises(ValueError):
        f1.dot(f2)


def test_cross(mesh_3d):
    # Zero vectors
    f1 = df.Field(mesh_3d, nvdim=3, value=(0, 0, 0))
    res = f1.cross(f1)
    assert res.nvdim == 3
    assert np.allclose(res.mean(), (0, 0, 0))

    # Orthogonal vectors
    f1 = df.Field(mesh_3d, nvdim=3, value=(1, 0, 0))
    f2 = df.Field(mesh_3d, nvdim=3, value=(0, 1, 0))
    f3 = df.Field(mesh_3d, nvdim=3, value=(0, 0, 1))
    assert np.allclose((f1.cross(f2)).mean(), (0, 0, 1))
    assert np.allclose((f1.cross(f3)).mean(), (0, -1, 0))
    assert np.allclose((f2.cross(f3)).mean(), (1, 0, 0))
    assert np.allclose((f1.cross(f1)).mean(), (0, 0, 0))
    assert np.allclose((f2.cross(f2)).mean(), (0, 0, 0))
    assert np.allclose((f3.cross(f3)).mean(), (0, 0, 0))

    # Constants
    assert np.allclose((f1.cross((0, 1, 0))).mean(), (0, 0, 1))

    # Check if not comutative
    assert f1.cross(f2) == -(f2.cross(f1))
    assert f1.cross(f3) == -(f3.cross(f1))
    assert f2.cross(f3) == -(f3.cross(f2))

    f1 = df.Field(mesh_3d, nvdim=3, value=lambda point: (point[0], point[1], point[2]))
    f2 = df.Field(mesh_3d, nvdim=3, value=lambda point: (point[2], point[0], point[1]))

    # The cross product should be
    # (y**2-x*z, z**2-x*y, x**2-y*z)
    assert np.allclose(
        (f1.cross(f2))((35e-9, 15e-9, 7.5e-9)),
        (-3.75e-17, -4.6875e-16, 1.1125e-15),
        atol=0,
    )
    assert np.allclose(
        (f2.cross(f1))((35e-9, 15e-9, 7.5e-9)),
        (3.75e-17, 4.6875e-16, -1.1125e-15),
        atol=0,
    )
    assert np.allclose(
        (f1.cross(f2))((45e-9, 25e-9, 12.5e-9)),
        (6.25e-17, -9.6875e-16, 1.7125e-15),
        atol=0,
    )

    # Exceptions
    f1 = df.Field(mesh_3d, nvdim=1, value=1.2)
    f2 = df.Field(mesh_3d, nvdim=3, value=(-1, -3, -5))
    with pytest.raises(TypeError):
        res = f1.cross(2)
    with pytest.raises(ValueError):
        res = f1.cross(f2)

    # Fields defined on different meshes
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
    f1 = df.Field(mesh1, nvdim=3, value=(1, 2, 3))
    f2 = df.Field(mesh2, nvdim=3, value=(3, 2, 1))
    with pytest.raises(ValueError):
        res = f1.cross(f2)


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_lshift(valid_mesh, nvdim):
    f_list = [df.Field(valid_mesh, nvdim=1, value=i + 1) for i in range(nvdim)]
    res = f_list[0]
    for f in f_list[1:]:
        res = res << f

    assert res.nvdim == nvdim
    assert np.allclose(res.mean(), tuple(range(1, nvdim + 1)))

    # Test for different dimensions
    f1 = df.Field(valid_mesh, nvdim=1, value=1.2)
    f2 = df.Field(valid_mesh, nvdim=nvdim, value=tuple(range(-nvdim, 0)))
    res = f1 << f2
    assert np.allclose(res.mean(), (1.2, *range(-nvdim, 0)))

    # Test for constants
    f1 = df.Field(valid_mesh, nvdim=1, value=1.2)
    res = f1 << tuple(range(nvdim))
    assert np.allclose(res.mean(), (1.2, *range(nvdim)))
    res = tuple(range(nvdim)) << f1
    assert np.allclose(res.mean(), (*range(nvdim), 1.2))

    with pytest.raises(TypeError):
        _ = "a" << f1
    with pytest.raises(TypeError):
        _ = f1 << "a"


def test_lshift_different_mesh():
    mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), n=(1, 1, 1))
    mesh2 = df.Mesh(p1=(0, 0, 0), p2=(3, 3, 3), n=(1, 1, 1))
    f1 = df.Field(mesh1, nvdim=1, value=1.2)
    f2 = df.Field(mesh2, nvdim=1, value=1)
    with pytest.raises(ValueError):
        _ = f1 << f2


def test_all_operators():
    p1 = (0, 0, 0)
    p2 = (5e-9, 5e-9, 10e-9)
    n = (2, 2, 1)
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    f1 = df.Field(mesh, nvdim=1, value=2)
    f2 = df.Field(mesh, nvdim=3, value=(-4, 0, 1))
    res = (
        ((+f1 / 2 + f2.x) ** 2 - 2 * f1 * 3) / (-f2.z)
        - 2 * f2.y
        + 1 / f2.z**2
        + f2.dot(f2)
    )
    assert np.all(res.array == 21)

    res = 1 + f1 + 0 * f2.x - 3 * f2.y / 3
    assert res.mean() == 3


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "mode", ["constant", "reflect", "symmetric", "median"]
)  # Selection of possible modes
def test_pad(valid_mesh, nvdim, mode):
    field = df.Field(valid_mesh, nvdim=nvdim, value=(1,) * nvdim)

    for dim in valid_mesh.region.dims:
        pf = field.pad({dim: (1, 1)}, mode=mode)
        pad_size = valid_mesh.n.copy()
        pad_size[valid_mesh.region._dim2index(dim)] += 2
        assert pf.array.shape == (*pad_size, nvdim)
        index = [
            (0, 0),
        ] * (valid_mesh.region.ndim + 1)
        index[valid_mesh.region._dim2index(dim)] = (1, 1)
        nppad = np.pad(field.array, index, mode=mode)
        assert np.allclose(pf.array, nppad)


def test_pad_explicit():
    mesh = df.Mesh(p1=(0.0), p2=(1.0), n=(5))
    f = mesh.coordinate_field()

    pad_f = f.pad({"x": (1, 1)}, mode="constant")
    assert np.allclose(pad_f.array[0, 0], 0)
    assert np.allclose(pad_f.array[-1, 0], 0)

    pad_f = f.pad({"x": (1, 1)}, mode="symmetric")
    assert np.allclose(pad_f.array[0, 0], 0.1)
    assert np.allclose(pad_f.array[-1, 0], 0.9)

    mesh = df.Mesh(p1=(0.0, 0.0, 0.0), p2=(1.0, 1.0, 1.0), n=(5, 5, 5))
    f = mesh.coordinate_field()

    pad_f = f.pad({"z": (1, 1)}, mode="constant")
    assert np.allclose(pad_f.array[:, :, 0, 2], 0)
    assert np.allclose(pad_f.array[:, :, -1, 2], 0)

    pad_f = f.pad({"z": (1, 2)}, mode="symmetric")
    assert np.allclose(pad_f.array[:, :, 0, 2], 0.1)
    assert np.allclose(pad_f.array[:, :, -1, 2], 0.7)
    assert np.allclose(pad_f.array[:, :, -2, 2], 0.9)

    pad_f = f.pad({"z": (1, 2), "x": (1, 0)}, mode="symmetric")
    assert np.allclose(pad_f.array[:, :, 0, 2], 0.1)
    assert np.allclose(pad_f.array[:, :, -1, 2], 0.7)
    assert np.allclose(pad_f.array[:, :, -2, 2], 0.9)
    assert np.allclose(pad_f.array[0, :, :, 0], 0.1)
    assert np.allclose(pad_f.array[-1, :, :, 0], 0.9)
    assert np.allclose(pad_f.array[-2, :, :, 0], 0.7)


def test_diff():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)

    # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    f = df.Field(mesh, nvdim=1, value=0)

    assert isinstance(f.diff("x"), df.Field)
    assert np.allclose(f.diff("x", order=1).mean(), 0)
    assert np.allclose(f.diff("y", order=1).mean(), 0)
    assert np.allclose(f.diff("z", order=1).mean(), 0)
    assert np.allclose(f.diff("x", order=2).mean(), 0)
    assert np.allclose(f.diff("y", order=2).mean(), 0)
    assert np.allclose(f.diff("z", order=2).mean(), 0)

    # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def value_fun(point):
        x, y, z = point
        return x + y + z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x", order=1).mean(), 1)
    assert np.allclose(f.diff("y", order=1).mean(), 1)
    assert np.allclose(f.diff("z", order=1).mean(), 1)
    assert np.allclose(f.diff("x", order=2).mean(), 0)
    assert np.allclose(f.diff("y", order=2).mean(), 0)
    assert np.allclose(f.diff("z", order=2).mean(), 0)

    # f(x, y, z) = x*y + 2*y + x*y*z ->
    # grad(f) = (y+y*z, x+2+x*z, x*y)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def value_fun(point):
        x, y, z = point
        return x * y + 2 * y + x * y * z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x")((7, 5, 1)), 10)
    assert np.allclose(f.diff("y")((7, 5, 1)), 16)
    assert np.allclose(f.diff("z")((7, 5, 1)), 35)
    assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 0)
    assert np.allclose(f.diff("y", order=2)((1, 1, 1)), 0)
    assert np.allclose(f.diff("z", order=2)((1, 1, 1)), 0)

    # f(x, y, z) = (0, 0, 0)
    # -> dfdx = (0, 0, 0)
    # -> dfdy = (0, 0, 0)
    # -> dfdz = (0, 0, 0)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

    assert isinstance(f.diff("y"), df.Field)
    assert np.allclose(f.diff("x").mean(), (0, 0, 0))
    assert np.allclose(f.diff("y").mean(), (0, 0, 0))
    assert np.allclose(f.diff("z").mean(), (0, 0, 0))

    # f(x, y, z) = (x,  y,  z)
    # -> dfdx = (1, 0, 0)
    # -> dfdy = (0, 1, 0)
    # -> dfdz = (0, 0, 1)
    def value_fun(point):
        x, y, z = point
        return (x, y, z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.diff("x").mean(), (1, 0, 0))
    assert np.allclose(f.diff("y").mean(), (0, 1, 0))
    assert np.allclose(f.diff("z").mean(), (0, 0, 1))

    # f(x, y, z) = (x*y, y*z, x*y*z)
    # -> dfdx = (y, 0, y*z)
    # -> dfdy = (x, z, x*z)
    # -> dfdz = (0, y, x*y)
    def value_fun(point):
        x, y, z = point
        return (x * y, y * z, x * y * z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.diff("x")((3, 1, 3)), (1, 0, 3))
    assert np.allclose(f.diff("y")((3, 1, 3)), (3, 3, 9))
    assert np.allclose(f.diff("z")((3, 1, 3)), (0, 1, 3))
    assert np.allclose(f.diff("x")((5, 3, 5)), (3, 0, 15))
    assert np.allclose(f.diff("y")((5, 3, 5)), (5, 5, 25))
    assert np.allclose(f.diff("z")((5, 3, 5)), (0, 3, 15))

    # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
    # -> dfdx = (y, 1, y*z)
    # -> dfdy = (x, -2, x*z)
    # -> dfdz = (0, 0, x*y)
    def value_fun(point):
        x, y, z = point
        return (3 + x * y, x - 2 * y, x * y * z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.diff("x")((7, 5, 1)), (5, 1, 5))
    assert np.allclose(f.diff("y")((7, 5, 1)), (7, -2, 7))
    assert np.allclose(f.diff("z")((7, 5, 1)), (0, 0, 35))
    assert np.allclose(f.diff("x", order=2).mean(), 0)
    assert np.allclose(f.diff("y", order=2).mean(), 0)
    assert np.allclose(f.diff("z", order=2).mean(), 0)

    # f(x, y, z) = 2*x*x + 2*y*y + 3*z*z
    # -> grad(f) = (4, 4, 6)
    def value_fun(point):
        x, y, z = point
        return 2 * x * x + 2 * y * y + 3 * z * z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x", order=2).mean(), 4)
    assert np.allclose(f.diff("y", order=2).mean(), 4)
    assert np.allclose(f.diff("z", order=2).mean(), 6)

    # f(x, y, z) = (2*x*x, 2*y*y, 3*z*z)
    def value_fun(point):
        x, y, z = point
        return (2 * x * x, 2 * y * y, 3 * z * z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.diff("x", order=2).mean(), (4, 0, 0))
    assert np.allclose(f.diff("y", order=2).mean(), (0, 4, 0))
    assert np.allclose(f.diff("z", order=2).mean(), (0, 0, 6))

    # Test invalid direction
    with pytest.raises(ValueError):
        f.diff("q")


def test_diff_small():
    p1 = (0, 0, 0)
    p2 = (3, 3, 3)
    n = (3, 3, 3)

    # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    f = df.Field(mesh, nvdim=1, value=0)

    assert isinstance(f.diff("x"), df.Field)
    assert np.allclose(f.diff("x", order=1).mean(), 0)
    assert np.allclose(f.diff("y", order=1).mean(), 0)
    assert np.allclose(f.diff("z", order=1).mean(), 0)

    # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    def value_fun(point):
        x, y, z = point
        return x + y + z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x", order=1).mean(), 1)
    assert np.allclose(f.diff("y", order=1).mean(), 1)
    assert np.allclose(f.diff("z", order=1).mean(), 1)

    # f(x, y, z) = x*y + 2*y + x*y*z ->
    # grad(f) = (y+y*z, x+2+x*z, x*y)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    def value_fun(point):
        x, y, z = point
        return x * y + 2 * y + x * y * z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x")((2.5, 1.5, 1.5)), 3.75)
    assert np.allclose(f.diff("y")((1.5, 2.5, 1.5)), 5.75)
    assert np.allclose(f.diff("z")((1.5, 1.5, 2.5)), 2.25)

    # f(x, y, z) = (0, 0, 0)
    # -> dfdx = (0, 0, 0)
    # -> dfdy = (0, 0, 0)
    # -> dfdz = (0, 0, 0)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

    assert np.allclose(f.diff("x").mean(), (0, 0, 0))
    assert np.allclose(f.diff("y").mean(), (0, 0, 0))
    assert np.allclose(f.diff("z").mean(), (0, 0, 0))

    # f(x, y, z) = (x,  y,  z)
    # -> dfdx = (1, 0, 0)
    # -> dfdy = (0, 1, 0)
    # -> dfdz = (0, 0, 1)
    def value_fun(point):
        x, y, z = point
        return (x, y, z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.diff("x").mean(), (1, 0, 0))
    assert np.allclose(f.diff("y").mean(), (0, 1, 0))
    assert np.allclose(f.diff("z").mean(), (0, 0, 1))

    # f(x, y, z) = (x*y, y*z, x*y*z)
    # -> dfdx = (y, 0, y*z)
    # -> dfdy = (x, z, x*z)
    # -> dfdz = (0, y, x*y)
    def value_fun(point):
        x, y, z = point
        return (x * y, y * z, x * y * z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.diff("x")((1.5, 1.5, 1.5)), (1.5, 0, 2.25))
    assert np.allclose(f.diff("x")((2.5, 1.5, 1.5)), (1.5, 0, 2.25))
    assert np.allclose(f.diff("y")((1.5, 1.5, 1.5)), (1.5, 1.5, 2.25))
    assert np.allclose(f.diff("y")((1.5, 2.5, 1.5)), (1.5, 1.5, 2.25))
    assert np.allclose(f.diff("z")((1.5, 1.5, 1.5)), (0, 1.5, 2.25))
    assert np.allclose(f.diff("z")((1.5, 1.5, 2.5)), (0, 1.5, 2.25))

    p1 = (0, 0, 0)
    p2 = (4, 4, 4)
    n = (4, 4, 4)
    # f(x, y, z) = 0 -> grad(f) = (0, 0, 0)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    f = df.Field(mesh, nvdim=1, value=0)

    assert np.allclose(f.diff("x", order=2).mean(), 0)
    assert np.allclose(f.diff("y", order=2).mean(), 0)
    assert np.allclose(f.diff("z", order=2).mean(), 0)

    # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    def value_fun(point):
        x, y, z = point
        return x + y + z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x", order=2).mean(), 0)
    assert np.allclose(f.diff("y", order=2).mean(), 0)
    assert np.allclose(f.diff("z", order=2).mean(), 0)

    # f(x, y, z) = x*y + 2*y + x*y*z ->
    # grad(f) = (y+y*z, x+2+x*z, x*y)
    # No BC
    mesh = df.Mesh(p1=p1, p2=p2, n=n)

    def value_fun(point):
        x, y, z = point
        return x * y + 2 * y + x * y * z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x", order=2)((1.5, 1.5, 1.5)), 0)
    assert np.allclose(f.diff("y", order=2)((1.5, 1.5, 1.5)), 0)
    assert np.allclose(f.diff("z", order=2)((1.5, 1.5, 1.5)), 0)

    # f(x, y, z) = 2*x*x + 2*y*y + 3*z*z
    # -> grad(f) = (4, 4, 6)
    def value_fun(point):
        x, y, z = point
        return 2 * x * x + 2 * y * y + 3 * z * z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    assert np.allclose(f.diff("x", order=2).mean(), 4)
    assert np.allclose(f.diff("y", order=2).mean(), 4)
    assert np.allclose(f.diff("z", order=2).mean(), 6)

    # f(x, y, z) = (2*x*x, 2*y*y, 3*z*z)
    def value_fun(point):
        x, y, z = point
        return (2 * x * x, 2 * y * y, 3 * z * z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.diff("x", order=2)((1.5, 1.5, 1.5)), (4, 0, 0))
    assert np.allclose(f.diff("x", order=2)((3.5, 1.5, 1.5)), (4, 0, 0))
    assert np.allclose(f.diff("y", order=2)((1.5, 1.5, 1.5)), (0, 4, 0))
    assert np.allclose(f.diff("y", order=2)((1.5, 3.5, 1.5)), (0, 4, 0))
    assert np.allclose(f.diff("z", order=2)((1.5, 1.5, 1.5)), (0, 0, 6))
    assert np.allclose(f.diff("z", order=2)((1.5, 1.5, 3.5)), (0, 0, 6))


def test_diff_pbc():
    p1 = (0.0, 0.0, 0.0)
    p2 = (12.0, 8.0, 6.0)
    cell = (2, 2, 2)

    mesh_nopbc = df.Mesh(p1=p1, p2=p2, cell=cell)
    mesh_pbc = df.Mesh(p1=p1, p2=p2, cell=cell, bc="xyz")

    # Scalar field
    def value_fun(point):
        x, y, z = point
        return x * y * z

    # No PBC
    f = df.Field(mesh_nopbc, nvdim=1, value=value_fun)
    assert np.allclose(f.diff("x")((11, 1, 1)), 1)
    assert np.allclose(f.diff("y")((1, 7, 1)), 1)
    assert np.allclose(f.diff("z")((1, 1, 5)), 1)

    # PBC
    f = df.Field(mesh_pbc, nvdim=1, value=value_fun)
    assert np.allclose(f.diff("x")((11, 1, 1)), -2)
    assert np.allclose(f.diff("y")((1, 7, 1)), -1)
    assert np.allclose(f.diff("z")((1, 1, 5)), -0.5)

    # Vector field
    def value_fun(point):
        x, y, z = point
        return (x * y * z,) * 3

    # No PBC
    f = df.Field(mesh_nopbc, nvdim=3, value=value_fun)
    assert np.allclose(f.diff("x")((11, 1, 1)), (1, 1, 1))
    assert np.allclose(f.diff("y")((1, 7, 1)), (1, 1, 1))
    assert np.allclose(f.diff("z")((1, 1, 5)), (1, 1, 1))

    # PBC
    f = df.Field(mesh_pbc, nvdim=3, value=value_fun)
    assert np.allclose(f.diff("x")((11, 1, 1)), (-2, -2, -2))
    assert np.allclose(f.diff("y")((1, 7, 1)), (-1, -1, -1))
    assert np.allclose(f.diff("z")((1, 1, 5)), (-0.5, -0.5, -0.5))

    # Higher order derivatives
    def value_fun(point):
        x, y, z = point
        return x**2

    f = df.Field(mesh_nopbc, nvdim=1, value=value_fun)
    assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 2)

    f = df.Field(mesh_pbc, nvdim=1, value=value_fun)
    assert np.allclose(f.diff("x", order=2)((1, 1, 1)), 32)


def test_diff_single_cell():
    p1 = (0, 0, 0)
    p2 = (10, 10, 2)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # Scalar field: f(x, y, z) = x + y + z
    # -> grad(f) = (1, 1, 1)
    def value_fun(point):
        x, y, z = point
        return x + y + z

    f = df.Field(mesh, nvdim=1, value=value_fun)

    # only one cell in the z-direction
    assert f.sel(x=(0, 1)).diff("x").mean() == 0
    assert f.sel(y=(0, 1)).diff("y").mean() == 0
    assert f.diff("z").mean() == 0

    # Vector field: f(x, y, z) = (x, y, z)
    # -> grad(f) = (1, 1, 1)
    def value_fun(point):
        x, y, z = point
        return (x, y, z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    # only one cell in the z-direction
    assert np.allclose(f.sel(x=(0, 1)).diff("x").mean(), (0, 0, 0))
    assert np.allclose(f.sel(y=(0, 1)).diff("y").mean(), (0, 0, 0))
    assert np.allclose(f.diff("z").mean(), (0, 0, 0))


def test_diff_valid():
    # 1d mesh
    mesh = df.Mesh(p1=0e-9, p2=10e-9, n=10)
    valid = [True, False, False, True, True, True, False, True, False, False]
    f = df.Field(mesh, nvdim=1, value=lambda p: p[0] ** 2, valid=valid)

    assert np.allclose(f.diff("x").array[:3], 0)
    assert np.allclose(f.diff("x").array[3:6, 0], 2 * f.mesh.cells[0][3:6])
    assert np.allclose(f.diff("x").array[6:], 0)
    assert np.allclose(
        f.diff("x", restrict2valid=False).array[..., 0], 2 * f.mesh.cells[0]
    )

    # 3d mesh
    p1 = (0, 0, 0)
    p2 = (20, 10, 10)
    cell = (2, 2, 2)

    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def value_fun(point):
        x, y, z = point
        return (x, y, z)

    def valid_fun(point):
        x, y, z = point
        if x > 6:
            return False
        else:
            return True

    f = df.Field(mesh, nvdim=3, value=value_fun, valid=valid_fun)

    assert np.allclose(f.diff("x").array[:3, ...], (1, 0, 0))
    assert np.allclose(f.diff("x").array[3:, ...], (0, 0, 0))
    assert np.allclose(f.diff("x", restrict2valid=False).array, (1, 0, 0))
    assert np.allclose(f.diff("y").array[:3, ...], (0, 1, 0))
    assert np.allclose(f.diff("y").array[3:, ...], (0, 0, 0))
    assert np.allclose(f.diff("y", restrict2valid=False).array, (0, 1, 0))
    assert np.allclose(f.diff("z").array[:3, ...], (0, 0, 1))
    assert np.allclose(f.diff("z").array[3:, ...], (0, 0, 0))
    assert np.allclose(f.diff("z", restrict2valid=False).array, (0, 0, 1))

    def valid_fun(point):
        x, y, z = point
        if x > 2 and x < 8:
            return True
        else:
            return False

    f = df.Field(mesh, nvdim=3, value=value_fun, valid=valid_fun)
    assert np.allclose(f.diff("x").array[1:4, ...], (1, 0, 0))
    assert np.allclose(f.diff("x").array[0, ...], (0, 0, 0))
    assert np.allclose(f.diff("x").array[4:, ...], (0, 0, 0))
    assert np.allclose(f.diff("x", restrict2valid=False).array, (1, 0, 0))
    assert np.allclose(f.diff("y").array[1:4, ...], (0, 1, 0))
    assert np.allclose(f.diff("y").array[0, ...], (0, 0, 0))
    assert np.allclose(f.diff("y").array[4:, ...], (0, 0, 0))
    assert np.allclose(f.diff("y", restrict2valid=False).array, (0, 1, 0))
    assert np.allclose(f.diff("z").array[1:4, ...], (0, 0, 1))
    assert np.allclose(f.diff("z").array[0, ...], (0, 0, 0))
    assert np.allclose(f.diff("z").array[4:, ...], (0, 0, 0))
    assert np.allclose(f.diff("z", restrict2valid=False).array, (0, 0, 1))

    def valid_fun(point):
        x, y, z = point
        if x > 4 and x < 8 and y < 5:
            return False
        elif x > 12 and x < 15:
            return False
        else:
            return True

    f = df.Field(mesh, nvdim=3, value=value_fun, valid=valid_fun)
    assert np.allclose(f.diff("x").array[f.valid[..., 0]], (1, 0, 0))
    assert np.allclose(f.diff("x").array[~f.valid[..., 0]], (0, 0, 0))
    assert np.allclose(f.diff("x", restrict2valid=False).array, (1, 0, 0))
    assert np.allclose(f.diff("y").array[f.valid[..., 0]], (0, 1, 0))
    assert np.allclose(f.diff("y").array[~f.valid[..., 0]], (0, 0, 0))
    assert np.allclose(f.diff("y", restrict2valid=False).array, (0, 1, 0))
    assert np.allclose(f.diff("z").array[f.valid[..., 0]], (0, 0, 1))
    assert np.allclose(f.diff("z").array[~f.valid[..., 0]], (0, 0, 0))
    assert np.allclose(f.diff("z", restrict2valid=False).array, (0, 0, 1))


def test_grad(valid_mesh):
    # f() = 0 -> grad(f) = (0, 0, 0)
    f = df.Field(valid_mesh, nvdim=1, value=0)
    assert isinstance(f.grad, df.Field)
    assert np.allclose(f.grad.mean(), (0,) * valid_mesh.region.ndim)

    # f(x, y, z) = x + y + z -> grad(f) = (1, 1, 1)
    def value_fun(point):
        return np.sum(point)

    f = df.Field(valid_mesh, nvdim=1, value=value_fun)
    assert isinstance(f.grad, df.Field)
    assert np.allclose(f.grad.mean(), [0 if num == 1 else 1 for num in valid_mesh.n])

    # f(x, y, z) = x^2 + y^2 + z^2 -> grad(f) = (2x, 2y, 2z)
    def value_fun(point):
        return np.sum(np.square(point))

    f = df.Field(valid_mesh, nvdim=1, value=value_fun)
    assert isinstance(f.grad, df.Field)
    assert np.allclose(
        f.grad.mean(),
        [
            0 if num == 1 else 2 * cent
            for num, cent in zip(valid_mesh.n, valid_mesh.region.center)
        ],
    )

    # Test mixing terms
    # f(x, y, z) = x^2 + x*y + x*z -> grad(f) = (2x+y+z, x, x)
    def value_fun(point):
        return np.sum(point[0] * point)

    f = df.Field(valid_mesh, nvdim=1, value=value_fun)
    assert isinstance(f.grad, df.Field)
    result = np.full(valid_mesh.region.ndim, valid_mesh.region.center[0])
    result[0] += np.sum(valid_mesh.region.center)
    assert np.allclose(
        f.grad.mean(),
        [0 if num == 1 else result[i] for i, num in enumerate(valid_mesh.n)],
    )


@pytest.mark.parametrize("nvdim", [2, 3, 4, 5])
def test_grad_exception(valid_mesh, nvdim):
    # Grad only on scalar fields
    f = df.Field(valid_mesh, nvdim=nvdim, value=0)
    with pytest.raises(ValueError):
        f.grad


def test_div(valid_mesh):
    nvdim = valid_mesh.region.ndim
    vdims = ["v" + d for d in valid_mesh.region.dims]  # ensure unique vdims
    vdim_mapping = dict(zip(vdims, valid_mesh.region.dims))
    f = df.Field(valid_mesh, nvdim=nvdim, vdims=vdims, vdim_mapping=vdim_mapping)

    assert isinstance(f.div, df.Field)
    assert np.allclose(f.div.mean(), 0)

    # f(x, y, z) = (x, y, z) -> div(f) = 1 + 1 + 1
    f = df.Field(
        valid_mesh,
        nvdim=nvdim,
        vdims=vdims,
        vdim_mapping=vdim_mapping,
        value=valid_mesh.coordinate_field(),
    )
    assert np.allclose(
        f.div.mean(), np.sum([0 if num == 1 else 1 for num in valid_mesh.n])
    )

    # f(x, y, z) = (x^2, y^2, z^2) -> div(f) = 2x + 2y + 1z
    def value_fun(point):
        return np.square(point)

    f = df.Field(
        valid_mesh, nvdim=nvdim, vdims=vdims, vdim_mapping=vdim_mapping, value=value_fun
    )
    assert np.allclose(
        f.div.mean(),
        np.sum(
            [
                0 if num == 1 else 2 * cent
                for num, cent in zip(valid_mesh.n, valid_mesh.region.center)
            ]
        ),
    )


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_div_exception(valid_mesh, nvdim):
    f = df.Field(valid_mesh, nvdim=nvdim)
    if valid_mesh.region.ndim != nvdim:
        # Wrong dimensions
        with pytest.raises(ValueError):
            f.div
    else:
        vdims = ["v" + d for d in valid_mesh.region.dims]
        f.vdims = vdims
        # Empty dictionary
        f.vdim_mapping = {}
        with pytest.raises(ValueError):
            f.div
        # Incorect dictionary
        if nvdim > 1:
            vdim_mapping = dict(zip(vdims, valid_mesh.region.dims))
            vdim_mapping[vdims[0]] = None
            f.vdim_mapping = vdim_mapping
            with pytest.raises(ValueError):
                f.div


def test_curl():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (2, 2, 2)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # f(x, y, z) = (0, 0, 0)
    # -> curl(f) = (0, 0, 0)
    f = df.Field(mesh, nvdim=3, value=(0, 0, 0))

    assert isinstance(f.curl, df.Field)
    assert f.curl.nvdim == 3
    assert np.allclose(f.curl.mean(), (0, 0, 0))

    # f(x, y, z) = (x, y, z)
    # -> curl(f) = (0, 0, 0)
    def value_fun(point):
        x, y, z = point
        return (x, y, z)

    f = df.Field(mesh, nvdim=3, value=value_fun)
    assert np.allclose(f.curl.mean(), (0, 0, 0))

    # f(x, y, z) = (x*y, y*z, x*y*z)
    # -> curl(f) = (x*z-y, -y*z, -x)
    def value_fun(point):
        x, y, z = point
        return (x * y, y * z, x * y * z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.curl((3, 1, 3)), (8, -3, -3))
    assert np.allclose(f.curl((5, 3, 5)), (22, -15, -5))

    # f(x, y, z) = (3+x*y, x-2*y, x*y*z)
    # -> curl(f) = (x*z, -y*z, 1-x)
    def value_fun(point):
        x, y, z = point
        return (3 + x * y, x - 2 * y, x * y * z)

    f = df.Field(mesh, nvdim=3, value=value_fun)

    assert np.allclose(f.curl((7, 5, 1)), (7, -5, -6))

    # Test vdims and dims
    # f(a, b, c) = (3+a*b, a-2*b, a*b*c)
    # -> curl(f) = (a*c, -b*c, 1-a)
    def value_fun(point):
        a, b, c = point
        return (3 + a * b, a - 2 * b, a * b * c)

    dims = ["a", "b", "c"]
    vdims = ["va", "vb", "vc"]
    vdim_mapping = dict(zip(vdims, dims))

    region = df.Region(p1=p1, p2=p2, dims=dims)
    mesh = df.Mesh(region=region, cell=cell)
    f = df.Field(mesh, nvdim=3, value=value_fun, vdims=vdims, vdim_mapping=vdim_mapping)

    assert np.allclose(f.curl((7, 5, 1)), (7, -5, -6))


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_curl_exception(valid_mesh, nvdim):
    # Exception
    f = df.Field(valid_mesh, nvdim=nvdim)

    if valid_mesh.region.ndim != 3 or nvdim != 3:
        with pytest.raises(ValueError):
            f.curl

    else:
        vdims = ["v" + d for d in valid_mesh.region.dims]
        f.vdims = vdims
        # Empty dictionary
        f.vdim_mapping = {}
        with pytest.raises(ValueError):
            f.curl

        # Incorect dictionary
        vdim_mapping = dict(zip(vdims, valid_mesh.region.dims))
        vdim_mapping[vdims[0]] = None
        f.vdim_mapping = vdim_mapping
        with pytest.raises(ValueError):
            f.curl


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_laplace(valid_mesh, nvdim):
    # f(x, y, z) = (0, 0, 0)
    # -> laplace(f) = 0
    f = df.Field(valid_mesh, nvdim=nvdim)

    assert isinstance(f.laplace, df.Field)
    assert f.laplace.nvdim == nvdim
    assert np.allclose(f.laplace.mean(), 0, atol=1e-5)
    f = df.Field(valid_mesh, nvdim=nvdim, value=(5,) * nvdim)
    assert np.allclose(f.laplace.mean(), 0, atol=1e-5)

    # f(x, y, z) = (x, y, z)
    # -> laplace(f) = 0
    f = valid_mesh.coordinate_field()
    assert isinstance(f.laplace, df.Field)
    assert np.allclose(f.laplace.mean(), 0, atol=1e-5)

    # f(x, y, z) = (x^2, y^2, z^2)
    # -> laplace(f) = (2, 2, 2)
    def value_fun(point):
        return np.full(nvdim, point[0] * point[0])

    f = df.Field(valid_mesh, nvdim=nvdim, value=value_fun)

    assert np.allclose(f.laplace.mean(), (2,) * nvdim if valid_mesh.n[0] > 1 else 0)


def test_integrate_volume():
    # 3d mesh
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (0.5, 0.5, 0.5)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    f = df.Field(mesh, nvdim=1, value=0)
    assert f.integrate() == 0

    f = df.Field(mesh, nvdim=1, value=2)
    assert f.integrate() == 2000

    f = df.Field(mesh, nvdim=3, value=(-1, 0, 3))
    assert np.allclose(f.integrate(), (-1000, 0, 3000))

    # 1d mesh
    mesh = df.Mesh(p1=-10e-9, p2=10e-9, n=10)

    def value_fun(point):
        if point <= 0:
            return (-1, -2)
        else:
            return (1, 2)

    f = df.Field(mesh, nvdim=2, value=value_fun)
    assert np.allclose(f.integrate(), (0, 0))


def test_integrate_surface():
    p1 = (0, 0, 0)
    p2 = (10, 5, 3)
    cell = (0.5, 0.5, 0.5)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    # 2d integral on a plane: ∫ f ds
    f = df.Field(mesh, nvdim=1, value=0)
    assert f.sel("x").integrate() == 0

    f = df.Field(mesh, nvdim=1, value=2)
    assert f.sel("x").integrate() == 30
    assert f.sel("y").integrate() == 60
    assert f.sel("z").integrate() == 100

    # surface integral with vector ds: ∫ f • ds
    f = df.Field(mesh, nvdim=3, value=(-1, 0, 3))
    assert f.sel("x").dot([1, 0, 0]).integrate() == -15
    assert f.sel("y").dot([0, 1, 0]).integrate() == 0
    assert f.sel("z").dot([0, 0, 1]).integrate() == 150

    # 1d integral along a line in y
    assert f.sel("z").sel("x").dot([1, 0, 0]).integrate() == -5

    # 4d field -> 3d integral
    mesh = df.Mesh(p1=(0, 0, 0, 0), p2=(20, 15, 10, 5), cell=(0.5, 0.5, 0.5, 0.5))
    f = df.Field(mesh, nvdim=2, value=(2, 3))
    assert np.allclose(f.sel("x0").integrate(), (1500, 2250), atol=0)


def test_integrate_directional():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (0.5, 0.5, 0.5)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    f = df.Field(mesh, nvdim=3, value=(1, 1, 1))

    # 3d -> 2d
    res = f.integrate(direction="x")
    assert isinstance(res, df.Field)
    assert res.nvdim == 3
    assert np.array_equal(res.mesh.n, [20, 20])
    assert np.allclose(res.mean(), (10, 10, 10))

    # 3d -> [2d ->] 1d
    res = f.integrate(direction="x").integrate(direction="y")
    assert isinstance(res, df.Field)
    assert res.nvdim == 3
    assert np.array_equal(res.mesh.n, [20])
    assert np.allclose(res.mean(), (100, 100, 100))

    # 3d -> [2d -> 1d ->] 0d
    res = f.integrate("x").integrate("y").integrate("z")
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, (1000, 1000, 1000))

    # explicit and implicit 3d -> 0d
    assert np.allclose(
        f.integrate("x").integrate("y").integrate("z").mean(), f.integrate()
    )


def test_integrate_cumulative():
    # 3d mesh
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (0.5, 0.5, 0.5)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    f = df.Field(mesh, nvdim=3, value=(1, 1, 1))

    f_int = f.integrate(direction="x", cumulative=True)
    assert isinstance(f_int, df.Field)
    assert f_int.nvdim == 3
    assert f_int.mesh == f.mesh
    assert np.allclose(f_int.mean(), (5, 5, 5))
    assert np.allclose(f_int((0, 0, 0)), (0.25, 0.25, 0.25))
    assert np.allclose(f_int((0.9, 0.9, 0.9)), (0.75, 0.75, 0.75))
    assert np.allclose(f_int((10, 10, 10)), (9.75, 9.75, 9.75))
    assert np.allclose(f_int.diff("x").array, f.array)

    for i, d in enumerate("xyz"):
        f = df.Field(mesh, nvdim=1, value=lambda p: p[i])
        assert np.allclose(f.integrate(d, cumulative=True).diff(d).array, f.array)
        assert np.allclose(f.diff(d).integrate(d, cumulative=True).array, f.array)

    # 1d mesh
    mesh = df.Mesh(p1=0, p2=10e-9, n=5)
    field = df.Field(mesh, nvdim=1, value=lambda p: p)
    assert np.allclose(
        field.integrate("x", cumulative=True).array,
        [1e-18, 5e-18, 13e-18, 24e-18, 41e-18],
    )

    # 2d mesh with one cell in integration direction
    mesh = df.Mesh(p1=(0, 0), p2=(10, 1), cell=(1, 1))
    f = df.Field(mesh, nvdim=2, value=(2, 2.5))
    f_int = f.integrate("y", cumulative=True)
    assert f_int.mesh == f.mesh
    assert np.allclose(f_int.mean(), (1, 1.25))


def test_integrate_exceptions():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (0.5, 0.5, 0.5)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    f = df.Field(mesh, nvdim=3, value=(1, 1, 1))

    with pytest.raises(ValueError):  # cumulative without specifying a direction
        f.integrate(cumulative=True)

    with pytest.raises(ValueError):  # invalid direction name
        f.integrate(direction="a")

    with pytest.raises(TypeError):  # invalid direction type
        f.integrate(1)

    with pytest.raises(TypeError):
        f.integrate(direction=["x", "y"])


def test_abs(valid_mesh):
    f = df.Field(valid_mesh, nvdim=1, value=-1)
    assert abs(f).mean() == 1

    f = df.Field(valid_mesh, nvdim=3, value=(-1, -1, -1))
    assert np.allclose(abs(f).mean(), (1, 1, 1))

    f = df.Field(valid_mesh, nvdim=4, value=(-1, -1, -1, -2))
    assert np.allclose(abs(f).mean(), (1, 1, 1, 2))

    f = df.Field(valid_mesh, nvdim=1, value=-1j)
    assert np.allclose(abs(f).mean(), 1)

    f = df.Field(valid_mesh, nvdim=4, value=(-1j, -1j, -1j, -2j))
    assert np.allclose(abs(f).mean(), (1, 1, 1, 2))

    f = df.Field(valid_mesh, nvdim=1, value=1 - 1j)
    assert np.allclose(abs(f).mean(), np.sqrt(2))


def test_line():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), n=(10, 10, 10))
    f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
    assert isinstance(f, df.Field)

    line = f.line(p1=(0, 0, 0), p2=(5, 5, 5), n=20)
    assert isinstance(line, df.Line)

    assert line.n == 20
    assert line.dim == 3


@pytest.mark.parametrize(
    "mesh, nvdim, value, range_",
    [
        [df.Mesh(p1=0, p2=10, n=10), 1, lambda p: p, (0, 2)],
        [df.Mesh(p1=(-10e-9, -5e-9), p2=(10e-9, 5e-9), n=(1, 1)), 3, (0, 1, 2), 0],
        [df.Mesh(p1=(0, 0, 0), p2=(30e-9, 20e-9, 10e-9), n=(7, 5, 3)), 2, (1, 2), None],
    ],
)
def test_sel(mesh, nvdim, value, range_):
    f = df.Field(mesh, nvdim=nvdim, value=value)
    for dim in f.mesh.region.dims:
        f_sel = f.sel(dim) if range_ is None else f.sel(**{dim: range_})
        if range_ is None:
            assert f_sel.mesh == f.sel(dim).mesh
            assert f_sel.array.shape == (*f.sel(dim).mesh.n, f.nvdim)
        else:
            assert f_sel.mesh == f.sel(**{dim: range_}).mesh
            assert f_sel.array.shape == (*f.sel(**{dim: range_}).mesh.n, f.nvdim)
        assert f_sel.nvdim == f.nvdim
        assert f_sel.vdims == f.vdims
        assert f_sel.unit == f.unit


def test_sel_subregions():
    mesh = df.Mesh(
        p1=(-50e-9, -20e-9, 0),
        p2=(50e-9, 40e-9, 30e-9),
        cell=(1e-9, 2e-9, 3e-9),
        subregions={
            "sr_x": df.Region(p1=(-50e-9, -20e-9, 0), p2=(0, 40e-9, 30e-9)),
            "sr_y": df.Region(p1=(-50e-9, 0, 0), p2=(50e-9, 40e-9, 30e-9)),
            "total": df.Region(p1=(-50e-9, -20e-9, 0), p2=(50e-9, 40e-9, 30e-9)),
        },
    )
    f = df.Field(mesh, nvdim=4, value=lambda p: [*p, 4])

    f_sel = f.sel(x=(-50e-9, -0.5e-9))
    assert len(f_sel.mesh.subregions) == 3
    assert sorted(f_sel.mesh.subregions) == ["sr_x", "sr_y", "total"]
    assert f_sel.mesh == f.mesh.sel(x=(-50e-9, -0.5e-9))
    assert f_sel.nvdim == f.nvdim
    assert f_sel.vdims == f.vdims
    assert f_sel.unit == f.unit
    assert f_sel.array.shape == (*f.mesh.sel(x=(-50e-9, -0.5e-9)).n, f.nvdim)
    # f.__getitem__ does not preserve subregions
    f_sel.mesh.subregions = {}
    assert f_sel.allclose(f["sr_x"])

    f_sel = f.sel(y=(0, 40e-9))
    assert sorted(f_sel.mesh.subregions) == ["sr_x", "sr_y", "total"]
    assert f_sel.mesh == f.mesh.sel(y=(0, 40e-9))
    assert f_sel.nvdim == f.nvdim
    assert f_sel.vdims == f.vdims
    assert f_sel.unit == f.unit
    assert f_sel.array.shape == (*f.mesh.sel(y=(0, 40e-9)).n, f.nvdim)
    # f.__getitem__ does not preserve subregions
    f_sel.mesh.subregions = {}
    assert f_sel.allclose(f["sr_y"])

    f_sel = f.sel(z=(0, 30e-9))
    assert f_sel == f
    assert sorted(f_sel.mesh.subregions) == ["sr_x", "sr_y", "total"]
    assert f_sel.mesh == f.mesh.sel(z=(0, 30e-9))
    assert f_sel.nvdim == f.nvdim
    assert f_sel.vdims == f.vdims
    assert f_sel.unit == f.unit
    assert f_sel.array.shape == (*f.mesh.sel(z=(0, 30e-9)).n, f.nvdim)
    # f.__getitem__ does not preserve subregions
    f_sel.mesh.subregions = {}
    assert f_sel.allclose(f["total"])

    assert np.allclose(
        f.sel(x=4.5e-9).sel(y=5.5e-9).sel(z=6.5e-9), f((4.5e-9, 5.5e-9, 6.5e-9))
    )
    assert np.allclose(f.sel("x").sel("y").sel("z"), f(f.mesh.region.center))

    mesh = df.Mesh(p1=0, p2=10, n=5, subregions={"sr": df.Region(p1=0, p2=2)})
    f = df.Field(mesh, nvdim=2, value=lambda p: (p, p**2))

    f_sel = f.sel(x=(2, 4))
    assert f_sel.mesh == f.mesh.sel(x=(2, 4))
    assert f_sel.nvdim == f.nvdim
    assert f_sel.vdims == f.vdims
    assert f_sel.unit == f.unit
    assert f_sel.array.shape == (*f.mesh.sel(x=(2, 4)).n, f.nvdim)
    assert len(f_sel.mesh.subregions) == 0

    f_sel = f.sel(x=3)
    assert np.allclose(f_sel, f((3,)))


def test_sel_invalid(test_field):
    with pytest.raises(ValueError):
        test_field.sel("a")  # invalid dimension name

    with pytest.raises(ValueError):
        test_field.sel(x=20)  # outside the region

    with pytest.raises(ValueError):
        test_field.sel(x=(0, 20))  # outside the region

    with pytest.raises(TypeError):
        test_field.sel(z=slice(0, 5e-9))


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_resample(valid_mesh, nvdim):
    test_field = df.Field(valid_mesh, nvdim=nvdim, value=(1,) * nvdim)
    desired_n = tuple(max(i - 1, 1) for i in valid_mesh.n)
    resampled = test_field.resample(desired_n)
    assert np.allclose(resampled.mesh.n, desired_n)
    assert resampled.mesh.region == test_field.mesh.region
    assert np.allclose(resampled.array, 1)

    desired_n = list(np.ones(valid_mesh.region.ndim, dtype=int))
    resampled = test_field.resample(desired_n)
    assert np.allclose(resampled.mesh.n, desired_n)
    assert resampled.mesh.region == test_field.mesh.region
    assert np.allclose(resampled.array, 1)

    desired_n[-1] = 0
    with pytest.raises(ValueError):
        test_field.resample(desired_n)

    desired_n = list(np.ones(valid_mesh.region.ndim, dtype=float))
    with pytest.raises(TypeError):
        test_field.resample(desired_n)


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_getitem_no_subregions(valid_mesh, nvdim):
    f = df.Field(valid_mesh, nvdim=nvdim, value=(1,) * nvdim)

    with pytest.raises(KeyError):
        f["no_sub"]


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_getitem(ndim, nvdim):
    p1 = np.full(ndim, 0.0)
    p2 = np.full(ndim, 50.0)
    cell = np.full(ndim, 5.0)

    p2_r1 = np.full(ndim, 50.0)
    p2_r1[0] = 30.0

    p1_r2 = np.full(ndim, 0.0)
    p1_r2[0] = 30.0

    subregions = {
        "r1": df.Region(p1=p1, p2=p2_r1),
        "r2": df.Region(p1=p1_r2, p2=p2),
    }
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)

    def value_fun(point):
        if point[0] <= 30:
            return (1,) * nvdim
        else:
            return (0,) * nvdim

    f = df.Field(mesh, nvdim=nvdim, value=value_fun)
    assert isinstance(f, df.Field)
    assert isinstance(f["r1"], df.Field)
    assert isinstance(f["r2"], df.Field)
    assert isinstance(f[subregions["r2"]], df.Field)
    assert isinstance(f[subregions["r1"]], df.Field)

    assert np.allclose(f["r1"].mean(), (1,) * nvdim)
    assert np.allclose(f["r2"].mean(), (0,) * nvdim)
    assert np.allclose(f[subregions["r1"]].mean(), (1,) * nvdim)
    assert np.allclose(f[subregions["r2"]].mean(), (0,) * nvdim)

    assert len(f["r1"].mesh) + len(f["r2"].mesh) == len(f.mesh)

    # Meshes are not aligned
    p1_sub = np.full(ndim, 1.1)
    p2_sub = np.full(ndim, 9.9)
    subregion = df.Region(p1=p1_sub, p2=p2_sub)
    assert f[subregion].array.shape == (2,) * ndim + (nvdim,)


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_angle(valid_mesh, nvdim):
    v = np.zeros(nvdim)
    v[0] = 1.0
    f = df.Field(valid_mesh, nvdim=nvdim, value=v)
    for i in range(nvdim):
        v = np.zeros(nvdim)
        v[i] = 1.0
        assert f.angle(v).array.shape == (*valid_mesh.n, 1)
        assert f.angle(v).nvdim == 1
        assert f.angle(v).unit == "rad"
        if i == 0:
            assert np.allclose(f.angle(v).mean(), 0.0)
        else:
            assert np.allclose(f.angle(v).mean(), np.pi / 2)


def test_rotate90():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(40e-9, 20e-9, 10e-9), n=(40, 10, 5))

    # uniform scalar field
    field = df.Field(mesh, nvdim=1, value=1)
    rotated = field.rotate90("x", "y")
    assert np.allclose(rotated.mesh.region.pmin, (10e-9, -10e-9, 0e-9))
    assert np.allclose(rotated.mesh.region.pmax, (30e-9, 30e-9, 10e-9))
    assert np.allclose(rotated.mesh.n, (10, 40, 5))
    assert rotated.array.shape == (10, 40, 5, 1)

    # the reference point does not affect rotating the array
    rotated_ref = field.rotate90("x", "y", reference_point=(0, 0, 0))
    assert np.allclose(rotated_ref.mesh.region.pmin, (-20e-9, 0, 10e-9))
    assert np.allclose(rotated.array, rotated_ref.array)

    # scalar field with local variations
    field = df.Field(mesh, nvdim=1, value=lambda p: p[2])
    rotated = field.rotate90("z", "x")
    assert np.allclose(rotated.mesh.n, (5, 10, 40))
    assert rotated.array.shape == (5, 10, 40, 1)
    assert np.allclose(rotated.sel("y").sel("z").array, [1e-9, 3e-9, 5e-9, 7e-9, 9e-9])

    # uniform vector field
    field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
    # 90° rotation in xy plane
    rotated = field.rotate90("x", "y")
    assert rotated.array.shape == (10, 40, 5, 3)
    assert np.allclose(rotated(rotated.mesh.region.centre), (-2, 1, 3))
    assert np.allclose(rotated.mean(), (-2, 1, 3))
    assert rotated.vdims == ["x", "y", "z"]
    # original field is not modified
    assert np.allclose(field(field.mesh.region.centre), (1, 2, 3))
    # -90° rotation in xz plane
    rotated = field.rotate90("x", "z", k=-1)
    assert rotated.array.shape == (5, 10, 40, 3)
    assert np.allclose(rotated.array[0, 0, 0], (3, 2, -1))
    assert np.allclose(rotated(rotated.mesh.region.centre), (3, 2, -1))
    # -90° rotation is equivalent to rotation in opposite direction
    assert field.rotate90("x", "y", k=-1).allclose(field.rotate90("y", "x"))
    # 270° rotation is equivalent to -90° rotation
    assert field.rotate90("x", "y", k=-1).allclose(field.rotate90("x", "y", k=3))
    # 360° rotation has no effect
    assert field.allclose(field.rotate90("x", "z", k=4))
    assert field.rotate90("x", "z").allclose(field.rotate90("x", "z", k=5))

    # in-place rotation
    rotated = field.rotate90("x", "y", k=2, inplace=True, reference_point=(0, 0, 0))
    assert rotated == field  # in-place rotation returns self
    assert np.allclose(field.mesh.region.pmin, (-40e-9, -20e-9, 0), atol=0)
    assert np.allclose(field.mesh.region.pmax, (0, 0, 10e-9), atol=0)
    assert field.array.shape == (40, 10, 5, 3)
    assert np.allclose(field(field.mesh.region.centre), (-1, -2, 3))

    # 2 dimensional field with cells not containing valid data
    mesh = df.Mesh(p1=(-20e-9, -10e-9), p2=(20e-9, 10e-9), cell=(2e-9, 2e-9))
    field = df.Field(
        mesh, nvdim=1, value=1, norm=lambda p: 10 if p[0] < 4e-9 else 0, valid="norm"
    )
    # check correct initialisation
    assert field((-5e-9, 1e-9)) == 10
    assert field((3e-9, 1e-9)) == 10
    assert field((5e-9, 1e-9)) == 0
    assert field.valid[0, 0]
    assert not field.valid[-1, 0]
    assert field.valid[0, -1]
    # check mean values in each sub-part; cell centres are located at 3e-9 and 5e-9
    assert field.sel(x=(-19e-9, 3e-9)).mean() == 10
    assert field.sel(x=(5e-9, 19e-9)).mean() == 0
    assert field.valid.shape == (20, 10)
    # rotate about 90°
    rotated = field.rotate90("x", "y")
    assert rotated.array.shape == (10, 20, 1)
    assert rotated.valid.shape == (10, 20)
    assert rotated.valid[0, 0]
    assert rotated.valid[-1, 0]
    assert not rotated.valid[0, -1]
    assert rotated.sel(y=(-19e-9, 3e-9)).mean() == 10
    assert rotated.sel(y=(5e-9, 19e-9)).mean() == 0
    # number of invalid cells stays the same
    assert np.count_nonzero(rotated.array == 0) == np.count_nonzero(field.array == 0)
    assert np.count_nonzero(rotated.valid == 0) == np.count_nonzero(field.valid == 0)
    # original field has not changed
    assert field.valid[0, 0]
    assert not field.valid[-1, 0]
    assert field.valid[0, -1]

    # mismatch between dims and vdims
    field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
    # no default vdim mapping -> rotation is not "safe" because the relation between
    # spatial directions and vector directions is unknown
    with pytest.raises(RuntimeError):
        rotated = field.rotate90("x", "y")

    # manually add vdim_mapping
    # we use other vdims that the default x, y, z to make the vdim_mapping easier to
    # read; the mz component does not point along any spatial direction
    field = df.Field(
        mesh,
        nvdim=3,
        value=(1, 2, 3),
        vdims=["mx", "my", "mz"],
        vdim_mapping={"mx": "x", "my": "y", "mz": None},
    )
    rotated = field.rotate90("x", "y")
    assert np.allclose(rotated.mean(), (-2, 1, 3))
    # vector names, like axis/dimension names, do not change when rotating the object
    assert rotated.vdims == ["mx", "my", "mz"]

    # change mesh to yz plane
    mesh.region.dims = ["y", "z"]
    field = df.Field(
        mesh,
        nvdim=3,
        value=(1, 2, 3),
        vdims=["mx", "my", "mz"],
        vdim_mapping={"mx": None, "my": "y", "mz": "z"},
    )
    field.rotate90("y", "z", k=3, inplace=True)
    assert np.allclose(field.mean(), (1, 3, -2))


def test_write_read_ovf(tmp_path):
    representations = ["txt", "bin4", "bin8"]
    filename = "testfile.ovf"
    p1 = (0, 0, 0)
    p2 = (8e-9, 5e-9, 3e-9)
    cell = (1e-9, 1e-9, 1e-9)
    subregions = {
        "sr1": df.Region(p1=p1, p2=(2e-9, 2e-9, 1e-9)),
        "sr2": df.Region(p1=(3e-9, 0, 0), p2=p2),
    }
    units = ("nm", "nm", "nm")  # test saving units to file
    region = df.Region(p1=p1, p2=p2, units=units)
    assert region.units == units  # sanity check: data type of region.units
    mesh = df.Mesh(region=region, cell=cell, subregions=subregions)

    # Write/read
    for nvdim, value in [
        (1, lambda point: point[0] + point[1] + point[2]),
        (2, lambda point: (point[0], point[1] + point[2])),
        (3, lambda point: (point[0], point[1], point[2])),
    ]:
        f = df.Field(mesh, nvdim=nvdim, value=value, unit="A/m")
        for rep in representations:
            tmpfilename = tmp_path / filename
            f.to_file(tmpfilename, representation=rep)
            f_read = df.Field.from_file(tmpfilename)

            assert f.allclose(f_read)
            assert f_read.unit == "A/m"
            assert f_read.mesh.region.units == units
            assert f.mesh.subregions == f_read.mesh.subregions

            tmpfilename = tmp_path / f"no_sr_{filename}"
            f.to_file(tmpfilename, representation=rep, save_subregions=False)
            f_read = df.Field.from_file(tmpfilename)

            assert f.allclose(f_read)
            assert f_read.unit == "A/m"
            assert f_read.mesh.region.units == units
            assert f_read.mesh.subregions == {}

        # Directly write with wrong representation (no data is written)
        with pytest.raises(ValueError):
            f._to_ovf("fname.ovf", representation="bin5")

    # multiple different units (not supported by discretisedfield)
    f = df.Field(mesh, nvdim=3, value=(1, 1, 1), unit="m s kg")
    tmpfilename = str(tmp_path / filename)
    f.to_file(tmpfilename, representation=rep)
    with pytest.warns(UserWarning, match=r"multiple units.+Unit is set to None"):
        f_read = df.Field.from_file(tmpfilename)

    assert f.allclose(f_read)
    assert f_read.unit is None

    # Extend scalar
    for rep in representations:
        # large mesh required to detect bugs when saving data in chunks
        large_mesh = df.Mesh(p1=(0, 0, 0), p2=(1, 1, 1), n=(480, 200, 12))
        f = df.Field(large_mesh, nvdim=1, value=1)
        tmpfilename = tmp_path / "extended-scalar.ovf"
        f.to_file(tmpfilename, representation=rep, extend_scalar=True)
        f_read = df.Field.from_file(tmpfilename)

        assert f.allclose(f_read.x)

    # Read different OOMMF representations
    # (OVF1, OVF2) x (txt, bin4, bin8)
    filenames = [
        "oommf-ovf2-txt.omf",
        "oommf-ovf2-bin4.omf",
        "oommf-ovf2-bin8.omf",
        "oommf-ovf1-txt.omf",
        "oommf-ovf1-bin4.omf",
        "oommf-ovf1-bin8.omf",
    ]
    dirname = os.path.join(os.path.dirname(__file__), "test_sample")
    for filename in filenames:
        omffilename = os.path.join(dirname, filename)
        f_read = df.Field.from_file(omffilename)

        if "ovf2" in filename:
            # The magnetisation is in the x-direction in OVF2 files.
            assert abs(f_read.orientation.x.mean() - 1) < 1e-2
        else:
            # The norm of magnetisation is known.
            assert abs(f_read.norm.mean() - 1261566.2610100) < 1e-3

    # Read component names (single-word and multi-word with and without hyphen)
    # from OOMMF files
    assert df.Field.from_file(os.path.join(dirname, "oommf-ovf2-bin8.omf")).vdims == [
        "x",
        "y",
        "z",
    ]
    assert df.Field.from_file(os.path.join(dirname, "oommf-ovf2-bin8.ohf")).vdims == [
        "x",
        "y",
        "z",
    ]
    assert df.Field.from_file(os.path.join(dirname, "oommf-ovf2-bin8.oef")).vdims == [
        "Total_energy_density"
    ]

    # Read different mumax3 bin4 and txt files (made on linux and windows)
    filenames = [
        "mumax-bin4-linux.ovf",
        "mumax-bin4-windows.ovf",
        "mumax-txt-linux.ovf",
    ]
    dirname = os.path.join(os.path.dirname(__file__), "test_sample")
    for filename in filenames:
        omffilename = os.path.join(dirname, filename)
        f_read = df.Field.from_file(omffilename)

        # We know the saved magentisation.
        f_saved = df.Field(f_read.mesh, nvdim=3, value=(1, 0.1, 0), norm=1)
        assert f_saved.allclose(f_read)

    # Read other ovf files that were reported as problematic by users
    filenames = [
        "ovf2-bin8_different-case.ovf",  # lower-case "Begin: data binary 8"
    ]
    for filename in filenames:
        omffilename = os.path.join(dirname, filename)
        f_read = df.Field.from_file(omffilename)
        # test that some data has been read without errors; the exact content of the
        # files can vary, so we cannot easily perform more thorough checks
        assert f_read.array.nbytes > 0


def test_write_read_vtk(tmp_path):
    filename = "testfile.vtk"

    p1 = (0, 0, 0)
    p2 = (5e-9, 2e-9, 1e-9)
    cell = (1e-9, 1e-9, 1e-9)
    subregions = {
        "sr1": df.Region(p1=p1, p2=(2e-9, 2e-9, 1e-9)),
        "sr2": df.Region(p1=(3e-9, 0, 0), p2=p2),
    }
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell, subregions=subregions)

    for nvdim, value, vdims in zip(
        [1, 2, 3, 4],
        [1.2, (1, 2.5), (1e-3, -5e6, 5e6), np.random.random(4)],
        [None, ("m_mag", "m_phase"), None, list("abcd")],
    ):
        for repr in ["txt", "bin", "xml"]:
            f = df.Field(mesh, nvdim=nvdim, value=value, vdims=vdims)
            tmpfilename = tmp_path / filename
            f.to_file(tmpfilename, representation=repr)
            f_read = df.Field.from_file(tmpfilename)

            assert np.allclose(f.array, f_read.array)
            assert np.allclose(f.mesh.region.pmin, f_read.mesh.region.pmin)
            assert np.allclose(f.mesh.region.pmax, f_read.mesh.region.pmax)
            assert np.allclose(f.mesh.cell, f_read.mesh.cell)
            assert np.all(f.mesh.n == f_read.mesh.n)
            assert f.vdims == f_read.vdims
            assert f.mesh.subregions == f_read.mesh.subregions

            tmpfilename = tmp_path / f"no_sr_{filename}"
            f.to_file(tmpfilename, representation=repr, save_subregions=False)
            f_read = df.Field.from_file(tmpfilename)

            assert f.allclose(f_read)
            assert f_read.mesh.subregions == {}

    dirname = os.path.join(os.path.dirname(__file__), "test_sample")
    f = df.Field.from_file(os.path.join(dirname, "vtk-file.vtk"))
    assert isinstance(f, df.Field)
    assert np.all(f.mesh.n == (5, 1, 2))
    assert f.array.shape == (5, 1, 2, 3)
    assert f.nvdim == 3

    # test reading legacy vtk file (written with discretisedfield<=0.61.0)
    dirname = os.path.join(os.path.dirname(__file__), "test_sample")
    f = df.Field.from_file(os.path.join(dirname, "vtk-vector-legacy.vtk"))
    assert isinstance(f, df.Field)
    assert np.all(f.mesh.n == (8, 1, 1))
    assert f.array.shape == (8, 1, 1, 3)
    assert f.nvdim == 3

    dirname = os.path.join(os.path.dirname(__file__), "test_sample")
    f = df.Field.from_file(os.path.join(dirname, "vtk-scalar-legacy.vtk"))
    assert isinstance(f, df.Field)
    assert np.all(f.mesh.n == (5, 1, 2))
    assert f.array.shape == (5, 1, 2, 1)
    assert f.nvdim == 1

    # test invalid arguments
    f = df.Field(mesh, nvdim=3, value=(0, 0, 1))
    with pytest.raises(ValueError):
        f.to_file(str(tmp_path / filename), representation="wrong")
    f._vdims = None  # manually remove component labels
    with pytest.raises(AttributeError):
        f.to_file(str(tmp_path / filename))


@pytest.mark.parametrize("extension", ["ovf", "vtk"])
@pytest.mark.parametrize("ndim", [1, 2, 4])
def test_write_invalid_ndim(ndim, extension):
    mesh = df.Mesh(p1=[0] * ndim, p2=[1] * ndim, n=[10] * ndim)
    field = df.Field(mesh, nvdim=1)

    with pytest.raises(RuntimeError):
        field.to_file(f"field.{extension}")


@pytest.mark.parametrize("norm", [None, lambda p: 100 if p[0] < 5e-12 else 0])
@pytest.mark.parametrize("nvdim,value", [(1, -1.23), (3, (1e-3 + np.pi, -5e6, 6e6))])
@pytest.mark.parametrize(
    "subregions",
    [
        {},
        {
            "sr1": df.Region(p1=(0, 0, 0), p2=(2e-12, 2e-12, 1e-12)),
            "sr2": df.Region(p1=(3e-12, 0, 0), p2=(10e-12, 5e-12, 5e-12)),
        },
    ],
)
@pytest.mark.parametrize("filename", ["testfile.hdf5", "testfile.h5"])
def test_write_read_hdf5(norm, nvdim, value, subregions, filename, tmp_path):
    p1 = (0, 0, 0)
    p2 = (10e-12, 5e-12, 5e-12)
    cell = (1e-12, 1e-12, 1e-12)
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell, subregions=subregions)

    f = df.Field(mesh, nvdim=nvdim, value=value, norm=norm, valid="norm")

    # sanity checks
    if norm is not None:
        assert f.valid[0, 0, 0]
        assert not f.valid[-1, -1, -1]
    else:
        assert f.valid.all()

    tmpfilename = tmp_path / filename
    f.to_file(tmpfilename)
    f_read = df.Field.from_file(tmpfilename)

    assert f == f_read
    assert f.mesh.subregions == f_read.mesh.subregions  # not checked in __eq__

    tmpfilename = tmp_path / f"no_sr_{filename}"
    f.to_file(tmpfilename, save_subregions=False)
    # subregions are always saved in hdf5, 'save_subregions' is ignored
    f_read = df.Field.from_file(tmpfilename)
    assert f == f_read
    assert f.mesh.subregions == f_read.mesh.subregions  # not checked in __eq__


def test_write_read_invalid_extension():
    filename = "testfile.jpg"

    p1 = (0, 0, 0)
    p2 = (10e-12, 5e-12, 3e-12)
    cell = (1e-12, 1e-12, 1e-12)
    mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

    f = df.Field(mesh, nvdim=1, value=5e-12)
    with pytest.raises(ValueError):
        f.to_file(filename)
    with pytest.raises(ValueError):
        df.Field.from_file(filename)


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_fft(valid_mesh, nvdim):
    def _init_random(p):
        return np.random.rand(nvdim) * 2 - 1

    f = df.Field(valid_mesh, nvdim=nvdim, value=_init_random, norm=1)

    ifft_f = f.fftn().ifftn()
    ifft_f.mesh.translate(f.mesh.region.centre, inplace=True)
    assert f.allclose(ifft_f)
    irfft_f = f.rfftn().irfftn(shape=f.mesh.n)
    irfft_f.mesh.translate(f.mesh.region.centre, inplace=True)
    assert f.allclose(ifft_f)


def test_fft_Fourier_slice_theoreme():
    p1 = (-10, -10, -5)
    p2 = (10, 10, 5)
    cell = (1, 1, 1)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def _init_random(p):
        return np.random.rand(3) * 2 - 1

    f = df.Field(mesh, nvdim=3, value=_init_random, norm=1)

    for i in "xyz":
        plane = f.integrate(i)
        assert plane.allclose(f.fftn().sel(**{"k_" + i: 0}).ifftn().real)
        assert (
            df.Field(mesh, nvdim=3)
            .integrate(i)
            .allclose(f.fftn().sel(**{"k_" + i: 0}).ifftn().imag)
        )


def test_rfft_no_shift_last_dim():
    a = np.zeros((5, 5))
    a[2, 3] = 1

    p1 = (0, 0)
    p2 = (10, 10)
    cell = (2.0, 2.0)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    f = df.Field(mesh, nvdim=1, value=a)

    field_ft = f.rfftn()
    ft = spfft.fftshift(spfft.rfftn(a), axes=[0])

    assert np.array_equal(field_ft.array[..., 0], ft)


def test_1d_fft():
    mesh = df.Mesh(p1=0, p2=10, cell=2)
    f = mesh.coordinate_field()
    field_ft = f.fftn()
    expected_array = np.fft.fftshift(np.fft.fftn(f.array))
    assert np.allclose(expected_array, field_ft.array)


def test_mpl_scalar(test_field):
    # No axes
    for comp in test_field.vdims:
        getattr(test_field, comp).sel("x").resample((3, 4)).mpl.scalar()

    # Axes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for comp in test_field.vdims:
        getattr(test_field, comp).sel("x").mpl.scalar(ax=ax)

    # All arguments
    for comp in test_field.vdims:
        getattr(test_field, comp).sel("x").mpl.scalar(
            figsize=(10, 10),
            filter_field=test_field.norm.sel("x"),
            colorbar=True,
            colorbar_label="something",
            multiplier=1e-6,
            cmap="hsv",
            clim=(-1, 1),
        )

    # Saving plot
    filename = "testfigure.pdf"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, filename)
        test_field.a.sel("x").mpl.scalar(filename=tmpfilename)

    # Exceptions
    with pytest.raises(RuntimeError):
        test_field.a.mpl.scalar()  # not sliced
    with pytest.raises(RuntimeError):
        test_field.sel("z").mpl.scalar()  # vector field
    with pytest.raises(ValueError):
        # wrong filter field
        test_field.a.sel("z").mpl.scalar(filter_field=test_field)
    plt.close("all")


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_mpl_dimension_scalar(valid_mesh, nvdim):
    field = df.Field(valid_mesh, nvdim=nvdim)

    if valid_mesh.region.ndim != 2 or nvdim != 1:
        with pytest.raises(RuntimeError):
            field.mpl.scalar()
    else:
        field.mpl.scalar()

    plt.close("all")


def test_mpl_lightess(test_field):
    filenames = ["skyrmion.omf", "skyrmion-disk.omf"]
    for i in filenames:
        filename = os.path.join(os.path.dirname(__file__), "test_sample", i)

        field = df.Field.from_file(filename)
        # TODO test all directions "xyz" (check samples first, presumably a single
        # layer, causes problems with x and y "planes").
        for plane in "z":
            field.sel(plane).mpl.lightness()
            field.sel(plane).mpl.lightness(
                lightness_field=-field.z.sel(plane), filter_field=field.norm.sel(plane)
            )
        fig, ax = plt.subplots()
        field.sel("z").mpl.lightness(
            ax=ax, clim=[0, 0.5], colorwheel_xlabel="mx", colorwheel_ylabel="my"
        )
        # Saving plot
        filename = "testfigure.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            field.sel("z").mpl.lightness(filename=tmpfilename)

    # Exceptions
    with pytest.raises(RuntimeError):
        test_field.mpl.lightness()  # not sliced
    with pytest.raises(ValueError):
        # wrong filter field
        test_field.sel("z").mpl.lightness(filter_field=test_field)
    with pytest.raises(ValueError):
        # wrong lightness field
        test_field.sel("z").mpl.lightness(lightness_field=test_field)
    plt.close("all")


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_mpl_dimension_lightness(valid_mesh, nvdim):
    field = df.Field(valid_mesh, nvdim=nvdim)

    if valid_mesh.region.ndim != 2 or nvdim > 3:
        with pytest.raises(RuntimeError):
            field.mpl.lightness()
    elif nvdim == 3:
        field.vdim_mapping = dict(zip(field.vdims, [*valid_mesh.region.dims, None]))
        field.mpl.lightness()
    else:
        field.mpl.lightness()

    plt.close("all")


@pytest.mark.filterwarnings("error")
def test_mpl_lightness_handles_invalid_parts(test_field, tmp_path):
    """
    We did set rgb values in invalid parts to np.nan. The array is internally converted
    to dtype np.uint8. The nan values result in a warning.
    To avoid this we now set invalid parts to zero.
    """
    assert (~test_field.valid).any()
    # save field to trigger the warning
    test_field.sel("z").mpl.lightness(filename=str(tmp_path / "test.pdf"))


@pytest.mark.filterwarnings("ignore:Automatic coloring")
def test_mpl_vector(test_field):
    # No axes
    test_field.sel("x").resample((3, 4)).mpl.vector()

    # Axes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    test_field.sel("x").mpl.vector(ax=ax)

    # All arguments
    test_field.sel("x").mpl.vector(
        figsize=(10, 10),
        color_field=test_field.b.sel("x"),
        colorbar=True,
        colorbar_label="something",
        multiplier=1e-6,
        cmap="hsv",
        clim=(-1, 1),
    )

    # 2d vector field
    plane_2d = test_field.sel("z").a << test_field.sel("z").b
    # automatic mapping between two spatial and two vector dimensions:
    # __lshift__ resets vdims to ["x", "y"] and sets vdim_mapping
    plane_2d.mpl.vector()
    # renaming vdims does update vdim_mapping
    plane_2d.vdims = ["a", "b"]
    plane_2d.mpl.vector()
    # manually remove vdim_mapping
    plane_2d.vdim_mapping = {}
    with pytest.raises(ValueError):
        plane_2d.mpl.vector()
    plane_2d.mpl.vector(vdims=["a", "b"])
    plane_2d.mpl.vector(vdims=["a", None])
    plane_2d.mpl.vector(vdims=[None, "b"])
    with pytest.raises(ValueError):
        plane_2d.mpl.vector(vdims=[None, None])

    # Saving plot
    filename = "testfigure.pdf"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, filename)
        test_field.sel("x").mpl.vector(filename=tmpfilename)

    # Exceptions
    with pytest.raises(RuntimeError):
        test_field.mpl.vector()  # not sliced
    with pytest.raises(ValueError):
        test_field.b.sel("z").mpl.vector()  # scalar field
    with pytest.raises(ValueError):
        # wrong color field
        test_field.sel("z").mpl.vector(color_field=test_field)

    plt.close("all")


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_mpl_dimension_vector(valid_mesh, nvdim):
    field = df.Field(valid_mesh, nvdim=nvdim)

    if valid_mesh.region.ndim != 2:
        with pytest.raises(RuntimeError):
            field.mpl.vector()
    else:
        if nvdim == 1:
            field.vdim_mapping = dict(zip([field.vdims], valid_mesh.region.dims))
            with pytest.raises(ValueError):
                field.mpl.vector()
        else:
            field.vdim_mapping = dict(
                zip(field.vdims, [*valid_mesh.region.dims, None, None])
            )
            field.mpl.vector()

        if nvdim > 1:
            field.mpl.vector(vdims=field.vdims[:2])

    plt.close("all")


def test_mpl_contour(test_field):
    # No axes
    test_field.sel("z").c.mpl.contour()

    # Axes
    fig, ax = plt.subplots()
    test_field.sel("z").c.mpl.contour(ax=ax)

    # All arguments
    test_field.sel("z").c.mpl.contour(
        figsize=(10, 10),
        multiplier=1e-6,
        filter_field=test_field.norm.sel("z"),
        colorbar=True,
        colorbar_label="something",
    )

    # Saving plot
    filename = "testfigure.pdf"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, filename)
        test_field.sel("z").c.mpl.contour(filename=tmpfilename)

    # Exceptions
    with pytest.raises(RuntimeError):
        test_field.mpl.contour()  # not sliced
    with pytest.raises(RuntimeError):
        test_field.sel("z").mpl.contour()  # vector field
    with pytest.raises(ValueError):
        # wrong filter field
        test_field.sel("z").c.mpl.contour(filter_field=test_field)

    plt.close("all")


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_mpl_dimension_contour(valid_mesh, nvdim):
    field = df.Field(valid_mesh, nvdim=nvdim)

    if valid_mesh.region.ndim != 2 or nvdim != 1:
        with pytest.raises(RuntimeError):
            field.mpl.contour()
    else:
        field.mpl.contour()

    plt.close("all")


def test_mpl(test_field):
    # No axes
    test_field.sel("x").resample((3, 4)).mpl()

    # Axes
    fig = plt.figure()
    ax = fig.add_subplot(111)
    test_field.a.sel("x").mpl(ax=ax)

    test_field.c.sel("x").mpl(
        figsize=(12, 6),
        scalar_kw={
            "filter_field": test_field.norm.sel("x"),
            "colorbar_label": "scalar",
            "cmap": "twilight",
        },
        vector_kw={
            "color_field": test_field.b.sel("x"),
            "use_color": True,
            "colorbar": True,
            "colorbar_label": "vector",
            "cmap": "hsv",
            "clim": (0, 1e6),
        },
        multiplier=1e-12,
    )

    # Saving plot
    filename = "testfigure.pdf"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpfilename = os.path.join(tmpdir, filename)
        test_field.sel("x").mpl(filename=tmpfilename)

    # Exception
    with pytest.raises(RuntimeError):
        test_field.mpl()

    plt.close("all")


@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_mpl_dimension(valid_mesh, nvdim):
    field = df.Field(valid_mesh, nvdim=nvdim)

    if valid_mesh.region.ndim != 2 or nvdim > 3:
        with pytest.raises(RuntimeError):
            field.mpl()
    else:
        if nvdim == 3:
            field.vdim_mapping = dict(zip(field.vdims, [*valid_mesh.region.dims, None]))
        field.mpl()

    plt.close("all")


@pytest.mark.parametrize(
    "selection", [dict(x=4e-9), dict(y=-2e-9), dict(z=0, vdims="b")]
)
def test_hv_data_selection(test_field, selection):
    """
    Test selecting parts of the data and returning them as xarray as done in the hv
    plotting methods.
    """
    hv_plane = test_field._hv_data_selection(**selection)
    field_plane = test_field
    for key, value in selection.items():
        if key == "vdims":
            field_plane = getattr(field_plane, value)
        else:
            field_plane = field_plane.sel(**{key: value})
    valid_plane = field_plane.valid.squeeze()
    assert np.allclose(hv_plane.data[valid_plane], field_plane.array[valid_plane])
    assert np.allclose(hv_plane.data[~valid_plane], np.nan, equal_nan=True)

    for dim in hv_plane.dims:
        assert all(hv_plane[dim].data == field_plane.to_xarray()[dim].data)


def test_hv_scalar(test_field):
    for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
        normal = (set("xyz") - set(kdims)).pop()
        kdim_str = f"[{','.join(kdims)}]"
        check_hv(
            test_field.hv.scalar(kdims=kdims),
            [f"DynamicMap [{normal},vdims]", f"Image {kdim_str}"],
        )
        check_hv(
            test_field.hv.scalar(kdims=kdims, roi=test_field.norm),
            [f"DynamicMap [{normal},vdims]", f"Image {kdim_str}"],
        )

        # additional kwargs and plane
        check_hv(
            test_field.sel(normal).hv.scalar(kdims=kdims, clim=(-1, 1)),
            ["DynamicMap [vdims]", f"Image {kdim_str}"],
        )

        for c in test_field.vdims:
            check_hv(
                getattr(test_field, c).hv.scalar(kdims=kdims),
                [f"DynamicMap [{normal}]", f"Image {kdim_str}"],
            )
            check_hv(
                getattr(test_field, c).sel(normal).hv.scalar(kdims=kdims),
                [f"Image {kdim_str}"],
            )

    with pytest.raises(ValueError):
        check_hv(test_field.hv.scalar(kdims=["wrong_name", "x"]), ...)

    with pytest.raises(ValueError):
        check_hv(test_field.hv.scalar(kdims=["x", "y", "z"]), ...)

    with pytest.raises(TypeError):
        check_hv(test_field.hv.scalar(kdims=["x", "y"], roi="z"), ...)

    with pytest.raises(ValueError):
        check_hv(test_field.hv.scalar(kdims=["x", "y"], roi=test_field), ...)

    with pytest.raises(ValueError):
        check_hv(
            test_field.sel("z").hv.scalar(kdims=["x", "y"], roi=test_field.norm),
            ...,
        )

    with pytest.raises(ValueError):
        check_hv(
            test_field[
                df.Region(p1=(-5e-9, -5e-9, -5e-9), p2=(5e-9, 5e-9, -1e-9))
            ].hv.scalar(kdims=["x", "y"], roi=test_field.norm.sel(z=4e-9)),
            ...,
        )


@pytest.mark.filterwarnings("ignore:Automatic coloring")
def test_hv_vector(test_field):
    for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
        normal = (set("xyz") - set(kdims)).pop()
        kdim_str = f"[{','.join(kdims)}]"
        check_hv(
            test_field.hv.vector(kdims=kdims),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        check_hv(
            test_field.sel(normal).hv.vector(kdims=kdims),
            [f"VectorField {kdim_str}"],
        )
        check_hv(
            test_field.hv.vector(kdims=kdims, roi=test_field.norm),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        check_hv(
            test_field.hv.vector(kdims=kdims, n=(10, 10)),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )

        # additional kwargs
        check_hv(
            test_field.hv.vector(kdims=kdims, use_color=False, color="blue"),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )

        for vdim in test_field.vdims:
            check_hv(
                test_field.hv.vector(kdims=kdims, cdim=vdim),
                [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
            )

        with pytest.raises(ValueError):
            check_hv(test_field.hv.vector(kdims=kdims, cdim="wrong"), ...)

        with pytest.raises(TypeError):
            check_hv(test_field.hv.vector(kdims=kdims, cdim=test_field.norm), ...)

        with pytest.raises(ValueError):
            check_hv(test_field.hv.vector(kdims=kdims, vdims=["a", "b", "c"]), ...)

        # 2d field
        with pytest.raises(ValueError):
            check_hv((test_field.a << test_field.b).hv.vector(kdims=kdims), ...)

        field_2d = test_field.a << test_field.b
        field_2d.vdims = ["a", "b"]
        check_hv(
            field_2d.hv.vector(kdims=kdims, vdims=["a", "b"]),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        check_hv(
            field_2d.hv.vector(kdims=kdims, vdims=[None, "b"]),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        check_hv(
            field_2d.hv.vector(kdims=kdims, vdims=["a", None]),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        with pytest.raises(ValueError):
            check_hv(field_2d.hv.vector(kdims=kdims, vdims=[None, None]), ...)

        # 4d field
        field_4d = test_field.a << test_field.b << test_field.a << test_field.b
        field_4d.vdims = ["a", "b", "c", "d"]
        with pytest.raises(ValueError):
            check_hv(field_4d.hv.vector(kdims=kdims), ...)
        check_hv(
            field_4d.hv.vector(kdims=kdims, vdims=["c", "d"]),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        check_hv(
            field_4d.hv.vector(kdims=kdims, vdims=["c", "d"]),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        check_hv(
            field_4d.hv.vector(kdims=kdims, vdims=[None, "b"]),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )

    with pytest.raises(ValueError):
        check_hv(test_field.hv.contour(kdims=["wrong_name", "x"]), ...)

    with pytest.raises(ValueError):
        check_hv(test_field.hv.vector(kdims=["x", "y"], n=(10, 10, 10)), ...)

    # scalar field, same implementation for all ndim, testing with ndim=3 sufficient
    with pytest.raises(ValueError):
        check_hv(field_2d.a.hv.vector(kdims=["x", "y"]), ...)


def test_hv_contour(test_field):
    for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
        normal = (set("xyz") - set(kdims)).pop()
        kdim_str = f"[{','.join(kdims)}]"
        # Required for the tests (not in a notebook):
        # If not specified the plot creation for the test fails because the width
        # and height of the plot cannot be calculated (NaN values). By manually
        # setting frame_width and frame_height this can be avoided.
        # It is not fully clear why this does not happen for the other plots.
        # Presumably, because we use a different method to create the contour plot.
        opts = dict(frame_width=300, frame_height=300)
        check_hv(
            test_field.hv.contour(kdims=kdims).opts(**opts),
            [f"DynamicMap [{normal},vdims]", f"Contours {kdim_str}"],
        )
        check_hv(
            test_field.hv.contour(kdims=kdims, roi=test_field.norm).opts(**opts),
            [f"DynamicMap [{normal},vdims]", f"Contours {kdim_str}"],
        )

        # additional kwargs
        check_hv(
            test_field.sel(normal).hv.contour(kdims=kdims, clim=(-1, 1)).opts(**opts),
            ["DynamicMap [vdims]", f"Contours {kdim_str}"],
        )

        for c in test_field.vdims:
            check_hv(
                getattr(test_field, c).hv.contour(kdims=kdims).opts(**opts),
                [f"DynamicMap [{normal}]", f"Contours {kdim_str}"],
            )

    with pytest.raises(ValueError):
        check_hv(test_field.hv.contour(kdims=["wrong_name", "x"]), ...)


def test_hv(test_field):
    for kdims in [["x", "y"], ["x", "z"], ["y", "z"]]:
        normal = (set("xyz") - set(kdims)).pop()
        kdim_str = f"[{','.join(kdims)}]"
        # 1d field
        check_hv(
            test_field.a.hv(kdims=kdims),
            [f"DynamicMap [{normal}]", f"Image {kdim_str}"],
        )
        check_hv(test_field.a.sel(normal).hv(kdims=kdims), [f"Image {kdim_str}"])

        # 2d field
        field_2d = test_field.b << test_field.c
        check_hv(
            field_2d.hv(kdims=kdims),
            [f"DynamicMap [{normal},vdims]", f"Image {kdim_str}"],
        )
        check_hv(
            field_2d.hv(kdims=kdims, vdims=["x", "y"]),
            [f"DynamicMap [{normal}]", f"VectorField {kdim_str}"],
        )
        check_hv(
            field_2d.sel(normal).hv(kdims=kdims),
            ["DynamicMap [vdims]", f"Image {kdim_str}"],
        )

        # 3d field
        check_hv(
            test_field.hv(kdims=kdims),
            [
                f"DynamicMap [{normal}]",
                f"Image {kdim_str}",
                f"VectorField {kdim_str}",
            ],
        )
        check_hv(
            test_field.hv(kdims=kdims, vdims=["a", "b"]),
            [
                f"DynamicMap [{normal}]",
                f"Image {kdim_str}",
                f"VectorField {kdim_str}",
            ],
        )
        check_hv(
            test_field.sel(normal).hv(kdims=kdims),
            [f"Image {kdim_str}", f"VectorField {kdim_str}"],
        )

        # additional kwargs
        check_hv(
            test_field.hv(
                kdims=kdims,
                scalar_kw={"clim": (-1, 1)},
                vector_kw={"cmap": "cividis"},
            ),
            [
                f"DynamicMap [{normal}]",
                f"Image {kdim_str}",
                f"VectorField {kdim_str}",
            ],
        )

        # 4d field
        field_4d = test_field.b << test_field.c << test_field.a << test_field.a
        field_4d.vdims = ["v1", "v2", "v3", "v4"]
        check_hv(
            field_4d.hv(kdims=kdims),
            [f"DynamicMap [{normal},vdims]", f"Image {kdim_str}"],
        )

        check_hv(
            field_4d.hv(kdims=kdims, vdims=["v2", "v1"]),
            [
                f"DynamicMap [{normal},vdims]",
                f"Image {kdim_str}",
                f"VectorField {kdim_str}",
            ],
        )

        check_hv(
            field_4d.sel(normal).hv(kdims=kdims),
            ["DynamicMap [vdims]", f"Image {kdim_str}"],
        )

        check_hv(
            field_4d.sel(normal).hv(
                kdims=kdims, vdims=["v2", "v1"], vector_kw={"cdim": "v4"}
            ),
            [
                "DynamicMap [vdims]",
                f"Image {kdim_str}",
                f"VectorField {kdim_str}",
            ],
        )


@pytest.mark.parametrize("method", ["__call__", "scalar", "vector", "contour"])
def test_hv_ndim_1(method):
    """
    Plotting ndim=1 fields is not supported. This is only indirectly checked:
    ``kdims`` must have length 2 and if we pass an unknown kdim (region.dim) a
    ``ValueError`` is raised.

    All plotting methods behave the same.
    """
    field = df.Mesh(p1=[0], p2=[1], n=[10]).coordinate_field()
    with pytest.raises(ValueError):
        getattr(field.hv, method)(kdims=["x"])
    with pytest.raises(ValueError):
        getattr(field.hv, method)(kdims=["x", "y"])


@pytest.mark.parametrize("ndim", range(2, 5))
@pytest.mark.parametrize("nvdim", range(1, 5))
def test_hv_scalar_ndim(ndim, nvdim):
    """
    Scalar can plot any fields with arbitrary ndim and nvdim.

    Contour is based on scalar and can also show fields with arbitrary ndim and nvdim.
    Testing is however more difficult because contour does not work for spatially
    constant data (due to restrictions in HoloViews that we cannot easily test for).
    We restrict testing of contour to the 3d case.
    """
    mesh = df.Mesh(p1=[0] * ndim, p2=[1] * ndim, n=[10] * ndim)
    field = df.Field(mesh, nvdim=nvdim, value=[1] * nvdim)

    static_dims = ",".join(field.mesh.region.dims[:2])
    dyn_dims = ",".join(field.mesh.region.dims[2:])
    if nvdim > 1:
        dyn_dims += ("," if dyn_dims != "" else "") + "vdims"

    if ndim > 2 or nvdim > 1:
        reference = [f"DynamicMap [{dyn_dims}]", f"Image [{static_dims}]"]
    else:
        reference = [f"Image [{static_dims}]"]

    check_hv(field.hv.scalar(kdims=list(field.mesh.region.dims[:2])), reference)


@pytest.mark.filterwarnings("ignore:Automatic coloring")
@pytest.mark.parametrize("ndim", range(2, 5))
@pytest.mark.parametrize("nvdim", range(2, 5))
def test_hv_vector_ndim(ndim, nvdim):
    mesh = df.Mesh(p1=[0] * ndim, p2=[1] * ndim, n=[10] * ndim)
    field = df.Field(mesh, nvdim=nvdim, value=[1] * nvdim)

    static_dims = ",".join(field.mesh.region.dims[:2])
    dyn_dims = ",".join(field.mesh.region.dims[2:])

    if ndim > 2:
        reference = [f"DynamicMap [{dyn_dims}]", f"VectorField [{static_dims}]"]
    else:
        reference = [f"VectorField [{static_dims}]"]

    if ndim == nvdim:
        check_hv(field.hv.vector(kdims=list(field.mesh.region.dims[:2])), reference)
    else:
        check_hv(
            field.hv.vector(
                kdims=list(field.mesh.region.dims[:2]), vdims=field.vdims[:2]
            ),
            reference,
        )


@pytest.mark.parametrize("ndim", range(2, 5))
@pytest.mark.parametrize("nvdim", range(1, 5))
def test_hv_ndim(ndim, nvdim):
    mesh = df.Mesh(p1=[0] * ndim, p2=[1] * ndim, n=[10] * ndim)
    field = df.Field(mesh, nvdim=nvdim, value=[1] * nvdim)

    static_dims = ",".join(field.mesh.region.dims[:2])
    dyn_dims = ",".join(field.mesh.region.dims[2:])
    if nvdim > 3 or (nvdim > 1 and ndim != nvdim):
        dyn_dims += ("," if dyn_dims != "" else "") + "vdims"

    if ndim == 2 and nvdim == 1:
        reference = [f"Image [{static_dims}]"]
    elif ndim == 2 and nvdim == 2:
        reference = [f"VectorField [{static_dims}]"]
    elif ndim == nvdim:
        reference = [
            f"DynamicMap [{dyn_dims}]",
            f"Image [{static_dims}]",
            f"VectorField [{static_dims}]",
        ]
    else:
        reference = [f"DynamicMap [{dyn_dims}]", f"Image [{static_dims}]"]

    check_hv(field.hv(kdims=list(field.mesh.region.dims[:2])), reference)


def test_k3d(valid_mesh):
    f = df.Field(valid_mesh, nvdim=3, value=(1, 1, 1))
    if f.mesh.region.ndim != 3:
        with pytest.raises(RuntimeError):
            f.k3d.vector()
    else:
        f.k3d.vector()
        f.x.k3d.scalar()
        f.norm.k3d.nonzero()


def test_k3d_nonzero(test_field):
    # Default
    test_field.norm.k3d.nonzero()

    # Color
    test_field.a.k3d.nonzero(color=0xFF00FF)

    # Multiplier
    test_field.b.k3d.nonzero(color=0xFF00FF, multiplier=1e-6)

    # Interactive field
    range_ = (test_field.mesh.region.pmin[2], test_field.mesh.region.pmin[2])
    test_field.c.sel(z=range_).k3d.nonzero(
        color=0xFF00FF, multiplier=1e-6, interactive_field=test_field
    )

    # kwargs
    test_field.a.sel(z=range_).k3d.nonzero(
        color=0xFF00FF,
        multiplier=1e-6,
        interactive_field=test_field,
        wireframe=True,
    )

    # Plot
    plot = k3d.plot()
    plot.display()
    test_field.b.sel(z=range_).k3d.nonzero(
        plot=plot, color=0xFF00FF, multiplier=1e-6, interactive_field=test_field
    )

    # Continuation for interactive plot testing.
    test_field.c.sel(z=range_).k3d.nonzero(
        plot=plot, color=0xFF00FF, multiplier=1e-6, interactive_field=test_field
    )

    assert len(plot.objects) == 2

    with pytest.raises(ValueError):
        test_field.k3d.nonzero()


def test_k3d_scalar(test_field):
    # Default
    test_field.a.k3d.scalar()

    # Filter field
    test_field.b.k3d.scalar(filter_field=test_field.norm)

    # Colormap
    test_field.c.k3d.scalar(filter_field=test_field.norm, cmap="hsv", color=0xFF00FF)

    # Multiplier
    test_field.a.k3d.scalar(
        filter_field=test_field.norm, color=0xFF00FF, multiplier=1e-6
    )

    # Interactive field
    test_field.b.k3d.scalar(
        filter_field=test_field.norm,
        color=0xFF00FF,
        multiplier=1e-6,
        interactive_field=test_field,
    )

    # kwargs
    test_field.c.k3d.scalar(
        filter_field=test_field.norm,
        color=0xFF00FF,
        multiplier=1e-6,
        interactive_field=test_field,
        wireframe=True,
    )

    # Plot
    plot = k3d.plot()
    plot.display()
    range_ = (test_field.mesh.region.pmin[2], test_field.mesh.region.pmin[2])
    test_field.a.sel(z=range_).k3d.scalar(
        plot=plot,
        filter_field=test_field.norm,
        color=0xFF00FF,
        multiplier=1e-6,
        interactive_field=test_field,
    )

    # Continuation for interactive plot testing.
    test_field.b.sel(z=range_).k3d.scalar(
        plot=plot,
        filter_field=test_field.norm,
        color=0xFF00FF,
        multiplier=1e-6,
        interactive_field=test_field,
    )

    assert len(plot.objects) == 2

    # Exceptions
    with pytest.raises(ValueError):
        test_field.k3d.scalar()
    with pytest.raises(ValueError):
        test_field.c.k3d.scalar(filter_field=test_field)  # filter field nvdim=3


def test_k3d_vector(test_field):
    # Default
    test_field.k3d.vector()

    # Color field
    test_field.k3d.vector(color_field=test_field.a)

    # Colormap
    test_field.k3d.vector(color_field=test_field.norm, cmap="hsv")

    # Head size
    test_field.k3d.vector(color_field=test_field.norm, cmap="hsv", head_size=3)

    # Points
    test_field.k3d.vector(
        color_field=test_field.norm, cmap="hsv", head_size=3, points=False
    )

    # Point size
    test_field.k3d.vector(
        color_field=test_field.norm,
        cmap="hsv",
        head_size=3,
        points=False,
        point_size=1,
    )

    # Vector multiplier
    test_field.k3d.vector(
        color_field=test_field.norm,
        cmap="hsv",
        head_size=3,
        points=False,
        point_size=1,
        vector_multiplier=1,
    )

    # Multiplier
    test_field.k3d.vector(
        color_field=test_field.norm,
        cmap="hsv",
        head_size=3,
        points=False,
        point_size=1,
        vector_multiplier=1,
        multiplier=1e-6,
    )

    # Interactive field
    range_ = (test_field.mesh.region.pmin[2], test_field.mesh.region.pmin[2])
    test_field.sel(z=range_).k3d.vector(
        color_field=test_field.norm,
        cmap="hsv",
        head_size=3,
        points=False,
        point_size=1,
        vector_multiplier=1,
        multiplier=1e-6,
        interactive_field=test_field,
    )

    # Plot
    plot = k3d.plot()
    plot.display()
    test_field.sel(z=range_).k3d.vector(plot=plot, interactive_field=test_field)

    # Continuation for interactive plot testing.
    test_field.sel(z=range_).k3d.vector(plot=plot, interactive_field=test_field)

    assert len(plot.objects) == 3

    # Exceptions
    with pytest.raises(ValueError):
        test_field.a.k3d.vector()
    with pytest.raises(ValueError):
        test_field.k3d.vector(color_field=test_field)  # filter field nvdim=3


def test_plot_large_sample():
    p1 = (0, 0, 0)
    p2 = (50e9, 50e9, 50e9)
    cell = (25e9, 25e9, 25e9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    value = (1e6, 1e6, 1e6)
    field = df.Field(mesh, nvdim=3, value=value)

    field.sel("z").mpl()
    field.norm.k3d.nonzero()
    field.x.k3d.scalar()
    field.k3d.vector()


# ##################################


def test_complex(test_field):
    mesh = df.Mesh(p1=(-5e-9, -5e-9, -5e-9), p2=(5e-9, 5e-9, 5e-9), n=(5, 5, 5))

    # real field
    real_field = test_field.real
    assert isinstance(real_field, df.Field)
    assert np.allclose(real_field((-2e-9, 0, 0)), (0, 0, 1e5))
    assert np.allclose(real_field((2e-9, 0, 0)), (0, 0, -1e5))

    imag_field = test_field.imag
    assert isinstance(imag_field, df.Field)
    assert df.Field(mesh, nvdim=3).allclose(imag_field)
    assert df.Field(mesh, nvdim=3).allclose(np.mod(test_field.phase, np.pi))

    # complex field
    field = df.Field(mesh, nvdim=1, value=1 + 1j)
    real_field = field.real
    assert isinstance(real_field, df.Field)
    assert df.Field(mesh, nvdim=1, value=1).allclose(real_field)

    imag_field = field.imag
    assert isinstance(imag_field, df.Field)
    assert df.Field(mesh, nvdim=1, value=1).allclose(imag_field)
    assert df.Field(mesh, nvdim=1, value=np.pi / 4).allclose(field.phase)


# TODO: Test at method, multiple np array
@pytest.mark.parametrize("ufunc", [np.add, np.multiply, np.power])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_numpy_ufunc_two_input(valid_mesh, nvdim, ufunc):
    field = df.Field(valid_mesh, nvdim=nvdim, value=tuple(range(nvdim)))

    # Test with another field
    assert np.allclose(
        ufunc(field, field).array,
        ufunc(field.array, field.array),
    )

    # Test with a scalar
    assert np.allclose(
        ufunc(field, 2).array,
        ufunc(field.array, 2),
    )

    # Test with an ndarray
    array = np.array(range(nvdim))
    assert np.allclose(
        ufunc(field, array).array,
        ufunc(field.array, array),
    )


@pytest.mark.parametrize("ufunc", [np.sin, np.exp])
@pytest.mark.parametrize("nvdim", [1, 2, 3, 4])
def test_numpy_ufunc_single_input(valid_mesh, nvdim, ufunc):
    field = df.Field(valid_mesh, nvdim=nvdim, value=tuple(range(nvdim)))
    assert np.allclose(
        ufunc(field).array,
        ufunc(field.array),
    )


@pytest.mark.parametrize("value, dtype", vfuncs)
def test_to_xarray_valid_args_vector(valid_mesh, value, dtype):
    f = df.Field(valid_mesh, nvdim=3, value=value, dtype=dtype)
    fxa = f.to_xarray()
    assert isinstance(fxa, xr.DataArray)
    assert f.nvdim == fxa["vdims"].size
    assert sorted([*fxa.attrs]) == [
        "cell",
        "nvdim",
        "pmax",
        "pmin",
        "tolerance_factor",
        "units",
    ]
    assert np.allclose(fxa.attrs["cell"], f.mesh.cell)
    assert np.allclose(fxa.attrs["pmin"], f.mesh.region.pmin)
    assert np.allclose(fxa.attrs["pmax"], f.mesh.region.pmax)
    assert np.allclose(fxa.attrs["tolerance_factor"], f.mesh.region.tolerance_factor)
    for i in f.mesh.region.dims:
        assert np.array_equal(getattr(f.mesh.cells, i), fxa[i].values)
        assert fxa[i].attrs["units"] == f.mesh.region.units[f.mesh.region.dims.index(i)]
    assert all(fxa["vdims"].values == f.vdims)
    assert np.array_equal(f.array, fxa.values)


@pytest.mark.parametrize("value, dtype", sfuncs)
def test_to_xarray_valid_args_scalar(valid_mesh, value, dtype):
    f = df.Field(valid_mesh, nvdim=1, value=value, dtype=dtype)
    fxa = f.to_xarray()
    assert isinstance(fxa, xr.DataArray)
    assert sorted([*fxa.attrs]) == [
        "cell",
        "nvdim",
        "pmax",
        "pmin",
        "tolerance_factor",
        "units",
    ]
    assert np.allclose(fxa.attrs["cell"], f.mesh.cell)
    assert np.allclose(fxa.attrs["pmin"], f.mesh.region.pmin)
    assert np.allclose(fxa.attrs["pmax"], f.mesh.region.pmax)
    assert np.allclose(fxa.attrs["tolerance_factor"], f.mesh.region.tolerance_factor)
    for i in f.mesh.region.dims:
        assert np.array_equal(getattr(f.mesh.cells, i), fxa[i].values)
        assert fxa[i].attrs["units"] == f.mesh.region.units[f.mesh.region.dims.index(i)]
    assert "vdims" not in fxa.dims
    assert np.array_equal(f.array.squeeze(axis=-1), fxa.values)


def test_to_xarray_6d_field(test_field):
    f6d = test_field << test_field
    f6d_xa = f6d.to_xarray()
    assert f6d_xa["vdims"].size == 6
    assert "vdims" in f6d_xa.coords
    assert [*f6d_xa["vdims"].values] == [f"v{i}" for i in range(6)]
    f6d.vdims = ["a", "c", "b", "e", "d", "f"]
    f6d_xa2 = f6d.to_xarray()
    assert "vdims" in f6d_xa2.coords
    assert [*f6d_xa2["vdims"].values] == ["a", "c", "b", "e", "d", "f"]

    # test name and units defaults
    f3d_xa = test_field.to_xarray()
    assert f3d_xa.name == "field"
    assert f3d_xa.attrs["units"] is None

    # test name and units
    f3d_xa_2 = test_field.to_xarray(name="m", unit="A/m")
    assert f3d_xa_2.name == "m"
    assert f3d_xa_2.attrs["units"] == "A/m"


@pytest.mark.parametrize(
    "name, unit",
    [
        ["m", 42.0],
        [21.0, 42],
        [21, "A/m"],
        [{"name": "m"}, {"unit": "A/m"}],
        [["m"], ["A/m"]],
        [["m", "A/m"], None],
        [("m", "A/m"), None],
        [{"name": "m", "unit": "A/m"}, None],
    ],
)
def test_to_xarray_invalid_args(name, unit, test_field):
    with pytest.raises(TypeError):
        test_field.to_xarray(name, unit)


@pytest.mark.parametrize("value, dtype", vfuncs)
def test_from_xarray_valid_args_vector(valid_mesh, value, dtype):
    f = df.Field(valid_mesh, nvdim=3, value=value, dtype=dtype)
    fxa = f.to_xarray()
    f_new = df.Field.from_xarray(fxa)
    assert f_new == f


@pytest.mark.parametrize("value, dtype", sfuncs)
def test_from_xarray_valid_args_scalar(valid_mesh, value, dtype):
    f = df.Field(valid_mesh, nvdim=1, value=value, dtype=dtype)
    fxa = f.to_xarray()
    f_new = df.Field.from_xarray(fxa)
    assert f_new == f


def test_from_xarray_valid_args(test_field):
    f_plane = test_field.sel("z")
    f_plane_xa = f_plane.to_xarray()
    f_plane_new = df.Field.from_xarray(f_plane_xa)
    assert f_plane_new == f_plane

    f6d = test_field << test_field
    f6d_xa = f6d.to_xarray()
    f6d_new = df.Field.from_xarray(f6d_xa)
    assert f6d_new == f6d

    good_darray1 = xr.DataArray(
        np.ones((20, 20, 5, 3)),
        dims=["x", "y", "z", "vdims"],
        coords=dict(
            x=np.arange(0, 20),
            y=np.arange(0, 20),
            z=np.arange(0, 5),
            vdims=["x", "y", "z"],
        ),
        name="mag",
        attrs=dict(units="A/m", nvdim=3),
    )

    good_darray2 = xr.DataArray(
        np.ones((20, 20, 1, 3)),
        dims=["x", "y", "z", "vdims"],
        coords=dict(
            x=np.arange(0, 20), y=np.arange(0, 20), z=[5.0], vdims=["x", "y", "z"]
        ),
        name="mag",
        attrs=dict(units="A/m", cell=[1.0, 1.0, 1.0], nvdim=3),
    )

    good_darray3 = xr.DataArray(
        np.ones((20, 20, 1, 3)),
        dims=["x", "y", "z", "vdims"],
        coords=dict(
            x=np.arange(0, 20), y=np.arange(0, 20), z=[5.0], vdims=["x", "y", "z"]
        ),
        name="mag",
        attrs=dict(
            units="A/m",
            cell=[1.0, 1.0, 1.0],
            p1=[1.0, 1.0, 1.0],
            p2=[21.0, 21.0, 2.0],
            nvdim=3,
        ),
    )

    fg_1 = df.Field.from_xarray(good_darray1)
    assert isinstance(fg_1, df.Field)
    fg_2 = df.Field.from_xarray(good_darray2)
    assert isinstance(fg_2, df.Field)
    fg_3 = df.Field.from_xarray(good_darray3)
    assert isinstance(fg_3, df.Field)


def test_from_xarray_invalid_args_and_DataArrays():
    args = [
        int(),
        float(),
        str(),
        list(),
        dict(),
        xr.Dataset(),
        np.empty((20, 20, 20, 3)),
    ]

    bad_dim_no2 = xr.DataArray(
        np.ones((20, 20), dtype=float),
        dims=["x", "y"],
        coords=dict(x=np.arange(0, 20), y=np.arange(0, 20)),
        name="mag",
        attrs=dict(units="A/m", nvdim=3),
    )

    bad_dim3 = xr.DataArray(
        np.ones((20, 20, 5), dtype=float),
        dims=["a", "b", "c"],
        coords=dict(a=np.arange(0, 20), b=np.arange(0, 20), c=np.arange(0, 5)),
        name="mag",
        attrs=dict(units="A/m", nvdim=3),
    )

    bad_dim4 = xr.DataArray(
        np.ones((20, 20, 5, 3), dtype=float),
        dims=["x", "y", "z", "c"],
        coords=dict(
            x=np.arange(0, 20),
            y=np.arange(0, 20),
            z=np.arange(0, 5),
            c=["x", "y", "z"],
        ),
        name="mag",
        attrs=dict(units="A/m", nvdim=3),
    )

    bad_attrs = xr.DataArray(
        np.ones((20, 20, 1, 3), dtype=float),
        dims=["x", "y", "z", "vdims"],
        coords=dict(
            x=np.arange(0, 20), y=np.arange(0, 20), z=[5.0], vdims=["x", "y", "z"]
        ),
        name="mag",
        attrs=dict(units="A/m"),
    )

    def bad_coord_gen():
        rng = np.random.default_rng()
        for coord in "xyz":
            coord_dict = {coord: rng.normal(size=20)}
            for other_coord in "xyz".translate({ord(coord): None}):
                coord_dict[other_coord] = np.arange(0, 20)
            coord_dict["vdims"] = ["x", "y", "z"]

            yield xr.DataArray(
                np.ones((20, 20, 20, 3), dtype=float),
                dims=["x", "y", "z", "vdims"],
                coords=coord_dict,
                name="mag",
                attrs=dict(units="A/m", nvdim=3),
            )

    for arg in args:
        with pytest.raises(TypeError):
            df.Field.from_xarray(arg)
    with pytest.raises(ValueError):
        df.Field.from_xarray(bad_dim_no2)
    with pytest.raises(ValueError):
        df.Field.from_xarray(bad_dim3)
    with pytest.raises(ValueError):
        df.Field.from_xarray(bad_dim4)
    for bad_coord_geo in bad_coord_gen():
        with pytest.raises(ValueError):
            df.Field.from_xarray(bad_coord_geo)
    with pytest.raises(KeyError):
        df.Field.from_xarray(bad_attrs)
