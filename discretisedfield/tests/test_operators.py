import numpy as np
import pytest
import sympy as sp

import discretisedfield as df
from discretisedfield.operators import (
    _1d_diff,
    _split_array_on_idx,
    _split_diff_combine,
)


def test_integrate():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (1, 1, 1)
    region = df.Region(p1=p1, p2=p2)
    mesh = df.Mesh(region=region, cell=cell)
    field = df.Field(mesh, nvdim=3, value=(1, -2, 3))

    assert np.allclose(df.integrate(field), (1000, -2000, 3000))
    assert np.allclose(df.integrate(field * 2), (2000, -4000, 6000))

    assert df.integrate(field.sel("z").dot([0, 0, 1])) == 300


@pytest.mark.parametrize(
    "valid",
    [
        [True, True, True],
        [True, True, False],
        [True, False, True],
        [False, True, True],
        [False, False, False],
        [True, True, False, True],
        [True, True, False, False, True],
        [True, False, True, False, True],
        [False, False, False, True, True, False],
    ],
)
def test_split_array_on_idx(valid):
    array = np.arange(len(valid))
    idx = np.where(np.invert(valid))[0]
    split_list = _split_array_on_idx(array, idx)

    assert isinstance(split_list, list)
    assert all(isinstance(sublist, np.ndarray) for sublist in split_list)

    # Check total number of elements
    assert sum(len(sublist) for sublist in split_list) == sum(valid)

    # Check all elements are present
    flat_list = [num for sublist in split_list for num in sublist]
    assert np.array_equal(flat_list, array[valid])

    # Check Missing elements
    missing = [i for i, x in enumerate(valid) if not x]

    # Check that missing elements are not in any sublist
    assert all(num not in flat_list for num in missing)


def test_split_diff_combine():
    valid = [True, True, False, False, True, True, False, True]
    array = np.arange(len(valid))
    out = _split_diff_combine(array, valid, 1, 1)
    assert len(out) == len(array)
    assert np.allclose(out, [1, 1, 0, 0, 1, 1, 0, 0])

    valid = [False, False, False, False, False, False, False, False]
    out = _split_diff_combine(array, valid, 1, 1)
    assert len(out) == len(array)
    assert np.allclose(out, [0, 0, 0, 0, 0, 0, 0, 0])


@pytest.mark.parametrize("array_len", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dx", [1, 2, 0.5])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize(
    "func",
    [
        lambda x: x,
        lambda x: x * x,
        lambda x: 1 / (x + 1),
        lambda x: x * x * x,
        lambda x: sp.sin(2 * sp.pi * x / 10),
    ],
)
def test_1d_derivative_sympy(func, order, array_len, dx):
    # Create array with sympy function
    x = sp.symbols("x")
    fx = func(x)
    lam_f = sp.lambdify(x, fx, "numpy")

    x_array = np.arange(array_len) * dx
    y_array = lam_f(x_array)

    # Calculate derivative
    diff_array = _1d_diff(order, y_array, dx)

    assert diff_array.shape == y_array.shape

    if order == 1:
        if array_len == 1:
            assert np.allclose(diff_array, 0)
        elif array_len == 2:
            # Accuracy is 1
            sp_expected = float(
                sp.apply_finite_diff(order, x_array, y_array, x_array[0])
            )
            assert np.allclose(diff_array, sp_expected)
        elif array_len > 3:
            # Accuracy is 2
            sp_expected = float(
                sp.apply_finite_diff(order, x_array[:3], y_array[:3], x_array[0])
            )
            assert np.allclose(diff_array[0], sp_expected)
            sp_expected = float(
                sp.apply_finite_diff(order, x_array[-3:], y_array[-3:], x_array[-1])
            )
            assert np.allclose(diff_array[-1], sp_expected)
            # test middle values
            cent = int(array_len / 2)
            sp_expected = float(
                sp.apply_finite_diff(
                    order,
                    x_array[cent - 1 : cent + 2],
                    y_array[cent - 1 : cent + 2],
                    x_array[cent],
                )
            )
            assert np.allclose(diff_array[cent], sp_expected)
    elif order == 2:
        if array_len == 1 or array_len == 2:
            assert np.allclose(diff_array, 0)
        elif array_len == 3:
            # Accuracy of 2nd order is 1
            sp_expected = float(
                sp.apply_finite_diff(order, x_array, y_array, x_array[0])
            )
            assert np.allclose(diff_array, sp_expected)
        elif array_len > 4:
            # Accuracy of 2nd order is 2
            sp_expected = float(
                sp.apply_finite_diff(order, x_array[:4], y_array[:4], x_array[0])
            )
            assert np.allclose(diff_array[0], sp_expected)
            sp_expected = float(
                sp.apply_finite_diff(order, x_array[-4:], y_array[-4:], x_array[-1])
            )
            assert np.allclose(diff_array[-1], sp_expected)
            # test middle values
            cent = int(array_len / 2)
            sp_expected = float(
                sp.apply_finite_diff(
                    order,
                    x_array[cent - 1 : cent + 2],
                    y_array[cent - 1 : cent + 2],
                    x_array[cent],
                )
            )
            assert np.allclose(diff_array[cent], sp_expected)
