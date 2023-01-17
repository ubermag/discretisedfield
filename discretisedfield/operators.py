import numpy as np


def integrate(field, direction=None, cumulative=False):
    """Integral.

    This function calls ``integral`` method of the ``discrteisedfield.Field``
    object.

    For details, please refer to :py:func:`~discretisedfield.Field.integral`

    """
    return field.integrate(direction=direction, cumulative=cumulative)


def _1d_diff(order, array, dx):
    """Differentiate 1D array."""

    # Directional derivative cannot be computed if less or an equal number of
    # discretisation cells exists in a specified direction than the order.
    # In that case, a zero array is returned.
    if len(array) < order + 1:
        return np.zeros_like(array)

    if order == 1:
        if len(array) < 3:
            # Second order accuracy is in the center of the array and
            # first order at the boundaries.
            derivative_array = np.gradient(array, dx, edge_order=1)
        else:
            # Second order accuracy at the boundaries.
            derivative_array = np.gradient(array, dx, edge_order=2)

    elif order == 2:
        # The derivative is computed using the central difference
        # this stencil will give incorrect result at the boundary.
        derivative_array = np.convolve(array, [1, -2, 1], "same")
        if len(array) >= 4:
            # Second order accuracy at the boundaries
            # These stencil coefficients are taken from FinDiff.
            derivative_array[0] = 2 * array[0] - 5 * array[1] + 4 * array[2] - array[3]
            derivative_array[-1] = (
                2 * array[-1] - 5 * array[-2] + 4 * array[-3] - array[-4]
            )
        else:
            # First order accuracy at the boundaries
            # These stencil coefficients are taken from FinDiff.
            derivative_array[0] = array[0] - 2 * array[1] + array[2]
            derivative_array[-1] = array[-1] - 2 * array[-2] + array[-3]
        derivative_array = derivative_array / dx**2

    return derivative_array


def _split_array_on_idx(array, loc):
    """Split a 1D array on based on indices.
    For a 100 element array, this method is 15.3 µs ± 63.1 ns
    compared to itertools.groupby which is 70.3 µs ± 719 ns."""
    loc = np.concatenate(([-1], loc, [len(array)]))
    # loc[i] is the location of a False hence we want to start
    # at the next element which is loc[i] + 1.
    # We then create a slice to the next False element which
    # is at loc[i + 1]. If the next False element is the same
    # as the current one, we do not want to create a slice.
    return [
        array[loc[i] + 1 : loc[i + 1]]
        for i in range(len(loc) - 1)
        if loc[i + 1] != loc[i] + 1
    ]


def _split_diff_combine(array, valid, order, dx):
    """Split a 1D array (with spacing dx)
    based on contiguous valid values,
    compute the derivative of certain order,
    and recombine the array."""
    # Find indices of invalid cells. The [0] is needed because
    # np.where returns a tuple of ndarray and we are only ever
    # in the case where we have a single element tuple.
    idx = np.where(np.invert(valid))[0]
    split = _split_array_on_idx(array, idx)
    diff = [_1d_diff(order, arr, dx) for arr in split]
    out = np.zeros_like(array)
    if len(diff) == 0:
        return out
    else:
        out[valid] = np.concatenate(diff)
        return out
