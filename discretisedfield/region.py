import collections
import numbers
import warnings

import numpy as np
import ubermagutil.units as uu

import discretisedfield.plotting as dfp

from . import html
from .io import _RegionIO


class Region(_RegionIO):
    r"""Region.

    A cuboid region spans between two corner points :math:`\mathbf{p}_1` and
    :math:`\mathbf{p}_2`. Points ``p1`` and ``p2`` can be any two
    diagonally-opposite points. If any of the edge lengths of the cuboid region
    is zero, ``ValueError`` is raised.

    Parameters
    ----------
    p1 / p2 : array_like

        Diagonally-opposite corner points of the region, for example in three
        dimensions :math:`\mathbf{p}_i = (p_x, p_y, p_z)`.

    dims : array_like of str, optional

        Name of the respective geometrical dimensions of the region.

        Up to three dimensions, this defaults to ``x``, ``y``, and ``z``. For more than
        three dimensions, it defaults to ``x1``, ``x2``, ``x3``, ``x4``, etc.

    units : array_like of str, optional

        Physical units of the region. This is mainly used for labelling plots.
        Defaults to ``m`` for all the dimensions.

    tolerance_factor : float, optional

        This factor is used to obtain a tolerance for comparison operations,
        e.g. ``region1 in region2``. It is internally multiplied with the
        minimum of the edge lengths to adjust the tolerance to the region size
        and have more accurate floating-point comparisons. Defaults to
        ``1e-12``.

    Raises
    ------
    ValueError

        If any of the region's edge lengths is zero.

    Examples
    --------
    1. Defining a nano-sized region.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> region = df.Region(p1=p1, p2=p2)
    ...
    >>> region
    Region(...)

    2. An attempt to define a region whose one of the edge lengths is zero.

    >>> # The edge length in the z-direction is zero.
    >>> p1 = (-25, 3, 1)
    >>> p2 = (25, 6, 1)
    >>> region = df.Region(p1=p1, p2=p2)
    Traceback (most recent call last):
        ...
    ValueError: ...

    """
    __slots__ = ["_pmin", "_pmax", "_dims", "_units", "_tolerance_factor"]

    def __init__(
        self, p1=None, p2=None, dims=None, units=None, tolerance_factor=1e-12, **kwargs
    ):
        # Allow pmin and pmax instead of p1 and p2 to simplify the internal code and the
        # conversion from a dict to a Region. Users should generally use p1 and p2 in
        # their code.
        if "pmin" in kwargs and "pmax" in kwargs:
            pmin, pmax = kwargs["pmin"], kwargs["pmax"]
            if not all(np.asarray(pmin) < np.asarray(pmax)):
                raise ValueError(
                    f"The values in {pmin=} must be element-wise smaller than in"
                    f" {pmax=}; use p1 and p2 if the input values are unordered."
                )
            p1, p2 = pmin, pmax

        # scalar data types for 1d regions
        if isinstance(p1, numbers.Real):
            p1 = [p1]
        if isinstance(p2, numbers.Real):
            p2 = [p2]

        if not isinstance(p1, (tuple, list, np.ndarray)) or not isinstance(
            p2, (tuple, list, np.ndarray)
        ):
            raise TypeError(
                "p1 and p2 must be real numbers (1d) or sequences of real numbers. Not"
                f" {type(p1)=} and {type(p2)=}."
            )

        if len(p1) != len(p2):
            raise ValueError(
                "The length of p1 and p2 must be the same. Not"
                f" {len(p1)=} and {len(p2)=}."
            )

        if len(p1) == 0:
            raise ValueError("p1 and p2 must not be empty.")

        if not all(isinstance(i, numbers.Real) for i in p1):
            raise TypeError("p1 can only contain elements of type numbers.Real.")

        if not all(isinstance(i, numbers.Real) for i in p2):
            raise TypeError("p2 can only contain elements of type numbers.Real.")

        self._pmin = np.minimum(p1, p2)
        self._pmax = np.maximum(p1, p2)
        self.dims = dims
        self.units = units
        self.tolerance_factor = tolerance_factor

        if not np.all(self.edges):
            raise ValueError(
                f"At least one of the region's edge lengths is zero: {self.edges=}."
            )

    @property
    def pmin(self):
        r"""Point with minimum coordinates in the region.

        The :math:`i`-th component of :math:`\mathbf{p}_\text{min}` is computed
        from points :math:`p_1` and :math:`p_2`, between which the region
        spans: :math:`p_\text{min}^i = \text{min}(p_1^i, p_2^i)`.

        Returns
        -------
        numpy.ndarray

            Point with minimum coordinates. E.g. for three dimensions
            :math:`(p_x^\text{min}, p_y^\text{min}, p_z^\text{min})`.

        Examples
        --------
        1. Getting region's point with minimum coordinates.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.pmin
        array([-1.1,  0. , -0.1])

        .. seealso:: :py:func:`~discretisedfield.Region.pmax`

        """
        return self._pmin

    @property
    def pmax(self):
        r"""Point with maximum coordinates in the region.

        The :math:`i`-th component of :math:`\mathbf{p}_\text{max}` is computed
        from points :math:`p_1` and :math:`p_2`, between which the region
        spans: :math:`p_\text{max}^i = \text{max}(p_1^i, p_2^i)`.

        Returns
        -------
        numpy.ndarray

            Point with maximum coordinates. E.g. for three dimensions
            :math:`(p_x^\text{max}, p_y^\text{max}, p_z^\text{max})`.

        Examples
        --------
        1. Getting region's point with maximum coordinates.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.pmax
        array([5. , 2.9, 0. ])

        .. seealso:: :py:func:`~discretisedfield.Region.pmin`

        """
        return self._pmax

    @property
    def ndim(self):
        r"""Number of dimensions.

        Calculates the number of dimensions of the region.

        Returns
        -------
        int

            Number of dimensions of the region.

        Examples
        --------
        1. Getting number of dimensions of the region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.ndim
        3

        .. seealso:: :py:func:`~discretisedfield.Region.dims`

        """
        return len(self.pmin)

    @property
    def dims(self):
        r"""Names of the region's dimensions.

        Returns
        -------
        tuple of str

            Names of the region's dimensions.

        Examples
        --------
        1. Getting region's dimension names.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.dims
        ('x', 'y', 'z')

        .. seealso:: :py:func:`~discretisedfield.Region.ndim`
        """
        return self._dims

    @dims.setter
    def dims(self, dims):
        # TODO: Think about correct defaults
        if dims is None:
            if self.ndim <= 3:
                dims = ["x", "y", "z"][: self.ndim]
            else:
                dims = [f"x{i}" for i in range(self.ndim)]
        elif isinstance(dims, (tuple, list, np.ndarray, str)):
            if isinstance(dims, str):
                dims = [dims]
            if len(dims) != self.ndim:
                raise ValueError(
                    "dims must have the same length as p1 and p2."
                    f" Not len(dims)={len(dims)} and ndim={self.ndim}."
                )
            if not all(isinstance(dim, str) for dim in dims):
                raise TypeError("dims can only contain elements of type str.")
            if len(dims) != len(set(dims)):
                raise ValueError("dims must be unique.")
        else:
            raise TypeError(
                "dims must be of type tuple, list, or None (for default behaviour)."
                f" Not {type(dims)}."
            )

        self._dims = tuple(dims)

    def _dim2index(self, dim):
        try:
            return self.dims.index(dim)
        except ValueError:
            raise ValueError(f"'{dim}' not in region.dims={self.dims}.") from None

    @property
    def units(self):
        r"""Units of the region's dimensions.

        Returns
        -------
        tuple of str

            Units of the region's dimensions.

        Examples
        --------
        1. Getting region's dimension units.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.units
        ('m', 'm', 'm')

        """
        return self._units

    @units.setter
    def units(self, units):
        if units is None:
            units = ["m"] * self.ndim
        elif isinstance(units, (tuple, list, np.ndarray, str)):
            if isinstance(units, str):
                units = [units]
            if len(units) != self.ndim:
                raise ValueError(
                    "units must have the same length as p1 and p2."
                    f" Not {len(units)=} and {self.ndim=}."
                )
            if not all(isinstance(unit, str) for unit in units):
                raise TypeError("units can only contain elements of type str.")
        else:
            raise TypeError(
                "units must be of type tuple, list, or None (for default behaviour)."
                f" Not {type(units)}."
            )

        self._units = tuple(units)

    @property
    def tolerance_factor(self):
        r"""Tolerance factor for floating-point comparisons.

        The tolerance factor is used for allclose and ``in`` if no other tolerance is
        provided. It is multiplied with the minimum edge length of the region to obtain
        reasonable relative and absolute tolerance.

        """
        return self._tolerance_factor

    @tolerance_factor.setter
    def tolerance_factor(self, tolerance_factor):
        if not isinstance(tolerance_factor, numbers.Number):
            raise TypeError(
                "tolerance_factor must be of type numbers.Number. Not"
                f" tolerance_factor={type(tolerance_factor)}."
            )
        self._tolerance_factor = tolerance_factor

    @property
    def edges(self):
        r"""Region's edge lengths.

        Edge length is computed from the points between which the region spans
        :math:`\mathbf{p}_1` and :math:`\mathbf{p}_2`:

        .. math::

            \mathbf{l} = (|p_2^x - p_1^x|, |p_2^y - p_1^y|, |p_2^z - p_1^z|).

        Returns
        -------
        numpy.ndarray

             Edge lengths. E.g. in three dimensions :math:`(l_{x}, l_{y}, l_{z})`.

        Examples
        --------
        1. Getting edge lengths of the region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5)
        >>> p2 = (5, 15, 15)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.edges
        array([ 5, 15, 20])

        """
        return self.pmax - self.pmin

    @property
    def center(self):
        r"""Center point.

        Center point is computed as the middle point between region's points
        with minimum and maximum coordinates:

        .. math::

            \mathbf{p}_\text{center} = \frac{1}{2} (\mathbf{p}_\text{min}
            + \mathbf{p}_\text{max}).

        Returns
        -------
        numpy.ndarray

            Center point. E.g. in three dimensions :math:`(p_c^x, p_c^y, p_c^z)`.

        Examples
        --------
        1. Getting the center point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 15, 20)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.center
        array([ 2.5,  7.5, 10. ])

        """
        return 0.5 * np.add(self.pmin, self.pmax)

    @property
    def centre(self):
        return self.center

    @property
    def volume(self):
        r"""Region's volume.

        It is computed by multiplying edge lengths of the region.
        E.g. in three dimensions

        .. math::

            V = l_x l_y l_z.

        Returns
        -------
        numbers.Real

            Volume of the region.

        Examples
        --------
        1. Computing the volume of the region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 10, 2)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.volume
        100

        """
        return np.prod(self.edges)

    def __repr__(self):
        r"""Representation string.

        Internally `self._repr_html_()` is called and all html tags are removed
        from this string.

        Returns
        -------
        str

           Representation string.

        Example
        -------
        1. Getting representation string.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region
        Region(pmin=[0, 0, 0], pmax=[2, 2, 1], ...)

        """
        return html.strip_tags(self._repr_html_())

    def _repr_html_(self):
        """Show HTML-based representation in Jupyter notebook."""
        return html.get_template("region").render(region=self)

    def __eq__(self, other):
        r"""Relational operator ``==``.

        Two regions are considered to be equal if they have the same minimum
        and maximum coordinate points, the same units, and the same dimension names.

        Parameters
        ----------
        other : discretisedfield.Region

            Second operand.

        Returns
        -------
        bool

            ``True`` if two regions are equal and ``False`` otherwise.

        Examples
        --------
        1. Usage of relational operator ``==``.

        >>> import discretisedfield as df
        ...
        >>> region1 = df.Region(p1=(0, 0, 0), p2=(5, 5, 5))
        >>> region2 = df.Region(p1=(0.0, 0, 0), p2=(5.0, 5, 5))
        >>> region3 = df.Region(p1=(1, 1, 1), p2=(5, 5, 5))
        ...
        >>> region1 == region2
        True
        >>> region1 != region2
        False
        >>> region1 == region3
        False
        >>> region1 != region3
        True

        """
        if isinstance(other, self.__class__):
            return (
                np.array_equal(self.pmin, other.pmin)
                and np.array_equal(self.pmax, other.pmax)
                and self.dims == other.dims
                and self.units == other.units
            )

        return False

    def allclose(
        self,
        other,
        rtol=None,
        atol=None,
    ):
        r"""Check if two regions are close.

        Two regions are considered to be equal if they have the same minimum
        and maximum coordinate points: :math:`\mathbf{p}^\text{max}_1 =
        \mathbf{p}^\text{max}_2` and :math:`\mathbf{p}^\text{min}_1 =
        \mathbf{p}^\text{min}_2` within a tolerance.

        Parameters
        ----------
        other : discretisedfield.Region

            Second operand.

        atol : numbers.Number, optional

            Absolute tolerance. If ``None``, the default value is
            the smallest edge length of the region multiplied by
            the tolerance factor.

        rtol : numbers.Number, optional

            Relative tolerance. If ``None``, ``region.tolerance_factor`` is used.

        Returns
        -------
        bool

            ``True`` if two regions are equal (within floating-point accuracy) and
            ``False`` otherwise.

        Examples
        --------
        1. Usage of ``allclose`` method.

        >>> import discretisedfield as df
        ...
        >>> region1 = df.Region(p1=(0, 0, 0), p2=(5, 5, 5))
        >>> region2 = df.Region(p1=(0.0, 0, 0), p2=(5.0, 5, 5))
        >>> region3 = df.Region(p1=(1, 1, 1), p2=(5, 5, 5))
        ...
        >>> region1.allclose(region2)
        True
        >>> region1.allclose(region3)
        False
        >>> region2.allclose(region3)
        False

        """
        if isinstance(other, self.__class__):
            if atol is None:
                atol = np.min(self.edges) * self.tolerance_factor
            elif not isinstance(atol, numbers.Number):
                raise TypeError(f"{type(atol)=} is not a number.")

            if rtol is None:
                rtol = self.tolerance_factor
            elif not isinstance(rtol, numbers.Number):
                raise TypeError(f"{type(rtol)=} is not a number.")

            return np.allclose(
                self.pmin, other.pmin, atol=atol, rtol=rtol
            ) and np.allclose(self.pmax, other.pmax, atol=atol, rtol=rtol)

        raise TypeError(
            f"Unsupported {(type(other))=}; only objects of type Region are allowed for"
            " method allclose."
        )

    def __contains__(self, other):
        """Determine if a point or another region belong to the region.

        Point is considered to be in the region if

        .. math::

            p^\\text{min}_{i} \\le p_{i} \\le p^\\text{max}_{i}, \\text{for}\\,
            i in dims.

        Similarly, if the second operand is ``discretisedfield.Region`` object,
        it is considered to be in the region if both its ``pmin`` and ``pmax``
        belong to the region.

        Parameters
        ----------
        other : array_like or discretisedfield.Region

            The point coordinate (E.g. in three dimensions
            :math:`(p_{x}, p_{y}, p_{z})`) or a region object.

        Returns
        -------
        bool

            ``True`` if ``other`` is inside the region and ``False`` otherwise.

        Example
        -------
        1. Check if point is inside the region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> (1, 1, 1) in region
        True
        >>> (1, 3, 1) in region
        False
        >>> # corner points are considered to be in the region
        >>> p1 in region
        True
        >>> p2 in region
        True

        2. Check if another region belongs to the region.

        >>> df.Region(p1=(0, 0, 0), p2=(1, 1, 1)) in region
        True
        >>> df.Region(p1=(0, 0, 0), p2=(2, 2, 2)) in region
        False
        >>> # Region is considered to be in itself
        >>> region in region
        True

        """
        if isinstance(other, (numbers.Real, collections.abc.Iterable)):
            atol = np.min(self.edges) * self.tolerance_factor
            rtol = self.tolerance_factor
            return np.all(
                np.logical_and(
                    np.less_equal(self.pmin, other)
                    | np.isclose(self.pmin, other, rtol=rtol, atol=atol),
                    np.greater_equal(self.pmax, other)
                    | np.isclose(self.pmax, other, rtol=rtol, atol=atol),
                )
            )
        if isinstance(other, self.__class__):
            return other.pmin in self and other.pmax in self

        return False

    def __or__(self, other):
        """Old implementation to find facing surfaces.

        :meta private:
        """
        raise AttributeError(
            "This operator has been removed. Please use the `facing_surface` method."
        )

    def facing_surface(self, other):
        """Facing surface.

        Parameters
        ----------
        other : discretisedfield.Region

            Second operand.

        Returns
        -------
        tuple : (ndims,)

            The first element is the axis facing surfaces are perpendicular to.
            If we start moving along that axis (e.g. from minus infinity) the
            first region we are going to enter is the region which is the
            second element of the tuple. When we leave that region, we enter
            the second region, which is the third element of the tuple.

        Examples
        --------
        1. Find facing surfaces.

        >>> import discretisedfield as df
        ...
        >>> p11 = (0, 0, 0)
        >>> p12 = (100e-9, 50e-9, 20e-9)
        >>> region1 = df.Region(p1=p11, p2=p12)
        ...
        >>> p21 = (0, 0, 20e-9)
        >>> p22 = (100e-9, 50e-9, 30e-9)
        >>> region2 = df.Region(p1=p21, p2=p22)
        ...
        >>> res = region1.facing_surface(region2)
        >>> res[0]
        'z'
        >>> res[1] == region1
        True
        >>> res[2] == region2
        True

        """
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot find facing surface for {type(other)}.")

        for i in range(self.ndim):
            if self.pmin[i] >= other.pmax[i]:
                return (self.dims[i], other, self)
            if other.pmin[i] >= self.pmax[i]:
                return (self.dims[i], self, other)
        else:
            msg = "Cannot find facing surface."
            raise ValueError(msg)

    @property
    def multiplier(self):
        """Compute multiplier for the region."""
        return uu.si_max_multiplier(self.edges)

    def scale(self, factor, reference_point=None, inplace=False):
        """Scale the region.

        This method scales the region about its ``center`` point or a
        ``reference_point`` if provided. If ``factor`` is a number the same scaling is
        applied along all dimensions. If ``factor`` is array-like its length must match
        ``region.ndim`` and different factors are applied along the different directions
        (based on their order). A new object is created unless ``inplace=True`` is
        specified.

        Parameters
        ----------
        factor : numbers.Real or array-like of numbers.Real

            Factor to scale the region.

        reference_point : array_like, optional

            The position of the reference point is fixed when scaling the region. If not
            specified the region is scaled about its ``center``.

        inplace : bool, optional

            If True, the Region object is modified in-place. Defaults to False.

        Returns
        -------
        discretisedfield.Region

            Resulting region.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Scale region uniformly.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> res = region.scale(5)
        >>> res.pmin
        array([-20., -20., -20.])
        >>> res.pmax
        array([30., 30., 30.])

        2. Scale the region inplace.

        >>> import discretisedfield as df
        >>> p1 = (-10, -10, -10)
        >>> p2 = (10, 10, 10)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.scale(5, inplace=True)
        Region(...)
        >>> region.pmin
        array([-50., -50., -50.])
        >>> region.pmax
        array([50., 50., 50.])

        3. Scale region with different factors along different directions.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> res = region.scale((2, 3, 4))
        >>> res.pmin
        array([ -5., -10., -15.])
        >>> res.pmax
        array([15., 20., 25.])

        """
        if isinstance(factor, numbers.Real):
            pass
        elif not isinstance(factor, (tuple, list, np.ndarray)):
            raise TypeError(f"Unsupported type {type(factor)} for scale.")
        elif len(factor) != self.ndim:
            raise ValueError(
                f"Wrong length for array-like argument: {len(factor)}; expected length"
                f" {len(self.pmin)}."
            )
        else:
            for elem in factor:
                if not isinstance(elem, numbers.Real):
                    raise TypeError(
                        f"Unsupported element {elem} of type {type(elem)} for scale."
                    )

        if reference_point is None:
            reference_point = self.center
        elif isinstance(reference_point, numbers.Real):
            reference_point = [reference_point]
        elif not isinstance(reference_point, (tuple, list, np.ndarray)):
            raise TypeError(
                "'reference_point' must be a sequence (or a real number for 1d) or"
                f" None (for default behaviour). Not {type(reference_point)=}."
            )

        if len(reference_point) != self.ndim:
            raise ValueError(
                f"The 'reference_point' must contain {self.ndim} elements, not"
                f" {len(reference_point)=}."
            )
        elif any(not isinstance(i, numbers.Real) for i in reference_point):
            raise ValueError("Elements of 'reference_point' must be real numbers.")

        pmin = reference_point - (reference_point - self.pmin) * factor
        pmax = pmin + self.edges * factor

        if inplace:
            self._pmin = pmin
            self._pmax = pmax
            return self
        else:
            return self.__class__(
                p1=pmin,
                p2=pmax,
                dims=self.dims,
                units=self.units,
                tolerance_factor=self.tolerance_factor,
            )

    def translate(self, vector, inplace=False):
        """Translate the region.

        This method translates the region by adding ``vector`` to ``pmin`` and ``pmax``.
        The ``vector`` must have ``Region.ndim`` elements. A new object is created
        unless ``inplace=True`` is specified.

        Parameters
        ----------
        vector : array-like of numbers.Number

            Vector to translate the region.

        inplace : bool, optional

            If True, the Region object is modified in-place. Defaults to False.

        Returns
        -------
        discretisedfield.Region

            Resulting region.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Examples
        --------
        1. Translate the region.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> res = region.translate((2, -2, 5))
        >>> res.pmin
        array([ 2, -2,  5])
        >>> res.pmax
        array([12,  8, 15])

        2. Translate the region inplace.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.translate((2, -2, 5), inplace=True)
        Region(...)
        >>> region.pmin
        array([ 2, -2,  5])
        >>> region.pmax
        array([12,  8, 15])

        """
        # allow scalar values for 1d regions
        if isinstance(vector, numbers.Real):
            vector = [vector]
        if not isinstance(vector, (tuple, list, np.ndarray)):
            raise TypeError(f"Unsupported type {type(vector)} for translate.")
        elif len(vector) != self.ndim:
            raise ValueError(
                f"Wrong length for array-like argument: {len(vector)}; expected length"
                f" {len(self.pmin)}."
            )
        for elem in vector:
            if not isinstance(elem, numbers.Number):
                raise TypeError(
                    f"Unsupported element {elem} of type {type(elem)} for translate."
                )
        if inplace:
            self._pmin = np.add(self.pmin, vector)
            self._pmax = np.add(self.pmax, vector)
            return self
        else:
            return self.__class__(
                p1=np.add(self.pmin, vector),
                p2=np.add(self.pmax, vector),
                dims=self.dims,
                units=self.units,
                tolerance_factor=self.tolerance_factor,
            )

    def rotate90(self, ax1, ax2, k=1, reference_point=None, inplace=False):
        """Rotate region by 90 degrees.

        Rotate the region ``k`` times by 90 degrees in the plane defined by ``ax1`` and
        ``ax2``. The rotation direction is from ``ax1`` to ``ax2``, the two must be
        different.

        Parameters
        ----------
        ax1 : str

            Name of the first dimension.

        ax2 : str

            Name of the second dimension.

        k : int, optional

            Number of 90° rotations, defaults to 1.

        reference_point : array_like, optional

            Point around which the region is rotated. If not provided the region's
            centre point is used.

        inplace : bool, optional

            If ``True``, the rotation is applied in-place. Defaults to ``False``.

        Returns
        -------
        discretisedfield.Region

            The rotated region object. Either a new object or a reference to the
            existing region for ``inplace=True``.

        Examples
        --------

        >>> import discretisedfield as df
        >>> import numpy as np
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 8, 6)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> rotated = region.rotate90('x', 'y')
        >>> rotated.pmin
        array([ 1., -1.,  0.])
        >>> rotated.pmax
        array([9., 9., 6.])

        See also
        --------
        :py:func:`~discretisedfield.Mesh.rotate90`
        :py:func:`~discretisedfield.Field.rotate90`

        """
        if ax1 == ax2:
            raise ValueError(f"{ax1=} and {ax2=} must have different values.")
        if not isinstance(k, int):
            raise TypeError(f"k must be an integer, not {type(k)=}.")

        if reference_point is None:
            reference_point = self.centre
        elif not isinstance(reference_point, (tuple, list, np.ndarray)):
            raise TypeError(
                f"reference_point must be array_like, not {type(reference_point)=}."
            )
        elif len(reference_point) != self.ndim:
            raise ValueError(
                f"reference_point must have length {self.ndim}, not"
                f" {len(reference_point)=}."
            )

        idx1 = self._dim2index(ax1)
        idx2 = self._dim2index(ax2)
        p1 = self.pmin.copy().astype("float")
        p2 = self.pmax.copy().astype("float")

        ref_1 = reference_point[idx1]
        ref_2 = reference_point[idx2]
        p1_inplane = np.array([p1[idx1] - ref_1, p1[idx2] - ref_2])
        p2_inplane = np.array([p2[idx1] - ref_1, p2[idx2] - ref_2])
        rot_matrix = np.array(
            [
                [np.cos(k * np.pi / 2), -np.sin(k * np.pi / 2)],
                [np.sin(k * np.pi / 2), np.cos(k * np.pi / 2)],
            ]
        )
        p1_rot = np.dot(rot_matrix, p1_inplane)
        p2_rot = np.dot(rot_matrix, p2_inplane)
        p1[idx1] = ref_1 + p1_rot[0]
        p1[idx2] = ref_2 + p1_rot[1]
        p2[idx1] = ref_1 + p2_rot[0]
        p2[idx2] = ref_2 + p2_rot[1]

        units = list(self.units)
        if k % 2 == 1:
            units[idx1], units[idx2] = units[idx2], units[idx1]

        if inplace:
            self._pmin = np.minimum(p1, p2)
            self._pmax = np.maximum(p1, p2)
            return self
        else:
            return self.__class__(
                p1=p1,
                p2=p2,
                dims=self.dims,
                units=units,
                tolerance_factor=self.tolerance_factor,
            )

    @property
    def mpl(self):
        r"""``matplotlib`` plot.

        If ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. The colour of lines depicting the region can be specified
        using ``color`` argument, which must be a valid ``matplotlib`` color.
        The plot is saved in PDF-format if ``filename`` is passed.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\,\text{nm}` and :math:`\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        This method is based on ``matplotlib.pyplot.plot``, so any keyword
        arguments accepted by it can be passed (for instance, ``linewidth``,
        ``linestyle``, etc.).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the plot is added. Defaults to ``None`` - axes are
            created internally.

        figsize : (2,) tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        color : int, str, tuple, optional

            A valid ``matplotlib`` color for lines depicting the region.
            Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        box_aspect : str, array_like (3), optional

            Set the aspect-ratio of the plot. If set to `'auto'` the aspect
            ratio is determined from the edge lengths of the region. To set
            different aspect ratios a tuple can be passed. Defaults to
            ``'auto'``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Examples
        --------
        1. Visualising the region using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.mpl()

        """
        return dfp.MplRegion(self)

    @property
    def k3d(self):
        """``k3d`` plot.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The colour of the region can be specified using
        ``color`` argument.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally.

        color : int, optional

            Colour of the region. Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        Examples
        --------
        1. Visualising the region using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.k3d()
        Plot(...)

        """
        return dfp.K3dRegion(self)

    def to_dict(self):
        """Convert region object to dict.

        Convert region object to dict with items pmin, pmax, dims,
        units, and tolerance_factor.

        """
        return {
            "pmin": self.pmin,
            "pmax": self.pmax,
            "dims": self.dims,
            "units": self.units,
            "tolerance_factor": self.tolerance_factor,
        }

    def random_point(self):
        r"""Return a random point in the region."""
        warnings.warn(
            "This method will be removed and should not be used anymore.",
            DeprecationWarning,
        )
        return tuple(np.random.random(self.ndim) * self.edges + self.pmin)
