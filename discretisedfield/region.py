import collections
import functools
import numbers

import numpy as np
import ubermagutil.typesystem as ts
import ubermagutil.units as uu

import discretisedfield.plotting as dfp
import discretisedfield.util as dfu

from . import html


@ts.typesystem(
    p1=ts.Vector(size=3, const=True),
    p2=ts.Vector(size=3, const=True),
    unit=ts.Name(const=True),
    tolerance_factor=ts.Scalar(expected_type=float, positive=True),
)
class Region:
    r"""Region.

    A cuboid region spans between two corner points :math:`\mathbf{p}_1` and
    :math:`\mathbf{p}_2`. Points ``p1`` and ``p2`` can be any two
    diagonally-opposite points. If any of the edge lengths of the cuboid region
    is zero, ``ValueError`` is raised.

    Parameters
    ----------
    p1 / p2 : (3,) array_like

        Diagonally-opposite corner points of the region :math:`\mathbf{p}_i =
        (p_x, p_y, p_z)`.

    unit : str, optional

        Physical unit of the region. This is mainly used for labelling plots.
        Defaults to ``m``.

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

    def __init__(self, p1, p2, unit="m", tolerance_factor=1e-12):
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.unit = unit
        self.tolerance_factor = tolerance_factor

        if not np.all(self.edges):
            msg = f"One of the region's edge lengths is zero: {self.edges=}."
            raise ValueError(msg)

    @functools.cached_property
    def pmin(self):
        r"""Point with minimum coordinates in the region.

        The :math:`i`-th component of :math:`\mathbf{p}_\text{min}` is computed
        from points :math:`p_1` and :math:`p_2`, between which the region
        spans: :math:`p_\text{min}^i = \text{min}(p_1^i, p_2^i)`.

        Returns
        -------
        tuple (3,)

            Point with minimum coordinates :math:`(p_x^\text{min},
            p_y^\text{min}, p_z^\text{min})`.

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
        (-1.1, 0.0, -0.1)

        .. seealso:: :py:func:`~discretisedfield.Region.pmax`

        """
        return dfu.array2tuple(np.minimum(self.p1, self.p2))

    @functools.cached_property
    def pmax(self):
        r"""Point with maximum coordinates in the region.

        The :math:`i`-th component of :math:`\mathbf{p}_\text{max}` is computed
        from points :math:`p_1` and :math:`p_2`, between which the region
        spans: :math:`p_\text{max}^i = \text{max}(p_1^i, p_2^i)`.

        Returns
        -------
        tuple (3,)

            Point with maximum coordinates :math:`(p_x^\text{max},
            p_y^\text{max}, p_z^\text{max})`.

        Examples
        --------
        1. Getting region's point with maximum coordinates.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.pmin
        (5.0, 2.9, 0.0)

        .. seealso:: :py:func:`~discretisedfield.Region.pmin`

        """
        return dfu.array2tuple(np.maximum(self.p1, self.p2))

    @functools.cached_property
    def edges(self):
        r"""Region's edge lengths.

        Edge length is computed from the points between which the region spans
        :math:`\mathbf{p}_1` and :math:`\mathbf{p}_2`:

        .. math::

            \mathbf{l} = (|p_2^x - p_1^x|, |p_2^y - p_1^y|, |p_2^z - p_1^z|).

        Returns
        -------
        tuple (3,)

             Edge lengths :math:`(l_{x}, l_{y}, l_{z})`.

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
        (5, 15, 20)

        """
        return dfu.array2tuple(np.abs(np.subtract(self.p1, self.p2)))

    @functools.cached_property
    def centre(self):
        r"""Centre point.

        Centre point is computed as the middle point between region's points
        with minimum and maximum coordinates:

        .. math::

            \mathbf{p}_\text{centre} = \frac{1}{2} (\mathbf{p}_\text{min}
            + \mathbf{p}_\text{max}).

        Returns
        -------
        tuple (3,)

            Centre point :math:`(p_c^x, p_c^y, p_c^z)`.

        Examples
        --------
        1. Getting the centre point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 15, 20)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.centre
        (2.5, 7.5, 10.0)

        """
        return dfu.array2tuple(0.5 * np.add(self.pmin, self.pmax))

    @functools.cached_property
    def volume(self):
        r"""Region's volume.

        It is computed by multiplying edge lengths of the region:

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
        100.0

        """
        return dfu.array2tuple(np.prod(self.edges))

    def random_point(self):
        r"""Regions random point.

        The use of this function is mostly for writing tests. This method is
        not a property and it is called as
        ``discretisedfield.Region.random_point()``.

        Returns
        -------
        tuple (3,)

            Random point coordinates :math:`\mathbf{p}_\text{r} =
            (p_x^\text{r}, p_y^\text{r}, p_z^\text{r})`.

        Examples
        --------
        1. Generating a random point in the region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (200e-9, 200e-9, 1e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.random_point()
        (...)

        .. note::

           In this example, ellipsis is used instead of an exact tuple because
           the result differs each time
           ``discretisedfield.Region.random_point()`` method is called.

        """
        return dfu.array2tuple(np.random.random(3) * self.edges + self.pmin)

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
        Region(p1=(0, 0, 0), p2=(2, 2, 1))

        """
        return html.strip_tags(self._repr_html_())

    def _repr_html_(self):
        """Show HTML-based representation in Jupyter notebook."""
        return html.get_template("region").render(region=self)

    def __eq__(self, other):
        r"""Relational operator ``==``.

        Two regions are considered to be equal if they have the same minimum
        and maximum coordinate points: :math:`\mathbf{p}^\text{max}_1 =
        \mathbf{p}^\text{max}_2` and :math:`\mathbf{p}^\text{min}_1 =
        \mathbf{p}^\text{min}_2`.

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
            return np.allclose(self.pmin, other.pmin, atol=0) and np.allclose(
                self.pmax, other.pmax, atol=0
            )

        return False

    def __contains__(self, other):
        """Determine if a point or another region belong to the region.

        Point is considered to be in the region if

        .. math::

            p^\\text{min}_{i} \\le p_{i} \\le p^\\text{max}_{i}, \\text{for}\\,
            i = x, y, z.

        Similarly, if the second operand is ``discretisedfield.Region`` object,
        it is considered to be in the region if both its ``pmin`` and ``pmax``
        belong to the region.

        Parameters
        ----------
        other : (3,) array_like or discretisedfield.Region

            The point coordinate :math:`(p_{x}, p_{y}, p_{z})` or a region
            object.

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
        if isinstance(other, collections.abc.Iterable):
            tol = np.min(self.edges) * self.tolerance_factor
            return np.all(
                np.logical_and(
                    np.less_equal(self.pmin, other)
                    | np.allclose(self.pmin, other, rtol=tol, atol=tol),
                    np.greater_equal(self.pmax, other)
                    | np.allclose(self.pmax, other, rtol=tol, atol=tol),
                )
            )
        if isinstance(other, self.__class__):
            return other.pmin in self and other.pmax in self

        return False

    def __or__(self, other):
        """Facing surface.

        Parameters
        ----------
        other : discretisedfield.Region

            Second operand.

        Returns
        -------
        tuple : (3,)

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
        >>> res = region1 | region2
        >>> res[0]
        'z'
        >>> res[1] == region1
        True
        >>> res[2] == region2
        True

        """
        if not isinstance(other, self.__class__):
            msg = (
                f"Unsupported operand type(s) for |: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        for i in range(3):
            if self.pmin[i] >= other.pmax[i]:
                return (dfu.raxesdict[i], other, self)
            if other.pmin[i] >= self.pmax[i]:
                return (dfu.raxesdict[i], self, other)
        else:
            msg = "Cannot find facing surface."
            raise ValueError(msg)

    @functools.cached_property
    def multiplier(self):
        """Compute multiplier for the region."""
        return uu.si_max_multiplier(self.edges)

    def __mul__(self, other):
        """Binary ``*`` operator.

        It can be applied only between ``discretisedfield.Region`` and
        ``numbers.Real``. The result is a region whose ``pmax`` and ``pmin``
        are multiplied by ``other``.

        Parameters
        ----------
        other : numbers.Real

            Second operand.

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
        1. Multiply region with a scalar.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> res = region * 5
        ...
        >>> res.pmin
        (0, 0, 0)
        >>> res.pmax
        (50, 50, 50)

        .. seealso:: :py:func:`~discretisedfield.Region.__truediv__`

        """
        if not isinstance(other, numbers.Real):
            msg = (
                f"Unsupported operand type(s) for *: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        return self.__class__(
            p1=np.multiply(self.pmin, other),
            p2=np.multiply(self.pmax, other),
            unit=self.unit,
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Binary ``/`` operator.

        It can be applied only between ``discretisedfield.Region`` and
        ``numbers.Real``. The result is a region whose ``pmax`` and ``pmin``
        are divided by ``other``.

        Parameters
        ----------
        other : numbers.Real

            Second operand.

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
        1. Divide region with a scalar.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> res = region / 2
        ...
        >>> res.pmin
        (0.0, 0.0, 0.0)
        >>> res.pmax
        (5.0, 5.0, 5.0)

        .. seealso:: :py:func:`~discretisedfield.Region.__mul__`

        """
        return self * other ** (-1)

    @property
    def mpl(self):
        return dfp.MplRegion(self)

    @property
    def k3d(self):
        return dfp.K3dRegion(self)
