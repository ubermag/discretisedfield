import collections
import numbers

import numpy as np
import ubermagutil.typesystem as ts

import discretisedfield.plotting as dfp
import discretisedfield.util as dfu

from . import html


@ts.typesystem(
    # units=ts.Name(const=True),  # TODO
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
    p1 / p2 : array_like

        Diagonally-opposite corner points of the region :math:`\mathbf{p}_i =
        (p_x, p_y, p_z)`.

    units : list[str], optional

        Physical units of the region. This is mainly used for labelling plots.
        Defaults to ``m`` in all directions.

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

    def __init__(self, p1, p2, units=None, tolerance_factor=1e-12):
        if len(p1) != len(p2):
            raise ValueError("p1 and p2 must have the same length.")
        self.pmin = np.minimum(p1, p2)
        self.pmax = np.maximum(p1, p2)

        if units is None:
            self.units = ["m"] * len(p1)
        else:
            if len(units) != len(p1):
                raise ValueError(
                    "The number of units is different from the length of p1 and p2."
                )
            self.units = units
        self.tolerance_factor = tolerance_factor

        if not np.all(self.edges):
            msg = f"One of the region's edge lengths is zero: {self.edges=}."
            raise ValueError(msg)

    @property
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
        return self._pmin

    @pmin.setter
    def pmin(self, pmin):
        self._pmin = pmin

    @property
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
        return self._pmax

    @pmax.setter
    def pmax(self, pmax):
        self._pmax = pmax

    @property
    def dim(self):
        """TODO"""  # names for dim and vector dim
        return len(self.pmin)

    @property
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
        return np.abs(self.pmax - self.pmin)

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
        np.ndarray  TODO

            Center point :math:`(p_c^x, p_c^y, p_c^z)`.

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
        (2.5, 7.5, 10.0)

        """
        return 0.5 * np.add(self.pmin, self.pmax)

    @property
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
        # TODO rtol, atol ?
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
                    | np.isclose(self.pmin, other, rtol=tol, atol=tol),
                    np.greater_equal(self.pmax, other)
                    | np.isclose(self.pmax, other, rtol=tol, atol=tol),
                )
            )
        if isinstance(other, self.__class__):
            return other.pmin in self and other.pmax in self

        return False

    # TODO rename
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

    def __mul__(self, other):
        """Binary ``*`` operator.  TODO this summary line does not provide relevant information

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
        # TODO can we omit this check and let numpy test it?
        if not isinstance(other, numbers.Real):
            raise TypeError(
                f"Unsupported operand type(s) for *: {type(self)=} and {type(other)=}."
            )

        return self.__class__(
            p1=np.multiply(self.pmin, other),
            p2=np.multiply(self.pmax, other),
            units=self.units,
            tolerance_factor=self.tolerance_factor,
        )

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Binary ``/`` operator.  TODO: Scale region.

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
        # TODO new implementation ?
        return self * other ** (-1)

    def __add__(self, other):
        """Move region.

        TODO
        """
        # TODO checks (?)
        return self.__class__(
            p1=self.pmin + other,
            p2=self.pmax + other,
            units=self.units,
            tolerance_factor=self.tolerance_factor,
        )

    def __radd__(self, other):
        self.pmin += other
        self.pmax += other
        return self

    def __sub__(self, other):
        """Move region.

        TODO
        """
        # TODO checks (?)
        return self.__class__(
            p1=self.pmin - other,
            p2=self.pmax - other,
            units=self.units,
            tolerance_factor=self.tolerance_factor,
        )

    def __rsub__(self, other):
        """Move subregion.

        TODO
        """
        self.pmin -= other
        self.pmax -= other
        return self

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
        """Convert region object to dict with items p1, p2, unit, tolerance_factor."""
        return {
            "pmin": list(self.pmin),  # TODO list on ndarray ?
            "pmax": list(self.pmax),  # TODO
            "units": self.units,
            "tolerance_factor": self.tolerance_factor,
        }
