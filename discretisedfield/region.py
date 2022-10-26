import collections
import numbers

import numpy as np
import ubermagutil.units as uu

import discretisedfield.plotting as dfp
import discretisedfield.util as dfu

from . import html


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

    def __init__(
        self,
        p1,
        p2,
        dims=None,
        units=None,
        tolerance_factor=1e-12,
    ):

        if not isinstance(p1, (tuple, list, np.ndarray)) or not isinstance(
            p2, (tuple, list, np.ndarray)
        ):
            raise TypeError(
                "p1 and p2 must be of type tuple, list, or np.ndarray. Not"
                f" p1={type(p1)} and p2={type(p2)}."
            )

        if not all(isinstance(i, numbers.Number) for i in p1):
            msg = "p1 can only contain elements of type numbers.Number."
            raise TypeError(msg)

        if not all(isinstance(i, numbers.Number) for i in p2):
            msg = "p2 can only contain elements of type numbers.Number."
            raise TypeError(msg)

        ndim = len(p1)
        if not (len(p1) == len(p2)):
            raise ValueError(
                "The length of p1 and p2 must be the same. Not"
                f" len(p1)={len(p1)} and len(p2)={len(p2)}."
            )

        # TODO: Remove for ndim != 3 functionality.
        if ndim != 3:
            raise NotImplementedError("Only 3D regions are supported at the moment.")

        if dims is None:
            if ndim == 3:
                dims = ("x", "y", "z")
            else:
                dims = [f"x{i}" for i in range(ndim)]

        if units is None:
            units = ["m" for i in range(ndim)]

        self._pmin = np.minimum(p1, p2)
        self._pmax = np.maximum(p1, p2)
        self.dims = dims
        self.units = units
        self.tolerance_factor = tolerance_factor

        if not np.all(self.edges):
            raise ValueError(
                f"One of the region's edge lengths is zero: {self.edges=}."
            )

    @property
    def pmin(self):
        r"""Point with minimum coordinates in the region.

        The :math:`i`-th component of :math:`\mathbf{p}_\text{min}` is computed
        from points :math:`p_1` and :math:`p_2`, between which the region
        spans: :math:`p_\text{min}^i = \text{min}(p_1^i, p_2^i)`.

        Returns
        -------
        numpy.ndarray (3,)

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

    @property
    def pmax(self):
        r"""Point with maximum coordinates in the region.

        The :math:`i`-th component of :math:`\mathbf{p}_\text{max}` is computed
        from points :math:`p_1` and :math:`p_2`, between which the region
        spans: :math:`p_\text{max}^i = \text{max}(p_1^i, p_2^i)`.

        Returns
        -------
        numpy.ndarray (3,)

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

    @property
    def ndim(self):
        r"""Number of dimentions.

        Calculates the number of dimensions of the region.

        Returns
        -------
        int

            Number of dimensions in the region.

        Examples
        --------
        1. Getting region's point with minimum coordinates.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        ...
        >>> region.ndim
        3

        .. seealso:: :py:func:`~discretisedfield.Region.pmax`

        """
        return len(self.pmin)

    @property
    def dims(self):
        r"""Names of the region's dimensions.

        Returns
        -------
        tuple, list of str

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

        """
        return self._dims

    @dims.setter
    def dims(self, dims):
        if not isinstance(dims, (tuple, list)):
            raise TypeError(f"dims must be of type tuple or list. Not {type(dims)}.")

        if not all(isinstance(i, str) for i in dims):
            msg = "dims can only contain elements of type str."
            raise TypeError(msg)

        if len(dims) != self.ndim:
            raise ValueError(
                "dims must have the same length as p1 and p2."
                f" Not len(dims)={len(dims)} and ndim={self.ndim}."
            )

        self._dims = dims

    @property
    def units(self):
        r"""Units of the region's dimensions.

        Returns
        -------
        tuple, list of str

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
        if not isinstance(units, (tuple, list)):
            raise TypeError(f"units must be of type tuple or list. Not {type(units)}.")

        if not all(isinstance(i, str) for i in units):
            msg = "units can only contain elements of type str."
            raise TypeError(msg)

        if len(units) != self.ndim:
            raise ValueError(
                "units must have the same length as p1 and p2."
                f" Not len(units)={len(units)} and ndim={self.ndim}."
            )

        self._units = units

    @property
    def edges(self):
        r"""Region's edge lengths.

        Edge length is computed from the points between which the region spans
        :math:`\mathbf{p}_1` and :math:`\mathbf{p}_2`:

        .. math::

            \mathbf{l} = (|p_2^x - p_1^x|, |p_2^y - p_1^y|, |p_2^z - p_1^z|).

        Returns
        -------
        numpy.ndarray (3,)

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
        return np.abs(np.subtract(self.pmin, self.pmax))

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
        numpy.ndarray (3,)

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

    def random_point(self):
        r"""Regions random point.

        The use of this function is mostly for writing tests. This method is
        not a property and it is called as
        ``discretisedfield.Region.random_point()``.

        Returns
        -------
        numpy.ndarray (3,)

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
        return np.random.random(3) * self.edges + self.pmin

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
        and maximum coordinate points, the same units, and the same dimention names.

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

    @property
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
        if not isinstance(other, numbers.Number):
            msg = (
                f"Unsupported operand type(s) for *: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        return self.__class__(
            p1=np.multiply(self.pmin, other),
            p2=np.multiply(self.pmax, other),
            dims=self.dims,
            units=self.units,
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

    def allclose(
        self,
        other,
        atol=None,
        rtol=None,
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
        >>> region1.isclose(region2)
        True
        >>> region1.isclose(region2)
        False
        >>> region1.isclose(region3)
        False
        >>> region1.isclose(region3)
        True

        """
        if isinstance(other, self.__class__):
            if atol is None:
                atol = np.min(self.edges) * self.tolerance_factor
            elif not isinstance(atol, numbers.Number):
                raise TypeError(f"{type(atol)=} is not a number.")

            if rtol is None:
                rtol = np.min(self.edges) * self.tolerance_factor
            elif not isinstance(rtol, numbers.Number):
                raise TypeError(f"{type(rtol)=} is not a number.")

            return np.allclose(
                self.pmin, other.pmin, atol=atol, rtol=rtol
            ) and np.allclose(self.pmax, other.pmax, atol=atol, rtol=rtol)

        return False

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
        """Convert region object to dict

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
