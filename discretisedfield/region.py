import k3d
import random
import numpy as np
import matplotlib.pyplot as plt
import ubermagutil.units as uu
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu


@ts.typesystem(p1=ts.Vector(size=3, const=True),
               p2=ts.Vector(size=3, const=True))
class Region:
    """A cuboid region.

    A cuboid region spans between two corner points :math:`\\mathbf{p}_{1}` and
    :math:`\\mathbf{p}_{2}`. Points ``p1`` and ``p2`` can be any two diagonally
    opposite points. If any of the edge lengths of the cuboid region is zero,
    ``ValueError`` is raised.

    Parameters
    ----------
    p1 / p2 : (3,) array_like

        Diagonnaly opposite corner points :math:`\\mathbf{p}_{i} = (p_{x},
        p_{y}, p_{z})`.

    Raises
    ------
    ValueError

        If any region's edge length is zero.

    Examples
    --------
    1. Defining a nano-sized region.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> region = df.Region(p1=p1, p2=p2)
    >>> region
    Region(...)

    2. An attempt to define a region, where one of the edge lengths is zero.

    >>> # The edge length in the z-direction is zero.
    >>> p1 = (-25, 3, 1)
    >>> p2 = (25, 6, 1)
    >>> region = df.Region(p1=p1, p2=p2)
    Traceback (most recent call last):
        ...
    ValueError: ...

    """
    def __init__(self, p1, p2):
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)

        if np.equal(self.edges, 0).any():
            msg = f'One of the region edge lengths is zero: {self.edges=}.'
            raise ValueError(msg)

    @property
    def pmin(self):
        """Point with minimum coordinates in the region.

        The :math:`i`-th component of :math:`\\mathbf{p}_\\text{min}` is
        computed from points :math:`p_{1}` and :math:`p_{2}` between which the
        cuboid region spans: :math:`p_\\text{min}^{i} = \\text{min}(p_{1}^{i},
        p_{2}^{i})`.

        Returns
        -------
        tuple (3,)

            Point with minimum coordinates :math:`(p_{x}^\\text{min},
            p_{y}^\\text{min}, p_{z}^\\text{min})`.

        Examples
        --------
        1. Getting the minimum coordinate point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.pmin
        (-1.1, 0.0, -0.1)

        .. seealso:: :py:func:`~discretisedfield.Region.pmax`

        """
        return dfu.array2tuple(np.minimum(self.p1, self.p2))

    @property
    def pmax(self):
        """Point with maximum coordinates in the region.

        The :math:`i`-th component of :math:`\\mathbf{p}_\\text{max}` is
        computed from points :math:`p_{1}` and :math:`p_{2}` between which the
        cuboid region spans: :math:`p_\\text{max}^{i} = \\text{max}(p_{1}^{i},
        p_{2}^{i})`.

        Returns
        -------
        tuple (3,)

            Point with maximum coordinates :math:`(p_{x}^\\text{max},
            p_{y}^\\text{max}, p_{z}^\\text{max})`.

        Examples
        --------
        1. Getting the maximum coordinate point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.pmax
        (5.0, 2.9, 0.0)

        .. seealso:: :py:func:`~discretisedfield.Region.pmin`

        """
        return dfu.array2tuple(np.maximum(self.p1, self.p2))

    @property
    def edges(self):
        """Edge lengths of the region.

        Edge length is computed from the points between which the region spans
        :math:`\\mathbf{p}_{1}` and :math:`\\mathbf{p}_{2}`:

        .. math::

            \\mathbf{l} = (|p_{2}^{x} - p_{1}^{x}|, |p_{2}^{y} - p_{1}^{y}|,
            |p_{2}^{z} - p_{1}^{z}|).

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
        >>> region.edges
        (5, 15, 20)

        """
        return dfu.array2tuple(np.abs(np.subtract(self.p1, self.p2)))

    @property
    def centre(self):
        """Centre point.

        It is computed as the middle point between minimum and maximum point
        coordinates:

        .. math::

            \\mathbf{p}_\\text{centre} = \\frac{1}{2} (\\mathbf{p}_\\text{min}
            + \\mathbf{p}_\\text{max}).

        Returns
        -------
        tuple (3,)

            Centre point :math:`(p_{c}^{x}, p_{c}^{y}, p_{c}^{z})`.

        Examples
        --------
        1. Getting the centre point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 15, 20)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.centre
        (2.5, 7.5, 10.0)

        """
        return dfu.array2tuple(np.multiply(np.add(self.pmin, self.pmax), 0.5))

    @property
    def volume(self):
        """Region volume.

        It is computed by multiplying edge lengths of the region:

        .. math::

            V = l_{x} l_{y} l_{z}.

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
        >>> region.volume
        100.0

        """
        return float(np.prod(self.edges))

    def random_point(self):
        """Generate a random point in the region.

        The use of this function is mostly for writing tests. This method is
        not a property and it is called as
        ``discretisedfield.Region.random_point()``.

        Returns
        -------
        tuple (3,)

            Random point coordinates :math:`\\mathbf{p}_\\text{r} =
            (p_{x}^\\text{r}, p_{y}^\\text{r}, p_{z}^\\text{r})`.

        Examples
        --------
        1. Generating a random point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (200e-9, 200e-9, 1e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> region.random_point()
        (...)

        .. note::

           In this example, ellipsis is used instead of an exact tuple because
           the result differs each time
           ``discretisedfield.Region.random_point`` method is called.

        """
        res = np.add(self.pmin, np.multiply(np.random.random(3), self.edges))
        return dfu.array2tuple(res)

    def __repr__(self):
        """Representation string.

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
        >>> repr(region)
        'Region(p1=(0, 0, 0), p2=(2, 2, 1))'

        """
        return f'Region(p1={self.pmin}, p2={self.pmax})'

    def __eq__(self, other):
        """Relational operator ``==``.

        Two regions are considered to be equal if they have the same minimum
        and maximum coordinate points: :math:`\\mathbf{p}^\\text{max}_{1} =
        \\mathbf{p}^\\text{max}_{2}` and :math:`\\mathbf{p}^\\text{min}_{1} =
        \\mathbf{p}^\\text{min}_{2}`.

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
        1. Check if regions are equal.

        >>> import discretisedfield as df
        ...
        >>> region1 = df.Region(p1=(0, 0, 0), p2=(5, 5, 5))
        >>> region2 = df.Region(p1=(0.0, 0, 0), p2=(5.0, 5, 5))
        >>> region3 = df.Region(p1=(1, 1, 1), p2=(5, 5, 5))
        >>> region1 == region2
        True
        >>> region1 != region2
        False
        >>> region1 == region3
        False
        >>> region1 != region3
        True

        """
        atol = 1e-15
        rtol = 1e-5
        if not isinstance(other, self.__class__):
            return False
        elif (np.allclose(self.pmin, other.pmin, atol=atol, rtol=rtol) and
              np.allclose(self.pmax, other.pmax, atol=atol, rtol=rtol)):
            return True
        else:
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
        if isinstance(other, self.__class__):
            return other.pmin in self and other.pmax in self
        elif np.logical_or(np.less(other, self.pmin),
                           np.greater(other, self.pmax)).any():
            return False

        return True

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
            msg = (f'Unsupported operand type(s) for |: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

        for i in range(3):
            if self.pmin[i] >= other.pmax[i]:
                return (dfu.raxesdict[i], other, self)
            if other.pmin[i] >= self.pmax[i]:
                return (dfu.raxesdict[i], self, other)
        else:
            msg = 'Cannot find facing surfaces'
            raise ValueError(msg)

    def mpl(self, *, ax=None, figsize=None, color=dfu.cp_hex[0],
            multiplier=None, filename=None, **kwargs):
        """``matplotlib`` plot.

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
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
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
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.edges)

        unit = f'({uu.rsi_prefixes[multiplier]}m)'

        pmin = np.divide(self.pmin, multiplier)
        pmax = np.divide(self.pmax, multiplier)

        dfu.plot_box(ax=ax, pmin=pmin, pmax=pmax, color=color, **kwargs)

        ax.set(xlabel=f'x {unit}', ylabel=f'y {unit}', zlabel=f'z {unit}')

        # Overwrite default plotting parameters.
        ax.set_facecolor('#ffffff')  # white face color
        ax.tick_params(axis='both', which='major', pad=0)  # no pad for ticks

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def k3d(self, *, plot=None, color=dfu.cp_int[0], multiplier=None,
            **kwargs):
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
        if plot is None:
            plot = k3d.plot()
            plot.display()

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.edges)

        unit = f'({uu.rsi_prefixes[multiplier]}m)'

        plot_array = np.ones((1, 1, 1)).astype(np.uint8)  # avoid k3d warning

        bounds = [i for sublist in
                  zip(np.divide(self.pmin, multiplier),
                      np.divide(self.pmax, multiplier))
                  for i in sublist]

        plot += k3d.voxels(plot_array, color_map=color, bounds=bounds,
                           outlines=False, **kwargs)

        plot.axes = [i + r'\,\text{{{}}}'.format(unit)
                     for i in dfu.axesdict.keys()]
