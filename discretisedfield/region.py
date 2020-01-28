import random
import numpy as np
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu


@ts.typesystem(p1=ts.Vector(size=3, const=True),
               p2=ts.Vector(size=3, const=True))
class Region:
    """A cuboid region.

    A cuboid region spans between two corner points :math:`\\mathbf{p}_{1}` and
    :math:`\\mathbf{p}_{2}`.

    Parameters
    ----------
    p1, p2 : (3,) array_like
        Points between which the cuboid region spans :math:`\\mathbf{p} =
        (p_{x}, p_{y}, p_{z})`.

    Raises
    ------
    ValueError
        If the length of one or more region edges is zero.

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

    >>> import discretisedfield as df
    ...
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

        # Is any edge length of the region equal to zero?
        if np.equal(self.edges, 0).any():
            msg = f'One of the region edges {self.edges} is zero.'
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

        Edge length in any direction :math:`i` is computed from the points
        between which the region spans :math:`l^{i} = |p_{2}^{i} - p_{1}^{i}|`.

        Returns
        -------
        tuple (3,)
             Edge lengths :math:`(l_{x}, l_{y}, l_{z})`.

        Examples
        --------
        1. Getting edge lengths.

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

        It is computed as the middle point between minimum and maximum
        coordinates :math:`\\mathbf{p}_\\text{c} = \\frac{1}{2}
        (\\mathbf{p}_\\text{min} + \\mathbf{p}_\\text{max})`.

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
        """Volume of the region.

        It is computed by multiplying all elements of `self.edges`:
        :math:`V = l_{x} l_{y} l_{z}`.

        Returns
        -------
        float
            Volume of the region

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

        The use of this function is mostly limited for writing tests for
        packages based on `discretisedfield`. This method is not a property.
        Therefore, it is called as `discretisedfield.Region.random_point()`.

        Returns
        -------
        tuple (3,)
            Random point coordinates :math:`(x_\\text{rand}, y_\\text{rand},
            z_\\text{rand})`.

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

           In the example, ellipsis is used instead of an exact tuple because
           the result differs each time `discretisedfield.Region.random_point`
           method is called.

        """
        res = np.add(self.pmin, np.multiply(np.random.random(3), self.edges))
        return dfu.array2tuple(res)

    def __repr__(self):
        """Region representation string.

        This method returns the string that can be copied in another Python
        script so that exactly the same region object can be defined.

        Returns
        -------
        str
           Region representation string.

        Example
        -------
        1. Getting region representation string.

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
        """Determine whether two regions are equal.

        Two regions are considered to be equal if they have the same minimum
        and maximum coordinate points: :math:`p^\\text{max}_{1} =
        p^\\text{max}_{2}` and :math:`p^\\text{min}_{1} = p^\\text{min}_{2}`.

        Parameters
        ----------
        other : discretisedfield.Region
            Region object, which is compared to `self`.

        Returns
        -------
        bool
            `True` if two regions are equal and `False` otherwise.

        Examples
        --------
        1. Check if regions are equal.

        >>> import discretisedfield as df
        ...
        >>> region1 = df.Region(p1=(0, 0, 0), p2=(5, 5, 5))
        >>> region2 = df.Region(p1=(0, 0, 0), p2=(5, 5, 5))
        >>> region3 = df.Region(p1=(1, 1, 1), p2=(5, 5, 5))
        >>> region1 == region2
        True
        >>> region1 == region3
        False

        .. seealso:: :py:func:`~discretisedfield.Region.__ne__`

        """
        if not isinstance(other, self.__class__):
            return False
        elif self.pmin == other.pmin and self.pmax == other.pmax:
            return True
        else:
            return False

    def __ne__(self, other):
        """Determine whether two regions are not equal.

        This method returns `not self == other`. For details, please
        refer to `discretisedfield.Region.__eq__()` method.

        """
        return not self == other

    def __contains__(self, point):
        """Determine whether `point` belongs to the region.

        Point is considered to be in the region if :math:`p^\\text{min}_{i}
        \\le p_{i} \\le p^\\text{max}_{i}` for :math:`i = x, y, z`.

        Parameters
        ----------
        point : (3,) array_like
            The point coordinate :math:`(p_{x}, p_{y}, p_{z})`.

        Returns
        -------
        bool
            `True` if `point` is inside the region and `False` otherwise.

        Example
        -------
        1. Check whether point is inside the region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> (1, 1, 1) in region
        True
        >>> (1, 3, 1) in region
        False

        """
        if np.logical_or(np.less(point, self.pmin),
                         np.greater(point, self.pmax)).any():
            return False
        else:
            return True
