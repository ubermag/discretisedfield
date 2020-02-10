import numbers
import numpy as np
import seaborn as sns
import ubermagutil.units as uu
import matplotlib.pyplot as plt


class Line:
    """Field sampled on the line.

    Sampling field on the line is getting the coordinates of points on which
    the field is sampled as well as the values of the field at those points.
    This class provides some convenience functions for the analysis of data on
    the line.

    Parameters
    ----------
    points : list

        Points at which the field is samples. It is a list of length-3 tuples.

    values : list

        Values sampled at ``points``.

    Raises
    ------
    ValueError

        If the numbers of points is not the same as the number of values.

    Example
    -------
    1. Defining ``Line`` object, which contains scalar values.

    >>> import discretisedfield as df
    ...
    >>> points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    >>> values = [1, 2, 3]  # scalar values
    >>> line = df.Line(points=points, values=values)
    >>> line
    Line(...)

    """
    def __init__(self, points, values):
        if len(points) != len(values):
            msg = (f'Cannot define Line, because the number of points '
                   f'({len(points)}) is not the same as the number of '
                   f'values ({values}).')
            raise ValueError(msg)

        self.dictionary = dict(zip(points, values))

    @property
    def points(self):
        """Points on the line.

        Returns
        -------
        list

            List of point coordinates at which the values were sampled.

        Example
        -------
        1. Getting the points from the line.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (1, 1, 1)]
        >>> values = [1, 2]  # scalar values
        >>> line = df.Line(points=points, values=values)
        >>> line.points
        [(0, 0, 0), (1, 1, 1)]

        """
        return list(self.dictionary.keys())

    @property
    def values(self):
        """Values on the line.

        Returns
        -------
        list

            List of values on the line.

        Example
        -------
        1. Getting the values from the line.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (1, 1, 1)]
        >>> values = [(1, 0, 0), (0, 1, 0)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> line.values
        [(1, 0, 0), (0, 1, 0)]

        """
        return list(self.dictionary.values())

    @property
    def length(self):
        """Line length.

        Length of the line is defined as the distance between the first and the
        last point in ``points``.

        Returns
        -------
        float

            Line length.

        Example
        -------
        1. Getting the length of the line.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> line.length
        4.0

        """
        r_vector = np.subtract(self.points[-1], self.points[0])
        return np.linalg.norm(r_vector)

    @property
    def n(self):
        """The number of points on the line.

        Returns
        -------
        int

            Number of points on the line.

        Example
        -------
        1. Getting the number of points.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> line.n
        3

        """
        return len(self.points)

    @property
    def dim(self):
        """Dimension of the value.

        This method extracts the dimension of the value. For instance for
        scalar values ``dim=1``, whereas for vector fields ``dim=3``.

        Returns
        -------
        int

            Value dimension.

        Example
        -------
        1. Getting the dimension of the value.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> line.dim
        3

        """
        if isinstance(self.values[0], numbers.Real):
            return 1
        else:
            return len(self.values[0])

    def __call__(self, point):
        """Value at ``point``.

        Returns
        -------
        tuple, numbers.Real

           Value at ``point``.

        Example
        -------
        1. Sampling the value on the line.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        >>> line = df.Line(points=points, values=values)
        >>> p = (2, 0, 0)
        >>> line(p)
        (0, 1, 0)

        """
        return self.dictionary[point]

    def __repr__(self):
        """Line representation string.

        Returns
        -------
        str

           Line representation string.

        Example
        -------
        1. Getting line representation string.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> repr(line)
        'Line(points=..., values=...)'

        """
        return 'Line(points=..., values=...)'

    def mpl(self, ax=None, figsize=None, multiplier=None, **kwargs):
        """Plots the values on the line.

        If ``ax`` is not passed, axes will be created automaticaly. In that
        case, the figure size can be changed using ``figsize``. It is often the
        case that the region size is small (e.g. on a nanoscale) or very large
        (e.g. in units of kilometers). Accordingly, ``multiplier`` can be
        passed as :math:`10^{n}`, where :math:`n` is a multiple of 3 (..., -6,
        -3, 0, 3, 6,...). According to that value, the axes will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, all points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels. If ``multiplier``
        is not passed, the optimum one is computed internally.

        This method plots the mesh using ``matplotlib.pyplot.plot()`` function,
        so any keyword arguments accepted by it can be passed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which line plot should be added. Defaults to ``None`` - new
            axes will be created in figure with size defined as ``figsize``.

        figsize : (2,) tuple, optional

            Length-2 tuple passed to ``matplotlib.pyplot.figure()`` to create a
            figure and axes if ``ax=None``. Defaults to ``None``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the line points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        Examples
        --------
        1. Visualising the values on the line using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> line.mpl()

        """
        sns.set()
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        if multiplier is None:
            multiplier = uu.si_multiplier(self.length)

        x_array = np.linspace(0, self.length, self.n)
        x_array = np.divide(x_array, multiplier)

        if self.dim == 1:
            with sns.axes_style('darkgrid'):
                ax.plot(x_array, self.values, **kwargs)
        else:
            vals = list(zip(*self.values))
            for val, label in zip(vals, 'xyz'):
                with sns.axes_style('darkgrid'):
                    ax.plot(x_array, val, label=label, **kwargs)
            ax.legend()

        ax.set_xlabel(f'r ({uu.rsi_prefixes[multiplier]}m)')
        ax.set_ylabel('value')
