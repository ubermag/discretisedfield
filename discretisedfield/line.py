import ipywidgets
import numpy as np
import pandas as pd
import ubermagutil.units as uu
import matplotlib.pyplot as plt
import discretisedfield.util as dfu


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

    """
    def __init__(self, points, values):
        if len(points) != len(values):
            msg = (f'The number of points ({len(points)}) is not the same '
                   f'as the number of values ({len(values)}).')
            raise ValueError(msg)

        points = np.array(points)
        values = np.array(values).reshape((points.shape[0], -1))

        # Calculate distance from the first point.
        r = np.linalg.norm(points - points[0, :], axis=1)

        self.data = pd.DataFrame({'r': r})

        for i, component in enumerate(dfu.axesdict.keys()):
            self.data['p' + component] = points[..., i]

        for i, component in zip(range(values.shape[-1]), dfu.axesdict.keys()):
            self.data['v' + component] = values[..., i]

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
        return self.data.shape[0]

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
        return self.data[[i for i in self.data if i.startswith('v')]].shape[-1]

    @property
    def points(self):
        points_columns = [i for i in self.data if i.startswith('p')]
        return self.data[points_columns].to_numpy().tolist()

    @property
    def values(self):
        points_columns = [i for i in self.data if i.startswith('v')]
        return np.squeeze(self.data[points_columns]).to_numpy().tolist()

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
        return self.data['r'].iloc[-1]

    def __repr__(self):
        return repr(self.data)

    def mpl(self, ax=None, figsize=None, y=None, xlim=None,
            multiplier=None, filename=None, **kwargs):
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
        is not passed, the optimum one is computed internally. If ``filename``
        is passed, figure is saved.

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

        filename: str

            Filename to which the plot is saved.

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
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        if multiplier is None:
            multiplier = uu.si_multiplier(self.length)

        if y is None:
            y = [i for i in self.data if i.startswith('v')]

        for i in y:
            ax.plot(np.divide(self.data['r'].to_numpy(), multiplier),
                    self.data[i],
                    label=i,
                    **kwargs)

        ax.set_xlabel(f'r ({uu.rsi_prefixes[multiplier]}m)')
        ax.set_ylabel('v')

        ax.grid(True)  # grid is turned off by default for field plots
        ax.legend()

        if xlim is not None:
            plt.xlim(*np.divide(xlim, multiplier))

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def slider(self, multiplier=None, **kwargs):
        if multiplier is None:
            multiplier = uu.si_multiplier(self.length)

        values = self.data['r'].to_numpy()
        labels = np.around(values/multiplier, decimals=2)
        options = list(zip(labels, values))
        slider_description = f'r ({uu.rsi_prefixes[multiplier]}m):'

        return ipywidgets.SelectionRangeSlider(options=options,
                                               value=(values[0], values[-1]),
                                               description=slider_description,
                                               **kwargs)

    def multipleselector(self, **kwargs):
        options = [i for i in self.data if i.startswith('v')]
        return ipywidgets.SelectMultiple(options=options,
                                         value=options,
                                         rows=3,
                                         description='y-axis:',
                                         disabled=False,
                                         **kwargs)
