import numbers
import ipywidgets
import numpy as np
import pandas as pd
import ubermagutil.units as uu
import matplotlib.pyplot as plt
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu


@ts.typesystem(dim=ts.Scalar(expected_type=int, positive=True, const=True),
               n=ts.Scalar(expected_type=int, positive=True, const=True))
class Line:
    """Line class.

    This class implements the field sampled on the line. It is based on
    ``pandas.DataFrame``, which is generated from two lists: ``points`` and
    ``values``. ``points`` is a list of length-3 tuples representing the points
    on the line on which the field was sampled. On the other hand, ``values``
    is a list of field values, which can be ``numbers.Real`` for scalar fields
    or ``array_like`` for vector fields. During the initialisation of the
    object, ``r`` column is added to the table and it represents the distance
    of the point from the first point.

    By default the columns where points data is stored are labelled as ``px``,
    ``py``, and ``pz``, storing the x, y, and z components of the point,
    respectively. Similarly, for scalar fields, values are stored in column
    ``v``, whereas for vector fields, data is stored in ``vx``, ``vy``, and
    ``vz``. The default names of columns can be changed by passing
    ``point_columns`` and ``value_columns`` lists. Both lists are composed of
    strings and must have appropriate lengths.

    Data in the form of ``pandas.DataFrame`` can be exposed as ``line.data``.

    Parameters
    ----------
    points : list

        Points at which the field was sampled. It is a list of length-3 tuples.

    values : list

        Values sampled at ``points``.

    point_columns : list

        Point column names. Defaults to None.

    value_columns : list

        Value column names. Defaults to None.

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
    def __init__(self, points, values, point_columns=None, value_columns=None):
        if len(points) != len(values):
            msg = (f'The number of points ({len(points)}) is not the same '
                   f'as the number of values ({len(values)}).')
            raise ValueError(msg)

        # Set the dimension (const descriptor).
        if isinstance(values[0], numbers.Real):
            self.dim = 1
        else:
            self.dim = len(values[0])

        # Set the number of values (const descriptor).
        self.n = len(points)

        points = np.array(points)
        values = np.array(values).reshape((points.shape[0], -1))

        self.data = pd.DataFrame()
        self.data['r'] = np.linalg.norm(points - points[0, :], axis=1)
        for i, column in enumerate(self.point_columns):
            self.data[column] = points[..., i]
        for i, column in zip(range(values.shape[-1]), self.value_columns):
            self.data[column] = values[..., i]

        if point_columns is not None:
            self.point_columns = point_columns

        if value_columns is not None:
            self.value_columns = value_columns

    @property
    def point_columns(self):
        if not hasattr(self, '_point_columns'):
            return [f'p{i}' for i in dfu.axesdict.keys()]
        else:
            return self._point_columns

    @point_columns.setter
    def point_columns(self, val):
        if len(val) != 3:
            msg = (f'Cannot change column names with a '
                   f'list of lenght {len(val)}.')
            raise ValueError(msg)

        self.data = self.data.rename(dict(zip(self.point_columns, val)),
                                     axis=1)
        self._point_columns = val

    @property
    def value_columns(self):
        if not hasattr(self, '_value_columns'):
            return [f'v{i}' for i in list(dfu.axesdict.keys())[:self.dim]]
        else:
            return self._value_columns

    @value_columns.setter
    def value_columns(self, val):
        if len(val) != self.dim:
            msg = (f'Cannot change column names with a '
                   f'list of lenght {len(val)}.')
            raise ValueError(msg)

        self.data = self.data.rename(dict(zip(self.value_columns, val)),
                                     axis=1)
        self._value_columns = val

    @property
    def points(self):
        return self.data[self.point_columns].to_numpy().tolist()

    @property
    def values(self):
        return self.data[self.value_columns].to_numpy().tolist()

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

    def mpl(self, ax=None, figsize=None, yaxis=None, xlim=None,
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

        if yaxis is None:
            yaxis = self.value_columns

        for i in yaxis:
            ax.plot(np.divide(self.data['r'].to_numpy(), multiplier),
                    self.data[i], label=i, **kwargs)

        ax.set_xlabel(f'r ({uu.rsi_prefixes[multiplier]}m)')
        ax.set_ylabel('value')

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
        return ipywidgets.SelectMultiple(options=self.value_columns,
                                         value=self.value_columns,
                                         rows=3,
                                         description='y-axis:',
                                         disabled=False,
                                         **kwargs)
