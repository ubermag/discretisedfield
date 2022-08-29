import numbers

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ubermagutil.typesystem as ts
import ubermagutil.units as uu

import discretisedfield.util as dfu


@ts.typesystem(
    dim=ts.Scalar(expected_type=int, positive=True, const=True),
    n=ts.Scalar(expected_type=int, positive=True, const=True),
)
class Line:
    """Line class.

    This class represents field sampled on the line. It is based on
    ``pandas.DataFrame``, which is generated from two lists: ``points`` and
    ``values`` of the same length. ``points`` is a list of length-3 tuples
    representing the points on the line on which the field was sampled. On the
    other hand, ``values`` is a list of field values, which are
    ``numbers.Real`` for scalar fields or ``array_like`` for vector fields.
    During the initialisation of the object, ``r`` column is added to
    ``pandas.DataFrame`` and it represents the distance of the point from the
    first point in ``points``.

    By default the columns where points data is stored are labelled as ``px``,
    ``py``, and ``pz``, storing the x, y, and z components of the point,
    respectively. Similarly, for scalar fields, values are stored in column
    ``v``, whereas for vector fields, data is stored in ``vx``, ``vy``, and
    ``vz``. The default names of columns can be changed by passing
    ``point_columns`` and ``value_columns`` lists. Both lists are composed of
    strings and must have appropriate lengths.

    The number of points can be retrieved as ``discretisedfield.Line.n`` and
    the dimension of the value can be retrieved using
    ``discretisedfield.Line.dim``.

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
    >>> line.n  # the number of points
    3
    >>> line.dim
    1

    2. Defining ``Line`` for vector values.

    >>> points = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]
    >>> values = [(0, 0, 1), (0, 0, 2), (0, 0, 3), (0, 0, 4)]  # vector values
    >>> line = df.Line(points=points, values=values)
    >>> line.n  # the number of points
    4
    >>> line.dim
    3

    """

    def __init__(self, points, values, point_columns=None, value_columns=None):
        if len(points) != len(values):
            msg = (
                f"The number of points ({len(points)}) must be the same "
                f"as the number of values ({len(values)})."
            )
            raise ValueError(msg)

        # Set the dimension (const descriptor).
        if isinstance(values[0], numbers.Complex):
            self.dim = 1
        else:
            self.dim = len(values[0])

        # Set the number of values (const descriptor).
        self.n = len(points)

        points = np.array(points)
        values = np.array(values).reshape((points.shape[0], -1))

        self.data = pd.DataFrame()
        self.data["r"] = np.linalg.norm(points - points[0, :], axis=1)
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
        """The names of point columns.

        This method returns a list of strings denoting the names of columns
        storing three coordinates of points. Similarly, by assigning a list of
        strings to this property, the columns can be renamed.

        Parameters
        ----------
        val : list

            Point column names used to rename them.

        Returns
        -------
        list

            List of point column names.

        Raises
        ------
        ValueError

            If a list of inappropriate length is passed.

        Examples
        --------
        1. Getting and setting the column names.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        >>> values = [1, 2, 3]  # scalar values
        >>> line = df.Line(points=points, values=values)
        >>> line.point_columns
        ['px', 'py', 'pz']
        >>> line.point_columns = ['p0', 'p1', 'p2']
        >>> line.data.columns
        Index(['r', 'p0', 'p1', 'p2', 'v'], dtype='object')

        """
        if not hasattr(self, "_point_columns"):
            return [f"p{i}" for i in dfu.axesdict.keys()]
        else:
            return self._point_columns

    @point_columns.setter
    def point_columns(self, val):
        if len(val) != 3:
            msg = f"Cannot change column names with a list of lenght {len(val)}."
            raise ValueError(msg)

        self.data = self.data.rename(dict(zip(self.point_columns, val)), axis=1)
        self._point_columns = val

    @property
    def value_columns(self):
        """The names of value columns.

        This method returns a list of strings denoting the names of columns
        storing values. The length of the list is the same as the dimension of
        the value. Similarly, by assigning a list of strings to this property,
        the columns can be renamed.

        Parameters
        ----------
        val : list

            Value column names used to rename them.

        Returns
        -------
        list

            List of value column names.

        Raises
        ------
        ValueError

            If a list of inappropriate length is passed.

        Examples
        --------
        1. Getting and setting the column names.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        >>> values = [1, 2, 3]  # scalar values
        >>> line = df.Line(points=points, values=values)
        >>> line.value_columns
        ['v']
        >>> line.value_columns = ['my_interesting_value']
        >>> line.data.columns
        Index(['r', 'px', 'py', 'pz', 'my_interesting_value'], dtype='object')

        """
        if not hasattr(self, "_value_columns"):
            if self.dim == 1:
                return ["v"]
            else:
                return [f"v{i}" for i in list(dfu.axesdict.keys())[: self.dim]]
        else:
            return self._value_columns

    @value_columns.setter
    def value_columns(self, val):
        if len(val) != self.dim:
            msg = f"Cannot change column names with a list of lenght {len(val)}."
            raise ValueError(msg)

        self.data = self.data.rename(dict(zip(self.value_columns, val)), axis=1)
        self._value_columns = val

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
        return self.data["r"].iloc[-1]

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
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> repr(line)
        '...

        """
        return repr(self.data)

    def mpl(
        self,
        ax=None,
        figsize=None,
        yaxis=None,
        xlim=None,
        multiplier=None,
        filename=None,
        **kwargs,
    ):
        """Line values plot.

        This method plots the values (scalar or individual components) as a
        function of the distance ``r``. ``mpl`` adds the plot to
        ``matplotlib.axes.Axes`` passed via ``ax`` argument. If ``ax`` is not
        passed, ``matplotlib.axes.Axes`` object is created automatically and
        the size of a figure can be specified using ``figsize``. To choose
        particular value columns to be plotted ``yaxis`` can be passed as a
        list of column names. The range of ``r``values on the horizontal axis
        can be defined by passing a lenth-2 tuple. It is often the case that
        the line length is small (e.g. on a nanoscale) or very large (e.g. in
        units of kilometers). Accordingly, ``multiplier`` can be passed as
        :math:`10^{n}`, where :math:`n` is a multiple of 3  (..., -6, -3, 0, 3,
        6,...). According to that value, the horizontal axis will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, all mesh points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels. If ``multiplier``
        is not passed, the best one is calculated internally. The plot can be
        saved as a PDF when ``filename`` is passed.

        This method plots the mesh using ``matplotlib.pyplot.plot()`` function,
        so any keyword arguments accepted by it can be passed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the field plot is added. Defaults to ``None`` - axes
            are created internally.

        figsize : tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        yaxis : list, optional

            A list of value columns to be plotted.

        xlim : tuple

            A length-2 tuple setting the limits of the horizontal axis.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

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
            ax.plot(
                np.divide(self.data["r"].to_numpy(), multiplier),
                self.data[i],
                label=i,
                **kwargs,
            )

        ax.set_xlabel(f"r ({uu.rsi_prefixes[multiplier]}m)")
        ax.set_ylabel("value")

        ax.grid(True)  # grid is turned off by default for field plots
        ax.legend()

        if xlim is not None:
            plt.xlim(*np.divide(xlim, multiplier))

        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", pad_inches=0)

    def slider(self, multiplier=None, **kwargs):
        """Slider for interactive plotting.

        Based on the values in the ``r`` column,
        ``ipywidgets.SelectionRangeSlider`` is returned for navigating
        interactive plots.

        This method is based on ``ipywidgets.SelectionRangeSlider``, so any
        keyword argument accepted by it can be passed.

        Parameters
        ----------
        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the values will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the slider points will
            be divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units
            will be used in the description. If ``multiplier`` is not passed,
            the optimum one is computed internally. Defaults to ``None``.

        Returns
        -------
        ipywidgets.SelectionRangeSlider

            ``r`` range slider.

        Example
        -------
        1. Get the slider for the horizontal axis.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> line.slider()
        SelectionRangeSlider(...)

        """
        if multiplier is None:
            multiplier = uu.si_multiplier(self.length)

        values = self.data["r"].to_numpy()
        labels = np.around(values / multiplier, decimals=2)
        options = list(zip(labels, values))
        slider_description = f"r ({uu.rsi_prefixes[multiplier]}m):"

        return ipywidgets.SelectionRangeSlider(
            options=options,
            value=(values[0], values[-1]),
            description=slider_description,
            **kwargs,
        )

    def selector(self, **kwargs):
        """Selection list for interactive plotting.

        Based on the value columns, ``ipywidgets.SelectMultiple`` widget is
        returned for selecting the value columns to be plotted.

        This method is based on ``ipywidgets.SelectMultiple``, so any
        keyword argument accepted by it can be passed.

        Returns
        -------
        ipywidgets.SelectMultiple

            Selection list.

        Example
        -------
        1. Get the widget for selecting value columns.

        >>> import discretisedfield as df
        ...
        >>> points = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]
        >>> values = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # vector values
        >>> line = df.Line(points=points, values=values)
        >>> line.selector()
        SelectMultiple(...)

        """
        return ipywidgets.SelectMultiple(
            options=self.value_columns,
            value=self.value_columns,
            rows=3,
            description="y-axis:",
            disabled=False,
            **kwargs,
        )
