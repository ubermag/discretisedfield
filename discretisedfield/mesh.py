import k3d
import itertools
import matplotlib
import ipywidgets
import numpy as np
import seaborn as sns
import discretisedfield as df
import matplotlib.pyplot as plt
import ubermagutil.units as uu
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu
from mpl_toolkits.mplot3d import Axes3D


@ts.typesystem(region=ts.Typed(expected_type=df.Region),
               cell=ts.Vector(size=3, positive=True, const=True),
               n=ts.Vector(size=3, component_type=int, unsigned=True,
                           const=True),
               pbc=ts.Subset(sample_set='xyz', unpack=True),
               subregions=ts.Dictionary(
                   key_descriptor=ts.Name(),
                   value_descriptor=ts.Typed(expected_type=df.Region),
                   allow_empty=True))
class Mesh:
    """Finite difference cubic mesh.

    Mesh discretises cubic ``discretisedfield.Region``, passed as ``region``,
    using a regular finite difference mesh. Since cubic region spans between
    two points :math:`\\mathbf{p}_{1}` and :math:`\\mathbf{p}_{2}`, these
    points can be passed as ``p1`` and ``p2``, instead of passing
    ``discretisedfield.Region`` object. In this case
    ``discretisedfield.Region`` is created internally, based on points ``p1``
    and ``p2``. Either ``region`` or ``p1`` and ``p2`` must be passed, not
    both. The region is discretised using a finite difference cell, whose
    dimensions are defined with ``cell``. Alternatively, the domain can be
    discretised by passing the number of discretisation cells ``n`` in all
    three dimensions. Either ``cell`` or ``n`` should be passed to discretise
    the region, not both. Periodic boundary conditions can be specified by
    passing ``pbc`` argument, which is an iterable containing one or more
    elements from ``{'x', 'y', 'z'}``. If it is necessary to define subregions
    in the mesh, a dictionary can be passed as ``subregions``. More precisely,
    dictionary keys are strings (as valid Python variable names), whereas
    values are ``discretisedfield.Region`` objects.

    In order to properly define a mesh, mesh region must be an aggregate of
    discretisation cells.

    Parameters
    ----------
    region : discretisedfield.Region, optional

        Cubic region to be discretised on a regular mesh. Either ``region`` or
        ``p1`` and ``p2`` should be defined, not both.

    p1/p2 : (3,) array_like, optional

        Points between which the mesh region spans :math:`\\mathbf{p} = (p_{x},
        p_{y}, p_{z})`. Either ``region`` or ``p1`` and ``p2`` should be
        defined, not both.

    cell : (3,) array_like, optional

        Discretisation cell size :math:`(d_{x}, d_{y}, d_{z})`. Either ``cell``
        or ``n`` should be defined, not both.

    n : (3,) array_like, optional

        The number of discretisation cells :math:`(n_{x}, n_{y}, n_{z})`.
        Either ``cell`` or ``n`` should be defined, not both.

    pbc : iterable, optional

        Periodic boundary conditions in x, y, or z directions. Its value is an
        iterable consisting of one or more characters ``'x'``, ``'y'``, or
        ``'z'``, denoting the direction(s) along which the mesh is periodic.

    subregions : dict, optional

        A dictionary defining subregions in the mesh. The keys of the
        dictionary are the region names (str), whereas the values are
        ``discretisedfield.Region`` objects.

    Raises
    ------
    ValueError

        If mesh domain is not an aggregate of discretisation cells.
        Alternatively, if both ``region`` as well as ``p1`` and ``p2`` or both
        ``cell`` and ``n`` are passed.

    Examples
    --------
    1. Defining a nano-sized thin film mesh by passing ``region`` and ``cell``
    parameters.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> cell = (1e-9, 1e-9, 0.1e-9)
    >>> region = df.Region(p1=p1, p2=p2)
    >>> mesh = df.Mesh(region=region, cell=cell)
    >>> mesh
    Mesh(...)

    2. Defining a nano-sized thin film mesh by passing ``p1``, ``p2`` and ``n``
    parameters.

    >>> n = (100, 50, 5)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> mesh
    Mesh(...)

    3. Defining a mesh with periodic boundary conditions in x and y directions.

    >>> pbc = 'xy'
    >>> region = df.Region(p1=p1, p2=p2)
    >>> mesh = df.Mesh(region=region, n=n, pbc=pbc)
    >>> mesh
    Mesh(...)

    4. Defining a mesh with two subregions.

    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
    ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
    >>> mesh
    Mesh(...)

    5. An attempt to define a mesh, whose region is not an aggregate of
    discretisation cells in the :math:`z` direction.

    >>> p1 = (-25, 3, 0)
    >>> p2 = (25, 6, 1)
    >>> cell = (5, 3, 0.4)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    Traceback (most recent call last):
        ...
    ValueError: ...

    """
    def __init__(self, region=None, p1=None, p2=None, n=None, cell=None,
                 pbc=set(), subregions={}):
        if region is not None and p1 is None and p2 is None:
            self.region = region
        elif region is None and p1 is not None and p2 is not None:
            self.region = df.Region(p1=p1, p2=p2)
        else:
            msg = ('Either region or p1 and p2 must be passed, not both.')
            raise ValueError(msg)

        if cell is not None and n is None:
            self.cell = tuple(cell)
            n = np.divide(self.region.edges, self.cell).round().astype(int)
            self.n = dfu.array2tuple(n)
        elif n is not None and cell is None:
            self.n = tuple(n)
            cell = np.divide(self.region.edges, self.n).astype(float)
            self.cell = dfu.array2tuple(cell)
        else:
            msg = ('Either n or cell must be passed, not both.')
            raise ValueError(msg)

        # Check if the mesh region is an aggregate of the discretisation cell.
        tol = 1e-12  # picometre tolerance
        rem = np.remainder(self.region.edges, self.cell)
        if np.logical_and(np.greater(rem, tol),
                          np.less(rem, np.subtract(self.cell, tol))).any():
            msg = 'Mesh region is not an aggregate of the discretisation cell.'
            raise ValueError(msg)

        self.pbc = pbc
        self.subregions = subregions

    def __len__(self):
        """Number of discretisation cells in the mesh.

        It is computed by multiplying all elements of ``n``:

        .. math::

            n_\\text{total} = n_{x} n_{y} n_{z}.

        Returns
        -------
        int

            Total number of discretisation cells.

        Examples
        --------
        1. Getting the number of discretisation cells in a mesh.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 5, 0)
        >>> p2 = (5, 15, 2)
        >>> cell = (1, 0.1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        >>> mesh.n
        (5, 100, 2)
        >>> len(mesh)
        1000

        """
        return int(np.prod(self.n))

    @property
    def indices(self):
        """Generator yielding indices of all mesh cells.

        Yields
        ------
        tuple (3,)

            Mesh cell indices :math:`(i_{x}, i_{y}, i_{z})`.

        Examples
        --------
        1. Getting indices of all mesh cells.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (3, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> list(mesh.indices)
        [(0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0)]

        .. seealso:: :py:func:`~discretisedfield.Mesh.__iter__`

        """
        for index in itertools.product(*map(range, reversed(self.n))):
            yield tuple(reversed(index))

    def __iter__(self):
        """Generator yielding coordinates of all mesh cell centres.

        The discretisation cell's coordinate corresponds to its centre point.

        Yields
        ------
        tuple (3,)

            Mesh cell's centre point :math:`\\mathbf{p} = (p_{x}, p_{y},
            p_{z})`.

        Examples
        --------
        1. Getting coordinates of all mesh cells.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        >>> list(mesh)
        [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 1.5, 0.5), (1.5, 1.5, 0.5)]

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        for index in self.indices:
            yield self.index2point(index)

    def __eq__(self, other):
        """Relational operator ``==``.

        Two meshes are considered to be equal if:

          1. Regions of both meshes are equal.

          2. They have the same number of discretisation cells in all three
          directions :math:`n^{1}_{i} = n^{2}_{i}`, for :math:`i = x, y, z`.

        Periodic boundary conditions ``pbc`` and ``subregions`` are not
        considered to be necessary conditions for determining equality.

        Parameters
        ----------
        other : discretisedfield.Mesh

            Mesh compared to ``self``.

        Returns
        -------
        bool

            ``True`` if two meshes are equal and ``False`` otherwise.

        Examples
        --------
        1. Check if meshes are equal.

        >>> import discretisedfield as df
        ...
        >>> mesh1 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        >>> mesh2 = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        >>> mesh3 = df.Mesh(p1=(1, 1, 1), p2=(5, 5, 5), cell=(2, 2, 2))
        >>> mesh1 == mesh2
        True
        >>> mesh1 != mesh2
        False
        >>> mesh1 == mesh3
        False
        >>> mesh1 != mesh3
        True

        .. seealso:: :py:func:`~discretisedfield.Mesh.__ne__`

        """
        if not isinstance(other, self.__class__):
            return False
        if self.region == other.region and self.n == other.n:
            return True
        else:
            return False

    def __ne__(self, other):
        """Relational operator ``!=``.

        This method returns ``not self == other``. For details, please
        refer to ``discretisedfield.Mesh.__eq__`` method.

        .. seealso:: :py:func:`~discretisedfield.Mesh.__eq__`

        """
        return not self == other

    def __repr__(self):
        """Mesh representation string.

        This method returns a string that can be copied so that exactly the
        same mesh object can be defined.

        Returns
        -------
        str

           Mesh representation string.

        Example
        -------
        1. Getting mesh representation string.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> pbc = 'x'
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, pbc=pbc)
        >>> repr(mesh)
        "Mesh(region=Region(p1=(0, 0, 0), p2=(2, 2, 1)), n=(2, 2, 1), ...)"

        """
        return (f'Mesh(region={repr(self.region)}, n={self.n}, '
                f'pbc={self.pbc}, subregions={self.subregions})')

    def index2point(self, index):
        """Convert cell's index to the its centre point coordinate.

        Parameters
        ----------
        index : (3,) array_like

            The cell's index :math:`(i_{x}, i_{y}, i_{z})`.

        Returns
        -------
        (3,) tuple

            The cell's centre point :math:`\\mathbf{p} = (p_{x}, p_{y},
            p_{z})`.

        Raises
        ------
        ValueError

            If ``index`` is out of range.

        Examples
        --------
        1. Converting cell's index to its centre point coordinate.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.index2point((0, 0, 0))
        (0.5, 0.5, 0.5)
        >>> mesh.index2point((0, 1, 0))
        (0.5, 1.5, 0.5)

        .. seealso:: :py:func:`~discretisedfield.Mesh.point2index`

        """
        if np.logical_or(np.less(index, 0),
                         np.greater_equal(index, self.n)).any():
            msg = f'Index {index} out of range.'
            raise ValueError(msg)

        point = np.add(self.region.pmin,
                       np.multiply(np.add(index, 0.5), self.cell))
        return dfu.array2tuple(point)

    def point2index(self, point):
        """Convert point to the index of a cell which contains it.

        Parameters
        ----------
        point : (3,) array_like

            Point :math:`\\mathbf{p} = (p_{x}, p_{y}, p_{z})`.

        Returns
        -------
        (3,) tuple

            The cell's index :math:`(i_{x}, i_{y}, i_{z})`.

        Raises
        ------
        ValueError

            If ``point`` is outside the mesh.

        Examples
        --------
        1. Converting point to the cell's index.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        >>> mesh.point2index((0.2, 1.7, 0.3))
        (0, 1, 0)

        .. seealso:: :py:func:`~discretisedfield.Mesh.index2point`

        """
        if point not in self.region:
            msg = f'Point {point} is outside the mesh region.'
            raise ValueError(msg)

        index = np.subtract(np.divide(np.subtract(point, self.region.pmin),
                                      self.cell), 0.5).round().astype(int)
        # If index is rounded to the out-of-range values.
        index = np.clip(index, 0, np.subtract(self.n, 1))

        return dfu.array2tuple(index)

    def line(self, p1, p2, n=100):
        """Line generator.

        Given two points ``p1`` and ``p2``, ``n`` points are generated:

        .. math::

           \\mathbf{r}_{i} = i\\frac{\\mathbf{p}_{2} - \\mathbf{p}_{1}}{n-1},
           \\text{for}\\, i = 0, ..., n-1

        and this method yields :math:`\\mathbf{r}_{i}` in :math:`n` iterations.

        Parameters
        ----------
        p1/p2 : (3,) array_like

            Points between which the line is generated :math:`\\mathbf{p} =
            (p_{x}, p_{y}, p_{z})`.

        n : int, optional

            Number of points on the line. Defaults to 100.

        Yields
        ------
        tuple (3,)

            :math:`\\mathbf{r}_{i}`

        Raises
        ------
        ValueError

            If ``p1`` or ``p2`` is outside the mesh region.

        Examples
        --------
        1. Creating line generator.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> line = mesh.line(p1=(0, 0, 0), p2=(2, 0, 0), n=2)
        >>> list(line)
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

        .. seealso:: :py:func:`~discretisedfield.Region.plane`

        """
        if p1 not in self.region or p2 not in self.region:
            msg = f'Point {p1} or point {p2} is outside the mesh region.'
            raise ValueError(msg)

        dl = np.subtract(p2, p1) / (n-1)
        for i in range(n):
            yield dfu.array2tuple(np.add(p1, i*dl))

    def plane(self, *args, n=None, **kwargs):
        """Extracts plane mesh.

        If one of the axes (``'x'``, ``'y'``, or ``'z'``) is passed as a
        string, a plane mesh perpendicular to that axis is extracted,
        intersecting the mesh region at its centre. Alternatively, if a keyword
        argument is passed (e.g. ``x=1e-9``), a plane perpendicular to the
        x-axis (parallel to yz-plane) and intersecting it at ``x=1e-9`` is
        extracted. The number of points in two dimensions on the plane can be
        defined using ``n`` tuple (e.g. ``n=(10, 15)``).

        The resulting mesh has an attribute ``info``, which is a dictionary
        containing basic information about the plane mesh.

        Parameters
        ----------
        n : (2,) tuple

            The number of points on the plane in two dimensions.

        Returns
        ------
        discretisedfield.Mesh

            An extracted mesh.

        Examples
        --------
        1. Extracting the plane mesh at a specific point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> plane_mesh = mesh.plane(y=1)

        2. Extracting the plane mesh at the mesh region centre.

        >>> plane_mesh = mesh.plane('z')

        3. Specifying the number of points.

        >>> plane_mesh = mesh.plane('z', n=(3, 3))

        .. seealso:: :py:func:`~discretisedfield.Region.line`

        """
        if args and not kwargs:
            if len(args) != 1:
                msg = f'Multiple args ({args}) passed.'
                raise ValueError(msg)

            # Only planeaxis is provided via args and the point is defined the
            # centre of the sample.
            planeaxis = dfu.axesdict[args[0]]
            point = self.region.centre[planeaxis]
        elif kwargs and not args:
            if len(kwargs) != 1:
                msg = f'Multiple kwargs ({kwargs}) passed.'
                raise ValueError(msg)

            planeaxis, point = list(kwargs.items())[0]
            planeaxis = dfu.axesdict[planeaxis]

            # Check if point is outside the mesh region.
            test_point = list(self.region.centre)
            test_point[planeaxis] = point
            if test_point not in self.region:
                msg = f'Point {test_point} is outside the mesh region.'
                raise ValueError(msg)
        else:
            msg = 'Either one arg or one kwarg can be passed, not both.'
            raise ValueError(msg)

        # Get indices of in-plane axes.
        axis1, axis2 = tuple(filter(lambda val: val != planeaxis,
                                    dfu.axesdict.values()))

        if n is None:
            n = (self.n[axis1], self.n[axis2])

        # Build plane-mesh.
        p1pm, p2pm, npm = np.zeros(3), np.zeros(3), np.zeros(3, dtype=int)
        ilist = [axis1, axis2, planeaxis]
        p1pm[ilist] = (self.region.pmin[axis1],
                       self.region.pmin[axis2],
                       point - self.cell[planeaxis]/2)
        p2pm[ilist] = (self.region.pmax[axis1],
                       self.region.pmax[axis2],
                       point + self.cell[planeaxis]/2)
        npm[ilist] = (*n, 1)

        plane_mesh = self.__class__(p1=p1pm, p2=p2pm, n=dfu.array2tuple(npm))

        # Add info dictionary, so that the mesh can be interpreted easier.
        info = dict()
        info['planeaxis'] = planeaxis
        info['point'] = point
        info['axis1'], info['axis2'] = axis1, axis2
        plane_mesh.info = info

        return plane_mesh

    def __getitem__(self, key):
        """Extracts the mesh of a subregion.

        If subregions were defined by passing ``subregions`` dictionary when
        the mesh was created, this method returns a mesh defined on a subregion
        ``subregions[key]`` with the same discretisation cell as the parent
        mesh.

        Parameters
        ----------
        key : str

            The key of a region in ``subregions`` dictionary.

        Returns
        -------
        disretisedfield.Mesh

            Mesh of a subregion.

        Example
        -------
        1. Extract subregion mesh.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (10, 10, 10)
        >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        ...
        >>> len(mesh)  # number of discretisation cells
        1000
        >>> mesh.region.pmin
        (0, 0, 0)
        >>> mesh.region.pmax
        (100, 100, 100)
        >>> submesh = mesh['r1']
        >>> len(submesh)
        500
        >>> submesh.region.pmin
        (0, 0, 0)
        >>> submesh.region.pmax
        (50, 100, 100)

        """
        return self.__class__(region=self.subregions[key], cell=self.cell)

    def mpl(self, ax=None, figsize=None, multiplier=None,
            color_palette=dfu.color_palette('deep', 10, 'rgb')[:2],
            linewidth=2, **kwargs):
        """Plots the mesh region and discretisation cell using ``matplotlib``
        3D plot.

        If ``ax`` is not passed, axes will be created automaticaly. In that
        case, the figure size can be changed using ``figsize``. It is often the
        case that the region size is small (e.g. on a nanoscale) or very large
        (e.g. in units of kilometers). Accordingly, ``multiplier`` can be
        passed as :math:`10^{n}`, where :math:`n` is a multiple of 3 (..., -6,
        -3, 0, 3, 6,...). According to that value, the axes will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, all mesh points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels. If ``multiplier``
        is not passed, the optimum one is computed internally. The colours of
        lines depicting the region and the discretisation cell can be
        determined using ``color_palette`` as an RGB-tuple. More precisely, the
        first element is the colour of the region, whereas the second value is
        the colour of the discretisation cell. Similarly, linewidth can be set
        up by passing ``linewidth``.

        This method plots the mesh using ``matplotlib.pyplot.plot()`` function,
        so any keyword arguments accepted by it can be passed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which mesh plot should be added. Defaults to ``None`` - new
            axes will be created in figure with size defined as ``figsize``.

        figsize : (2,) tuple, optional

            Length-2 tuple passed to ``matplotlib.pyplot.figure()`` to create a
            figure and axes if ``ax=None``. Defaults to ``None``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        color_palette : (2,) tuple, optional

            An RGB length-2 list, whose elements are length-3 tuples of RGB
            colours. Defaults to
            ``seaborn.color_pallette(palette='deep')[:2]``.

        linewidth : float, optional

            Width of the line. Defaults to 2.

        Examples
        --------
        1. Visualising the mesh using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, n=(50, 50, 5))
        >>> mesh.mpl()

        .. seealso:: :py:func:`~discretisedfield.Mesh.k3d`

        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.region.edges)

        cell_region = df.Region(p1=self.region.pmin,
                                p2=np.add(self.region.pmin, self.cell))
        self.region.mpl(ax=ax, multiplier=multiplier, color=color_palette[0],
                        linewidth=linewidth, **kwargs)
        cell_region.mpl(ax=ax, multiplier=multiplier, color=color_palette[1],
                        linewidth=linewidth, **kwargs)

    def mpl_subregions(self, ax=None, figsize=None, multiplier=None,
                       color_palette=dfu.color_palette('deep', 10, 'rgb'),
                       linewidth=2, **kwargs):
        """Plots the mesh subregions using ``matplotlib`` 3D plot.

        If ``ax`` is not passed, axes will be created automaticaly. In that
        case, the figure size can be changed using ``figsize``. It is often the
        case that the mesh region size is small (e.g. on a nanoscale) or very
        large (e.g. in units of kilometers). Accordingly, ``multiplier`` can be
        passed as :math:`10^{n}`, where :math:`n` is a multiple of 3 (..., -6,
        -3, 0, 3, 6,...). According to that value, the axes will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, all mesh points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels. If ``multiplier``
        is not passed, the optimum one is computed internally. The colours of
        lines depicting the region and the discretisation cell can be
        determined using ``color_palette`` as a list of RGB-tuples. Similarly,
        linewidth can be set up by passing ``linewidth``.

        This method plots the mesh using ``matplotlib.pyplot.plot()`` function,
        so any keyword arguments accepted by it can be passed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which subregions plot should be added. Defaults to ``None``
            - new axes will be created in figure with size defined as
            ``figsize``.

        figsize : (2,) tuple, optional

            Length-2 tuple passed to ``matplotlib.pyplot.figure()`` to create a
            figure and axes if ``ax=None``. Defaults to ``None``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        color_palette : list, optional

            A list of RGB tuples, whose elements are length-3 tuples of RGB
            colours. Defaults to
            ``seaborn.color_pallette(palette='deep')``.

        linewidth : float, optional

            Width of the line. Defaults to 2.

        Examples
        --------
        1. Visualising subregions using ``matplotlib``.

        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
        >>> mesh.mpl_subregions()

        .. seealso:: :py:func:`~discretisedfield.Mesh.k3d_subregions`

        """
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.region.edges)

        for i, subregion in enumerate(self.subregions.values()):
            subregion.mpl(ax=ax, multiplier=multiplier,
                          color=color_palette[i % len(color_palette)],
                          linewidth=linewidth,
                          **kwargs)

    def k3d(self, plot=None, multiplier=None,
            color_palette=dfu.color_palette('deep', 2, 'int'), **kwargs):
        """Plots the mesh region and discretisation cell using ``k3d`` voxels.

        If ``plot`` is not passed, ``k3d`` plot will be created automaticaly.
        It is often the case that the mesh region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometeres). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, the mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the optimum one is
        computed internally. The colours of the region and the discretisation
        cell can be determined using ``color_palette`` as a list of integers,
        where the first value is the colour of the mesh region and the second
        value is the colour of the discretisation cell.

        This method plots the region using ``k3d.voxels()`` function, so any
        keyword arguments accepted by it can be passed.

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which mesh plot should be added. Defaults to ``None`` - new
            plot will be created.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        color_palette : list, optional

            A length-2 list, whose elements are integers of colours. Defaults
            to ``seaborn.color_pallette(palette='deep')[:2]``. The first
            element is the colour of the mesh region, whereas the second colour
            is the colour of the discretisation cell.


        Examples
        --------
        1. Visualising the mesh using ``k3d``.

        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> mesh.k3d()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Mesh.mpl`

        """
        plot_array = np.ones(tuple(reversed(self.n)))
        plot_array[0, 0, -1] = 2  # mark the discretisation cell

        plot, multiplier = dfu.k3d_parameters(plot, multiplier,
                                              self.region.edges)

        plot += dfu.voxels(plot_array, pmin=self.region.pmin,
                           pmax=self.region.pmax, color_palette=color_palette,
                           multiplier=multiplier, **kwargs)

    def k3d_subregions(self, plot=None, multiplier=None,
                       color_palette=dfu.color_palette('deep', 10, 'int'),
                       **kwargs):
        """Plots the mesh subregions using ``k3d`` voxels.

        If ``plot`` is not passed, ``k3d`` plot will be created automaticaly.
        It is often the case that the mesh region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometeres). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, the mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the optimum one is
        computed internally. The colours of the subregions can be determined
        using ``color_palette`` as a list of integers.

        This method plots the region using ``k3d.voxels()`` function, so any
        keyword arguments accepted by it can be passed.

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which mesh subregions plot should be added. Defaults to
            ``None`` - new plot will be created.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        color_palette : list, optional

            List of integers for the colours of subregions. Defaults to
            ``seaborn.color_pallette(palette='deep')``.

        Examples
        --------
        1. Visualising subregions using ``k3d``.

        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
        >>> mesh.k3d_subregions()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Mesh.mpl_subregions`

        """
        plot_array = np.zeros(self.n)
        for index in self.indices:
            for i, region in enumerate(self.subregions.values()):
                if self.index2point(index) in region:
                    # +1 to avoid 0 value - invisible voxel
                    plot_array[index] = (i % len(color_palette)) + 1
        plot_array = np.swapaxes(plot_array, 0, 2)  # swap axes for k3d.voxels

        plot, multiplier = dfu.k3d_parameters(plot, multiplier,
                                              self.region.edges)

        plot += dfu.voxels(plot_array, pmin=self.region.pmin,
                           pmax=self.region.pmax, color_palette=color_palette,
                           multiplier=multiplier, **kwargs)

    def k3d_points(self, plot=None, point_size=None, multiplier=None,
                   color=dfu.color_palette('deep', 1, 'int')[0], **kwargs):
        """Plots the points at discretisation cell centres using ``k3d``.

        If ``plot`` is not passed, ``k3d`` plot will be created automaticaly.
        It is often the case that the mesh region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometeres). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, the mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the optimum one is
        computed internally. The colour of points can be determined using
        ``color`` as an integer, whereas the size of the points can be passed
        using ``point_size``. If ``point_size`` is not passed, optimum size is
        computed intenally.

        This method plots the points using ``k3d.points()`` function, so any
        keyword arguments accepted by it can be passed.

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which mesh points should be added. Defaults to ``None`` -
            new plot will be created.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        point_size : float, optional

            Size of points.

        color : int, optional

            Colour of points. Defaults to
            ``seaborn.color_pallette(palette='deep')[0]``.

        Examples
        --------
        1. Visualising the mesh points using ``k3d``.

        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> mesh.k3d_points()
        Plot(...)

        """
        coordinates = np.array(list(self))

        plot, multiplier = dfu.k3d_parameters(plot, multiplier,
                                              self.region.edges)

        if point_size is None:
            # If undefined, the size of the point is 1/4 of the smallest cell
            # dimension.
            point_size = np.divide(self.cell, multiplier).min() / 4

        plot += dfu.points(coordinates, color=color, point_size=point_size,
                           multiplier=multiplier, **kwargs)

    def slider(self, axis, multiplier=None, **kwargs):
        """Slider for interactive plotting.

        For ``axis``, ``'x'``, ``'y'``, or ``'z'`` can be passed. Based on that
        value, ``ipywidgets.SelectionSlider`` is returned for navigating
        interactive plots.

        This method plots the points using ``k3d.points()`` function, so any
        keyword arguments accepted by it can be passed.

        This method is based on ``ipywidgets.SelectionSlider``, so any keyword
        argument accepted by it can be passed.

        Parameters
        ----------
        axis : struct

            Axis for which the slider is returned.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the slider points will
            be divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units
            will be used in the description. If ``multiplier`` is not passed,
            the optimum one is computed internally. Defaults to ``None``.

        Returns
        -------
        ipywidgets.SelectionSlider

            Axis slider.

        Example
        -------
        1. Get the slider for the x-coordinate.

        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> mesh.slider('x')
        SelectionSlider(...)

        """
        if isinstance(axis, str):
            axis = dfu.axesdict[axis]

        if multiplier is None:
            multiplier = uu.si_multiplier(self.region.edges[axis])

        slider_min = self.region.pmin[axis] + self.cell[axis]/2
        slider_max = self.region.pmax[axis] - self.cell[axis]/2
        slider_step = self.cell[axis]
        slider_description = (f'{dfu.raxesdict[axis]} '
                              f'({uu.rsi_prefixes[multiplier]}m)')

        values = np.arange(slider_min, slider_max+1e-20, slider_step)
        labels = np.around(values / multiplier, decimals=3)
        options = list(zip(labels, values))

        # Select middle element for slider value
        slider_value = values[int(self.n[axis]/2)]

        return ipywidgets.SelectionSlider(options=options,
                                          value=slider_value,
                                          description=slider_description,
                                          **kwargs)

    def axis_selection(self, widget='dropdown', description='axis'):
        """Axis selection widget.

        For ``widget='dropdown'``, ``ipywidgets.Dropdown`` is returned, whereas
        for ``widget='radiobuttons'``, ``ipywidgets.RadioButtons`` is returned.
        Returned widget can later be used for navigating plots. Description of
        the widget can be passed using ``description``.

        Parameters
        ----------
        widget : str

            Type of widget to be returned. Defaults to ``'dropdown'``.

        description : str

            Widget description to be showed. Defaults to ``'axis'``.

        Returns
        -------
        ipywidgets.Dropdown, ipywidgets.RadioButtons

            Axis selection widget.

        Example
        -------
        1. Get the ``RadioButtons`` slider.

        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> mesh.axis_selection(widget='radiobuttons')
        RadioButtons(...)

        """
        if widget.lower() == 'dropdown':
            return ipywidgets.Dropdown(options=list('xyz'),
                                       value='z',
                                       description=description)
        elif widget == 'radiobuttons':
            return ipywidgets.RadioButtons(options=list('xyz'),
                                           value='z',
                                           description=description)
        else:
            msg = f'Widget {widget} is not supported.'
            raise ValueError(msg)
