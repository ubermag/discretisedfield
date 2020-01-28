import itertools
import matplotlib
import numpy as np
import discretisedfield as df
import matplotlib.pyplot as plt
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu
from mpl_toolkits.mplot3d import Axes3D


@ts.typesystem(region=ts.Typed(expected_type=df.Region),
               cell=ts.Vector(size=3, positive=True, const=True),
               n=ts.Vector(size=3, component_type=int, unsigned=True,
                           const=True),
               pbc=ts.Subset(sample_set='xyz', unpack=True),
               subregions=ts.Dictionary(key_descriptor=ts.Name(),
                                        value_descriptor=
                                        ts.Typed(expected_type=df.Region),
                                        allow_empty=True))
class Mesh:
    """Finite difference mesh.

    Mesh discretises cubic `discretisedfield.Region`, passed as `region`, using
    a regular finite difference mesh. Since cubic region spans between two
    points :math:`\\mathbf{p}_{1}` and :math:`\\mathbf{p}_{2}`, these points
    can be passed as `p1` and `p2`, instead of passing
    `discretisedfield.Region` object. In this case `discretisedfield.Region` is
    created internally, based on points `p1` and `p2`. Either `region` or `p1`
    and `p2` must be passed, not both. The region is discretised using a finite
    difference cell, whose dimensions are defined with `cell`. Alternatively,
    the domain can be discretised by passing the number of discretisation cells
    `n` in all three dimensions. Either `cell` or `n` should be defined to
    discretise the region, not both. Periodic boundary conditions can be
    specified by passing `pbc` argument, which is an iterable containing one or
    more elements from ``['x', 'y', 'z']``. If it is necessary to define
    subregions in the mesh, a dictionary can be passed as `subregions`. More
    precisely, dictionary keys are strings representing valid Python variable
    names, whereas values are `discretisedfield.Region` objects.

    In order to properly define a mesh, mesh region must be an aggregate of
    discretisation cells.

    Parameters
    ----------
    region : discretisedfield.Region
        Cubic region to be discretised on a regular mesh. Either `region` or
        `p1` and `p2` should be defined, not both.
    p1, p2 : (3,) array_like
        Points between which the mesh region spans :math:`\\mathbf{p} = (p_{x},
        p_{y}, p_{z})`. Either `region` or `p1` and `p2` should be defined, not
        both.
    cell : (3,) array_like, optional
        Discretisation cell size :math:`(d_{x}, d_{y}, d_{z})`. Either `cell`
        or `n` should be defined, not both.
    n : (3,) array_like, optional
        The number of discretisation cells :math:`(n_{x}, n_{y}, n_{z})`.
        Either `cell` or `n` should be defined, not both.
    pbc : iterable, optional
        Periodic boundary conditions in x, y, or z direction. Its value is an
        iterable consisting of one or more characters `x`, `y`, or `z`,
        denoting the direction(s) along which the mesh is periodic.
    subregions : dict, optional
        A dictionary defining subregions in the mesh. The keys of the
        dictionary are the region names (str), whereas the values are
        `discretisedfield.Region` objects.

    Raises
    ------
    ValueError
        If mesh domain is not an aggregate of discretisation cells.
        Alternatively, if both `region` and `p1`/`p2` or both `cell` and `n`
        are passed.

    Examples
    --------
    1. Defining a nano-sized thin film mesh by passing `region` and `cell`
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

    2. Defining a nano-sized thin film mesh by passing `p1`, `p2` and `n`
    parameters.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> n = (100, 50, 5)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> mesh
    Mesh(...)

    3. Defining a mesh with periodic boundary conditions in x and y
    directions.

    >>> import discretisedfield as df
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 1)
    >>> n = (100, 100, 1)
    >>> pbc = 'xy'
    >>> region = df.Region(p1=p1, p2=p2)
    >>> mesh = df.Mesh(region=region, n=n, pbc=pbc)
    >>> mesh
    Mesh(...)

    4. Defining a mesh with two subregions.

    >>> import discretisedfield as df
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> n = (10, 10, 10)
    >>> regions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
    ...            'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, subregions=subregions)
    >>> mesh
    Mesh(...)

    5. An attempt to define a mesh, whose region is not an aggregate of
    discretisation cells in the :math:`z` direction.

    >>> import discretisedfield as df
    ...
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

        # Is the mesh region not an aggregate of the discretisation cell?
        tol = 1e-12  # picometre tolerance
        rem = np.remainder(self.region.edges, self.cell)
        if np.logical_and(np.greater(rem, tol),
                          np.less(rem, np.subtract(self.cell, tol))).any():
            msg = 'Mesh region is not an aggregate of the discretisation cell.'
            raise ValueError(msg)

        self.pbc = pbc
        self.subregions = subregions

    @property
    def ntotal(self):
        """Number of discretisation cells.

        It is computed by multiplying all elements of `self.n`:
        :math:`n_\\text{total} = n_{x}n_{y}n_{z}`.

        Returns
        -------
        int
            Total number of discretisation cells

        Examples
        --------
        1. Getting the number of discretisation cells in a mesh.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 5, 0)
        >>> p2 = (5, 15, 2)
        >>> cell = (1, 0.1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        >>> mesh.ntotal
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

        .. seealso:: :py:func:`~discretisedfield.Mesh.coordinates`

        """
        for index in itertools.product(*map(range, reversed(self.n))):
            yield tuple(reversed(index))

    @property
    def coordinates(self):
        """Generator yielding coordinates of all mesh cells.

        The discretisation cell coordinate corresponds to its centre point.

        Yields
        ------
        tuple (3,)
            Mesh cell coordinates :math:`(p_{x}, p_{y}, p_{z})`.

        Examples
        --------
        1. Getting coordinates of all mesh cells.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        >>> list(mesh.coordinates)
        [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 1.5, 0.5), (1.5, 1.5, 0.5)]

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        for index in self.indices:
            yield self.index2point(index)

    def __iter__(self):
        """This method enables `discretisedfield.Mesh` object to be iterable.

        It iterates through the coodinates of the mesh cells
        (`df.Mesh.coordinates`).

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        return self.coordinates

    def __eq__(self, other):
        """Equality operator.

        Two meshes are considered to be equal if:

          1. Regions of both meshes are equal.

          2. They have the same number of discretisation cells in all three
          directions :math:`n^{1}_{i} = n^{2}_{i}`.

        Periodic boundary conditions `pbc` and `subregions` are not considered
        to be necessary conditions for determining equality.

        Parameters
        ----------
        other : discretisedfield.Mesh
            Mesh compared to self.

        Returns
        -------
        bool
            `True` if two regions are equal and `False` otherwise.

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
        """Inverse of equality operator.

        This method returns `not self == other`. For details, please
        refer to `discretisedfield.Mesh.__eq__()` method.

        """
        return not self == other

    def __repr__(self):
        """Mesh representation string.

        This method returns the string that can be copied in another Python
        script so that exactly the same mesh object can be defined.

        Returns
        -------
        str
            Mesh representation string.

        Examples
        --------
        1. Getting mesh representation string.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> pbc = 'x'
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, pbc=pbc)
        >>> repr(mesh)
        "Mesh(region=Region(p1=(0, 0, 0), p2=(2, 2, 1)), cell=(1, 1, 1), ...)"

        """
        return (f'Mesh(region={repr(self.region)}, n={self.n}, '
                f'pbc={self.pbc}, subregions={self.subregions})')

    def index2point(self, index):
        """Convert cell's index to the its coordinate.

        Parameters
        ----------
        index : (3,) array_like
            The cell's index :math:`(i_{x}, i_{y}, i_{z})`.

        Returns
        -------
            The cell's centre point :math:`(p_{x}, p_{y}, p_{z})`.

        Raises
        ------
        ValueError
            If `index` is out of range.

        Examples
        --------
        1. Converting cell's index to its coordinate.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.index2point((0, 1, 0))
        (0.5, 1.5, 0.5)

        .. seealso:: :py:func:`~discretisedfield.Mesh.point2index`

        """
        # Does index refer to a cell outside the mesh?
        if np.logical_or(np.less(index, 0),
                         np.greater_equal(index, self.n)).any():
            msg = f'Index {index} out of range.'
            raise ValueError(msg)

        point = np.add(self.region.pmin,
                        np.multiply(np.add(index, 0.5), self.cell))
        return dfu.array2tuple(point)

    def point2index(self, point):
        """Convert coordinate to cell's index.

        Parameters
        ----------
        p : (3,) array_like
            The point's coordinate :math:`(p_{x}, p_{y}, p_{z})`.

        Returns
        -------
            The cell's index :math:`(i_{x}, i_{y}, i_{z})`.

        Raises
        ------
        ValueError
            If `point` is outside the mesh.

        Examples
        --------
        1. Converting coordinate to cell's index.

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

        Given two points :math:`p_{1}` and :math:`p_{2}`, :math:`n` position
        coordinates are generated.

        .. math::

           \\mathbf{r}_{i} = i\\frac{\\mathbf{p}_{2} - \\mathbf{p}_{1}}{n-1}

        and this method yields :math:`\\mathbf{r}_{i}` in :math:`n` iterations.

        Parameters
        ----------
        p1, p2 : (3,) array_like
            Two points between which the line is generated.
        n : int
            Number of points on the line. Defaults to 100.

        Yields
        ------
        tuple
            :math:`\\mathbf{r}_{i}`

        Raises
        ------
        ValueError
            If `p1` or `p2` is outside the mesh region.

        Examples
        --------
        1. Creating line generator.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> list(mesh.line(p1=(0, 0, 0), p2=(2, 0, 0), n=2))
        [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]

        """
        if p1 not in self.region or p2 not in self.region:
            msg = f'Point {p1} or point {p2} is outside the mesh region.'
            raise ValueError(msg)

        dl = np.subtract(p2, p1) / (n-1)
        for i in range(n):
            yield dfu.array2tuple(np.add(p1, i*dl))

    def plane(self, *args, n=None, **kwargs):
        """Slices mesh with a plane.

        If one of the axes (`'x'`, `'y'`, or `'z'`) is passed as a string, a
        plane perpendicular to that axis is generated which intersects the mesh
        at its centre. Alternatively, if a keyword argument is passed (e.g.
        `x=1`), a plane perpendicular to the x-axis (parallel to yz-plane) and
        intersecting it at `x=1` is generated. The number of points in two
        dimensions on the plane can be defined using `n` (e.g. `n=(10, 15)`).
        Using the generated plane, a new "two-dimensional" mesh is created and
        returned. The resulting mesh has an attribute `info`, which is a
        dictionary containing basic information about the mesh plane.

        Parameters
        ----------
        n : tuple of length 2
            The number of points on the plane in two dimensions.

        Returns
        ------
        discretisedfield.Mesh
            A mesh obtained as an intersection of mesh and plane.

        Examples
        --------
        1. Intersecting the mesh with a plane.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.plane(y=1)
        Mesh(p1=(0.0, 0.5, 0.0), p2=(2.0, 1.5, 2.0), ...)

        """
        # The plane is defined with: planeaxis and point. They are extracted
        # from *args and **kwargs.
        if args and not kwargs:
            if len(args) != 1:
                msg = f'Multiple args {args} passed.'
                raise ValueError(msg)

            # Only planeaxis is provided via args and the point is defined
            # later as a centre of the sample.
            planeaxis = dfu.axesdict[args[0]]
            point = self.region.centre[planeaxis]
        elif kwargs and not args:
            if len(kwargs.keys()) != 1:
                msg = f'Multiple kwargs {kwargs} passed.'
                raise ValueError(msg)

            # Both planeaxis and point are provided via kwargs.
            planeaxis = dfu.axesdict[list(kwargs.keys())[0]]
            point = list(kwargs.values())[0]

            # Check if point is outside the mesh region.
            test_point = list(self.region.centre)
            test_point[planeaxis] = point
            if test_point not in self.region:
                msg = f'Point {test_point} is outside the mesh.'
                raise ValueError(msg)
        else:
            msg = 'Either one arg or one kwarg can be passed, not both.'
            raise ValueError(msg)

        # Get indices of in-plane axes.
        axis1, axis2 = tuple(filter(lambda val: val != planeaxis,
                                    dfu.axesdict.values()))

        # Determine the n tuple.
        if n is None:
            n = (self.n[axis1], self.n[axis2])

        # Build a mesh.
        p1s, p2s, ns = np.zeros(3), np.zeros(3), np.zeros(3)
        ilist = [axis1, axis2, planeaxis]
        p1s[ilist] = (self.region.pmin[axis1],
                      self.region.pmin[axis2],
                      point - self.cell[planeaxis]/2)
        p2s[ilist] = (self.region.pmax[axis1],
                      self.region.pmax[axis2],
                      point + self.cell[planeaxis]/2)
        ns[ilist] = (*n, 1)
        ns = dfu.array2tuple(ns.astype(int))

        plane_mesh = self.__class__(p1=p1s, p2=p2s, n=ns)

        info = dict()
        info['planeaxis'] = planeaxis
        info['point'] = point
        info['axis1'], info['axis2'] = axis1, axis2

        plane_mesh.info = info  # Add info so it can be interpreted easier

        return plane_mesh

    def mpl(self, figsize=None):
        """Plots the mesh domain and the discretisation cell using a
        `matplotlib` 3D plot.

        Parameters
        ----------
        figsize : tuple, optional
            Length-2 tuple passed to the `matplotlib.pyplot.figure`
            function.

        Examples
        --------
        1. Visualising the mesh using `matplotlib`

        >>> import discretisedfield as df
        ...
        >>> p1 = (-6, -3, -3)
        >>> p2 = (6, 3, 3)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.mpl()

        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        dfu.plot_box(ax, self.region.pmin, self.region.pmax,
                     'b-', linewidth=1.5)
        dfu.plot_box(ax, self.region.pmin, np.add(self.region.pmin, self.cell),
                     'r--', linewidth=1)

        ax.set(xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$')

    def k3d(self, colormap=dfu.colormap, plot=None, **kwargs):
        """Plots the mesh domain and emphasises the discretisation cell.

        The first element of `colormap` is the colour of the domain,
        whereas the second one is the colour of the discretisation
        cell. If `plot` is passed as a `k3d.plot.Plot`, plot is added
        to it. Otherwise, a new k3d plot is created. All arguments
        allowed in `k3d.voxels()` can be passed. This function is to
        be called in Jupyter notebook.

        Parameters
        ----------
        colormap : list, optional
            Length-2 list of colours in hexadecimal format. The first
            element is the colour of the domain, whereas the second
            one is the colour of the discretisation cell.
        plot : k3d.plot.Plot, optional
            If this argument is passed, plot is added to
            it. Otherwise, a new k3d plot is created.

        Examples
        --------
        1. Visualising the mesh using `k3d`

        >>> import discretisedfield as df
        ...
        >>> p1 = (-6, -3, -3)
        >>> p2 = (6, 3, 3)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.k3d()
        Plot(...)

        """
        plot_array = np.ones(tuple(reversed(self.n)))
        plot_array[0, 0, -1] = 2  # mark the discretisation cell

        # In the case of nano-sized samples, fix the order of
        # magnitude of the plot extent to avoid freezing the k3d plot.
        if np.any(np.divide(self.cell, 1e-9) < 1e3):
            pmin = np.divide(self.region.pmin, 1e-9)
            pmax = np.divide(self.region.pmax, 1e-9)
        else:
            pmin = self.region.pmin
            pmax = self.region.pmax

        dfu.voxels(plot_array, pmin=pmin, pmax=pmax,
                   colormap=colormap, plot=plot, **kwargs)

    def k3d_points(self, point_size=0.5, color=dfu.colormap[0],
                   plot=None, **kwargs):
        """Plots the points at discretisation cell centres.

        The size of points can be defined with `point_size` argument
        and their colours with `color`. If `plot` is passed as a
        `k3d.plot.Plot`, plot is added to it. Otherwise, a new k3d
        plot is created. All arguments allowed in `k3d.points()` can
        be passed. This function is to be called in Jupyter notebook.

        Parameters
        ----------
        point_size : float, optional
            The size of a single point.
        color : hex, optional
            Colour of a single point.
        plot : k3d.plot.Plot, optional
            If this argument is passed, plot is added to
            it. Otherwise, a new k3d plot is created.

        Examples
        --------
        1. Plotting discretisation cell centres using `k3d.points`

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (20, 20, 10)
        >>> n = (10, 10, 5)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> mesh.k3d_points()
        Plot(...)

        """
        plot_array = np.array(list(self.coordinates))
        dfu.points(plot_array, point_size=point_size, color=color,
                   plot=plot, **kwargs)

    def mpl_subregions(self, colormap=dfu.colormap, figsize=None, **kwargs):
        """Plots the mesh regions using a `matplotlib` 3D plot.

        Parameters
        ----------
        colormap : list, optional
            List of colours in hexadecimal format. The order of
            colours should be the same as the order of regions defined
            in `discretisedfield.Mesh.regions`. By default 6 colours
            are defined.
        figsize : tuple, optional
            Length-2 tuple passed to the `matplotlib.pyplot.figure`
            function.

        Examples
        --------
        1. Visualising the mesh regions using `matplotlib`

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> regions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...            'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, regions=regions)
        >>> mesh.mpl_regions()

        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Add random colours if necessary.
        colormap = dfu.add_random_colors(colormap, self.subregions)

        cmap = matplotlib.cm.get_cmap('hsv', 256)
        for i, name in enumerate(self.subregions.keys()):
            hc = matplotlib.colors.rgb2hex(cmap(colormap[i]/16777215)[:3])
            dfu.plot_box(ax, self.subregions[name].pmin,
                         self.subregions[name].pmax,
                         color=hc, linewidth=1.5)

        ax.set(xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$')

    def k3d_subregions(self, colormap=dfu.colormap, plot=None, **kwargs):
        """Plots the mesh domain and emphasises defined regions.

        The order of colours in `colormap` should be the same as the
        order of regions defined in
        `discretisedfield.Mesh.regions`. By default 6 colours are
        defined and for all additional regions, the colours will be
        generated randomly. If `plot` is passed as a `k3d.plot.Plot`,
        plot is added to it. Otherwise, a new k3d plot is created. All
        arguments allowed in `k3d.voxels()` can be passed. This
        function is to be called in Jupyter notebook.

        Parameters
        ----------
        colormap : list, optional
            List of colours in hexadecimal format. The order of
            colours should be the same as the order of regions defined
            in `discretisedfield.Mesh.regions`. By default 6 colours
            are defined.
        plot : k3d.plot.Plot, optional
            If this argument is passed, plot is added to
            it. Otherwise, a new k3d plot is created.

        Examples
        --------
        1. Visualising defined regions in the mesh

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> regions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...            'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n, regions=regions)
        >>> mesh.k3d_regions()
        Plot(...)

        """
        # Add random colours if necessary.
        colormap = dfu.add_random_colors(colormap, self.subregions)

        plot_array = np.zeros(self.n)
        for i, name in enumerate(self.subregions.keys()):
            for index in self.indices:
                if self.index2point(index) in self.subregions[name]:
                    plot_array[index] = i+1  # i+1 to avoid 0 value
        plot_array = np.swapaxes(plot_array, 0, 2)  # swap axes for k3d.voxels

        # In the case of nano-sized samples, fix the order of
        # magnitude of the plot extent to avoid freezing the k3d plot.
        if np.any(np.divide(self.cell, 1e-9) < 1e3):
            pmin = np.divide(self.region.pmin, 1e-9)
            pmax = np.divide(self.region.pmax, 1e-9)
        else:
            pmin = self.region.pmin
            pmax = self.region.pmax

        dfu.voxels(plot_array, pmin=pmin, pmax=pmax,
                   colormap=colormap, plot=plot, **kwargs)
