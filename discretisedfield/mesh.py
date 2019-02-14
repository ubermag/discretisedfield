import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import joommfutil.typesystem as ts
import discretisedfield.util as dfu
from .plot3d import k3d_vox, k3d_points
from mpl_toolkits.mplot3d import Axes3D


@ts.typesystem(p1=ts.Vector(size=3, const=True),
               p2=ts.Vector(size=3, const=True),
               cell=ts.Vector(size=3, positive=True, const=True),
               pbc=ts.Subset(sample_set="xyz"),
               name=ts.Name(const=True))
class Mesh:
    """Finite difference rectangular mesh.

    A rectangular mesh domain spans between two points `p1` and
    `p2`. They are defined as ``array_like`` objects of length 3
    (e.g. ``tuple``, ``list``, ``numpy.ndarray``), :math:`p = (p_{x},
    p_{y}, p_{z})`. The domain is then discretised into finite
    difference cells, where dimensions of a single cell are defined
    with a `cell` parameter. Similar to `p1` and `p2`, `cell`
    parameter is an ``array_like`` object :math:`(d_{x}, d_{y},
    d_{z})` of length 3. The parameter `name` is optional and defaults
    to "mesh".

    In order to properly define a mesh, the length of all mesh domain
    edges must not be zero and the mesh domain must be an aggregate
    of discretisation cells.

    Parameters
    ----------
    p1, p2 : (3,) array_like
        Points between which the mesh domain spans :math:`p = (p_{x},
        p_{y}, p_{z})`.
    cell : (3,) array_like
        Discretisation cell size :math:`(d_{x}, d_{y}, d_{z})`.
    pbc : iterable, optional
        Periodic boundary conditions in x, y, or z direction. Its value
        is a string consisting of one or more of the letters `x`, `y`,
        or `z`, denoting the direction(s) in which the mesh is periodic.
    name : str, optional
        Mesh name (the default is "mesh"). The mesh name must be a valid
        Python variable name string. More specifically, it must not
        contain spaces, or start with underscore or numeric character.

    Raises
    ------
    ValueError
        If the length of one or more mesh domain edges is zero, or
        mesh domain is not an aggregate of discretisation cells.

    Examples
    --------
    1. Creating a nano-sized thin film mesh

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> cell = (1e-9, 1e-9, 0.1e-9)
    >>> name = "mesh_name"
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name=name)

    2. An attempt to create a mesh with invalid parameters, so that
    the ``ValueError`` is raised. In this example, the mesh domain is
    not an aggregate of discretisation cells in the :math:`z`
    direction.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-25, 3, 0)
    >>> p2 = (25, 6, 1)
    >>> cell = (5, 3, 0.4)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name=name)
    Traceback (most recent call last):
        ...
    ValueError: ...

    """
    def __init__(self, p1, p2, cell, pbc=set(), name='mesh'):
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.cell = tuple(cell)
        self.pbc = pbc
        self.name = name

        # Is any edge length of the domain equal to zero?
        if np.equal(self.l, 0).any():
            msg = 'The length of one of the domain edges is zero.'
            raise ValueError(msg)

        # Is the mesh domain not an aggregate of discretisation cells?
        tol = 1e-12  # picometre tolerance
        rem = np.remainder(self.l, self.cell)
        if np.logical_and(np.greater(rem, tol),
                          np.less(rem, np.subtract(self.cell, tol))).any():
            msg = 'Mesh domain is not an aggregate of the discretisation cell'
            raise ValueError(msg)

    @property
    def pmin(self):
        """Mesh point with minimum coordinates.

        The :math:`i`-th component of :math:`p_\\text{min}` is
        computed from points :math:`p_{1}` and :math:`p_{2}` between
        which the mesh domain spans: :math:`p_\\text{min}^{i} =
        \\text{min}(p_{1}^{i}, p_{2}^{i})`.

        Returns
        -------
        tuple (3,)
            Point with minimum mesh coordinates :math:`(p_{x}^\\text{min},
            p_{y}^\\text{min}, p_{z}^\\text{min})`.

        Example
        -------
        1. Getting the minimum mesh point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> cell = (0.1, 0.1, 0.005)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name='mesh')
        >>> mesh.pmin
        (-1.1, 0.0, -0.1)

        .. seealso:: :py:func:`~discretisedfield.Mesh.pmax`

        """
        res = np.minimum(self.p1, self.p2)
        return dfu.array2tuple(res)

    @property
    def pmax(self):
        """Mesh point with maximum coordinates.

        The :math:`i`-th component of :math:`p_\\text{max}` is
        computed from points :math:`p_{1}` and :math:`p_{2}` between
        which the mesh domain spans: :math:`p_\\text{min}^{i} =
        \\text{max}(p_{1}^{i}, p_{2}^{i})`.

        Returns
        -------
        tuple (3,)
            Point with maximum mesh coordinates :math:`(p_{x}^\\text{max},
            p_{y}^\\text{max}, p_{z}^\\text{max})`.

        Example
        -------
        1. Getting the maximum mesh point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> cell = (0.1, 0.1, 0.005)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.pmax
        (5.0, 2.9, 0.0)

        .. seealso:: :py:func:`~discretisedfield.Mesh.pmin`

        """
        res = np.maximum(self.p1, self.p2)
        return dfu.array2tuple(res)

    @property
    def l(self):
        """Mesh domain edge lengths.

        Edge length in any direction :math:`i` is computed from the
        points between which the mesh domain spans :math:`l^{i} =
        |p_{2}^{i} - p_{1}^{i}|`.

        Returns
        -------
        tuple (3,)
            Lengths of mesh domain edges :math:`(l_{x}, l_{y}, l_{z})`.

        Example
        -------
        1. Getting mesh domain edge lengths.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5)
        >>> p2 = (5, 15, 15)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.l
        (5, 15, 20)

        .. seealso:: :py:func:`~discretisedfield.Mesh.n`

        """
        res = np.abs(np.subtract(self.p1, self.p2))
        return dfu.array2tuple(res)

    @property
    def n(self):
        """Number of discretisation cells in all directions.

        By dividing the lengths of mesh domain edges with
        discretisation in all directions, the number of cells is
        computed :math:`n^{i} = l^{i}/d^{i}`.

        Returns
        -------
        tuple (3,)
            The number of discretisation cells :math:`(n_{x},
            n_{y}, n_{z})`.

        Example
        -------
        1. Getting the number of discretisation cells in all three
        directions.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 5, 0)
        >>> p2 = (5, 15, 1)
        >>> cell = (0.5, 0.1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.n
        (10, 100, 1)

        .. seealso:: :py:func:`~discretisedfield.Mesh.ntotal`

        """
        res = np.divide(self.l, self.cell).round().astype(int)
        return dfu.array2tuple(res)

    @property
    def ntotal(self):
        """Total number of discretisation cells in the mesh.

        `ntotal` is obtained by multiplying all elements of `self.n`
        tuple.

        Returns
        -------
        int
            The total number of discretisation cells in the mesh

        Example
        -------
        1. Getting the number of discretisation cells in a mesh.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 5, 0)
        >>> p2 = (5, 15, 2)
        >>> cell = (1, 0.1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.ntotal
        1000

        .. seealso:: :py:func:`~discretisedfield.Mesh.n`

        """
        return int(np.prod(self.n))

    @property
    def centre(self):
        """Mesh domain centre point.

        This point does not necessarily coincide with the
        discretisation cell centre. It is computed as the middle point
        between minimum and maximum coordinate :math:`p_{c}^{i} =
        p_\\text{min}^{i} + 0.5l^{i}`, where :math:`p_\\text{min}^{i}`
        is the :math:`i`-th coordinate of the minimum mesh domain
        point and :math:`l^{i}` is the mesh domain edge length in the
        :math:`i`-th direction.

        Returns
        -------
        tuple (3,)
            Mesh domain centre point :math:`(p_{c}^{x}, p_{c}^{y},
            p_{c}^{z})`.

        Example
        -------
        1. Getting the mesh centre point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 15, 20)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.centre
        (2.5, 7.5, 10.0)

        """
        res = np.add(self.pmin, np.multiply(self.l, 0.5))
        return dfu.array2tuple(res)

    @property
    def indices(self):
        """Generator yielding indices of all mesh cells.

        Yields
        ------
        tuple (3,)
            Mesh cell indices :math:`(i_{x}, i_{y}, i_{z})`.

        Example
        -------
        1. Getting all mesh cell indices.

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

        The discretisation cell coordinate corresponds to the cell
        centre point.

        Yields
        ------
        tuple (3,)
            Mesh cell coordinates (`px`, `py`, `pz`).

        Example
        -------
        1. Getting all mesh cell coordinates.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> list(mesh.coordinates)
        [(0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 1.5, 0.5), (1.5, 1.5, 0.5)]

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        for index in self.indices:
            yield self.index2point(index)

    def __iter__(self):
        """This method makes `df.Mesh` object iterable.

        It iterates through the coodinates of the mesh cells
        (`df.Mesh.coordinates`).

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        return self.coordinates

    def __repr__(self):
        """Mesh representation.

        This method returns the string that can be copied in another
        Python script so that exactly the same mesh object could be
        created.

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
        >>> name = 'm'
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, pbc=pbc, name=name)
        >>> repr(mesh)
        "Mesh(p1=(0, 0, 0), p2=(2, 2, 1), cell=(1, 1, 1), pbc={'x'}, name='m')"

        """
        return (f'Mesh(p1={self.p1}, p2={self.p2}, cell={self.cell}, '
                f'pbc={self.pbc}, name=\'{self.name}\')')

    def random_point(self):
        """Generate the random point belonging to the mesh.

        The use of this function is mostly limited for writing tests
        for packages based on `discretisedfield`.

        Returns
        -------
        tuple (3,)
            Coordinates of a random point inside that belongs to the mesh
            :math:`(x_\\text{rand}, y_\\text{rand}, z_\\text{rand})`.

        Example
        -------
        1. Generating a random mesh point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (200e-9, 200e-9, 1e-9)
        >>> cell = (1e-9, 1e-9, 0.5e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.random_point()
        (...)

        .. note::

           In the example, ellipsis is used instead of an exact tuple
           because the result differs each time ``random_point``
           method is called.

        """
        res = np.add(self.pmin, np.multiply(np.random.random(3), self.l))
        return dfu.array2tuple(res)

    def index2point(self, index):
        """Convert cell's index to the cell's centre coordinate.

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
            If the cell's index is out of range.

        Example
        -------
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
        if np.logical_or(np.less(index, 0), np.greater_equal(index, self.n)).any():
            msg = 'Index out of range.'
            raise ValueError(msg)

        res = np.add(self.pmin, np.multiply(np.add(index, 0.5), self.cell))
        return dfu.array2tuple(res)

    def point2index(self, point):
        """Compute the index of a cell to which the point belongs to.

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
            If `point` is outside the mesh domain.

        Example
        -------
        1. Converting point's coordinate to the cell index.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.point2index((0.2, 1.7, 0.3))
        (0, 1, 0)

        .. seealso:: :py:func:`~discretisedfield.Mesh.index2point`

        """
        self.isinside(point, raise_exception=True)

        index = np.subtract(np.divide(np.subtract(point, self.pmin),
                                      self.cell), 0.5).round().astype(int)
        # If index is rounded to the out-of-range values.
        index = np.clip(index, 0, np.subtract(self.n, 1))

        return dfu.array2tuple(index)

    def isinside(self, point, raise_exception=False):
        """Raises ValueError if point is outside the mesh.

        Parameters
        ----------
        point : (3,) array_like
            The mesh point coordinate :math:`(p_{x}, p_{y}, p_{z})`.

        Returns
        -------
            None
                If `point` is inside the mesh.

        Raises
        ------
            ValueError
                If `point` is outside the mesh domain.

        Example
        -------
        Checking if point is outside the mesh.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> point1 = (1, 1, 1)
        >>> mesh._isoutside(point1)  # Nothing is returned.
        >>> point2 = (1, 3, 1)
        >>> mesh._isoutside(point2)
        Traceback (most recent call last):
        ...
        ValueError: ...

        """
        if np.logical_or(np.less(point, self.pmin),
                         np.greater(point, self.pmax)).any():
            if raise_exception:
                msg = 'Point is outside the mesh.'
                raise ValueError(msg)
            else:
                return False
        else:
            return True

    def line(self, p1, p2, n=100):
        """Line generator.

        Given two points :math:`p_{1}` and :math:`p_{2}`, :math:`n`
        position vectors are generated.

        .. math::

           \\mathbf{r}_{i} = i\\frac{\\mathbf{p}_{2} -
           \\mathbf{p}_{1}}{n-1}

        and this method yields :math:`\\mathbf{r}_{i}` in :math:`n`
        iterations.

        Parameters
        ----------
            p1, p2 : (3,) array_like
                Two points between which the line is generated.
            n : int
                Number of points on the line.

        Yields
        ------
            tuple
                :math:`\\mathbf{r}_{i}`

        Raises
        ------
            ValueError
                If `p` is outside the mesh domain.

        Example
        -------
        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> tuple(mesh.line(p1=(0, 0, 0), p2=(2, 0, 0), n=2))
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))

        """
        self.isinside(p1, raise_exception=True)
        self.isinside(p2, raise_exception=True)

        p1, p2 = np.array(p1), np.array(p2)
        dl = (p2-p1) / (n-1)
        for i in range(n):
            yield dfu.array2tuple(p1+i*dl)

    def plane(self, *args, x=None, y=None, z=None, n=None):
        info = dfu.plane_info(*args, x=x, y=y, z=z)

        if info["point"] is None:
            info["point"] = self.centre[info["slice"]]
        else:
            test_point = list(self.centre)
            test_point[info["slice"]] = info["point"]
            self.isinside(test_point, raise_exception=True)

        if n is None:
            n = (self.n[info["haxis"]], self.n[info["vaxis"]])

        dhaxis, dvaxis = self.l[info["haxis"]]/n[0], self.l[info["vaxis"]]/n[1]
        haxis = np.linspace(self.pmin[info["haxis"]]+dhaxis/2,
                            self.pmax[info["haxis"]]-dhaxis/2, n[0])
        vaxis = np.linspace(self.pmin[info["vaxis"]]+dvaxis/2,
                            self.pmax[info["vaxis"]]-dvaxis/2, n[1])

        for x, y in itertools.product(haxis, vaxis):
            point = 3*[None]
            point[info["slice"]] = info["point"]
            point[info["haxis"]] = x
            point[info["vaxis"]] = y
            yield tuple(point)

    def plot(self):
        """Plots the mesh domain and the discretisation cell.

        This function is called as a display function in Jupyter
        notebook.

        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal")

        dfu.plot_box(ax, self.pmin, self.pmax, "b-", linewidth=1.5)
        cell_point = tuple(pmini+celli for pmini, celli
                           in zip(self.pmin, self.cell))
        dfu.plot_box(ax, self.pmin, cell_point, "r-", linewidth=1)
        ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$z$")

    def _ipython_display_(self):
        """Jupyter notebook mesh representation.

        Mesh domain and discretisation cell are plotted by the
        :py:func:`~discretisedfield.Mesh.plot`.

        """
        self.plot()  # pragma: no cover

    def plot3d(self, k3d_plot=None, **kwargs):
        """Plots the mesh domain and the discretisation cell on 3D.

        This function is called as a display function in Jupyter
        notebook.

        Parameters
        ----------
        k3d_plot : k3d.plot.Plot, optional
            We transfer a k3d.plot.Plot object to add the current 3d figure
            to the canvas(?).

        """
        plot_array = np.ones(tuple(reversed((self.n))))
        plot_array[0, 0, -1] = 2  # mark the discretisation cell
        k3d_vox(plot_array, self, k3d_plot=k3d_plot, **kwargs)

    def plot3d_coordinates(self, k3d_plot=None, **kwargs):
        """Plots the mesh coordinates.

        This function is called as a display function in Jupyter
        notebook.

        Parameters
        ----------
        k3d_plot : k3d.plot.Plot, optional
            We transfer a k3d.plot.Plot object to add the current 3d figure
            to the canvas(?).

        """
        plot_array = np.array(list(self.coordinates))
        k3d_points(plot_array, k3d_plot=k3d_plot, **kwargs)

    @property
    def _script(self):
        """This abstract method should be implemented by a specific
        calculator.

        Raises
        ------
        NotImplementedError
            If not implemented by a specific calculator.

        """
        raise NotImplementedError
