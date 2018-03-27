import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import joommfutil.typesystem as ts
import discretisedfield.util as dfu
from mpl_toolkits.mplot3d import Axes3D


@ts.typesystem(p1=ts.ConstantRealVector(size=3),
               p2=ts.ConstantRealVector(size=3),
               cell=ts.ConstantPositiveRealVector(size=3),
               pbc=ts.FromCombinations(sample_set="xyz"),
               name=ts.ConstantObjectName)
class Mesh:
    """Finite difference rectangular mesh.

    The rectangular mesh domain spans between two points `p1` and
    `p2`. They are defined as ``array_like`` objects of length 3
    (e.g. ``tuple``, ``list``, ``numpy.ndarray``), :math:`p = (p_{x},
    p_{y}, p_{z})`. The domain is then discretised into finite
    difference cells, where dimensions of a single cell are defined
    with a `cell` parameter. Similar to `p1` and `p2`, `cell`
    parameter is an ``array_like`` object :math:`(d_{x}, d_{y},
    d_{z})` of length 3. The parameter `name` is optional and defaults
    to "mesh".

    In order to properly define a mesh, the length of all mesh domain
    edges must be positive, and the mesh domain must be an aggregate
    of discretisation cells.

    Parameters
    ----------
    p1, p2 : (3,) array_like
        Points between which the mesh domain spans :math:`p = (p_{x},
        p_{y}, p_{z})`.
    cell : (3,) array_like
        Discretisation cell size :math:`(d_{x}, d_{y}, d_{z})`.
    pbc : str, optional
        Periodic boundary conditions in x, y, or z direction. Its value
        is a string consisting of one or more of the letters `x`, `y`,
        or ``z, denoting the periodic direction(s).
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
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> cell = (1e-9, 1e-9, 0.1e-9)
    >>> name = "mesh_name"
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, pbc="xy", name=name)

    2. An attempt to create a mesh with invalid parameters, so that
    the ``ValueError`` is raised. In this example, the mesh domain is
    not an aggregate of discretisation cells in the :math:`z`
    direction.

    >>> import discretisedfield as df
    >>> p1 = (-25, 3, 0)
    >>> p2 = (25, 6, 1)
    >>> cell = (5, 3, 0.4)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, name=name)
    Traceback (most recent call last):
        ...
    ValueError: ...

    """
    def __init__(self, p1, p2, cell, pbc=None, name="mesh"):
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.cell = tuple(cell)
        self.pbc = pbc
        self.name = name

        # Is the length of any mesh domain edges zero?
        for i, li in enumerate(self.l):
            if li == 0:
                msg = "Mesh domain edge length is zero (l[{}]==0).".format(i)
                raise ValueError(msg)

        # Is the mesh domain not an aggregate of discretisation cells?
        tol = 1e-12  # picometre tolerance
        for i, (li, celli) in enumerate(zip(self.l, self.cell)):
            if tol < li % celli < celli - tol:
                msg = ("Mesh domain is not an aggregate of the discretisation "
                       "cell: abs(p2[{0}]-p1[{0}]) % cell[{0}].").format(i)
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
        Getting the minimum mesh point.

        >>> import discretisedfield as df
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> cell = (0.1, 0.1, 0.005)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.pmin
        (-1.1, 0, -0.1)

        .. note::

           Please note this method is a property and should be called
           as ``mesh.pmin``, not ``mesh.pmin()``.

        .. seealso:: :py:func:`~discretisedfield.Mesh.pmax`

        """
        return tuple(map(min, zip(self.p1, self.p2)))

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
        Getting the maximum mesh point.

        >>> import discretisedfield as df
        >>> p1 = (-1.1, 2.9, 0)
        >>> p2 = (5, 0, -0.1)
        >>> cell = (0.1, 0.1, 0.005)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.pmax
        (5, 2.9, 0)

        .. note::

           Please note this method is a property and should be called
           as ``mesh.pmax``, not ``mesh.pmax()``.

        .. seealso:: :py:func:`~discretisedfield.Mesh.pmin`

        """
        return tuple(map(max, zip(self.p1, self.p2)))

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
        Getting mesh domain edge lengths.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, -5)
        >>> p2 = (5, 15, 15)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.l
        (5, 15, 20)

        .. note::

           Please note this method is a property and should be called
           as ``mesh.l``, not ``mesh.l()``.

        """
        return tuple(abs(p1i-p2i) for p1i, p2i in zip(self.p1, self.p2))

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
        Getting the number of discretisation cells in all three
        directions.

        >>> import discretisedfield as df
        >>> p1 = (0, 5, 0)
        >>> p2 = (5, 15, 1)
        >>> cell = (0.5, 0.1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.n
        (10, 100, 1)

        .. note::

           Please note this method is a property and should be called
           as ``mesh.n``, not ``mesh.n()``.

        """
        return tuple(int(round(li/di)) for li, di in zip(self.l, self.cell))

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
        Getting the mesh centre point.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 15, 20)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.centre
        (2.5, 7.5, 10.0)

        .. note::

           Please note this method is a property and should be called
           as ``mesh.centre``, not ``mesh.centre()``.

        """
        return tuple(pmini+0.5*li for pmini, li in zip(self.pmin, self.l))

    @property
    def indices(self):
        """Generator iterating through all mesh cells and yielding their
        indices.

        Yields
        ------
        tuple (3,)
            Mesh cell indices :math:`(i_{x}, i_{y}, i_{z})`.

        Example
        -------
        Getting all mesh cell indices.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (3, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> tuple(mesh.indices)
        ((0, 0, 0), (1, 0, 0), (2, 0, 0), (0, 1, 0), (1, 1, 0), (2, 1, 0))

        .. note::

           Please note this method is a property and should be called
           as ``mesh.indices``, not ``mesh.indices()``.

        .. seealso:: :py:func:`~discretisedfield.Mesh.coordinates`

        """
        for index in itertools.product(*map(range, reversed(self.n))):
            yield tuple(reversed(index))

    @property
    def coordinates(self):
        """Generator iterating through all mesh cells and yielding their
        coordinates.

        The discretisation cell coordinate corresponds to the cell
        centre point.

        Yields
        ------
        tuple (3,)
            Mesh cell coordinates (`px`, `py`, `pz`).

        Example
        -------
        Getting all mesh cell coordinates.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> tuple(mesh.coordinates)
        ((0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 1.5, 0.5), (1.5, 1.5, 0.5))

        .. note::

           Please note this method is a property and should be called
           as ``mesh.coordinates``, not ``mesh.coordinates()``.

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        for index in self.indices:
            yield self.index2point(index)

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
        Getting mesh representation string.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> pbc = "xy"
        >>> name = "m"
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, pbc=pbc, name=name)
        >>> repr(mesh)
        'Mesh(p1=(0, 0, 0), p2=(2, 2, 1), cell=(1, 1, 1), pbc="xy", name="m")'

        """
        if self.pbc is not None:
            pbc = "\"{}\"".format("".join(sorted(self.pbc)))
        else:
            pbc = self.pbc
        return ("Mesh(p1={}, p2={}, cell={}, pbc={}, "
                "name=\"{}\")").format(self.p1, self.p2, self.cell,
                                       pbc, self.name)

    def random_point(self):
        """Generate the random point belonging to the mesh.

        Returns
        -------
        tuple (3,)
            Coordinates of a random point inside that belongs to the mesh
            :math:`(x_\\text{rand}, y_\\text{rand}, z_\\text{rand})`.

        Example
        -------
        Getting mesh representation string.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (200e-9, 200e-9, 1e-9)
        >>> cell = (1e-9, 1e-9, 0.5e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.random_point()
        (...)

        .. note::

           In the example, ellipsis is used instead of an exact tuple
           because the result differs each time the ``random_point``
           is called.

        """
        return tuple(pmini+random.random()*li
                     for pmini, li in zip(self.pmin, self.l))

    def index2point(self, index):
        """Convert the cell index to its centre's coordinate.

        Parameters
        ----------
        index : (3,) array_like
            The discretisation cell index :math:`(i_{x}, i_{y}, i_{z})`.

        Returns
        -------
            The discretisation cell centre point :math:`(p_{x}, p_{y}, p_{z})`.

        Raises
        ------
            ValueError
                If the discretisation cell index is out of range.

        Example
        -------
        Converting cell index to its coordinate.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.index2point((0, 1, 0))
        (0.5, 1.5, 0.5)

        .. seealso:: :py:func:`~discretisedfield.Mesh.point2index`

        """
        # Does index refer to a cell outside the mesh?
        for i, (indexi, ni) in enumerate(zip(index, self.n)):
            if indexi > ni - 1 or indexi < 0:
                msg = ("index[{}]={} out of range "
                       "[0, n[{}]].").format(i, indexi, ni)
                raise ValueError(msg)
        return tuple(pmini+(indexi+0.5)*celli for pmini, indexi, celli
                     in zip(self.pmin, index, self.cell))

    def point2index(self, p):
        """Convert the mesh coordinate to the cell index which contains it.

        Parameters
        ----------
        p : (3,) array_like
            The mesh point coordinate :math:`(p_{x}, p_{y}, p_{z})`.

        Returns
        -------
            The discretisation cell index :math:`(i_{x}, i_{y}, i_{z})`.

        Raises
        ------
            ValueError
                If `point` is outside the mesh domain.

        Example
        -------
        Converting point coordinate to the cell index.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.point2index((0.2, 1.7, 0.3))
        (0, 1, 0)

        .. seealso:: :py:func:`~discretisedfield.Mesh.index2point`

        """
        self._isoutside(p)

        index = tuple(int(round((pi-pmini)/celli - 0.5))
                      for pi, pmini, celli in zip(p, self.pmin, self.cell))

        # If rounded to the out-of-range values
        return tuple(max(min(ni-1, indexi), 0)
                     for ni, indexi in zip(self.n, index))

    def _isoutside(self, p):
        """Raises ValueError if point is outside the mesh.

        Parameters
        ----------
        p : (3,) array_like
            The mesh point coordinate :math:`(p_{x}, p_{y}, p_{z})`.

        Returns
        -------
            None
                If `point` is inside the mesh.

        Raises
        ------
            ValueError
                If `p` is outside the mesh domain.

        Example
        -------
        Checking if point is outside the mesh.

        >>> import discretisedfield as df
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
        for i, (pi, pmini, pmaxi) in enumerate(zip(p, self.pmin, self.pmax)):
            if pi < pmini or pi > pmaxi:
                msg = "Point p[{}]={} outside the mesh.".format(i, pi)
                raise ValueError(msg)

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
        self._isoutside(p1)
        self._isoutside(p2)

        p1, p2 = np.array(p1), np.array(p2)
        dl = (p2-p1) / (n-1)
        for i in range(n):
            yield tuple(p1+i*dl)

    def plane(self, *args, x=None, y=None, z=None, n=None):
        info = dfu.plane_info(*args, x=x, y=y, z=z)

        if info["point"] is None:
            info["point"] = self.centre[info["slice"]]
        else:
            test_point = list(self.centre)
            test_point[info["slice"]] = info["point"]
            self._isoutside(test_point)

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
