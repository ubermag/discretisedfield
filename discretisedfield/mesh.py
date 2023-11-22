import collections
import contextlib
import itertools
import numbers
import warnings
from numbers import Integral, Number

import ipywidgets
import numpy as np
import scipy.fft as spfft
import ubermagutil.units as uu

import discretisedfield as df
import discretisedfield.plotting as dfp
import discretisedfield.util as dfu

from . import html
from .io import _MeshIO


class Mesh(_MeshIO):
    """Finite-difference mesh.

    Mesh discretises the ``discretisedfield.Region``, passed as ``region``,
    using a regular finite-difference mesh. Since the region spans between
    two points :math:`\\mathbf{p}_{1}` and :math:`\\mathbf{p}_{2}`, these
    points can be passed as ``p1`` and ``p2``, instead of passing
    ``discretisedfield.Region`` object. In this case
    ``discretisedfield.Region`` is created internally. Either ``region`` or
    ``p1`` and ``p2`` can be passed, not both. The region is discretised using
    a finite-difference cell, whose dimensions are defined with ``cell``.
    Alternatively, the domain can be discretised by passing the number of
    discretisation cells ``n`` in all three dimensions. Either ``cell`` or
    ``n`` can be passed, not both.

    It is possible to define boundary conditions (bc) for the mesh by passing a string
    to ``bc``.

    If it is necessary to define subregions in the mesh, a dictionary can be
    passed using ``subregions``. More precisely, dictionary keys are strings
    (valid Python variable names), whereas values are
    ``discretisedfield.Region`` objects. It is necessary that subregions belong
    to the mesh region, are an aggregate of a discretisation cell, and are
    "aligned" with the mesh. If not, ``ValueError`` is raised.

    In order to properly define a mesh, mesh region must be an aggregate of
    discretisation cells. Otherwise, ``ValueError`` is raised.

    Parameters
    ----------
    region : discretisedfield.Region, optional

        Cubic region to be discretised on a regular mesh. Either ``region`` or
        ``p1`` and ``p2`` should be defined, not both. Defaults to ``None``.

    p1 / p2 : array_like, optional

        Diagonally-opposite region points, for example for three dimensions
        :math:`\\mathbf{p} = (p_{x}, p_{y}, p_{z})`. Either ``region`` or ``p1`` and
        ``p2`` should be defined, not both. Defaults to ``None``.

    cell : array_like, optional

        Discretisation cell size, for example for three dimensions
        :math:`(d_{x}, d_{y}, d_{z})`. Either ``cell`` or ``n`` should be defined, not
        both. Defaults to ``None``.

    n : array_like, optional

        The number of discretisation cells, for example for three dimensions
        :math:`(n_{x}, n_{y}, n_{z})`. Either ``cell`` or ``n`` should be defined, not
        both. Defaults to ``None``.

    bc : str, optional

        Periodic boundary conditions in geometrical directions. It is a string
        consisting of one or more characters representing the name of the direction(s)
        as present in ``self.region.dims``, denoting the direction(s) along which the
        mesh is periodic. In the case of Neumann or Dirichlet boundary condition, string
        ``'neumann'`` or ``'dirichlet'`` is passed. Defaults to an empty string.

    subregions : dict, optional

        A dictionary defining subregions in the mesh. The keys of the
        dictionary are the region names (``str``) as valid Python variable
        names, whereas the values are ``discretisedfield.Region`` objects.
        Defaults to an empty dictionary.

    Raises
    ------
    ValueError

        If mesh domain is not an aggregate of discretisation cells.
        Alternatively, if both ``region`` as well as ``p1`` and ``p2`` or both
        ``cell`` and ``n`` are passed.

        Alternatively if one of the subregions is: (i) not in the mesh region,
        (ii) it is not an aggregate of discretisation cell, or (iii) it is not
        aligned with the mesh.

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

    3. Defining a mesh with periodic boundary conditions in :math:`x` and
    :math:`y` directions.

    >>> bc = 'xy'
    >>> region = df.Region(p1=p1, p2=p2)
    >>> mesh = df.Mesh(region=region, n=n, bc=bc)
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

    6. An attempt to define a mesh, whose subregion is not aligned.

    >>> p1 = (0, 0, 0)
    >>> p2 = (100, 100, 100)
    >>> cell = (10, 10, 10)
    >>> subregions = {'r1': df.Region(p1=(2, 0, 0), p2=(52, 100, 100))}
    >>> mesh = df.Mesh(p1=p1, p2=p2, subregions=subregions)
    Traceback (most recent call last):
        ...
    ValueError: ...

    """

    __slots__ = ["_region", "_n", "_bc", "_subregions"]

    # removed attribute: new method/property
    # implemented in __getattr__
    # to exclude methods from tap completion and documentation
    _removed_attributes = {"midpoints": "cells", "points": "cells"}

    def __init__(
        self,
        *,
        region=None,
        p1=None,
        p2=None,
        n=None,
        cell=None,
        bc="",
        subregions=None,
    ):
        # TODO NO MUTABLE DEFAULT
        if region is not None and p1 is None and p2 is None:
            if not isinstance(region, df.Region):
                raise TypeError("region must be of class discretisedfield.Region.")
            self._region = region
        elif region is None and p1 is not None and p2 is not None:
            self._region = df.Region(p1=p1, p2=p2)
        else:
            raise ValueError(
                "region, p1, and p2 cannot be None or passed simultaneously. Either"
                " pass region or both p1 and p2."
            )

        if cell is not None and n is None:
            # scalar data types for 1d regions
            if isinstance(cell, numbers.Real):
                cell = [cell]

            if not isinstance(cell, (tuple, list, np.ndarray)):
                raise TypeError(
                    "Cell must be either a tuple, a list, or a numpy.ndarray."
                )
            if len(cell) != self.region.ndim:
                raise ValueError("The cell must have same dimensions as the region.")
            elif not all(isinstance(i, Number) for i in cell):
                raise TypeError("The values of cell must be numbers.")
            elif not all(i > 0 for i in cell):
                raise ValueError("The values of cell must be positive numbers.")
            # Check if the cell size exceeds the region size
            if (
                df.Region(p1=self.region.pmin, p2=self.region.pmin + cell)
                not in self.region
            ):
                raise ValueError(
                    f"The cell size ({cell=}) exceeds the region size ({self.region=})."
                )
            # Check if the mesh region is an aggregate of the discretisation cell.
            tol = np.min(cell) * 1e-3  # tolerance
            rem = np.remainder(self.region.edges, cell)
            if np.logical_and(
                np.greater(rem, tol), np.less(rem, np.subtract(cell, tol))
            ).any():
                raise ValueError(
                    "Region cannot be divided into "
                    f"discretisation cells of size {cell=}."
                )
            self._n = np.divide(self.region.edges, cell).round().astype(int)

        elif n is not None and cell is None:
            # scalar data types for 1d regions
            if isinstance(n, numbers.Real):
                n = [n]
            if not isinstance(n, (tuple, list, np.ndarray)):
                raise TypeError("n must be either a tuple, a list or a numpy.ndarray.")
            if len(n) != self.region.ndim:
                raise ValueError("n must have same dimensions as the region.")
            elif not all(isinstance(i, Integral) for i in n):
                raise TypeError("The values of n must be integers.")
            elif not all(i > 0 for i in n):
                raise ValueError("The values of n must be positive integers.")
            self._n = np.array(n, dtype=int)

        else:
            raise ValueError(
                "Both n and cell cannot be None or passed simultaneously. Either pass n"
                " or cell."
            )

        self.bc = bc

        self.subregions = subregions

    @property
    def bc(self):
        """Boundary condition for the mesh.

        Periodic boundary conditions can be specified by passing a string containing one
        or more characters from ``self.region.dims`` (e.g. ``'x'``, ``'yz'``, ``'xyz'``
        for three dimensions). Neumann or Dirichlet boundary conditions are defined by
        passing ``'neumann'`` or ``'dirichlet'`` string. Neumann and Dirichlet boundary
        conditions are still experimental.

        Returns
        -------
        str

            A string representing periodic boundary condition along one or more axes, or
            Dirichlet or Neumann boundary condition. The string is empty if no boundary
            condition is defined.
        """
        return self._bc

    @bc.setter
    def bc(self, bc):
        if not isinstance(bc, str):
            raise TypeError("Value of bc must be a string.")
        bc = bc.lower()
        if bc not in {"neumann", "dirichlet", ""}:
            for char in bc:
                if char not in self.region.dims:
                    raise ValueError(f"Axis {char} is absent in {self.region.dims}.")
                elif bc.count(char) > 1:
                    raise ValueError(f"Axis {char} is present more than once.")

        self._bc = bc

    @property
    def cell(self):
        """The cell size of the mesh.

        Returns
        -------
        numpy.ndarray

            A numpy array representing discretisation size along respective axes.
        """
        return np.divide(self.region.edges, self.n).astype(float)

    @property
    def n(self):
        """Number of cells along each dimension of the mesh.

        Returns
        -------
        numpy.ndarray

            A numpy array representing number of discretisation cells along respective
            axes.
        """
        return self._n

    @property
    def region(self):
        """Region on which the mesh is defined.

        Returns
        -------
        discretisedfield.Region

            A region over which the regular mesh is defined.
        """
        return self._region

    @property
    def subregions(self):
        """Subregions of the mesh.

        When setting subregions all attributes of the individual regions (e.g. dims)
        apart from ``pmin`` and ``pmax`` will be overwritten with the values from
        ``mesh.region``.

        Returns
        -------
        dict

            A dictionary defining subregions in the mesh. The keys of the
            dictionary are the region names (``str``) as valid Python variable
            names, whereas the values are ``discretisedfield.Region`` objects.

        """
        return self._subregions

    @subregions.setter
    def subregions(self, subregions):
        if subregions is None:
            subregions = {}

        if not isinstance(subregions, dict):
            raise TypeError(
                "Subregions must be a dictionary relating the name of a subregion"
                " with its region."
            )

        if not all(isinstance(key, str) for key in subregions):
            raise TypeError("The keys of subregion dictionary must be strings.")

        # Check if subregions are aligned with the mesh
        for key, value in subregions.items():
            # Is the subregion in the mesh region?
            if value not in self.region:
                raise ValueError(f"Subregion {key} is not in the mesh region.")

            # Is the subregion an aggregate of discretisation cell?
            try:
                self.__class__(region=value, cell=self.cell)
            except ValueError:
                msg = (
                    f"Subregion {key} cannot be divided into "
                    f"discretisation cells of size {self.cell=}."
                )
                raise ValueError(msg)

            # Is the subregion aligned with the mesh?
            if not self.is_aligned(self.__class__(region=value, cell=self.cell)):
                raise ValueError(f"Subregion {key} is not aligned with the mesh.")
        if "default" in subregions.keys():
            warnings.warn(
                "Subregion name ``default`` has a special meaning when "
                "initialising field values"
            )
        self._subregions = {
            name: df.Region(
                p1=sr.pmin,
                p2=sr.pmax,
                dims=self.region.dims,
                units=self.region.units,
                tolerance_factor=self.region.tolerance_factor,
            )
            for name, sr in subregions.items()
        }

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
        array([  5, 100,   2])
        >>> len(mesh)
        1000

        """
        return int(np.prod(self.n))

    @property
    def indices(self):
        """Generator yielding indices of all mesh cells.

        Yields
        ------
        tuple

            For three dimensions, mesh cell indices :math:`(i_{x}, i_{y}, i_{z})`.

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
        """Generator yielding coordinates of discretisation cells.

        The discretisation cell's coordinate corresponds to its center point.

        Yields
        ------
        numpy.ndarray

            For three dimensions, mesh cell's center point
            :math:`\\mathbf{p} = (p_{x}, p_{y}, p_{z})`.

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
        [array([0.5, 0.5, 0.5]), array([1.5, 0.5, 0.5]), array([0.5, 1.5, 0.5]),...]

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        yield from map(self.index2point, self.indices)

    @property
    def cells(self):
        """Midpoints of the cells of the mesh along the spatial directions.

        This method returns a named tuple containing numpy arrays with midpoints of the
        cells along the spatial directions. Individual directions can be accessed from
        the tuple.

        Returns
        -------
        collections.namedtuple

            Namedtuple with elements corresponding to geometrical directions, the cell
            midpoints along the directions as numpy arrays.

        Examples
        --------
        1. Getting midpoints along the ``x`` axis.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 1, 1)
        >>> cell = (2, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        ...
        >>> mesh.cells.x
        array([1., 3., 5., 7., 9.])

        """
        cells = collections.namedtuple("cells", self.region.dims)

        return cells(
            *(
                np.linspace(pmin + cell / 2, pmax - cell / 2, n)
                for pmin, pmax, cell, n in zip(
                    self.region.pmin, self.region.pmax, self.cell, self.n
                )
            )
        )

    @property
    def vertices(self):
        """Vertices of the cells of the mesh along the spatial directions.

        This method returns a named tuple containing numpy arrays with vertices of the
        cells along the spatial directions. Individual directions can be accessed from
        the tuple.

        Returns
        -------
        collections.namedtuple

            Namedtuple with elements corresponding to spatial directions, the cell
            vertices along the directions as numpy arrays.

        Examples
        --------
        1. Getting vertices along the ``x`` axis.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 1, 1)
        >>> cell = (2, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        ...
        >>> mesh.vertices.x
        array([ 0.,  2.,  4.,  6.,  8., 10.])

        """
        vertices = collections.namedtuple("vertices", self.region.dims)

        return vertices(
            *(
                np.linspace(pmin, pmax, n + 1)
                for pmin, pmax, n in zip(self.region.pmin, self.region.pmax, self.n)
            )
        )

    def __eq__(self, other):
        """Relational operator ``==``.

        Two meshes are considered to be equal if:

          1. Regions of both meshes are equal.

          2. Discretisation cell sizes are the same.

        Boundary conditions ``bc`` and ``subregions`` are not considered to be
        necessary conditions for determining equality.

        Parameters
        ----------
        other : discretisedfield.Mesh

            Second operand.

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

        """
        if not isinstance(other, self.__class__):
            return False
        if self.region == other.region and all(self.n == other.n):
            return True
        else:
            return False

    def allclose(self, other, rtol=None, atol=None):
        """Check if the mesh is close enough to the other based on a tolerance.

        This methods compares the two underlying regions using ``Region.allclose`` and
        the number of cells ``n`` of the two meshes. The value of relative tolerance
        (``rtol``) and absolute tolerance (``atol``) are passed on to
        ``Region.allclose`` for the comparison. If not provided default values of
        ``Region.allclose`` are used.

        Parameters
        ----------
        other : discretisedfield.Mesh

            The other mesh used for comparison.

        rtol : numbers.Real, optional

            Absolute tolerance. If ``None``, the default value is
            the smallest edge length of the region multipled by
            the ``region.tolerance_factor``.

        atol : numbers.Real, optional

            Relative tolerance. If ``None``, ``region.tolerance_factor`` is used.

        Returns
        -------
        bool

            ``True`` if other mesh is close enough, otherwise ``False``.

        Raises
        ------
        TypeError

            If the ``other`` argument is not of type ``discretisedfield.Mesh`` or if
            ``rtol`` and ``atol`` arguments are not of type ``numbers.Real``.

        ValueError

            If the dimensions of the mesh and the other mesh does not match.

        Example
        -------
        >>> p1 = (0, 0, 0)
        >>> p2 = (20e-9, 20e-9, 20e-9)
        >>> n = (10, 10, 10)
        >>> mesh1 = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (20e-9 + 1.2e-12, 20e-9 + 1e-13, 20e-9 + 2e-12)
        >>> n = (10, 10, 10)
        >>> mesh2 = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> mesh1.allclose(mesh2, atol=1e-11)
        True
        >>> mesh1.allclose(mesh2, atol=1e-13)
        False

        """

        if not isinstance(other, df.Mesh):
            raise TypeError(
                f"Expected argument of type discretisedfield.Mesh but got {type(other)}"
            )

        if self.region.dims != other.region.dims:
            raise ValueError("The mesh dimensions do not match.")

        return self.region.allclose(
            other.region, rtol=rtol, atol=atol
        ) and np.array_equal(self.n, other.n)

    def __repr__(self):
        """Representation string.

        Internally `self._repr_html_()` is called and all html tags are removed
        from this string.

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
        >>> cell = (1, 1, 1)
        >>> bc = 'x'
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, bc=bc)
        >>> mesh
        Mesh(Region(pmin=[0, 0, 0], pmax=[2, 2, 1], ...), n=[2, 2, 1], bc=x)

        """
        return html.strip_tags(self._repr_html_())

    def _repr_html_(self):
        """Show HTML-based representation in Jupyter notebook."""
        return html.get_template("mesh").render(mesh=self)

    def index2point(self, index, /):
        """Convert cell's index to its coordinate.

        Parameters
        ----------
        index : array_like

            For three dimensions, the cell's index :math:`(i_{x}, i_{y}, i_{z})`.

        Returns
        -------
        numpy.ndarray

            For three dimensions, the cell's coordinate
            :math:`\\mathbf{p} = (p_{x}, p_{y}, p_{z})`.

        Raises
        ------
        ValueError

            If ``index`` is out of range.

        Examples
        --------
        1. Converting cell's index to its center point coordinate.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.index2point((0, 0, 0))
        array([0.5, 0.5, 0.5])
        >>> mesh.index2point((0, 1, 0))
        array([0.5, 1.5, 0.5])

        .. seealso:: :py:func:`~discretisedfield.Mesh.point2index`

        """
        if isinstance(index, numbers.Integral):
            index = [index]
        elif isinstance(index, (np.ndarray, list, tuple)):
            if any(not isinstance(i, numbers.Integral) for i in index):
                raise TypeError(f"The elements of {index=} must be integer.")
        else:
            raise TypeError(
                f"The index is of the wrong type {type(index)=}. It must be an integer"
                " (1D) or a tuple/list/array of integers."
            )

        if len(index) != self.region.ndim:
            raise IndexError(
                f"Wrong dimensional index. {index=} but {self.region.ndim=}."
            )

        if np.logical_or(np.less(index, 0), np.greater_equal(index, self.n)).any():
            raise IndexError(f"Index {index=} out of range.")

        point = self.region.pmin + np.add(index, 0.5) * self.cell
        return point

    def point2index(self, point, /):
        """Convert point to the index of a cell which contains that point.

        This method uses half-open intervals for each cell,
        inclusive of the start point but exclusive of the endpoints.
        i.e. for each cell [).
        The exception to this is the very last cell contained in the region
        which has a closed interval i.e. [] and is inclusive of both the
        lower and upper bounds of the cell.

        Parameters
        ----------
        point : array_like

            For three dimensions, point :math:`\\mathbf{p} = (p_{x}, p_{y}, p_{z})`.

        Returns
        -------
        tuple

            For three dimensions, the cell's index :math:`(i_{x}, i_{y}, i_{z})`.

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
        if isinstance(point, (tuple, list, np.ndarray)):
            if any(not isinstance(i, numbers.Real) for i in point):
                raise TypeError(
                    f"The elements of point {point=} must be of type numbers.Real."
                )
        elif isinstance(point, numbers.Real):
            point = [point]
        else:
            raise TypeError(
                f"The point is of the wrong type {type(point)=}. It must be an integer"
                " (1D) or a tuple/list/array of integers."
            )

        if len(point) != self.region.ndim:
            raise ValueError(
                f"Wrong dimensional point. {point=} but {self.region.ndim=}."
            )

        if point not in self.region:
            raise ValueError(f"Point {point} is outside the region {self.region=}.")

        index = np.floor((point - self.region.pmin) / self.cell).astype(int)
        # If index is rounded to the out-of-range values.
        index = np.clip(index, 0, self.n - 1)

        return tuple(index)

    def region2slices(self, region):
        """Slices of indices that correspond to cells contained in the region.

        Parameters
        ----------
        region : df.Region

            Region to convert to slices.

        Returns
        -------
        tuple

            Tuple of slices of region indices.

        Examples
        --------
        1. Slices of a subregion
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 1)
        >>> cell = (1, 1, 1)
        >>> subregions = {'sr': df.Region(p1=p1, p2=(10, 5, 1))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        >>> mesh.region2slices(mesh.subregions['sr'])
        (slice(0, 10, None), slice(0, 5, None), slice(0, 1, None))
        """

        i1 = self.point2index(region.pmin + self.cell / 2)
        i2 = self.point2index(region.pmax - self.cell / 2)
        return tuple(slice(i1[i], i2[i] + 1) for i in range(self.region.ndim))

    def line(self, *, p1, p2, n):
        """Line generator.

        Given two points ``p1`` and ``p2`` line is defined and ``n`` points on
        that line are generated and yielded in ``n`` iterations:

        .. math::

           \\mathbf{r}_{i} = i\\frac{\\mathbf{p}_{2} - \\mathbf{p}_{1}}{n-1},
           \\text{for}\\, i = 0, ..., n-1

        Parameters
        ----------
        p1 / p2 : array_like

            For three dimensions, points between which the line is defined
            :math:`\\mathbf{p} = (p_{x}, p_{y}, p_{z})`.

        n : int

            Number of points on the line.

        Yields
        ------
        tuple

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
            msg = f"Point {p1=} or point {p2=} is outside the mesh region."
            raise ValueError(msg)

        dl = np.subtract(p2, p1) / (n - 1)
        for i in range(n):
            yield dfu.array2tuple(np.add(p1, i * dl))

    def sel(self, *args, **kwargs):
        """Select a part of the mesh.

        If one of the axis from ``region.dims`` is passed as a string, a mesh of a
        reduced dimension along the axis and perpendicular to it is extracted,
        intersecting the axis at its center. Alternatively, if a keyword (representing
        the axis) argument is passed with a real number value (e.g. ``x=1e-9``), a mesh
        of reduced dimensions intersects the axis at a point 'nearest' to the provided
        value is returned. If instead a tuple, list or a numpy array of length 2 is
        passed as a value containing two real numbers (e.g. ``x=(1e-9, 7e-9)``), a sub
        mesh is returned with minimum and maximum points along the selected axis,
        'nearest' to the minimum and maximum of the selected values, respectively.

        Parameters
        ----------
        args :

            A string corresponding to the selection axis that belongs to
            ``region.dims``.

        kwarg :

            A key corresponding to the selection axis that belongs to ``region.dims``.
            The values are either a ``numbers.Real`` or list, tuple, numpy array of
            length 2 containing ``numbers.Real`` which represents a point or a range of
            points to be selected from the mesh.

        Returns
        -------
        discretisedfield.Mesh

            An extracted mesh.

        Examples
        --------
        1. Extracting the mesh at a specific point (``y=1``).

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> mesh.region.ndim
        3
        >>> mesh.region.dims
        ('x', 'y', 'z')
        >>> plane_mesh = mesh.sel(y=1)
        >>> plane_mesh.region.ndim
        2
        >>> plane_mesh.region.dims
        ('x', 'z')

        2. Extracting the xy-plane mesh at the mesh region center.

        >>> plane_mesh = mesh.sel('z')
        >>> plane_mesh.region.ndim
        2
        >>> plane_mesh.region.dims
        ('x', 'y')

        3. Specifying a range of points along axis ``x`` to be selected from mesh.

        >>> selected_mesh = mesh.sel(x=(2, 4))
        >>> selected_mesh.region.ndim
        3
        >>> selected_mesh.region.dims
        ('x', 'y', 'z')

        """
        dim, dim_index, selection, _ = self._sel_convert_input(*args, **kwargs)

        sub_region = dict()
        if isinstance(selection, numbers.Real):
            idxs = [i for i in range(self.region.ndim) if i != dim_index]
            p_1 = list()
            p_2 = list()
            cell = list()
            dims = list()
            units = list()
            for j in idxs:
                p_1.append(self.region.pmin[j])
                p_2.append(self.region.pmax[j])
                cell.append(self.cell[j])
                dims.append(self.region.dims[j])
                units.append(self.region.units[j])

            if self.subregions is not None:
                for key, subreg in self.subregions.items():
                    if (
                        selection > subreg.pmax[dim_index]
                        or selection < subreg.pmin[dim_index]
                    ):
                        continue
                    else:
                        sub_p_1 = list()
                        sub_p_2 = list()
                        for j in idxs:
                            sub_p_1.append(subreg.pmin[j])
                            sub_p_2.append(subreg.pmax[j])
                        sub_region[key] = df.Region(
                            p1=sub_p_1,
                            p2=sub_p_2,
                        )
        else:
            step = self.cell[dim_index] / 2
            p_1 = self.region.pmin.copy().astype(
                max(self.region.pmin.dtype, type(step))
            )
            p_2 = self.region.pmax.copy().astype(
                max(self.region.pmax.dtype, type(step))
            )
            min_val = selection[0] - step
            max_val = selection[1] + step
            p_1[dim_index] = min_val
            p_2[dim_index] = max_val
            cell = self.cell
            dims = self.region.dims
            units = self.region.units
            if self.subregions is not None:
                for key, subreg in self.subregions.items():
                    sub_reg_p_min = subreg.pmin[dim_index]
                    sub_reg_p_max = subreg.pmax[dim_index]
                    if sub_reg_p_min >= max_val or min_val >= sub_reg_p_max:
                        continue
                    else:
                        sub_p_1 = subreg.pmin.copy().astype(
                            max(subreg.pmin.dtype, type(min_val))
                        )
                        sub_p_2 = subreg.pmax.copy().astype(
                            max(subreg.pmax.dtype, type(max_val))
                        )
                        sub_p_1[dim_index] = max(min_val, sub_reg_p_min)
                        sub_p_2[dim_index] = min(max_val, sub_reg_p_max)
                        sub_region[key] = df.Region(
                            p1=sub_p_1,
                            p2=sub_p_2,
                        )

        return self.__class__(
            region=df.Region(
                p1=p_1,
                p2=p_2,
                dims=dims,
                units=units,
                tolerance_factor=self.region.tolerance_factor,
            ),
            cell=cell,
            subregions=sub_region,
        )

    def _sel_convert_input(self, *args, **kwargs):
        """Convert input of 'sel' into (dim, dim_index, selection, selection_index).

        The value(s) in selection are cell centre points. If a range is selected a list
        is returned for selection and a slice for selection_index. The upper boundary
        for selection_index is increased by 1 to "make the slice inclusive".

        """
        if len(args) > 1 or len(kwargs) > 1:
            raise ValueError("Select method only accepts one dimension at a time.")

        if args and not kwargs:
            dim = args[0]
            range_ = None
        elif not args and kwargs:
            dim, range_ = list(kwargs.items())[0]
        else:
            raise ValueError(
                "Either one positional argument or a keyword argument can be passed."
            )

        dim_index = self.region._dim2index(dim)

        # Check input arguments
        if range_ is not None:
            if isinstance(range_, numbers.Real):
                if (
                    range_ < self.region.pmin[dim_index]
                    or range_ > self.region.pmax[dim_index]
                ):
                    raise ValueError(
                        f"Selected value {range_} is outside the mesh region."
                    )
                test_point = self.region.pmin.copy().astype(
                    max(self.region.pmin.dtype, type(range_))
                )
                test_point[dim_index] = range_
                selection = self.index2point(self.point2index(test_point))[dim_index]
                selection_index = self.point2index(test_point)[dim_index]
            elif isinstance(range_, (tuple, list, np.ndarray)):
                if len(range_) != 2:
                    raise ValueError(
                        "The points along the selected dimension must have two"
                        " real numbers."
                    )
                elif not all(isinstance(point, numbers.Real) for point in range_):
                    raise TypeError(
                        f"The elements of {type(range_)} passed as the value of keyword"
                        " argument must be real numbers."
                    )
                selection = list()
                selection_index = list()
                for point in sorted(range_):
                    if (
                        point < self.region.pmin[dim_index]
                        or point > self.region.pmax[dim_index]
                    ):
                        raise ValueError(
                            f"Selected value {point} is outside the mesh region"
                            f" {self.region}."
                        )
                    test_point = self.region.pmin.copy().astype(
                        max(self.region.pmin.dtype, type(point))
                    )
                    test_point[dim_index] = point
                    selection.append(
                        self.index2point(self.point2index(test_point))[dim_index]
                    )
                    selection_index.append(self.point2index(test_point)[dim_index])
                # increase upper boundary to "make slice inclusive"
                selection_index = slice(selection_index[0], selection_index[1] + 1)
            else:
                raise TypeError(
                    "The value passed to selected dimension must be a tuple, list,"
                    " array or real number."
                )
        else:
            selection = self.index2point(self.point2index(self.region.center))[
                dim_index
            ]
            selection_index = self.point2index(self.region.center)[dim_index]

        return dim, dim_index, selection, selection_index

    def __or__(self, other):
        # """Depricated method to check if meshes are aligned: use ``is_aligned``"""

        warnings.warn(
            "Bitwise OR (|) operator is deprecated; please use is_aligned",
            DeprecationWarning,
        )
        return self.is_aligned(other)

    def is_aligned(self, other, tolerance=1e-12):
        """Check if meshes are aligned.

        Two meshes are considered to be aligned if and only if:

            1. They have same discretisation cell size.

            2. They have common cell coordinates.

        for a given tolerance value.

        Parameters
        ----------
        other : discretisedfield.Mesh

            Other mesh to be checked if it is aligned with self.

        tolerance : int, float, optional

            The allowed extent of misalignment for discretisation cells and cell
            coordinates.

        Returns
        -------
        bool

            ``True`` if meshes are aligned, ``False`` otherwise.

        Raises
        ------
        TypeError

            If ``other`` argument is not of type ``discretisedfield.Mesh`` or if
            ``tolerance`` argument is not of type ``float`` or ``int``.

        Examples
        --------
        1. Check if two meshes are aligned.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -25e-9, 0)
        >>> p2 = (50e-9, 25e-9, 5e-9)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> region1 = df.Region(p1=p1, p2=p2)
        >>> mesh1 = df.Mesh(region=region1, cell=cell)
        ...
        >>> p1 = (-45e-9, -20e-9, 0)
        >>> p2 = (10e-9, 20e-9, 5e-9)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> region2 = df.Region(p1=p1, p2=p2)
        >>> mesh2 = df.Mesh(region=region2, cell=cell)
        ...
        >>> p1 = (-42e-9, -20e-9, 0)
        >>> p2 = (13e-9, 20e-9, 5e-9)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> region3 = df.Region(p1=p1, p2=p2)
        >>> mesh3 = df.Mesh(region=region3, cell=cell)
        ...
        >>> mesh1.is_aligned(mesh2)
        True
        >>> mesh1.is_aligned(mesh3)
        False
        >>> mesh1.is_aligned(mesh1)
        True
        >>> p_1 = (0, 0, 0)
        >>> p_2 = (0 + 1e-13, 0, 0)
        >>> p_3 = (0, 0, 0 + 1e-10)
        >>> p_4 = (20e-9, 20e-9, 20e-9)
        >>> p_5 = (20e-9 + 1e-13, 20e-9, 20e-9)
        >>> p_6 = (20e-9, 20e-9, 20e-9 + 1e-10)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> mesh4 = df.Mesh(p1=p_1, p2=p_4, cell=cell)
        >>> mesh5 = df.Mesh(p1=p_2, p2=p_5, cell=cell)
        >>> mesh6 = df.Mesh(p1=p_3, p2=p_6, cell=cell)
        ...
        >>> mesh4.is_aligned(mesh5, 1e-12)
        True
        >>> mesh4.is_aligned(mesh6, 1e-11)
        False

        """
        if not isinstance(other, df.Mesh):
            raise TypeError(
                f"Expected argument of type discretisedfield.Mesh but got {type(other)}"
            )
        if not isinstance(tolerance, numbers.Real):
            raise TypeError(
                "Expected tolerance to be either a float or an integer but got"
                f" {type(tolerance)}"
            )

        if not np.allclose(self.cell, other.cell, atol=tolerance):
            return False

        tol = tolerance
        for i in ["pmin", "pmax"]:
            diff = np.subtract(getattr(self.region, i), getattr(other.region, i))
            rem = np.remainder(abs(diff), self.cell)
            if np.logical_and(
                np.greater(rem, tol), np.less(rem, np.subtract(self.cell, tol))
            ).any():
                return False

        return True

    def __getitem__(self, item):
        """Extracts the mesh of a subregion.

        If subregions were defined by passing ``subregions`` dictionary when
        the mesh was created, this method returns a mesh defined on a subregion
        with key ``item``. Alternatively, a ``discretisedfield.Region``
        object can be passed and a minimum-sized mesh containing it will be
        returned. The resulting mesh has the same discretisation cell as the
        original mesh. This method uses closed intervals, inclusive of endpoints
        i.e. [], for extracting the new mesh.

        Parameters
        ----------
        item : str, discretisedfield.Region

            The key of a subregion in ``subregions`` dictionary or a region
            object.

        Returns
        -------
        disretisedfield.Mesh

            Mesh of a subregion.

        Example
        -------
        1. Extract subregion mesh by passing a subregion key.

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
        array([0, 0, 0])
        >>> mesh.region.pmax
        array([100, 100, 100])
        >>> submesh = mesh['r1']
        >>> len(submesh)
        500
        >>> submesh.region.pmin
        array([0, 0, 0])
        >>> submesh.region.pmax
        array([ 50, 100, 100])

        2. Extracting a submesh on a "newly-defined" region.

        >>> p1 = (-50e-9, -25e-9, 0)
        >>> p2 = (50e-9, 25e-9, 5e-9)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, cell=cell)
        ...
        >>> subregion = df.Region(p1=(0, 1e-9, 0), p2=(10e-9, 14e-9, 5e-9))
        >>> submesh = mesh[subregion]
        >>> submesh.cell
        array([5.e-09, 5.e-09, 5.e-09])
        >>> submesh.n
        array([2, 3, 1])

        """
        if isinstance(item, str):
            return self.__class__(region=self.subregions[item], cell=self.cell)

        if item not in self.region:
            msg = f"Subregion '{item}' is outside the mesh region '{self.region}'."
            raise ValueError(msg)

        hc = np.divide(self.cell, 2)  # half-cell
        p1 = np.subtract(self.index2point(self.point2index(item.pmin)), hc)

        # Calculate p2 index manually as point2index will give [) and we want [].
        p2_idx = (np.ceil((item.pmax - self.region.pmin) / self.cell) - 1).astype(int)
        p2 = np.add(self.index2point(p2_idx), hc)

        return self.__class__(
            region=df.Region(
                p1=p1,
                p2=p2,
                dims=self.region.dims,
                units=self.region.units,
                tolerance_factor=self.region.tolerance_factor,
            ),
            cell=self.cell,
        )

    def pad(self, pad_width):
        """Mesh padding.

        This method extends the mesh by adding (padding) discretisation cells
        in chosen direction(s). The way in which the mesh is going to be padded
        is defined by passing ``pad_width`` dictionary. The keys of the
        dictionary are the directions (axes), e.g. ``'x'``, ``'y'``, or
        ``'z'``, whereas the values are the tuples of length 2. The first
        integer in the tuple is the number of cells added in the negative
        direction, and the second integer is the number of cells added in the
        positive direction.

        Parameters
        ----------
        pad_width : dict

            The keys of the dictionary are the directions (axes), e.g. ``'x'``,
            ``'y'``, or ``'z'``, whereas the values are the tuples of length 2.
            The first integer in the tuple is the number of cells added in the
            negative direction, and the second integer is the number of cells
            added in the positive direction.

        Returns
        -------
        discretisedfield.Mesh

            Padded (extended) mesh.

        Examples
        --------
        1. Padding a mesh in the x and y directions by 1 cell.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> mesh.region.edges
        array([100, 100, 100])
        >>> padded_mesh = mesh.pad({'x': (1, 1), 'y': (1, 1), 'z': (0, 1)})
        >>> padded_mesh.region.edges
        array([120., 120., 110.])
        >>> padded_mesh.n
        array([12, 12, 11])

        """
        pmin = self.region.pmin.copy().astype(float)
        pmax = self.region.pmax.copy().astype(float)
        # Convert to np.ndarray to allow operations on them.
        for direction in pad_width.keys():
            axis = self.region._dim2index(direction)
            pmin[axis] -= pad_width[direction][0] * self.cell[axis]
            pmax[axis] += pad_width[direction][1] * self.cell[axis]

        return self.__class__(
            region=df.Region(
                p1=pmin,
                p2=pmax,
                dims=self.region.dims,
                units=self.region.units,
                tolerance_factor=self.region.tolerance_factor,
            ),
            cell=self.cell,
            bc=self.bc,
        )

    def __getattr__(self, attr):
        """Extracting the discretisation in a particular direction.

        For example in a three dimensional geometry with spatial dimensions ``'x'``,
        ``'y'``, and ``'z'``, if ``'dx'``, ``'dy'``, or ``'dz'`` is accessed, the
        discretisation cell size in that direction is returned.

        Parameters
        ----------
        attr : str

            Discretisation direction (eg. ``'dx'``, ``'dy'``, or ``'dz'``)

        Returns
        -------
        numbers.Real

            Discretisation in a particular direction.

        Examples
        --------
        1. Discretisation in the different directions.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (10, 25, 50)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> mesh.dx
        10.0
        >>> mesh.dy
        25.0
        >>> mesh.dz
        50.0

        """
        if attr in self._removed_attributes:
            raise AttributeError(
                f"'{attr}' has been removed; use '{self._removed_attributes[attr]}'"
                " instead."
            )
        if len(attr) > 1 and attr[0] == "d":
            with contextlib.suppress(ValueError):
                return self.cell[self.region._dim2index(attr[1:])]
        raise AttributeError(f"Object has no attribute {attr}.")

    def __dir__(self):
        """Extension of the ``dir(self)`` list.

        For example in a three dimensional geometry with spatial dimensions ``'x'``,
        ``'y'``, and ``'z'``, it adds ``'dx'``, ``'dy'``, and ``'dz'``.

        Returns
        -------
        list

            Avalilable attributes.

        """
        return dir(self.__class__) + [f"d{i}" for i in self.region.dims]

    @property
    def dV(self):
        """Discretisation cell volume.

        Returns
        -------
        float

            Discretisation cell volume.

        Examples
        --------
        1. Discretisation cell volume.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (1, 2, 4)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> mesh.dV
        8.0

        """
        return np.product(self.cell)

    def scale(self, factor, reference_point=None, inplace=False):
        """Scale the underlying region and all subregions.

        This method scales mesh.region and all subregions by a ``factor`` with respect
        to a ``reference_point``. If ``factor`` is a number the same scaling is applied
        along all dimensions. If ``factor`` is array-like its length must match
        ``region.ndim`` and different factors are applied along the different directions
        (based on their order). If ``reference_point`` is ``None``,
        ``mesh.region.center`` is used as the reference point. A new object is created
        unless ``inplace=True`` is specified.

        Scaling the mesh also scales ``mesh.cell``. The number of cells ``mesh.n`` stays
        constant.

        Parameters
        ----------
        factor : numbers.Real or array-like of numbers.Real

            Factor to scale the mesh.

        reference_point : array_like, optional

            The position of the reference point is fixed when scaling the mesh. If not
            specified the mesh is scaled about its ``mesh.region.center``.

        inplace : bool, optional

            If True, the mesh object is modified in-place. Defaults to False.

        Returns
        -------
        discretisedfield.Mesh

            Resulting mesh.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Scale a mesh without subregions.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 1))
        >>> res = mesh.scale(2)
        >>> res.region.pmin
        array([-5., -5., -5.])
        >>> res.region.pmax
        array([15., 15., 15.])

        2. Scale a mesh with subregions.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> sr = {'sub_reg': df.Region(p1=p1, p2=(5, 5, 5))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 1), subregions=sr)
        >>> res = mesh.scale(2)
        >>> res.region.pmin
        array([-5., -5., -5.])
        >>> res.region.pmax
        array([15., 15., 15.])
        >>> res.subregions['sub_reg'].pmin
        array([-5., -5., -5.])
        >>> res.subregions['sub_reg'].pmax
        array([5., 5., 5.])

        3. Scale a mesh with subregions in place.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> sr = {'sub_reg': df.Region(p1=p1, p2=(5, 5, 5))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 1), subregions=sr)
        >>> mesh.scale((2, 2, 5), inplace=True)
        Mesh(...)
        >>> mesh.region.pmin
        array([ -5.,  -5., -20.])
        >>> mesh.region.pmax
        array([15., 15., 30.])
        >>> mesh.subregions['sub_reg'].pmin
        array([ -5.,  -5., -20.])
        >>> mesh.subregions['sub_reg'].pmax
        array([5., 5., 5.])

        4. Scale with respect to the origin

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 1))
        >>> res = mesh.scale(2, reference_point=p1)
        >>> res.region.pmin
        array([0, 0, 0])
        >>> res.region.pmax
        array([20, 20, 20])

        See also
        --------
        ~discretisedfield.Region.scale

        """
        sr_ref = self.region.center if reference_point is None else reference_point
        if inplace:
            self.region.scale(factor, inplace=True, reference_point=reference_point)
            for sr in self.subregions.values():
                sr.scale(factor, inplace=True, reference_point=sr_ref)
            return self
        else:
            region = self.region.scale(factor, reference_point=reference_point)
            subregions = {
                key: sr.scale(factor, reference_point=sr_ref)
                for key, sr in self.subregions.items()
            }
            return self.__class__(
                region=region, n=self.n, bc=self.bc, subregions=subregions
            )

    def translate(self, vector, inplace=False):
        """Translate the underlying region and all subregions.

        This method translates mesh.region and all subregions by adding ``vector`` to
        ``pmin`` and ``pmax``. The ``vector`` must have ``Region.ndim`` elements. A new
        object is created unless ``inplace=True`` is specified.

        Parameters
        ----------
        vector : array-like of numbers.Number

            Vector to translate the underlying region.

        inplace : bool, optional

            If True, the Region objects are modified in-place. Defaults to False.

        Returns
        -------
        discretisedfield.Mesh

            Resulting mesh.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Examples
        --------
        1. Translate a mesh without subregions.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 1))
        >>> res = mesh.translate((2, -2, 5))
        >>> res.region.pmin
        array([ 2, -2,  5])
        >>> res.region.pmax
        array([12,  8, 15])

        2. Translate a mesh with subregions.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> sr = {'sub_reg': df.Region(p1=p1, p2=(5, 5, 5))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 1), subregions=sr)
        >>> res = mesh.translate((2, -2, 5))
        >>> res.region.pmin
        array([ 2, -2,  5])
        >>> res.region.pmax
        array([12,  8, 15])
        >>> res.subregions['sub_reg'].pmin
        array([ 2, -2,  5])
        >>> res.subregions['sub_reg'].pmax
        array([ 7,  3, 10])

        3. Translate a mesh with subregions in place.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> sr = {'sub_reg': df.Region(p1=p1, p2=(5, 5, 5))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=(1, 1, 1), subregions=sr)
        >>> mesh.translate((2, -2, 5), inplace=True)
        Mesh(...)
        >>> mesh.region.pmin
        array([ 2, -2,  5])
        >>> mesh.region.pmax
        array([12,  8, 15])
        >>> mesh.subregions['sub_reg'].pmin
        array([ 2, -2,  5])
        >>> mesh.subregions['sub_reg'].pmax
        array([ 7,  3, 10])

        See also
        --------
        ~discretisedfield.Region.translate

        """
        if inplace:
            self.region.translate(vector, inplace=True)
            for sr in self.subregions.values():
                sr.translate(vector, inplace=True)
            return self
        else:
            region = self.region.translate(vector)
            subregions = {
                key: sr.translate(vector) for key, sr in self.subregions.items()
            }
            return self.__class__(
                region=region, n=self.n, bc=self.bc, subregions=subregions
            )

    def rotate90(self, ax1, ax2, k=1, reference_point=None, inplace=False):
        """Rotate mesh by 90°.

        Rotate the mesh ``k`` times by 90 degrees in the plane defined by ``ax1`` and
        ``ax2``. The rotation direction is from ``ax1`` to ``ax2``, the two must be
        different.

        The rotate method does not rotate the string defining periodic boundary
        conditions, e.g. if a system has periodic boundary conditions in x and is
        rotated in the xy plane the new system will still have periodic boundary
        conditions in the new x direction, NOT in the new y direction. It is the
        user's task to update the ``bc`` string after rotation if required.

        Parameters
        ----------
        ax1 : str

            Name of the first dimension.

        ax2 : str

            Name of the second dimension.

        k : int, optional

            Number of 90° rotations, defaults to 1.

        reference_point : array_like, optional

            Point around which the mesh is rotated. If not provided the mesh.region's
            centre point is used.

        inplace : bool, optional

            If ``True``, the rotation is applied in-place. Defaults to ``False``.

        Returns
        -------
        discretisedfield.Mesh

            The rotated mesh object. Either a new object or a reference to the
            existing mesh for ``inplace=True``.

        Examples
        --------

        >>> import discretisedfield as df
        >>> import numpy as np
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 8, 6)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=(10, 4, 6))
        >>> rotated = mesh.rotate90('x', 'y')
        >>> rotated.region.pmin
        array([ 1., -1.,  0.])
        >>> rotated.region.pmax
        array([9., 9., 6.])
        >>> rotated.n
        array([ 4, 10,  6])

        See also
        --------
        :py:func:`~discretisedfield.Region.rotate90`
        :py:func:`~discretisedfield.Field.rotate90`

        """
        # all checks will be performed by region.rotate90
        region = self.region.rotate90(
            ax1=ax1, ax2=ax2, k=k, reference_point=reference_point, inplace=inplace
        )

        n = list(self.n)
        if k % 2 == 1:
            idx1 = self.region._dim2index(ax1)
            idx2 = self.region._dim2index(ax2)
            n[idx1], n[idx2] = n[idx2], n[idx1]

        if reference_point is None:
            reference_point = self.region.centre

        subregions = {
            name: subregion.rotate90(
                ax1=ax1, ax2=ax2, k=k, reference_point=reference_point, inplace=inplace
            )
            for name, subregion in self.subregions.items()
        }

        if inplace:
            self._n = np.array(n, dtype=int)
            return self
        else:
            return self.__class__(region=region, n=n, bc=self.bc, subregions=subregions)

    @property
    def mpl(self):
        """``matplotlib`` plot.

        If ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. The color of lines depicting the region and the
        discretisation cell can be specified using ``color`` length-2 tuple,
        where the first element is the colour of the region and the second
        element is the colour of the discretisation cell. The plot is saved in
        PDF-format if ``filename`` is passed.

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

        color : (2,) array_like

            A valid ``matplotlib`` color for lines depicting the region.
            Defaults to the default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        box_aspect : str, array_like (3), optional

            Set the aspect-ratio of the plot. If set to `'auto'` the aspect
            ratio is determined from the edge lengths of the region on which
            the mesh is defined. To set different aspect ratios a tuple can be
            passed. Defaults to ``'auto'``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Examples
        --------
        1. Visualising the mesh using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, 0)
        >>> p2 = (50e-9, 50e-9, 10e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, n=(50, 50, 5))
        ...
        >>> mesh.mpl()

        .. seealso:: :py:func:`~discretisedfield.Mesh.k3d`

        """
        return dfp.MplMesh(self)

    @property
    def k3d(self):
        """``k3d`` plot.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The color of the region and the discretisation cell can
        be specified using ``color`` length-2 tuple, where the first element is
        the colour of the region and the second element is the colour of the
        discretisation cell.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally.

        color : (2,) array_like

            Colour of the region and the discretisation cell. Defaults to the
            default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        Examples
        --------
        1. Visualising the mesh using ``k3d``.

        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> mesh.k3d()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Mesh.mpl`

        """
        return dfp.K3dMesh(self)

    def slider(self, axis, /, *, multiplier=None, description=None, **kwargs):
        """Axis slider.

        For ``axis``, the name of a spatial dimension is passed. Based on that
        value, ``ipywidgets.SelectionSlider`` is returned. Axis multiplier can
        be changed via ``multiplier``.

        This method is based on ``ipywidgets.SelectionSlider``, so any keyword
        argument accepted by it can be passed.

        Parameters
        ----------
        axis : str

            Axis for which the slider is returned (For eg., ``'x'``, ``'y'``, or
            ``'z'``).

        multiplier : numbers.Real, optional

            Axis multiplier. Defaults to ``None``.

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
        ...
        >>> mesh.slider('x')
        SelectionSlider(...)

        """
        if isinstance(axis, str):
            axis = self.region._dim2index(axis)

        if multiplier is None:
            multiplier = uu.si_multiplier(self.region.edges[axis])

        slider_min = self.index2point((0, 0, 0))[axis]
        slider_max = self.index2point(np.subtract(self.n, 1))[axis]
        slider_step = self.cell[axis]
        if description is None:
            description = (
                f"{self.region.dims[axis]} ({uu.rsi_prefixes[multiplier]}"
                f"{self.region.units[axis]})"
            )

        values = np.arange(slider_min, slider_max + 1e-20, slider_step)
        labels = np.around(values / multiplier, decimals=3)
        options = list(zip(labels, values))

        # Select middle element for slider value
        slider_value = values[int(self.n[axis] / 2)]

        return ipywidgets.SelectionSlider(
            options=options, value=slider_value, description=description, **kwargs
        )

    def axis_selector(self, *, widget="dropdown", description="axis"):
        """Axis selector.

        For ``widget='dropdown'``, ``ipywidgets.Dropdown`` is returned, whereas
        for ``widget='radiobuttons'``, ``ipywidgets.RadioButtons`` is returned.
        Default widget description can be changed using ``description``.

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
        ...
        >>> mesh.axis_selector(widget='radiobuttons')
        RadioButtons(...)

        """
        if widget.lower() == "dropdown":
            widget_cls = ipywidgets.Dropdown
        elif widget == "radiobuttons":
            widget_cls = ipywidgets.RadioButtons
        else:
            msg = f"Widget {widget} is not supported."
            raise ValueError(msg)

        return widget_cls(
            options=self.region.dims,
            value="z",
            description=description,
            disabled=False,
        )

    def coordinate_field(self):
        """Create a field whose values are the mesh coordinates.

        This method can be used to create a vector field with values equal to the
        coordinates of the cell midpoints. The result is equivalent to a field created
        with the following code:

        .. code-block::

            mesh = df.Mesh(...)
            df.Field(mesh, dim=mesh.region.ndim, value=lambda point: point)

        This method should be preferred over the manual creation with a callable because
        it provides much better performance.

        Returns
        -------
        discretisedfield.Field

            Field with coordinates as values.

        Examples
        --------
        1. Create a coordinate field.

        >>> import discretisedfield as df
        ...
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(4, 2, 1), cell=(1, 1, 1))
        >>> cfield = mesh.coordinate_field()
        >>> cfield
        Field(...)

        2. Extract its value at position (0.5, 0.5, 0.5)

        >>> cfield((0.5, 0.5, 0.5))
        array([0.5, 0.5, 0.5])

        3. Compare with manually created coordinate field

        >>> manually = df.Field(mesh, nvdim=3, value=lambda point: point)
        >>> cfield.allclose(manually)
        True

        """

        field = df.Field(
            self,
            nvdim=self.region.ndim,
            vdims=self.region.dims,
            vdim_mapping=dict(zip(self.region.dims, self.region.dims)),
        )
        for i, dim in enumerate(self.region.dims):
            cells = self.cells  # avoid re-computing cells
            field.array[..., i] = getattr(cells, dim).reshape(
                tuple(self.n[i] if i == j else 1 for j in range(self.region.ndim))
            )

        return field

    def fftn(self, rfft=False):
        """Performs an N-dimensional discrete Fast Fourier Transform (FFT) on the mesh.

        This method computes the FFT in an N-dimensional space. The FFT is a way to
        transform a spatial-domain into a frequency domain. Note that any information
        about subregions in the mesh is lost during this transformation.

        Parameters
        ----------
        rfft : bool, optional

            Determines if a real FFT is to be performed (if True) or a complex FFT
            (if False). Defaults to False, i.e., a complex FFT is performed by default.

        Returns
        -------
        discretisedfield.Mesh

            A mesh representing the Fourier transform of the original mesh. The returned
            mesh has dimensions labeled with frequency (k) and cells have coordinates
            that correspond to the correct frequencies in the frequency domain.

        Examples
        --------
        1. Create a mesh and perform a FFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=0, p2=10, cell=2)
        >>> fft_mesh = mesh.fftn()
        >>> fft_mesh.n
        array([5])
        >>> fft_mesh.cell
        array([0.1])
        >>> fft_mesh.region.pmin
        array([-0.25])
        >>> fft_mesh.region.pmax
        array([0.25])

        2. Perform a real FFT.
        >>> fft_mesh = mesh.fftn(rfft=True)
        >>> fft_mesh.n
        array([3])
        >>> fft_mesh.cell
        array([0.1])
        >>> fft_mesh.region.pmin
        array([-0.05])
        >>> fft_mesh.region.pmax
        array([0.25])

        3. Create a 2D mesh and perform a FFT. This demonstrates how the function works
        with higher dimensional meshes.
        >>> mesh = df.Mesh(p1=(0, 0), p2=(10, 10), cell=(2, 2))
        >>> fft_mesh = mesh.fftn()
        >>> fft_mesh.n
        array([5, 5])
        >>> fft_mesh.cell
        array([0.1, 0.1])
        >>> fft_mesh.region.pmin
        array([-0.25, -0.25])
        >>> fft_mesh.region.pmax
        array([0.25, 0.25])
        """

        p1 = []
        p2 = []
        n = []

        for i in range(self.region.ndim):
            if self.n[i] == 1:
                p1.append(0)
                p2.append(1 / self.cell[i])
                n.append(1)
            else:
                if rfft and i == self.region.ndim - 1:
                    # last frequency is different for rfft if it has more than 1 element
                    freqs = spfft.rfftfreq(self.n[i], self.cell[i])
                else:
                    freqs = spfft.fftfreq(self.n[i], self.cell[i])
                # Shift the region boundaries to get the correct coordinates of
                # mesh cells.
                # This effectively does the same as using fftshift
                dfreq = abs(freqs[1] - freqs[0]) / 2
                p1.append(min(freqs) - dfreq)
                p2.append(max(freqs) + dfreq)
                n.append(len(freqs))

        kdims = [f"k_{d}" for d in self.region.dims]
        kunits = [f"({u})" + "$^{-1}$" for u in self.region.units]
        region = df.Region(
            p1=p1,
            p2=p2,
            dims=kdims,
            units=kunits,
            tolerance_factor=self.region.tolerance_factor,
        )
        # Subregions cannot be kept as we loose the information about the
        # translation and size of the original subregions.
        mesh = df.Mesh(region=region, n=n)

        return mesh

    def ifftn(self, rfft=False, shape=None):
        """Performs an N-dimensional discrete inverse Fast Fourier Transform (iFFT)
        on the mesh.

        This function calculates the iFFT in an N-dimensional space. The iFFT is a
        method to convert a frequency-domain signal into a spatial-domain signal.
        If 'rfft' is set to True and 'shape' is None, the original mesh shape is
        assumed to be even in the last dimension.

        Please note that during Fourier transformations, the original position
        information is lost, causing the inverse Fourier transform to be centered at
        the origin. This can be rectified by `mesh.translate` to translate the mesh
        back to the desired position.

        Parameters
        ----------
        rfft : bool, optional

            If set to True, a real FFT is performed. If False, a complex FFT is
            performed. Defaults to False.

        shape : (tuple, np.ndarray, list), optional

            Specifies the shape of the original mesh. Defaults to None, which means the
            shape of the original mesh is used.

        Returns
        -------
        discretisedfield.Mesh

            A mesh representing the inverse Fourier transform of the mesh.

        Examples
        --------
        1. Create a mesh and perform an iFFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=0, p2=10, cell=2)
        >>> ifft_mesh = mesh.fftn().ifftn()
        >>> ifft_mesh.n
        array([5])
        >>> ifft_mesh.cell
        array([2.])
        >>> ifft_mesh.region.pmin
        array([-5.])
        >>> ifft_mesh.region.pmax
        array([5.])

        2. Perform a real iFFT.
        >>> ifft_mesh = mesh.fftn(rfft=True).ifftn(rfft=True, shape=mesh.n)
        >>> ifft_mesh.n
        array([5])
        >>> ifft_mesh.cell
        array([2.])
        >>> ifft_mesh.region.pmin
        array([-5.])
        >>> ifft_mesh.region.pmax
        array([5.])

        3. Perform a 2D iFFT.
        >>> mesh = df.Mesh(p1=(0, 0), p2=(10, 10), cell=(2, 2))
        >>> ifft_mesh = mesh.fftn().ifftn()
        >>> ifft_mesh.n
        array([5, 5])
        >>> ifft_mesh.cell
        array([2., 2.])
        >>> ifft_mesh.region.pmin
        array([-5., -5.])
        >>> ifft_mesh.region.pmax
        array([5., 5.])

        4. Perform a real 2D iFFT.
        >>> ifft_mesh = mesh.fftn(rfft=True).ifftn(rfft=True, shape=mesh.n)
        >>> ifft_mesh.n
        array([5, 5])
        >>> ifft_mesh.cell
        array([2., 2.])
        >>> ifft_mesh.region.pmin
        array([-5., -5.])
        >>> ifft_mesh.region.pmax
        array([5., 5.])

        """
        if shape is not None:
            if isinstance(shape, numbers.Number):
                shape = (shape,)

            if isinstance(shape, (tuple, list, np.ndarray)):
                if len(shape) != self.region.ndim:
                    raise ValueError(
                        "The shape must have the same number of dimensions as the mesh"
                        f" ({self.region.ndim=})."
                    )
                if not np.array_equal(shape[:-1], self.n[:-1]):
                    raise ValueError(
                        f"The shape apart from the last dimension must match {self.n=}."
                    )
            else:
                raise TypeError(
                    "Expected shape to be either int, tuple, list or np.ndarray but got"
                    f" {type(shape)}."
                )
            if shape[-1] // 2 + 1 != self.n[-1]:
                raise ValueError(
                    "The last dimension of the shape must match"
                    f" {(self.n[-1] - 1) * 2} or {(self.n[-1] - 1) * 2 + 1} not"
                    f" {shape[-1]}."
                )
        else:
            shape = self.n.copy()
            if rfft and self.n[-1] != 1:
                shape[-1] = (self.n[-1] - 1) * 2

        p1 = []
        p2 = []
        n = []
        for i in range(self.region.ndim):
            if shape[i] == 1:
                p1.append(0)
                p2.append(1 / self.cell[i])
                n.append(1)
            else:
                freqs = spfft.fftfreq(shape[i], self.cell[i])
                # Shift the region boundaries to get the correct coordinates of
                # mesh cells.
                dfreq = abs(freqs[1] - freqs[0]) / 2
                p1.append(min(freqs) - dfreq)
                p2.append(max(freqs) + dfreq)
                n.append(len(freqs))

        kdims = [d[2:] if d.startswith("k_") else d for d in self.region.dims]
        kunits = [
            u[1:-8] if u.startswith("(") and u.endswith(")$^{-1}$") else u
            for u in self.region.units
        ]

        region = df.Region(
            p1=p1,
            p2=p2,
            dims=kdims,
            units=kunits,
            tolerance_factor=self.region.tolerance_factor,
        )

        mesh = df.Mesh(region=region, n=n)

        # Shift the center of the mesh to the origin.
        mesh.translate(-mesh.region.center, inplace=True)

        return mesh
