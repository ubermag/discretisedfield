import k3d
import h5py
import pyvtk
import struct
import numbers
import itertools
import matplotlib
import numpy as np
import seaborn as sns
import mpl_toolkits.axes_grid1
import discretisedfield as df
import ubermagutil.units as uu
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu
import matplotlib.pyplot as plt

# TODO: tutorials (code polishing), check rtd requirements, remove numbers from
# tutorials, installation instructions (conda environment, k3d jupyterlab), go
# through other repo files

@ts.typesystem(mesh=ts.Typed(expected_type=df.Mesh, const=True),
               dim=ts.Scalar(expected_type=int, positive=True, const=True))
class Field:
    """Finite difference field.

    This class defines a finite difference field and enables certain operations
    for its analysis and visualisation. The field is defined on a finite
    difference mesh (`discretisedfield.Mesh`) passed by using ``mesh``. Another
    value that must be passed is the dimension of the value using ``dim``. More
    precisely, if the field is a scalar field ``dim=1`` must be passed. On the
    other hand, for a three-dimensional vector field ``dim=3`` is passed. The
    value of the field can be set by passing ``value``. For details on how the
    value can be set, please refer to ``discretisedfield.Field.value``.
    Similarly, if the field has ``dim>1``, the field can be normalised by
    passing ``norm``. For details on setting the norm, please refer to
    ``discretisedfield.Field.norm``.

    Parameters
    ----------
    mesh : discretisedfield.Mesh

        Finite difference rectangular mesh.

    dim : int

        Dimension of the field's value. For instance, if `dim=3` the field is a
        three-dimensional vector field and for `dim=1` the field is a scalar
        field.

    value : array_like, callable, optional

        Please refer to ``discretisedfield.Field.value`` property. Defaults to
        0, meaning that if the value is not provided in the initialisation,
        "zero-field" will be defined.

    norm : numbers.Real, callable, optional

        Please refer to ``discretisedfield.Field.norm`` property. Defaults to
        ``None`` (``norm=None`` defines no norm).

    Examples
    --------
    1. Defining a uniform three-dimensional vector field on a nano-sized thin
    film.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> cell = (1e-9, 1e-9, 0.1e-9)
    >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
    >>> dim = 3
    >>> value = (0, 0, 1)
    ...
    >>> field = df.Field(mesh=mesh, dim=dim, value=value)
    >>> field
    Field(mesh=...)
    >>> field.average
    (0.0, 0.0, 1.0)

    2. Defining a scalar field.

    >>> p1 = (-10, -10, -10)
    >>> p2 = (10, 10, 10)
    >>> n = (1, 1, 1)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> dim = 1
    >>> value = 3.14
    ...
    >>> field = df.Field(mesh=mesh, dim=dim, value=value)
    >>> field
    Field(mesh=...)
    >>> field.average
    3.14

    3. Defining a uniform three-dimensional normalised vector field.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e9, -25e9, 0)
    >>> p2 = (50e9, 25e9, 5e9)
    >>> cell = (1e9, 1e9, 0.1e9)
    >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
    >>> dim = 3
    >>> value = (0, 0, 8)
    >>> norm = 1
    ...
    >>> field = df.Field(mesh=mesh, dim=dim, value=value, norm=norm)
    >>> field
    Field(mesh=...)
    >>> field.average
    (0.0, 0.0, 1.0)

    .. seealso:: :py:func:`~discretisedfield.Mesh`

    """
    def __init__(self, mesh, dim, value=0, norm=None):
        self.mesh = mesh
        self.dim = dim
        self.value = value
        self.norm = norm

    @property
    def value(self):
        """Field value representation.

        This property returns a representation of the field value if it exists.
        Otherwise, ``discretisedfield.Field.array`` containing all field values
        is returned.

        The value of the field can be set using a scalar value for ``dim=1``
        fields (e.g. ``value=3``) or ``array_like`` value for ``dim>1`` fields
        (e.g. ``value=(1, 2, 3)``). Alternatively, the value can be defined
        using a callable object, which takes a point tuple as an input argument
        and returns a value of appropriate dimension. Internally, callable
        object is called for every point in the mesh on which the field is
        defined. For instance, callable object can be a Python function or
        another ``discretisedfield.Field``. Finally, ``numpy.ndarray`` with
        shape ``(*self.mesh.n, dim)`` can be passed.

        Parameters
        ----------
        value : numbers.Real, array_like, callable

            For scalar fields (``dim=1``) ``numbers.Real`` values are allowed.
            In the case of vector fields, ``array_like`` (list, tuple,
            numpy.ndarray) value with length equal to `dim` should be used.
            Finally, the value can also be a callable (e.g. Python function or
            another field), which for every coordinate in the mesh returns a
            valid value. If ``value=0``, all values in the field will be set to
            zero independent of the field dimension.

        Returns
        -------
        array_like, callable, numbers.Real, numpy.ndarray

            The value used (representation) for setting the field is returned.
            However, if the actual value of the field does not correspond to
            the initially used value anymore, a ``numpy.ndarray`` is returned
            containing all field values.

        Raises
        ------
        ValueError

            If unsupported type is passed.

        Examples
        --------
        1. Different ways of setting and getting the field value.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> value = (0, 0, 1)
        ...
        >>> # if value is not specified, zero-field is defined
        >>> field = df.Field(mesh=mesh, dim=3)
        >>> field.value
        0
        >>> field.value = (0, 0, 1)
        >>> field.value
        (0, 0, 1)
        >>> # Setting the field value using a Python function (callable).
        >>> def value_function(pos):
        ...     x, y, z = pos
        ...     if x <= 1:
        ...         return (0, 0, 1)
        ...     else:
        ...         return (0, 0, -1)
        >>> field.value = value_function
        >>> field.value
        <function value_function at ...>
        >>> # We now change the value of a single cell so that the
        >>> # representation used for initialising field is not valid
        >>> # anymore.
        >>> field.array[0, 0, 0, :] = (0, 0, 0)
        >>> field.value
        array(...)
        >>> field.value.shape
        (2, 2, 1, 3)

        .. seealso:: :py:func:`~discretisedfield.Field.array`

        """
        value_array = dfu.as_array(self.mesh, self.dim, self._value)
        if np.array_equal(self.array, value_array):
            return self._value
        else:
            return self.array

    @value.setter
    def value(self, val):
        self._value = val
        self.array = dfu.as_array(self.mesh, self.dim, val)

    @property
    def array(self):
        """Field value as ``numpy.ndarray``.

        The shape of the array is ``(*mesh.n, dim)``.

        Parameters
        ----------
        array : numpy.ndarray

            Array with shape ``(*mesh.n, dim)``.

        Returns
        -------
        numpy.ndarray

            Field values array.

        Raises
        ------
        ValueError

            If unsupported type or shape is passed.

        Examples
        --------
        1. Accessing and setting the field array.

        >>> import discretisedfield as df
        >>> import numpy as np
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (1, 1, 1)
        >>> cell = (0.5, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> value = (0, 0, 1)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=value)
        >>> field.array
        array(...)
        >>> field.average
        (0.0, 0.0, 1.0)
        >>> field.array.shape
        (2, 1, 1, 3)
        >>> field.array = np.ones_like(field.array)
        >>> field.array
        array(...)
        >>> field.average
        (1.0, 1.0, 1.0)

        .. seealso:: :py:func:`~discretisedfield.Field.value`

        """
        return self._array

    @array.setter
    def array(self, val):
        self._array = dfu.as_array(self.mesh, self.dim, val)

    @property
    def norm(self):
        """Norm of the field.

        Computes the norm of the field and returns it as
        ``discretisedfield.Field`` with ``dim=1``. Norm of a scalar field
        cannot be computed/set and ``ValueError`` is raised. Alternatively,
        ``discretisedfield.Field.__abs__`` can be called for obtaining the norm
        of the field.

        The field norm can be set by passing ``numbers.Real``,
        ``numpy.ndarray``, or callable. If the field has ``dim=1`` or it
        contains zero values, norm cannot be set and ``ValueError`` is raised.

        Parameters
        ----------
        numbers.Real, numpy.ndarray, callable

            Norm value

        Returns
        -------
        discretisedfield.Field

            ``dim=1`` norm field.

        Raises
        ------
        ValueError

            If the norm is set with wrong type, shape, or value. In addition,
            if the field is scalar (``dim=1``) or the field contains zero
            values.

        Examples
        --------
        1. Manipulating the field norm.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (1, 1, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field.norm
        Field(...)
        >>> field.norm.average
        1.0
        >>> field.norm = 2
        >>> field.average
        (0.0, 0.0, 2.0)
        >>> field.value = (1, 0, 0)
        >>> field.norm.average
        1.0
        >>> # An attempt to set the norm for a zero field.
        >>> field.value = 0
        >>> field.average
        (0.0, 0.0, 0.0)
        >>> field.norm = 1
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.__abs__`

        """
        if self.dim == 1:
            msg = f'Cannot compute norm for field with dim={self.dim}.'
            raise ValueError(msg)

        computed_norm = np.linalg.norm(self.array, axis=-1)[..., np.newaxis]
        return self.__class__(self.mesh, dim=1, value=computed_norm)

    @norm.setter
    def norm(self, val):
        if val is not None:
            if self.dim == 1:
                msg = f'Cannot set norm for field with dim={self.dim}.'
                raise ValueError(msg)

            if not np.all(self.norm.array):
                msg = 'Cannot normalise field with zero values.'
                raise ValueError(msg)

            self.array /= self.norm.array  # normalise to 1
            self.array *= dfu.as_array(self.mesh, dim=1, val=val)

    def __abs__(self):
        """Field norm.

        This method returns ``discretisedfield.Field.norm``.

        .. seealso:: :py:func:`~discretisedfield.Field.norm`

        """
        return self.norm

    @property
    def orientation(self):
        """Orientation field.

        This method computes the orientation (direction) of a vector field and
        returns ``discretisedfield.Field`` with the same dimension. More
        precisely, at every mesh discretisation cell, the vector is divided by
        its norm, so that a unit vector is obtained. However, if the vector at
        a discretisation cell is a zero-vector, it remains unchanged. In the
        case of a scalar (``dim=1``) field, ``ValueError`` is raised.

        Returns
        -------
        discretisedfield.Field

            Orientation field.

        Raises
        ------
        ValueError

            If the field is has ``dim=1``.

        Examples
        --------
        1. Computing the orientation field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(6, 0, 8))
        >>> field.orientation
        Field(...)
        >>> field.orientation.norm.average
        1.0

        """
        if self.dim == 1:
            msg = (f'Cannot compute orientation field for a '
                   f'dim={self.dim} field.')
            raise ValueError(msg)

        orientation_array = np.divide(self.array,
                                      self.norm.array,
                                      out=np.zeros_like(self.array),
                                      where=(self.norm.array != 0))
        return self.__class__(self.mesh, dim=self.dim, value=orientation_array)

    @property
    def average(self):
        """Field average.

        It computes the average of the field over the entire volume of the
        mesh. It returns a tuple with the length same as the dimension
        (``dim``) of the field.

        Returns
        -------
        tuple

            Field average tuple, whose length equals to the field's dimension.

        Examples
        --------
        1. Computing the vector field average.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field.average
        (0.0, 0.0, 1.0)

        2. Computing the scalar field average.

        >>> field = df.Field(mesh=mesh, dim=1, value=55)
        >>> field.average
        55.0

        """
        return dfu.array2tuple(self.array.mean(axis=(0, 1, 2)))

    def __repr__(self):
        """Field representation string.

        Returns
        -------
        str

            Field representation string.

        Example
        -------
        1. Getting field representation string.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh, dim=1, value=1)
        >>> repr(field)
        'Field(mesh=..., dim=1)'

        """
        return f'Field(mesh={repr(self.mesh)}, dim={self.dim})'

    def __call__(self, point):
        """Sample the field value at ``point``.

        It returns the value of the field in the discretisation cell to which
        ``point`` belongs to. It returns a tuple, whose length is the same as
        the dimension (``dim``) of the field.

        Parameters
        ----------
        point : (3,) array_like

            The mesh point coordinate :math:`\\mathbf{p} = (p_{x}, p_{y},
            p_{z})`.

        Returns
        -------
        tuple

            A tuple, whose length is the same as the dimension of the field.

        Example
        -------
        1. Sampling the field value.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (20, 20, 20)
        >>> n = (20, 20, 20)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        ...
        >>> field = df.Field(mesh, dim=3, value=(1, 3, 4))
        >>> point = (10, 2, 3)
        >>> field(point)
        (1.0, 3.0, 4.0)

        """
        return dfu.array2tuple(self.array[self.mesh.point2index(point)])

    def __getattr__(self, attr):
        """Extracting the component of the vector field.

        If ``'x'``, ``'y'``, or ``'z'`` is accessed, a scalar field of that
        component will be returned. This method is effective for vector fields
        with dimension 2 or 3 only.

        Parameters
        ----------
        attr : str

            Vector field component (``'x'``, ``'y'``, or ``'z'``)

        Returns
        -------
        discretisedfield.Field

            Scalar field with vector field component values.

        Examples
        --------
        1. Accessing the vector field components.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field.x
        Field(...)
        >>> field.x.average
        0.0
        >>> field.y
        Field(...)
        >>> field.y.average
        0.0
        >>> field.z
        Field(...)
        >>> field.z.average
        1.0
        >>> field.z.dim
        1

        """
        if attr in list(dfu.axesdict.keys())[:self.dim] and self.dim in (2, 3):
            attr_array = self.array[..., dfu.axesdict[attr]][..., np.newaxis]
            return Field(mesh=self.mesh, dim=1, value=attr_array)
        else:
            msg = f'Object has no attribute {attr}.'
            raise AttributeError(msg)

    def __dir__(self):
        """Extension of the ``dir(self)`` list.

        Adds ``'x'``, ``'y'``, or ``'z'``, depending on the dimension of the
        field, to the ``dir(self)`` list. Similarly, adds or removes methods
        (``grad``, ``div``,...) depending on the dimension of the field.

        Returns
        -------
        list

            Avalilable attributes.

        """
        dirlist = dir(self.__class__)
        if self.dim in (2, 3):
            dirlist += list(dfu.axesdict.keys())[:self.dim]
        if self.dim == 1:
            need_removing = ['div', 'curl', 'topological_charge',
                             'topological_charge_density', 'bergluescher',
                             'norm', 'orientation', 'quiver', 'k3d_vectors']
        if self.dim == 3:
            need_removing = ['grad', 'imshow', 'k3d_voxels', 'k3d_nonzero']

        for attr in need_removing:
            dirlist.remove(attr)

        return dirlist

    def __iter__(self):
        """Generator yielding coordinates and values of all mesh discretisation
        cells.

        Yields
        ------
        tuple (2,)

            The first value is the mesh cell coordinates :math:`\\mathbf{p} =
            (p_{x}, p_{y}, p_{z})`, whereas the second one is the field value.

        Examples
        --------
        1. Iterating through the field coordinates and values

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh, dim=3, value=(0, 0, 1))
        >>> for coord, value in field:
        ...     print (coord, value)
        (0.5, 0.5, 0.5) (0.0, 0.0, 1.0)
        (1.5, 0.5, 0.5) (0.0, 0.0, 1.0)
        (0.5, 1.5, 0.5) (0.0, 0.0, 1.0)
        (1.5, 1.5, 0.5) (0.0, 0.0, 1.0)

        .. seealso:: :py:func:`~discretisedfield.Mesh.indices`

        """
        for point in self.mesh:
            yield point, self(point)

    def __eq__(self, other):
        """Relational operator ``==``.

        Two fields are considered to be equal if:

          1. They are defined on the same mesh.

          2. They have the same dimension (``dim``).

          3. They both contain the same values in ``array``.

        Parameters
        ----------
        other : discretisedfield.Field

            Second operand.

        Returns
        -------
        bool

            ``True`` if two fields are equal, ``False`` otherwise.

        Examples
        --------
        1. Check if two fields are (not) equal.

        >>> import discretisedfield as df
        ...
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        ...
        >>> f1 = df.Field(mesh, dim=1, value=3)
        >>> f2 = df.Field(mesh, dim=1, value=4-1)
        >>> f3 = df.Field(mesh, dim=3, value=(1, 4, 3))
        >>> f1 == f2
        True
        >>> f1 != f2
        False
        >>> f1 == f3
        False
        >>> f1 != f3
        True
        >>> f2 == f3
        False
        >>> f1 == 'a'
        False

        .. seealso:: :py:func:`~discretisedfield.Field.__ne__`

        """
        if not isinstance(other, self.__class__):
            return False
        elif (self.mesh == other.mesh and self.dim == other.dim and
              np.array_equal(self.array, other.array)):
            return True
        else:
            return False

    def __ne__(self, other):
        """Relational operator ``!=``.

        This method returns ``not self == other``. For details, please refer to
        ``discretisedfield.Field.__eq__`` method.

        .. seealso:: :py:func:`~discretisedfield.Field.__eq__`

        """
        return not self == other

    def __pos__(self):
        """Unary ``+`` operator.

        This method defines the unary operator ``+``. It returns the field
        itself:

        .. math::

            +f(x, y, z) = f(x, y, z)

        Returns
        -------
        discretisedfield.Field

            Field itself.

        Example
        -------
        1. Applying unary ``+`` operator on a field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5e-9, 5e-9, 5e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> f = df.Field(mesh, dim=3, value=(0, -1000, -3))
        >>> res = +f
        >>> res.average
        (0.0, -1000.0, -3.0)
        >>> res == f
        True
        >>> +(+f) == f
        True

        """
        return self

    def __neg__(self):
        """Unary ``-`` operator.

        This method negates the value of each discretisation cell. It is
        equivalent to multiplication with -1:

        .. math::

            -f(x, y, z) = -1 \\cdot f(x, y, z)

        Returns
        -------
        discretisedfield.Field

            Field multiplied with -1.

        Example
        -------
        1. Applying unary ``-`` operator on a scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5e-9, 3e-9, 1e-9)
        >>> n = (10, 5, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> f = df.Field(mesh, dim=1, value=3.1)
        >>> res = -f
        >>> res.average
        -3.1
        >>> f == -(-f)
        True

        2. Applying unary negation operator on a vector field.

        >>> f = df.Field(mesh, dim=3, value=(0, -1000, -3))
        >>> res = -f
        >>> res.average
        (0.0, 1000.0, 3.0)

        """
        return -1 * self

    def __pow__(self, other):
        """Unary ``**`` operator.

        This method defines the ``**`` operator for scalar (``dim=1``) fields
        only. This operator is not defined for vector (``dim>1``) fields, and
        ``ValueError`` is raised.

        Parameters
        ----------
        other : numbers.Real

            Value to which the field is raised.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Applying unary ``**`` operator on a scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-25e-3, -25e-3, -25e-3)
        >>> p2 = (25e-3, 25e-3, 25e-3)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        ...
        >>> f = df.Field(mesh, dim=1, value=2)
        >>> res = f**(-1)
        >>> res
        Field(...)
        >>> res.average
        0.5
        >>> res = f**2
        >>> res.average
        4.0
        >>> f**f  # the power must be numbers.Real
        Traceback (most recent call last):
        ...
        TypeError: ...

        2. Attempt to apply power operator on a vector field.

        >>> p1 = (0, 0, 0)
        >>> p2 = (5e-9, 5e-9, 5e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> f = df.Field(mesh, dim=3, value=(0, -1, -3))
        >>> f**2
        Traceback (most recent call last):
        ...
        ValueError: ...

        """
        if self.dim != 1:
            msg = f'Cannot apply ** operator on dim={self.dim} field.'
            raise ValueError(msg)
        if not isinstance(other, numbers.Real):
            msg = (f'Unsupported operand type(s) for **: '
                   f'{type(self)} and {type(other)}.')
            raise TypeError(msg)

        return self.__class__(self.mesh, dim=1,
                              value=np.power(self.array, other))

    def __add__(self, other):
        """Binary ``+`` operator.

        It can be applied only between two ``discretisedfield.Field`` objects.
        Both ``discretisedfield.Field`` objects must be defined on the same
        mesh and have the same dimensions.

        Parameters
        ----------
        other : discretisedfield.Field

            Second operand.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Add two vector fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 3, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=3, value=(0, -1, -3.1))
        >>> f2 = df.Field(mesh, dim=3, value=(0, 1, 3.1))
        >>> res = f1 + f2
        >>> res.average
        (0.0, 0.0, 0.0)
        >>> f1 + f2 == f2 + f1
        True
        >>> f1 + 3
        Traceback (most recent call last):
        ...
        TypeError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.__sub__`

        """
        if not isinstance(other, self.__class__):
            msg = (f'Unsupported operand type(s) for +: '
                   f'{type(self)} and {type(other)}.')
            raise TypeError(msg)
        if self.dim != other.dim:
            msg = f'Cannot add dim={self.dim} and dim={other.dim} fields.'
            raise ValueError(msg)
        if self.mesh != other.mesh:
            msg = 'Cannot add fields defined on different meshes.'
            raise ValueError(msg)

        return self.__class__(self.mesh, dim=self.dim,
                              value=self.array + other.array)

    def __sub__(self, other):
        """Binary ``-`` operator.

        It can be applied only between two ``discretisedfield.Field`` objects.
        Both ``discretisedfield.Field`` objects must be defined on the same
        mesh and have the same dimensions.

        Parameters
        ----------
        other : discretisedfield.Field

            Second operand.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Subtract two vector fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 3, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=3, value=(0, 1, 6))
        >>> f2 = df.Field(mesh, dim=3, value=(0, 1, 3))
        >>> res = f1 - f2
        >>> res.average
        (0.0, 0.0, 3.0)
        >>> f1 - f2 == -(f2 - f1)
        True
        >>> f1 - 3.14
        Traceback (most recent call last):
        ...
        TypeError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.__add__`

        """
        return self + (-other)

    def __mul__(self, other):
        """Binary ``*`` operator.

        It can be applied between:

        1. Two scalar (``dim=1``) fields,

        2. A field of any dimension and ``numbers.Real``, or

        3. A field of any dimension and a scalar (``dim=1``) field.

        If both operands are ``discretisedfield.Field`` objects, they must be
        defined on the same mesh.

        Parameters
        ----------
        other : discretisedfield.Field, numbers.Real

            Second operand.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Multiply two scalar fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=1, value=5)
        >>> f2 = df.Field(mesh, dim=1, value=9)
        >>> res = f1 * f2
        >>> res.average
        45.0
        >>> f1 * f2 == f2 * f1
        True

        2. Multiply vector field with a scalar.

        >>> f1 = df.Field(mesh, dim=3, value=(0, 2, 5))
        ...
        >>> res = f1 * 5  # discretisedfield.Field.__mul__ is called
        >>> res.average
        (0.0, 10.0, 25.0)
        >>> res = 10 * f1  # discretisedfield.Field.__rmul__ is called
        >>> res.average
        (0.0, 20.0, 50.0)

        .. seealso:: :py:func:`~discretisedfield.Field.__truediv__`

        """
        if not isinstance(other, (self.__class__, numbers.Real)):
            msg = (f'Unsupported operand type(s) for *: '
                   f'{type(self)} and {type(other)}.')
            raise TypeError(msg)
        if isinstance(other, self.__class__):
            if self.mesh != other.mesh:
                msg = 'Cannot multiply fields defined on different meshes.'
                raise ValueError(msg)
            if not (self.dim == 1 or other.dim == 1):
                msg = (f'Cannot multiply dim={self.dim} and '
                       f'dim={other.dim} fields.')
                raise ValueError(msg)
            res_array = np.multiply(self.array, other.array)
        if isinstance(other, numbers.Real):
            res_array = np.multiply(self.array, other)

        return self.__class__(self.mesh, dim=res_array.shape[-1],
                              value=res_array)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Binary ``/`` operator.

        It can be applied between:

        1. Two scalar (``dim=1``) fields,

        2. A field of any dimension and ``numbers.Real``, or

        3. A field of any dimension and a scalar (``dim=1``) field.

        If both operands are ``discretisedfield.Field`` objects, they must be
        defined on the same mesh.

        Parameters
        ----------
        other : discretisedfield.Field, numbers.Real

            Second operand.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Divide two scalar fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=1, value=100)
        >>> f2 = df.Field(mesh, dim=1, value=20)
        >>> res = f1 / f2
        >>> res.average
        5.0
        >>> f1 / f2 == (f2 / f1)**(-1)
        True

        2. Divide vector field by a scalar.

        >>> f1 = df.Field(mesh, dim=3, value=(0, 10, 5))
        >>> res = f1 / 5  # discretisedfield.Field.__mul__ is called
        >>> res.average
        (0.0, 2.0, 1.0)
        >>> 10 / f1  # division by a vector is not allowed
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.__mul__`

        """
        return self * other**(-1)

    def __rtruediv__(self, other):
        return self**(-1) * other

    def __matmul__(self, other):
        """Binary ``@`` operator, defined as dot product.

        This method computes the dot product between two fields. Both fields
        must be three-dimensional (``dim=3``) and defined on the same mesh.

        Parameters
        ----------
        other : discretisedfield.Field

            Second operand.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError, TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Compute the dot product of two vector fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> cell = (2e-9, 2e-9, 2e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=3, value=(1, 3, 6))
        >>> f2 = df.Field(mesh, dim=3, value=(-1, -2, 2))
        >>> (f1@f2).average
        5.0

        """
        if not isinstance(other, df.Field):
            msg = (f'Unsupported operand type(s) for @: '
                   f'{type(self)} and {type(other)}.')
            raise TypeError(msg)
        if self.dim != 3 or other.dim != 3:
            msg = (f'Cannot compute the dot product on '
                   f'dim={self.dim} and dim={other.dim} fields.')
            raise ValueError(msg)
        if self.mesh != other.mesh:
            msg = ('Cannot compute the dot product of '
                   'fields defined on different meshes.')
            raise ValueError(msg)

        res_array = np.einsum('ijkl,ijkl->ijk', self.array, other.array)
        return df.Field(self.mesh, dim=1, value=res_array[..., np.newaxis])

    def derivative(self, direction):
        """Directional derivative.

        This method computes a directional derivative of the field and returns
        a field (with the same dimension). The direction in which the
        derivative is computed is passed via ``direction`` argument, which can
        be ``'x'``, ``'y'``, or ``'z'``. Alternatively, ``0``, ``1``, or ``2``
        can be passed, respectively.

        Directional derivative cannot be computed if only one discretisation
        cell exists in a specified direction. In that case, a zero field is
        returned. More precisely, it is assumed that the field does not change
        in that direction.

        Parameters
        ----------
        direction : str, int

            The direction in which the derivative is computed. It can be
            ``'x'``, ``'y'``, or ``'z'`` (alternatively, 0, 1, or 2,
            respectively).

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Example
        -------
        1. Compute the directional derivative of a scalar field in the
        y-direction of a spatially varying field. For the field we choose
        :math:`f(x, y, z) = 2x + 3y - 5z`. Accordingly, we expect the
        derivative in the y-direction to be to be a constant scalar field
        :math:`df/dy = 3`.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 10e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> value_fun = lambda pos: 2*pos[0] + 3*pos[1] - 5*pos[2]
        >>> f = df.Field(mesh, dim=1, value=value_fun)
        >>> f.derivative('y').average
        3.0

        2. Try to compute directional derivatives of the vector field which has
        only one discretisation cell in the z-direction. For the field we
        choose :math:`f(x, y, z) = (2x, 3y, -5z)`. Accordingly, we expect the
        directional derivatives to be: :math:`df/dx = (2, 0, 0)`,
        :math:`df/dy=(0, 3, 0)`, :math:`df/dz = (0, 0, -5)`. However, because
        there is only one discretisation cell in the z-direction, the
        derivative cannot be computed and a zero field is returned.

        >>> def value_fun(pos):
        ...     x, y, z = pos
        ...     return (2*x, 3*y, -5*z)
        ...
        >>> f = df.Field(mesh, dim=3, value=value_fun)
        >>> f.derivative('x').average
        (2.0, 0.0, 0.0)
        >>> f.derivative('y').average
        (0.0, 3.0, 0.0)
        >>> f.derivative('z').average  # derivative cannot be calculated
        (0.0, 0.0, 0.0)

        """
        if isinstance(direction, str):
            direction = dfu.axesdict[direction]

        if self.mesh.n[direction] == 1:
            derivative_array = 0  # derivative cannot be computed
        elif self.dim == 1:
            derivative_array = np.gradient(self.array[..., 0],
                                           self.mesh.cell[direction],
                                           axis=direction)[..., np.newaxis]
        else:
            derivative_array = np.gradient(self.array,
                                           self.mesh.cell[direction],
                                           axis=direction)

        return self.__class__(self.mesh, dim=self.dim, value=derivative_array)

    @property
    def grad(self):
        """Gradient.

        This method computes the gradient of a scalar (``dim=1``) field and
        returns a vector field:

        .. math::

            \\nabla f = (\\frac{\\partial f}{\\partial x},
                         \\frac{\\partial f}{\\partial y},
                         \\frac{\\partial f}{\\partial z})

        Directional derivative cannot be computed if only one discretisation
        cell exists in a certain direction. In that case, a zero field is
        considered to be that directional derivative. More precisely, it is
        assumed that the field does not change in that direction.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError

            If the dimension of the field is not 1.

        Example
        -------
        1. Compute gradient of a contant field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> cell = (2e-9, 2e-9, 2e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=1, value=5)
        >>> f.grad.average
        (0.0, 0.0, 0.0)

        2. Compute gradient of a spatially varying field. For a field we choose
        :math:`f(x, y, z) = 2x + 3y - 5z`. Accordingly, we expect the gradient
        to be a constant vector field :math:`\\nabla f = (2, 3, -5)`.

        >>> def value_fun(pos):
        ...     x, y, z = pos
        ...     return 2*x + 3*y - 5*z
        ...
        >>> f = df.Field(mesh, dim=1, value=value_fun)
        >>> f.grad.average
        (2.0, 3.0, -5.0)

        3. Attempt to compute the gradient of a vector field.

        >>> f = df.Field(mesh, dim=3, value=(1, 2, -3))
        >>> f.grad
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.dim != 1:
            msg = f'Cannot compute gradient for dim={self.dim} field.'
            raise ValueError(msg)

        return df.stack([self.derivative('x'),
                         self.derivative('y'),
                         self.derivative('z')])

    @property
    def div(self):
        """Divergence.

        This method computes the divergence of a vector (``dim=3``) field and
        returns a scalar (``dim=1``) field as a result.

        .. math::

            \\nabla\\cdot\\mathbf{v} = \\frac{\\partial v_{x}}{\\partial x} +
                                       \\frac{\\partial v_{y}}{\\partial y} +
                                       \\frac{\\partial v_{z}}{\\partial z}

        Directional derivative cannot be computed if only one discretisation
        cell exists in a certain direction. In that case, a zero field is
        considered to be that directional derivative. More precisely, it is
        assumed that the field does not change in that direction.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError

            If the dimension of the field is not 3.

        Example
        -------
        1. Compute the divergence of a vector field. For a field we choose
        :math:`\\mathbf{v}(x, y, z) = (2x, -2y, 5z)`. Accordingly, we expect
        the divergence to be to be a constant scalar field :math:`\\nabla\\cdot
        \\mathbf{v} = 5`.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 100e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> def value_fun(pos):
        ...     x, y, z = pos
        ...     return (2*x, -2*y, 5*z)
        ...
        >>> f = df.Field(mesh, dim=3, value=value_fun)
        >>> f.div.average
        5.0

        2. Attempt to compute the divergence of a scalar field.

        >>> f = df.Field(mesh, dim=1, value=3.14)
        >>> f.div
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.dim != 3:
            msg = f'Cannot compute divergence for dim={self.dim} field.'
            raise ValueError(msg)

        return (self.x.derivative('x') +
                self.y.derivative('y') +
                self.z.derivative('z'))

    @property
    def curl(self):
        """Curl.

        This method computes the curl of a vector (``dim=3``) field and returns
        a vector (``dim=3``) as a result:

        .. math::

            \\nabla \\times \\mathbf{v} = \\left(\\frac{\\partial
            v_{z}}{\\partial y} - \\frac{\\partial v_{y}}{\\partial z},
            \\frac{\\partial v_{x}}{\\partial z} - \\frac{\\partial
            v_{z}}{\\partial x}, \\frac{\\partial v_{y}}{\\partial x} -
            \\frac{\\partial v_{x}}{\\partial y},\\right)

        Directional derivative cannot be computed if only one discretisation
        cell exists in a certain direction. In that case, a zero field is
        considered to be that directional derivative. More precisely, it is
        assumed that the field does not change in that direction.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError

            If the dimension of the field is not 3.

        Example
        -------
        1. Compute curl of a vector field. For a field we choose
        :math:`\\mathbf{v}(x, y, z) = (2xy, -2y, 5xz)`. Accordingly, we expect
        the curl to be to be a constant vector field :math:`\\nabla\\times
        \\mathbf{v} = (0, -5z, -2x)`.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> def value_fun(pos):
        ...     x, y, z = pos
        ...     return (2*x*y, -2*y, 5*x*z)
        ...
        >>> f = df.Field(mesh, dim=3, value=value_fun)
        >>> f.curl((1, 1, 1))
        (0.0, -5.0, -2.0)

        2. Attempt to compute the curl of a scalar field.

        >>> f = df.Field(mesh, dim=1, value=3.14)
        >>> f.curl
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.dim != 3:
            msg = f'Cannot compute curl for dim={self.dim} field.'
            raise ValueError(msg)

        curl_x = self.z.derivative('y') - self.y.derivative('z')
        curl_y = self.x.derivative('z') - self.z.derivative('x')
        curl_z = self.y.derivative('x') - self.x.derivative('y')

        return df.stack([curl_x, curl_y, curl_z])

    @property
    def integral(self):
        """Volume integral.

        This method integrates the field over volume and returns a single
        (scalar or vector) value as ``tuple``. This value can be understood as
        the product of field's average value and the mesh volume, because the
        volume of all discretisation cells is the same.

        Returns
        -------
        tuple

            Volume integral.

        Example
        -------
        1. Compute the volume integral of a scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=1, value=5)
        >>> f.integral
        5000.0

        2. Compute the volume integral of a vector field.

        >>> f = df.Field(mesh, dim=3, value=(-1, -2, -3))
        >>> f.integral
        (-1000.0, -2000.0, -3000.0)

        .. seealso::

            :py:func:`~discretisedfield.Field.average`
            :py:func:`~discretisedfield.Mesh.volume`

        """
        cell_volume = self.mesh.region.volume / len(self.mesh)
        field_sum = np.sum(self.array, axis=(0, 1, 2))
        return dfu.array2tuple(field_sum * cell_volume)

    @property
    def topological_charge_density(self):
        """Topological charge density.

        This method computes the topological charge density for the vector
        (``dim=3``) field:

        .. math::

            q = \\frac{1}{4\\pi} \\mathbf{n} \\cdot \\left(\\frac{\\partial
            \\mathbf{n}}{\\partial x} \\times \\frac{\\partial
            \\mathbf{n}}{\\partial x} \\right),

        where :math:`\\mathbf{n}` is the orientation field. Topological charge
        is defined on two-dimensional samples only. Therefore, the field must
        be "sliced" using the ``discretisedfield.Field.plane`` method. If the
        field is not three-dimensional or the field is not sliced,
        ``ValueError`` is raised.

        Returns
        -------
        discretisedfield.Field

            Topological charge density as a scalar field.

        Raises
        ------
        ValueError

            If the field is not three-dimensional or the field is not sliced.

        Example
        -------
        1. Compute the topological charge density of a spatially constant
        vector field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=3, value=(1, 1, -1))
        >>> f.plane('z').topological_charge_density.average
        0.0

        2. Attempt to compute the topological charge density of a scalar field.

        >>> f = df.Field(mesh, dim=1, value=12)
        >>> f.plane('z').topological_charge_density
        Traceback (most recent call last):
        ...
        ValueError: ...

        3. Attempt to compute the topological charge density of a vector field,
        which is not sliced.

        >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
        >>> f.topological_charge_density
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.topological_charge`

        """
        if self.dim != 3:
            msg = (f'Cannot compute topological charge density '
                   f'for dim={self.dim} field.')
            raise ValueError(msg)
        if not hasattr(self.mesh, 'info'):
            msg = ('The field must be sliced before the topological '
                   'charge density can be computed.')
            raise ValueError(msg)

        of = self.orientation  # unit (orientation) field
        thickness = self.mesh.cell[self.mesh.info['planeaxis']]
        prefactor = 1 / (4 * np.pi * thickness)
        q = of @ df.cross(of.derivative(self.mesh.info['axis1']),
                          of.derivative(self.mesh.info['axis2']))
        return prefactor * q

    @property
    def bergluescher(self):
        """Topological charge computed using Berg-Luescher method.

        The details of this method can be found in Berg and Luescher, Nuclear
        Physics, Section B, Volume 190, Issue 2, p. 412-424.

        This method computes the topological charge for the vector field
        (``dim=3``). Topological charge is defined on two-dimensional samples.
        Therefore, the field must be "sliced" using
        ``discretisedfield.Field.plane`` method. If the field is not
        three-dimensional or the field is not sliced, ``ValueError`` is raised.

        Returns
        -------
        float

            Topological charge.

        Raises
        ------
        ValueError

            If the field does not have ``dim=3`` or the field is not sliced.

        Example
        -------
        1. Compute the topological charge of a spatially constant vector field.
        Zero value is expected.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=3, value=(1, 1, -1))
        >>> f.plane('z').bergluescher
        0.0
        >>> f.plane('z').bergluescher
        0.0

        .. seealso::

            :py:func:`~discretisedfield.Field.topological_charge`
            :py:func:`~discretisedfield.Field.tological_charge_density`

        """
        if self.dim != 3:
            msg = (f'Cannot compute Berg-Luescher topological charge '
                   f'for dim={self.dim} field.')
            raise ValueError(msg)
        if not hasattr(self.mesh, 'info'):
            msg = ('The field must be sliced before the Berg-Luescher '
                   'topological charge can be computed.')
            raise ValueError(msg)

        axis1 = self.mesh.info['axis1']
        axis2 = self.mesh.info['axis2']
        of = self.orientation  # unit (orientation) field

        topological_charge = 0
        for i, j in itertools.product(range(of.mesh.n[axis1]-1),
                                      range(of.mesh.n[axis2]-1)):
            v1 = of.array[dfu.assemble_index({axis1: i, axis2: j})]
            v2 = of.array[dfu.assemble_index({axis1: i+1, axis2: j})]
            v3 = of.array[dfu.assemble_index({axis1: i+1, axis2: j+1})]
            v4 = of.array[dfu.assemble_index({axis1: i, axis2: j+1})]

            triangle1 = dfu.bergluescher_angle(v1, v2, v4)
            triangle2 = dfu.bergluescher_angle(v2, v3, v4)

            topological_charge += triangle1 + triangle2

        return topological_charge

    def topological_charge(self, method='continuous'):
        """Topological charge.

        This method computes the topological charge for the vector field
        (``dim=3``). There are two possible methods, which can be chosen using
        ``method`` parameter:

        1. ``continuous``: Topological charge density is integrated.

        2. ``berg-luescher``: Topological charge is computed on a discrete
        lattice, as described in: Berg and Luescher, Nuclear Physics, Section
        B, Volume 190, Issue 2, p. 412-424.

        Topological charge is defined on two-dimensional samples. Therefore,
        the field must be "sliced" using ``discretisedfield.Field.plane``
        method. If the field is not three-dimensional or the field is not
        sliced, ``ValueError`` is raised.

        Parameters
        ----------
        method : str, optional

            Method how the topological charge is computed. It can be
            ``continuous`` or ``berg-luescher``. Defaults to ``continuous``.

        Returns
        -------
        float

            Topological charge.

        Raises
        ------
        ValueError

            If the field does not have ``dim=3`` or the field is not sliced.

        Example
        -------
        1. Compute the topological charge of a spatially constant vector field.
        Zero value is expected.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=3, value=(1, 1, -1))
        >>> f.plane('z').topological_charge(method='continuous')
        0.0
        >>> f.plane('z').topological_charge(method='berg-luescher')
        0.0

        2. Attempt to compute the topological charge of a scalar field.

        >>> f = df.Field(mesh, dim=1, value=12)
        >>> f.plane('z').topological_charge()
        Traceback (most recent call last):
        ...
        ValueError: ...

        3. Attempt to compute the topological charge of a vector field, which
        is not sliced.

        >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
        >>> f.topological_charge_density()
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso::

            :py:func:`~discretisedfield.Field.tological_charge_density`
            :py:func:`~discretisedfield.Field.bergluescher`

        """
        if method == 'continuous':
            return self.topological_charge_density.integral
        elif method == 'berg-luescher':
            return self.bergluescher
        else:
            msg = 'Method can be either continuous or berg-luescher'
            raise ValueError(msg)

    def line(self, p1, p2, n=100):
        """Sampling the field along the line.

        Given two points :math:`p_{1}` and :math:`p_{2}`, :math:`n`
        position coordinates are generated and the corresponding field
        values.

        .. math::

           \\mathbf{r}_{i} = i\\frac{\\mathbf{p}_{2} -
           \\mathbf{p}_{1}}{n-1}

        Parameters
        ----------
        p1, p2 : (3,) array_like
            Two points between which the line is generated.
        n : int
            Number of points on the line.

        Yields
        ------
        tuple
            The first element is the coordinate of the point on the
            line, whereas the second one is the value of the field.

        Raises
        ------
        ValueError
            If `p1` or `p2` is outside the mesh domain.

        Examples
        --------
        1. Sampling the field along the line.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh, dim=2, value=(0, 3))
        >>> field.line(p1=(0, 0, 0), p2=(2, 0, 0), n=5)
        Line(...)

        """
        points = list(self.mesh.line(p1=p1, p2=p2, n=n))
        values = [self(p) for p in points]
        return df.Line(points=points, values=values)

    def plane(self, *args, n=None, **kwargs):
        """Extracts field on the plane mesh.

        If one of the axes (``'x'``, ``'y'``, or ``'z'``) is passed as a
        string, a plane mesh perpendicular to that axis is extracted,
        intersecting the mesh region at its centre, and the field is sampled on
        that mesh. Alternatively, if a keyword argument is passed (e.g.
        ``x=1e-9``), a plane perpendicular to the x-axis (parallel to yz-plane)
        and intersecting it at ``x=1e-9`` is extracted. The number of points in
        two dimensions on the plane can be defined using ``n`` tuple (e.g.
        ``n=(10, 15)``).

        Parameters
        ----------
        n : (2,) tuple

            The number of points on the plane in two dimensions.

        Returns
        ------
        discretisedfield.Field

            An extracted field.

        Examples
        --------
        1. Extracting the field on a plane at a specific point.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> f = df.Field(mesh, dim=3, value=(0, 0, 1))
        ...
        >>> f.plane(y=1)
        Field(...)

        2. Extracting the field at the mesh region centre.

        >>> f.plane('z')
        Field(...)

        3. Specifying the number of points.

        >>> f.plane('z', n=(10, 10))
        Field(...)

        .. seealso:: :py:func:`~discretisedfield.Mesh.plane`

        """
        plane_mesh = self.mesh.plane(*args, n=n, **kwargs)
        return self.__class__(plane_mesh, dim=self.dim, value=self)

    def __getitem__(self, key):
        """Extracts the field on a subregion.

        If subregions were defined by passing ``subregions`` dictionary when
        the mesh was created, this method returns a field in a subregion
        ``subregions[key]`` with the same discretisation cell as the parent
        mesh.

        Parameters
        ----------
        key : str

            The key of a region in ``subregions`` dictionary.

        Returns
        -------
        disretisedfield.Field

            Field on a subregion.

        Example
        -------
        1. Extract field on the subregion.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (10, 10, 10)
        >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        >>> def value_fun(pos):
        ...     x, y, z = pos
        ...     if x <= 50:
        ...         return (1, 2, 3)
        ...     else:
        ...         return (-1, -2, -3)
        ...
        >>> f = df.Field(mesh, dim=3, value=value_fun)
        >>> f.average
        (0.0, 0.0, 0.0)
        >>> f['r1']
        Field(...)
        >>> f['r1'].average
        (1.0, 2.0, 3.0)
        >>> f['r2'].average
        (-1.0, -2.0, -3.0)

        """
        return self.__class__(self.mesh[key], dim=self.dim, value=self)

    def project(self, *args):
        """Projects the field along one direction and averages it out along
        that direction.

        One of the axes (``'x'``, ``'y'``, or ``'z'``) is passed and the field
        is projected (averaged) along that direction. For example
        ``project('z')`` would average the field in the z-direction and return
        the field which has only one discretisation cell in the z-direction.

        Returns
        ------
        discretisedfield.Field

            An extracted field.

        Example
        -------
        1. Projecting the field along a certain direction.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 3))
        ...
        >>> field.project('z')
        Field(...)
        >>> field.project('z').average
        (1.0, 2.0, 3.0)
        >>> field.project('z').array.shape
        (2, 2, 1, 3)

        """
        plane_mesh = self.mesh.plane(*args)
        project_array = self.array.mean(axis=plane_mesh.info['planeaxis'],
                                        keepdims=True)
        return self.__class__(plane_mesh, dim=self.dim, value=project_array)

    def write(self, filename, representation='txt', extend_scalar=False):
        """Write the field to OVF, HDF5, or VTK file.

        If the extension of ``filename`` is ``.vtk``, a VTK file is written
        (:py:func:`~discretisedfield.Field._writevtk`).

        For ``.ovf``, ``.omf``, or ``.ohf`` extensions, the field is saved to
        OVF file (:py:func:`~discretisedfield.Field._writeovf`). In that case,
        the representation of data (``'bin4'``, ``'bin8'``, or ``'txt'``) is
        passed as ``representation`` and if ``extend_scalar=True``, a scalar
        field will be saved as a vector field. More precisely, if the value at
        a cell is X, that cell will be saved as (X, 0, 0).

        Finally, if the extension of ``filename`` is ``.hdf5``, HDF5 file will
        be written (:py:func:`~discretisedfield.Field._writehdf5`).

        Parameters
        ----------
        filename : str

            Name of the file written.

        representation : str, optional

            In the case of OVF files (``.ovf``, ``.omf``, or ``.ohf``),
            representation can be specified (``'bin4'``, ``'bin8'``, or
            ``'txt'``). Defaults to ``'txt'``.

        extend_scalar : bool, optional

            If ``True``, a scalar field will be saved as a vector field. More
            precisely, if the value at a cell is 3, that cell will be saved as
            (3, 0, 0). This is valid only for the OVF file formats. Defaults to
            ``False``.

        Example
        -------
        1. Write field to the OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5e-9)
        >>> p2 = (5e-9, 15e-9, 15e-9)
        >>> n = (5, 15, 20)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(5, 6, 7))
        ...
        >>> filename = 'mytestfile.omf'
        >>> field.write(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.fromfile(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        2. Write field to the VTK file.

        >>> filename = 'mytestfile.vtk'
        >>> field.write(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> os.remove(filename)  # delete the file

        3. Write field to the HDF5 file.

        >>> filename = 'mytestfile.hdf5'
        >>> field.write(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.fromfile(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        .. seealso:: :py:func:`~discretisedfield.Field.fromfile`

        """
        if any([filename.endswith(ext) for ext in ['.omf', '.ovf', '.ohf']]):
            self._writeovf(filename, representation=representation,
                           extend_scalar=extend_scalar)
        elif any([filename.endswith(ext) for ext in ['.hdf5', '.h5']]):
            self._writehdf5(filename)
        elif filename.endswith('.vtk'):
            self._writevtk(filename)
        else:
            msg = (f'Writing file with extension {filename.split(".")[-1]} '
                   f'not supported.')
            raise ValueError(msg)

    def _writeovf(self, filename, representation='txt', extend_scalar=False):
        """Write the field to an OVF file.

        The extension of ``filename`` should be ``.ovf``, ``.omf``, or
        ``.ohf``. Data representation (``'bin4'``, ``'bin8'``, or ``'txt'``)
        can be passed using ``representation`` argument. If
        ``extend_scalar=True``, a scalar field will be saved as a vector field.
        More precisely, if the value at a cell is X, that cell will be saved as
        (X, 0, 0).

        Parameters
        ----------
        filename : str

            Name with an extension of the file written.

        representation : str, optional

            Representation can be specified by passing ``'bin4'``, ``'bin8'``,
            or ``'txt'``. Defaults to ``'txt'``.

        extend_scalar : bool, optional

            If ``True``, a scalar field will be saved as a vector field. More
            precisely, if the value at a cell is 3, that cell will be saved as
            (3, 0, 0). Defaults to ``False``.

        Example
        -------
        1. Write field to the OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 5e-9, 3e-9)
        >>> n = (10, 5, 3)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> value_fun = lambda pos: (pos[0], pos[1], pos[2])
        >>> field = df.Field(mesh, dim=3, value=value_fun)
        ...
        >>> filename = 'mytestfile.ohf'
        >>> field._writeovf(filename, representation='bin8')  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.fromfile(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        .. seealso:: :py:func:`~discretisedfield.Field.fromfile`

        """
        if self.dim != 1 and self.dim != 3:
            msg = (f'Cannot write dim={self.dim} field.')
            raise TypeError(msg)

        if extend_scalar and self.dim == 1:
            write_dim = 3
        else:
            write_dim = self.dim

        header = ['OOMMF OVF 2.0',
                  '',
                  'Segment count: 1',
                  '',
                  'Begin: Segment',
                  'Begin: Header',
                  '',
                  'Title: Field generated omf file',
                  'Desc: File generated by Field class',
                  'meshunit: m',
                  'meshtype: rectangular',
                  f'xbase: {self.mesh.region.pmin[0] + self.mesh.cell[0]/2}',
                  f'ybase: {self.mesh.region.pmin[1] + self.mesh.cell[1]/2}',
                  f'zbase: {self.mesh.region.pmin[2] + self.mesh.cell[2]/2}',
                  f'xnodes: {self.mesh.n[0]}',
                  f'ynodes: {self.mesh.n[1]}',
                  f'znodes: {self.mesh.n[2]}',
                  f'xstepsize: {self.mesh.cell[0]}',
                  f'ystepsize: {self.mesh.cell[1]}',
                  f'zstepsize: {self.mesh.cell[2]}',
                  f'xmin: {self.mesh.region.pmin[0]}',
                  f'ymin: {self.mesh.region.pmin[1]}',
                  f'zmin: {self.mesh.region.pmin[2]}',
                  f'xmax: {self.mesh.region.pmax[0]}',
                  f'ymax: {self.mesh.region.pmax[1]}',
                  f'zmax: {self.mesh.region.pmax[2]}',
                  f'valuedim: {write_dim}',
                  f'valuelabels: field_x field_y field_z',
                  'valueunits: A/m A/m A/m',
                  '',
                  'End: Header',
                  '']

        if representation == 'bin4':
            header.append('Begin: Data Binary 4')
            footer = ['End: Data Binary 4',
                      'End: Segment']
        elif representation == 'bin8':
            header.append('Begin: Data Binary 8')
            footer = ['End: Data Binary 8',
                      'End: Segment']
        elif representation == 'txt':
            header.append('Begin: Data Text')
            footer = ['End: Data Text',
                      'End: Segment']

        # Write header lines
        with open(filename, 'w') as f:
            f.write(''.join(map(lambda line: f'# {line}\n', header)))

        binary_reps = {'bin4': (1234567.0, 'f'),
                       'bin8': (123456789012345.0, 'd')}

        if representation in binary_reps:
            # Reopen with binary write, appending to the end of the file.
            with open(filename, 'ab') as f:

                # Add the 8 bit binary check value that OOMMF uses.
                packarray = [binary_reps[representation][0]]

                # Write data to the ovf file.
                for i in self.mesh.indices:
                    for vi in self.array[i]:
                        packarray.append(vi)

                pack_fmt = binary_reps[representation][1]*len(packarray)
                f.write(struct.pack(pack_fmt, *packarray))

        else:
            # Reopen with txt representation, appending to the end of the file.
            with open(filename, 'a') as f:
                for i in self.mesh.indices:
                    if self.dim == 3:
                        v = [vi for vi in self.array[i]]
                    else:
                        if extend_scalar:
                            v = [self.array[i][0], 0.0, 0.0]
                        else:
                            v = [self.array[i][0]]
                    for vi in v:
                        f.write(f' {str(vi)}')
                    f.write('\n')

        # Write footer lines to OOMMF file.
        with open(filename, 'a') as f:
            f.write(''.join(map(lambda line: f'# {line}\n', footer)))

    def _writevtk(self, filename):
        """Write the field to a VTK file.

        Parameters
        ----------
        filename : str

            Name with an extension of the file written.

        Example
        -------
        1. Write field to a VTK file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 5e-9, 3e-9)
        >>> n = (10, 5, 3)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> value_fun = lambda pos: (pos[0], pos[1], pos[2])
        >>> field = df.Field(mesh, dim=3, value=value_fun)
        ...
        >>> filename = 'mytestfile.vtk'
        >>> field._writevtk(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> os.remove(filename)  # delete the file

        """
        grid = [pmini + np.linspace(0, li, ni+1) for pmini, li, ni in
                zip(self.mesh.region.pmin,
                    self.mesh.region.edges,
                    self.mesh.n)]

        structure = pyvtk.RectilinearGrid(*grid)
        vtkdata = pyvtk.VtkData(structure)

        vectors = [self.__call__(coord) for coord in self.mesh]
        vtkdata.cell_data.append(pyvtk.Vectors(vectors, 'field'))
        for i, component in enumerate(dfu.axesdict.keys()):
            name = f'field_{component}'
            vtkdata.cell_data.append(pyvtk.Scalars(list(zip(*vectors))[i],
                                                   name))

        vtkdata.tofile(filename)

    def _writehdf5(self, filename):
        """Write the field to an HDF5 file.

        Parameters
        ----------
        filename : str

            Name with an extension of the file written.

        Example
        -------
        1. Write field to an HDF5 file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 5e-9, 3e-9)
        >>> n = (10, 5, 3)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> value_fun = lambda pos: (pos[0], pos[1], pos[2])
        >>> field = df.Field(mesh, dim=3, value=value_fun)
        ...
        >>> filename = 'mytestfile.h5'
        >>> field._writehdf5(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.fromfile(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        .. seealso:: :py:func:`~discretisedfield.Field.fromfile`

        """
        with h5py.File(filename, 'w') as f:
            # Set up the file structure
            gfield = f.create_group('field')
            gmesh = gfield.create_group('mesh')
            gregion = gmesh.create_group('region')

            # Save everything as datasets
            gregion.create_dataset('p1', data=self.mesh.region.p1)
            gregion.create_dataset('p2', data=self.mesh.region.p2)
            gmesh.create_dataset('n', dtype='i4', data=self.mesh.n)
            gfield.create_dataset('dim', dtype='i4', data=self.dim)
            gfield.create_dataset('array', data=self.array)

    @classmethod
    def fromfile(cls, filename):
        """Read the field from an OVF or HDF5 file.

        The extension of the ``filename`` should be suitable for OVF format
        (``.ovf``, ``.omf``, ``.ohf``) or for HDF5 (``.hdf5`` or ``.h5``). This
        is a ``classmethod`` and should be called as, for instance,
        ``discretisedfield.Field.fromfile('myfile.omf')``.

        Parameters
        ----------
        filename : str

            Name of the file to be read.

        Returns
        -------
        discretisedfield.Field

            Field read from the file.

        Example
        -------
        1. Read a field from the OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> dirname = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'mumax-output-linux.ovf')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(mesh=...)

        2. Read a field from the HDF5 file.

        >>> filename = os.path.join(dirname, 'testfile.hdf5')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(mesh=...)

        .. seealso:: :py:func:`~discretisedfield.Field.write`

        """
        if any([filename.endswith(ext) for ext in ['.omf', '.ovf', '.ohf']]):
            return cls._fromovf(filename)
        elif any([filename.endswith(ext) for ext in ['.hdf5', '.h5']]):
            return cls._fromhdf5(filename)
        else:
            msg = (f'Reading file with extension {filename.split(".")[-1]} '
                   f'not supported.')
            raise ValueError(msg)

    @classmethod
    def _fromovf(cls, filename):
        """Read the field from an OVF file.

        The extension of the ``filename`` should be suitable for OVF format
        (``.ovf``, ``.omf``, ``.ohf``). This is a ``classmethod`` and should be
        called as, for instance,
        ``discretisedfield.Field._fromovf('myfile.omf')``.

        Parameters
        ----------
        filename : str

            Name of the file to be read.

        Returns
        -------
        discretisedfield.Field

            Field read from the file.

        Example
        -------
        1. Read a field from the OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> dirname = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'mumax-output-linux.ovf')
        >>> field = df.Field._fromovf(filename)
        >>> field
        Field(mesh=...)

        .. seealso:: :py:func:`~discretisedfield.Field._writeovf`

        """
        mdatalist = ['xmin', 'ymin', 'zmin', 'xmax', 'ymax', 'zmax',
                     'xstepsize', 'ystepsize', 'zstepsize', 'valuedim']
        mdatadict = dict()

        try:
            with open(filename, 'r', encoding='utf-8') as ovffile:
                f = ovffile.read()
                lines = f.split('\n')

            mdatalines = filter(lambda s: s.startswith('#'), lines)
            datalines = np.loadtxt(filter(lambda s: not s.startswith('#'),
                                          lines))

            for line in mdatalines:
                for mdatum in mdatalist:
                    if mdatum in line:
                        mdatadict[mdatum] = float(line.split()[-1])
                        break

        except UnicodeDecodeError:
            with open(filename, 'rb') as ovffile:
                f = ovffile.read()
                lines = f.split(b'\n')

            mdatalines = filter(lambda s: s.startswith(bytes('#', 'utf-8')),
                                lines)

            for line in mdatalines:
                for mdatum in mdatalist:
                    if bytes(mdatum, 'utf-8') in line:
                        mdatadict[mdatum] = float(line.split()[-1])
                        break

            header = b'# Begin: Data Binary '
            data_start = f.find(header)
            header = f[data_start:data_start + len(header) + 1]

            data_start += len(b'# Begin: Data Binary 8')
            data_end = f.find(b'# End: Data Binary ')

            # ordered by length
            newlines = [b'\n\r', b'\r\n', b'\n']
            for nl in newlines:
                if f.startswith(nl, data_start):
                    data_start += len(nl)
                    break

            if b'4' in header:
                formatstr = '@f'
                checkvalue = 1234567.0
            elif b'8' in header:
                formatstr = '@d'
                checkvalue = 123456789012345.0

            listdata = list(struct.iter_unpack(formatstr,
                                               f[data_start:data_end]))
            datalines = np.array(listdata)

            if datalines[0] != checkvalue:
                # These two lines cannot be accessed via tests. Therefore, they
                # are excluded from coverage.
                msg = 'Binary Data cannot be read.'  # pragma: no cover
                raise AssertionError(msg)  # pragma: no cover

            datalines = datalines[1:]  # check value removal

        p1 = (mdatadict[key] for key in ['xmin', 'ymin', 'zmin'])
        p2 = (mdatadict[key] for key in ['xmax', 'ymax', 'zmax'])
        cell = (mdatadict[key] for key in ['xstepsize', 'ystepsize',
                                           'zstepsize'])
        dim = int(mdatadict['valuedim'])

        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        field = cls(mesh, dim=dim)

        r_tuple = (*tuple(reversed(field.mesh.n)), int(mdatadict['valuedim']))
        t_tuple = (*tuple(reversed(range(3))), 3)
        field.array = datalines.reshape(r_tuple).transpose(t_tuple)

        return field

    @classmethod
    def _fromhdf5(cls, filename):
        """Read the field from an HDF5 file.

        The extension of the ``filename`` should be suitable for the HDF5
        format (``.hdf5`` or ``.h5``). This is a ``classmethod`` and should be
        called as, for instance,
        ``discretisedfield.Field._fromhdf5('myfile.h5')``.

        Parameters
        ----------
        filename : str

            Name of the file to be read.

        Returns
        -------
        discretisedfield.Field

            Field read from the file.

        Example
        -------
        1. Read a field from the HDF5 file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> dirname = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'testfile.hdf5')
        >>> field = df.Field._fromhdf5(filename)
        >>> field
        Field(mesh=...)

        .. seealso:: :py:func:`~discretisedfield.Field._writehdf5`

        """
        with h5py.File(filename, 'r') as f:
            # Read data from the file.
            p1 = f['field/mesh/region/p1']
            p2 = f['field/mesh/region/p2']
            n = np.array(f['field/mesh/n']).tolist()
            dim = np.array(f['field/dim']).tolist()
            array = f['field/array']

            # Create field.
            region = df.Region(p1=p1, p2=p2)
            mesh = df.Mesh(region=region, n=n)
            return cls(mesh, dim=dim, value=array[:])

    def mpl(self, ax=None, figsize=None, multiplier=None):
        """Plots the field on a plane using ``matplotlib``.

        If ``ax`` is not passed, axes will be created automaticaly. In that
        case, the figure size can be changed using ``figsize``. It is often the
        case that the region size is small (e.g. on a nanoscale) or very large
        (e.g. in units of kilometers). Accordingly, ``multiplier`` can be
        passed as :math:`10^{n}`, where :math:`n` is a multiple of 3 (..., -6,
        -3, 0, 3, 6,...). According to that value, the axes will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, all mesh points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels. If ``multiplier``
        is not passed, the optimum one is computed internally.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). Otherwise, ``ValueError`` is raised. For vector
        fields, this method plots both ``quiver`` (vector) and ``imshow``
        (scalar) plots. The ``imshow`` plot represents the value of the
        out-of-plane vector component and the ``quiver`` plot is not coloured.
        On the other hand, only ``imshow`` is plotted for scalar fields. Where
        the norm of the field is zero, no vectors are shown and those
        ``imshow`` pixels are not coloured.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which field plot should be added. Defaults to ``None`` -
            new axes will be created in figure with size defined as
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

        Raises
        ------
        ValueError

            If the field has not been sliced with a plane.

        Example
        -------
        1. Visualising the field using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> field.plane(z=50, n=(5, 5)).mpl()

        .. seealso::

            :py:func:`~discretisedfield.Field.k3d_voxels`
            :py:func:`~discretisedfield.Field.k3d_vectors`

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        planeaxis = dfu.raxesdict[self.mesh.info['planeaxis']]

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        unit = f' ({uu.rsi_prefixes[multiplier]}m)'

        if self.dim > 1:
            # Vector field has both quiver and imshow plots.
            self.quiver(ax=ax, headwidth=5, multiplier=multiplier)
            scalar_field = getattr(self, planeaxis)
            coloredplot = scalar_field.imshow(ax=ax, filter_field=self.norm,
                                              cmap='cividis',
                                              multiplier=multiplier)
        else:
            # Scalar field has only imshow.
            coloredplot = self.imshow(ax=ax, filter_field=None, cmap='cividis',
                                      multiplier=multiplier)

        cbar = self.colorbar(ax, coloredplot)

        # Add labels.
        ax.set_xlabel(dfu.raxesdict[self.mesh.info['axis1']] + unit)
        ax.set_ylabel(dfu.raxesdict[self.mesh.info['axis2']] + unit)
        if self.dim > 1:
            cbar.ax.set_ylabel(planeaxis + ' component')

        ax.figure.tight_layout()

    def imshow(self, ax, filter_field=None, multiplier=1, **kwargs):
        """Plots the scalar field on a plane using
        ``matplotlib.pyplot.imshow``.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be of dimension 1
        (scalar field). Otherwise, ``ValueError`` is raised. ``imshow`` adds
        the plot to ``matplotlib.axes.Axes`` passed via ``ax`` argument. By
        passing ``filter_field`` the points at which the pixels are not
        coloured can be determined. More precisely, only discretisation cells
        where ``filter_field != 0`` are plotted. It is often the case that the
        region size is small (e.g. on a nanoscale) or very large (e.g. in units
        of kilometers). Accordingly, ``multiplier`` can be passed as
        :math:`10^{n}`, where :math:`n` is a multiple of 3  (..., -6, -3, 0, 3,
        6,...). According to that value, the axes will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, all mesh points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels.

        This method plots the mesh using ``matplotlib.pyplot.imshow()``
        function, so any keyword arguments accepted by it can be passed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes

            Axes to which field plot should be added.

        filter_field : discretisedfield.Field, optional

            A (scalar) field used for determining whether certain pixels should
            be coloured. More precisely, only discretisation cells where
            ``filter_field != 0`` are plotted.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to 1.

        Raises
        ------
        ValueError

            If the field has not been sliced with a plane or its dimension is
            not 1.

        Example
        -------
        1. Visualising the scalar field using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
        ...
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> field.plane('y').imshow(ax=ax)
        <matplotlib.image.AxesImage object at ...>

        .. seealso:: :py:func:`~discretisedfield.Field.quiver`

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        if self.dim != 1:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        points, values = list(zip(*list(self)))

        # If filter_field is passed, set values where norm=0 to np.nan,
        # so that they are not plotted.
        if filter_field is not None:
            values = list(values)  # tuple -> list to make values mutable
            for i, point in enumerate(points):
                if filter_field(point) == 0:
                    values[i] = np.nan

        pmin = np.divide(self.mesh.region.pmin, multiplier)
        pmax = np.divide(self.mesh.region.pmax, multiplier)

        extent = [pmin[self.mesh.info['axis1']],
                  pmax[self.mesh.info['axis1']],
                  pmin[self.mesh.info['axis2']],
                  pmax[self.mesh.info['axis2']]]
        n = (self.mesh.n[self.mesh.info['axis2']],
             self.mesh.n[self.mesh.info['axis1']])

        return ax.imshow(np.array(values).reshape(n), origin='lower',
                         extent=extent, **kwargs)

    def quiver(self, ax, color_field=None, multiplier=1, **kwargs):
        """Plots the vector field on a plane using
        ``matplotlib.pyplot.quiver``.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be of dimension 3
        (vector field). Otherwise, ``ValueError`` is raised. ``quiver`` adds
        the plot to ``matplotlib.axes.Axes`` passed via ``ax`` argument.
        Vectors can be coloured by passing ``color_field`` which is a scalar
        field defining the colour at different points. It is often the case
        that the region size is small (e.g. on a nanoscale) or very large (e.g.
        in units of kilometers). Accordingly, ``multiplier`` can be passed as
        :math:`10^{n}`, where :math:`n` is a multiple of 3  (..., -6, -3, 0, 3,
        6,...). According to that value, the axes will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, all mesh points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels.

        This method plots the mesh using ``matplotlib.pyplot.quiver()``
        function, so any keyword arguments accepted by it can be passed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes

            Axes to which field plot should be added.

        color_field : discretisedfield.Field, optional

            A (scalar) field used for colouring vectors.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to 1.

        Raises
        ------
        ValueError

            If the field has not been sliced with a plane or its dimension is
            not 3.

        Example
        -------
        1. Visualising the vector field using ``matplotlib`` and colour
        according to its z-component.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(2, 1, 0))
        ...
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> field.plane('y').quiver(ax=ax, color_field=field.z)
        <matplotlib.quiver.Quiver object at ...>

        .. seealso:: :py:func:`~discretisedfield.Field.imshow`

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        if self.dim != 3:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        points, values = list(zip(*list(self)))

        # Remove values where norm is 0
        points, values = list(points), list(values)  # make them mutable
        points = [p for p, v in zip(points, values)
                  if not np.equal(v, 0).all()]
        values = [v for v in values if not np.equal(v, 0).all()]
        if color_field is not None:
            colors = [color_field(p) for p in points]

        # "Unpack" values inside arrays and convert to np.ndarray.
        points = np.array(list(zip(*points)))
        values = np.array(list(zip(*values)))

        points = np.divide(points, multiplier)

        # Are there any vectors pointing out-of-plane? If yes, set the scale.
        if not any(values[self.mesh.info['axis1']] +
                   values[self.mesh.info['axis2']]):
            kwargs['scale'] = 1

        kwargs['pivot'] = 'mid'  # arrow at the centre of the cell

        if color_field is None:
            qvax = ax.quiver(points[self.mesh.info['axis1']],
                             points[self.mesh.info['axis2']],
                             values[self.mesh.info['axis1']],
                             values[self.mesh.info['axis2']],
                             **kwargs)

        else:
            qvax = ax.quiver(points[self.mesh.info['axis1']],
                             points[self.mesh.info['axis2']],
                             values[self.mesh.info['axis1']],
                             values[self.mesh.info['axis2']],
                             colors,
                             **kwargs)

        return qvax

    def colorbar(self, ax, coloredplot, cax=None, **kwargs):
        """Adds a colorbar to the axes using ``matplotlib.pyplot.colorbar``.

        Axes to which the colorbar should be added is passed via ``ax``
        argument. If the colorbar axes are made before the method is called,
        they should be passed as ``cax``. The plot to which the colorbar should
        correspond to is passed via ``coloredplot``. All other keyword
        arguments accepted by ``matplotlib.pyplot.colorbar`` can be
        passed.

        Parameters
        ----------
        ax : matplotlib.axes.Axes

            Axes object to which the colorbar will be added.

        coloredplot : matplotlib.quiver.Quiver, matplotlib.image.AxesImage

            A plot to which the colorbar should correspond.

        cax : matplotlib.axes.Axes, optional

            Colorbar axes.

        Example
        -------
        1. Add colorbar to the plot.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        ...
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> coloredplot = field.plane(z=50).quiver(ax=ax, color_field=field.z)
        >>> field.colorbar(ax=ax, coloredplot=coloredplot)
        <matplotlib.colorbar.Colorbar object at ...>

        """
        if cax is None:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)

        return plt.colorbar(coloredplot, cax=cax, **kwargs)

    def k3d_nonzero(self, plot=None, multiplier=None,
                    color=dfu.color_palette('deep', 10, 'int')[0], field=None,
                    interactive=False, **kwargs):
        """Plots the mesh discretisation cells where the value of the field is
        not zero using ``k3d`` voxels.

        If ``plot`` is not passed, ``k3d`` plot will be created automaticaly.
        It is often the case that the mesh region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometeres). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, the mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the optimum one is
        computed internally. The colour of the "non-zero region" can be
        determined using ``color`` as an integer. When sliced field is plotted,
        it is sometimes necessary to plot the region of the original field.
        This can be achieved by passing the field using ``field``. In
        interactive plots, ``field`` must be passed. ``interactive=True`` must
        be defined when the method is used for interactive plotting.

        This method plots the region using ``k3d.voxels()`` function, so any
        keyword arguments accepted by it can be passed.

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which plot should be added. Defaults to ``None`` - new plot
            will be created.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        color : list, optional

            Colour of the "non-zero" region. Defaults to
            ``seaborn.color_pallette(palette='deep')[0]``.

        field : discretisedfield.Field

            If ``field`` is passed, then the region of the field is plotted.
            Defaults to ``None``.

        interactive : bool

            For interactive plotting, ``True`` must be passed. Defaults to
            ``False``.

        Raises
        ------
        ValueError

            If the dimension of the field is not 1.

        Examples
        --------
        1. Visualising the "non-zero" region using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, -50e-9)
        >>> p2 = (50e-9, 50e-9, 50e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> def normfun(pos):
        ...     x, y, z = pos
        ...     if x**2 + y**2 < 30**2:
        ...         return 1
        ...     else:
        ...         return 0
        >>> field.norm = normfun
        >>> field.norm.k3d_nonzero()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Field.k3d_voxels`

        """
        if self.dim != 1:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        plot_array = np.ones_like(self.array)  # all voxels have the same color
        plot_array[self.array == 0] = 0  # remove voxels where field is zero
        plot_array = plot_array[..., 0]  # remove an empty dimension
        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)

        plot, multiplier = dfu.k3d_parameters(plot, multiplier,
                                              self.mesh.region.edges)

        if field is not None:
            dfu.k3d_plot_region(plot, field.mesh.region, multiplier)

        if interactive:
            dfu.k3d_setup_interactive_plot(plot)

        plot += dfu.voxels(plot_array, pmin=self.mesh.region.pmin,
                           pmax=self.mesh.region.pmax, color_palette=color,
                           multiplier=multiplier, **kwargs)

    def k3d_voxels(self, plot=None, filter_field=None, multiplier=None,
                   cmap='cividis', n=256, field=None, interactive=False,
                   **kwargs):
        """Plots the scalar field as a coloured ``k3d.voxels()`` plot.

        If ``plot`` is not passed, ``k3d`` plot will be created automaticaly.
        By passing ``filter_field``, the points at which the voxels are plotted
        can be determined. More precisely, only voxels where ``filter_field !=
        0`` are plotted. It is often the case that the mesh region size is
        small (e.g. on a nanoscale) or very large (e.g. in units of
        kilometeres). Accordingly, ``multiplier`` can be passed as
        :math:`10^{n}`, where :math:`n` is a multiple of 3 (..., -6, -3, 0, 3,
        6,...). According to that value, the axes will be scaled and
        appropriate units shown. For instance, if ``multiplier=1e-9`` is
        passed, the mesh points will be divided by :math:`1\\,\\text{nm}` and
        :math:`\\text{nm}` units will be used as axis labels. If ``multiplier``
        is not passed, the optimum one is computed internally. The colormap and
        the resolution of the colours can be set by passing ``cmap`` and ``n``.
        When sliced field is plotted, it is sometimes necessary to plot the
        region of the original field. This can be achieved by passing the field
        using ``field``. In interactive plots, ``field`` must be passed.
        ``interactive=True`` must be defined when the method is used for
        interactive plotting.

        This method plots the region using ``k3d.voxels()`` function, so any
        keyword arguments accepted by it can be passed.

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which plot should be added. Defaults to ``None`` - new plot
            will be created.

        filter_field : discretisedfield.Field, optional

            A (scalar) field used for determining whether certain voxels should
            be plotted. More precisely, only discretisation cells where
            ``filter_field != 0`` are plotted.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        cmap : str

            Colormap. Defaults to ``'cividis'``.

        n : int

            The resolution of the colormap. Defaults to 256, which is also the
            maximum possible value.

        field : discretisedfield.Field

            If ``field`` is passed, then the region of the field is plotted.
            Defaults to ``None``.

        interactive : bool

            For interactive plotting, ``True`` must be passed. Defaults to
            ``False``.

        Raises
        ------
        ValueError

            If the dimension of the field is not 1.

        Example
        -------
        1. Plot the scalar field using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50, -50, -50)
        >>> p2 = (50, 50, 50)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> field = df.Field(mesh, dim=1, value=5)
        >>> field.k3d_voxels()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Field.k3d_vectors`

        """
        if self.dim != 1:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        if filter_field is not None:
            if filter_field.dim != 1:
                msg = f'Cannot use dim={self.dim} filter_field.'
                raise ValueError(msg)

        if n > 256:
            msg = f'Cannot use n={n}. Maximum value is 256.'
            raise ValueError(msg)

        plot_array = np.copy(self.array)  # make a deep copy
        plot_array = plot_array[..., 0]  # remove an empty dimension

        # All values must be in (1, 255) -> (1, n-1), for n=256 range, with
        # maximum n=256. This is the limitation of k3d.voxels(). Voxels where
        # values are zero, are invisible.
        plot_array = dfu.normalise_to_range(plot_array, (1, n-1))

        # Remove voxels where filter_field = 0.
        if filter_field is not None:
            for index in self.mesh.indices:
                if filter_field(self.mesh.index2point(index)) == 0:
                    plot_array[index] = 0

        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)

        color_palette = dfu.color_palette(cmap, n, 'int')

        plot, multiplier = dfu.k3d_parameters(plot, multiplier,
                                              self.mesh.region.edges)

        if field is not None:
            dfu.k3d_plot_region(plot, field.mesh.region, multiplier)

        if interactive:
            dfu.k3d_setup_interactive_plot(plot)

        plot += dfu.voxels(plot_array, pmin=self.mesh.region.pmin,
                           pmax=self.mesh.region.pmax,
                           color_palette=color_palette, multiplier=multiplier,
                           **kwargs)

    def k3d_vectors(self, plot=None, color_field=None, points=True,
                    cmap='cividis', n=256,
                    point_color=dfu.color_palette('deep', 1, 'int')[0],
                    point_size=None, multiplier=None, vector_multiplier=None,
                    field=None, interactive=False, **kwargs):
        """Plots the vector field using ``k3d``.

        If ``plot`` is not passed, ``k3d`` plot will be created automaticaly.
        It is often the case that the mesh region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometeres). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, the mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the optimum one is
        computed internally. Similarly, the vectors can be too large or two
        small to be plotted. In that case, ``vector_multiplier`` can be passed,
        so that all vectors are divided by that scalar. If not passed, the
        optimum value is computed internally. The colour of vectors can be
        determined by passing ``color_field``, whereas the colormap and the
        resolution can be determined by passing ``cmap`` and ``n``. In addition
        to vectors, points at which the vectors are defined can be plotted if
        ``points=True``. The size of the points can be passed using
        ``point_size`` and if ``point_size`` is not passed, optimum size is
        computed intenally. Similarly, ``point_color`` can be passed as an
        integer. When sliced field is plotted, it is sometimes necessary to
        plot the region of the original field. This can be achieved by passing
        the field using ``field``. In interactive plots, ``field`` must be
        passed. ``interactive=True`` must be defined when the method is used
        for interactive plotting.

        This method plots the vectors using ``k3d.vectors()`` function, so any
        keyword arguments accepted by it can be passed.

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which vector plot should be added. Defaults to ``None`` -
            new plot will be created.

        color_field : discretisedfield.Field, optional

            Field determining the values according to which the vectors are
            coloured. Defults to ``None``.

        points : bool, optional

            If ``True``, points are added to the plot.

        point_size : float, optional

            Size of points.

        point_color : int, optional

            Colour of points. Defaults to
            ``seaborn.color_pallette(palette='deep')[0]``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. If ``multiplier`` is not passed, the
            optimum one is computed internally. Defaults to ``None``.

        vector_multiplier : numbers.Real, optional

            Value by which all vectors are divided to fit the plot.

        cmap : str

            Colormap. Defaults to ``'cividis'``.

        n : int

            The resolution of the colormap. Defaults to 256.

        field : discretisedfield.Field

            If ``field`` is passed, then the region of the field is plotted.
            Defaults to ``None``.

        interactive : bool

            For interactive plotting, ``True`` must be passed. Defaults to
            ``False``.

        Raises
        ------
        ValueError

            If the dimension of the field is not 3.

        Examples
        --------
        1. Visualising the vector field using ``k3d``.

        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> mesh.k3d_points()
        Plot(...)

        """
        if self.dim != 3:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        if color_field is not None:
            if color_field.dim != 1:
                msg = f'Cannot use dim={self.dim} color_field.'
                raise ValueError(msg)

        coordinates, vectors, color_values = [], [], []
        norm_field = self.norm  # assigned to be computed only once
        for coord, value in self:
            if norm_field(coord) > 0:
                coordinates.append(coord)
                vectors.append(value)
                if color_field is not None:
                    color_values.append(color_field(coord))
        coordinates, vectors = np.array(coordinates), np.array(vectors)

        plot, multiplier = dfu.k3d_parameters(plot, multiplier,
                                              self.mesh.region.edges)

        if field is not None:
            dfu.k3d_plot_region(plot, field.mesh.region, multiplier)

        if interactive:
            dfu.k3d_setup_interactive_plot(plot)

        if vector_multiplier is None:
            vector_multiplier = (vectors.max() /
                                 np.divide(self.mesh.cell, multiplier).min())

        if color_field is not None:
            color_values = dfu.normalise_to_range(color_values, (0, n-1))

            # Generate double pairs (body, head) for colouring vectors.
            color_palette = dfu.color_palette(cmap, n, 'int')
            colors = []
            for cval in color_values:
                colors.append(2*(color_palette[cval],))
        else:
            # Uniform colour.
            colors = (len(vectors) *
                      ([2*(dfu.color_palette('deep', 2, 'int')[1],)]))

        plot += dfu.vectors(coordinates, vectors, colors=colors,
                            multiplier=multiplier,
                            vector_multiplier=vector_multiplier, **kwargs)

        if points:
            if point_size is None:
                # If undefined, the size of the point is 1/4 of the smallest
                # cell dimension.
                point_size = np.divide(self.mesh.cell, multiplier).min() / 4

            plot += dfu.points(coordinates, color=point_color,
                               point_size=point_size, multiplier=multiplier)
