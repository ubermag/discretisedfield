import k3d
import h5py
import struct
import numbers
import inspect
import itertools
import matplotlib
import numpy as np
import discretisedfield as df
import ubermagutil.units as uu
import matplotlib.pyplot as plt
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu

# TODO: tutorials, fft, line operations


@ts.typesystem(mesh=ts.Typed(expected_type=df.Mesh, const=True),
               dim=ts.Scalar(expected_type=int, positive=True, const=True))
class Field:
    """Finite-difference field.

    This class specifies a finite-difference field and defines operations for
    its analysis and visualisation. The field is defined on a finite-difference
    mesh (`discretisedfield.Mesh`) passed using ``mesh``. Another value that
    must be passed is the dimension of the field's value using ``dim``. For
    instance, for a scalar field, ``dim=1`` and for a three-dimensional vector
    field ``dim=3`` must be passed. The value of the field can be set by
    passing ``value``. For details on how the value can be defined, refer to
    ``discretisedfield.Field.value``. Similarly, if the field has ``dim>1``,
    the field can be normalised by passing ``norm``. For details on setting the
    norm, please refer to ``discretisedfield.Field.norm``.

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
        >>> def value_function(point):
        ...     x, y, z = point
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

        Computes the norm of the field and returns ``discretisedfield.Field``
        with ``dim=1``. Norm of a scalar field is interpreted as an absolute
        value of the field. Alternatively, ``discretisedfield.Field.__abs__``
        can be called for obtaining the norm of the field.

        The field norm can be set by passing ``numbers.Real``,
        ``numpy.ndarray``, or callable. If the field has ``dim=1`` or it
        contains zero values, norm cannot be set and ``ValueError`` is raised.

        Parameters
        ----------
        numbers.Real, numpy.ndarray, callable

            Norm value.

        Returns
        -------
        discretisedfield.Field

            Norm of the field if ``dim>1`` or absolute value for ``dim=1``.

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
            res = abs(self.value)
        else:
            res = np.linalg.norm(self.array, axis=-1)[..., np.newaxis]

        return self.__class__(self.mesh, dim=1, value=res)

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

        This is a convenience operator and it returns
        ``discretisedfield.Field.norm``. For details, please refer to
        ``discretisedfield.Field.norm``.

        Returns
        -------
        discretisedfield.Field

            Norm of the field if ``dim>1`` or absolute value for ``dim=1``.

        Examples
        --------
        1. Computing the absolute value of a scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 10, 13)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=1, value=-5)
        >>> abs(field).average
        5.0

        .. seealso:: :py:func:`~discretisedfield.Field.norm`

        """
        return self.norm

    @property
    def zero(self):
        """Zero field.

        This method returns a zero field defined on the same mesh and with the
        same value dimension.

        Returns
        -------
        discretisedfield.Field

            Zero field.

        Examples
        --------
        1. Getting the zero-field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 10, 13)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(3, -1, 1))
        >>> zero_field = field.zero
        >>> zero_field.average
        (0.0, 0.0, 0.0)

        """
        return self.__class__(self.mesh, dim=self.dim, value=0)

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
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh, dim=1, value=1)
        >>> repr(field)
        "Field(mesh=..., dim=1)"

        """
        return f"Field(mesh={repr(self.mesh)}, dim={self.dim})"

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
            return self.__class__(mesh=self.mesh, dim=1, value=attr_array)
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
            need_removing = ['div', 'curl', 'orientation', 'mpl_vector',
                             'k3d_vector']
        if self.dim == 3:
            need_removing = ['grad', 'mpl_scalar', 'k3d_scalar', 'k3d_nonzero']

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

        """
        if not isinstance(other, self.__class__):
            return False
        elif (self.mesh == other.mesh and self.dim == other.dim and
              np.array_equal(self.array, other.array)):
            return True
        else:
            return False

    def allclose(self, other, rtol=1e-5, atol=1e-8):
        """Allclose method.

        This method determines whether two fields are:

          1. Defined on the same mesh.

          2. Have the same dimension (``dim``).

          3. All values in are within relative (``rtol``) and absolute
          (``atol``) tolerances.

        Parameters
        ----------
        other : discretisedfield.Field

            Field to be compared to.

        rtol : numbers.Real

            Relative tolerance. Defaults to 1e-5.

        atol : numbers.Real

            Absolute tolerance. Defaults to 1e-8.

        Returns
        -------
        bool

            ``True`` if two fields are within tolerance, ``False`` otherwise.

        Raises
        ------
        TypeError

            If a non field object is passed.

        Examples
        --------
        1. Check if two fields are within a tolerance.

        >>> import discretisedfield as df
        ...
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        ...
        >>> f1 = df.Field(mesh, dim=1, value=3)
        >>> f2 = df.Field(mesh, dim=1, value=3+1e-9)
        >>> f3 = df.Field(mesh, dim=1, value=3.1)
        >>> f1.allclose(f2)
        True
        >>> f1.allclose(f3)
        False
        >>> f1.allclose(f3, atol=1e-2)
        False

        """
        if not isinstance(other, self.__class__):
            msg = (f'Cannot apply allclose method between '
                   f'{type(self)=} and {type(other)=} objects.')
            raise TypeError(msg)

        if (self.mesh == other.mesh and self.dim == other.dim):
            return np.allclose(self.array, other.array, rtol=rtol, atol=atol)
        else:
            return False

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
            msg = f'Cannot apply ** operator on {self.dim=} field.'
            raise ValueError(msg)
        if not isinstance(other, numbers.Real):
            msg = (f'Unsupported operand type(s) for **: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

        return self.__class__(self.mesh, dim=1,
                              value=np.power(self.array, other))

    def __add__(self, other):
        """Binary ``+`` operator.

        It can be applied between two ``discretisedfield.Field`` objects or
        between a ``discretisedfield.Field`` object and a "constant". For
        instance if the field is a scalar field, a scalar field or
        ``numbers.Real`` can be the second operand. Similarly, for a vector
        field, either vector field or an iterable, such as ``tuple``, ``list``,
        or ``numpy.ndarray``, can be the second operand. If the second operand
        is a ``discretisedfield.Field`` object, both must be defined on the
        same mesh and have the same dimensions.

        Parameters
        ----------
        other : discretisedfield.Field, numbers.Real, tuple, list, np.ndarray

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
        1. Add vector fields.

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
        >>> res = f1 + (1, 2, 3.1)
        >>> res.average
        (1.0, 1.0, 0.0)
        >>> f1 + 5
        Traceback (most recent call last):
        ...
        TypeError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.__sub__`

        """
        if isinstance(other, self.__class__):
            if self.dim != other.dim:
                msg = (f'Cannot apply operator + on {self.dim=} '
                       f'and {other.dim=} fields.')
                raise ValueError(msg)
            if self.mesh != other.mesh:
                msg = ('Cannot apply operator + on fields '
                       'defined on different meshes.')
                raise ValueError(msg)
        elif self.dim == 1 and isinstance(other, numbers.Real):
            return self + self.__class__(self.mesh, dim=self.dim, value=other)
        elif self.dim == 3 and isinstance(other, (tuple, list, np.ndarray)):
            return self + self.__class__(self.mesh, dim=self.dim, value=other)
        else:
            msg = (f'Unsupported operand type(s) for +: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

        return self.__class__(self.mesh, dim=self.dim,
                              value=self.array + other.array)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        """Binary ``-`` operator.

        It can be applied between two ``discretisedfield.Field`` objects or
        between a ``discretisedfield.Field`` object and a "constant". For
        instance if the field is a scalar field, a scalar field or
        ``numbers.Real`` can be the second operand. Similarly, for a vector
        field, either vector field or an iterable, such as ``tuple``, ``list``,
        or ``numpy.ndarray``, can be the second operand. If the second operand
        is a ``discretisedfield.Field`` object, both must be defined on the
        same mesh and have the same dimensions.

        Parameters
        ----------
        other : discretisedfield.Field, numbers.Real, tuple, list, np.ndarray

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
        1. Subtract vector fields.

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
        >>> res = f1 - (0, 1, 0)
        >>> res.average
        (0.0, 0.0, 6.0)

        .. seealso:: :py:func:`~discretisedfield.Field.__add__`

        """
        # Ensure unary '-' can be applied to other.
        if isinstance(other, (list, tuple)):
            other = np.array(other)

        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        """Binary ``*`` operator.

        It can be applied between:

        1. Two scalar (``dim=1``) fields,

        2. A field of any dimension and ``numbers.Real``,

        3. A field of any dimension and a scalar (``dim=1``) field, or

        4. A field and an "abstract" integration variable (e.g. ``df.dV``)

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
        if isinstance(other, self.__class__):
            if self.dim == 3 and other.dim == 3:
                msg = (f'Cannot apply operator * on {self.dim=} '
                       f'and {other.dim=} fields.')
                raise ValueError(msg)
            if self.mesh != other.mesh:
                msg = ('Cannot apply operator * on fields '
                       'defined on different meshes.')
                raise ValueError(msg)
        elif isinstance(other, numbers.Real):
            return self * self.__class__(self.mesh, dim=1, value=other)
        elif self.dim == 1 and isinstance(other, (tuple, list, np.ndarray)):
            return self * self.__class__(self.mesh, dim=3, value=other)
        elif isinstance(other, df.DValue):
            return self * other(self)
        else:
            msg = (f'Unsupported operand type(s) for *: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

        res_array = np.multiply(self.array, other.array)
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
        if isinstance(other, self.__class__):
            if self.mesh != other.mesh:
                msg = ('Cannot apply operator @ on fields '
                       'defined on different meshes.')
                raise ValueError(msg)
            if self.dim != 3 or other.dim != 3:
                msg = (f'Cannot apply operator @ on {self.dim=} '
                       f'and {other.dim=} fields.')
                raise ValueError(msg)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self @ self.__class__(self.mesh, dim=3, value=other)
        elif isinstance(other, df.DValue):
            return self @ other(self)
        else:
            msg = (f'Unsupported operand type(s) for @: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

        res_array = np.einsum('ijkl,ijkl->ijk', self.array, other.array)
        return df.Field(self.mesh, dim=1, value=res_array[..., np.newaxis])

    def __rmatmul__(self, other):
        return self @ other

    def __and__(self, other):
        """Binary ``&`` operator, defined as cross product.

        This method computes the cross product between two fields. Both fields
        must be three-dimensional (``dim=3``) and defined on the same mesh.

        Parameters
        ----------
        other : discretisedfield.Field, tuple, list, numpy.ndarray

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
        1. Compute the cross product of two vector fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=3, value=(1, 0, 0))
        >>> f2 = df.Field(mesh, dim=3, value=(0, 1, 0))
        >>> (f1 & f2).average
        (0.0, 0.0, 1.0)
        >>> (f1 & (0, 0, 1)).average
        (0.0, -1.0, 0.0)

        """
        if isinstance(other, self.__class__):
            if self.mesh != other.mesh:
                msg = ('Cannot apply operator & on fields '
                       'defined on different meshes.')
                raise ValueError(msg)
            if self.dim != 3 or other.dim != 3:
                msg = (f'Cannot apply operator & on {self.dim=} '
                       f'and {other.dim=} fields.')
                raise ValueError(msg)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self & self.__class__(self.mesh, dim=3, value=other)
        else:
            msg = (f'Unsupported operand type(s) for &: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

        res_array = np.cross(self.array, other.array)
        return self.__class__(self.mesh, dim=3, value=res_array)

    def __rand__(self, other):
        return self & other

    def __lshift__(self, other):
        """Stacks multiple scalar fields in a single vector field.

        This method takes a list of scalar (``dim=1``) fields and returns a
        vector field, whose components are defined by the scalar fields passed.
        If any of the fields passed has ``dim!=1`` or they are not defined on
        the same mesh, an exception is raised. The dimension of the resulting
        field is equal to the length of the passed list.

        Parameters
        ----------
        fields : list

            List of ``discretisedfield.Field`` objects with ``dim=1``.

        Returns
        -------
        disrectisedfield.Field

            Resulting field.

        Raises
        ------
        ValueError

            If the dimension of any of the fields is not 1, or the fields
            passed are not defined on the same mesh.

        Example
        -------
        1. Stack 3 scalar fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=1, value=1)
        >>> f2 = df.Field(mesh, dim=1, value=5)
        >>> f3 = df.Field(mesh, dim=1, value=-3)
        ...
        >>> f = f1 << f2 << f3
        >>> f.average
        (1.0, 5.0, -3.0)
        >>> f.dim
        3
        >>> f.x == f1
        True
        >>> f.y == f2
        True
        >>> f.z == f3
        True

        """
        if isinstance(other, self.__class__):
            if self.mesh != other.mesh:
                msg = ('Cannot apply operator << on fields '
                       'defined on different meshes.')
                raise ValueError(msg)
        elif isinstance(other, numbers.Real):
            return self << self.__class__(self.mesh, dim=1, value=other)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self << self.__class__(self.mesh, dim=len(other),
                                          value=other)
        else:
            msg = (f'Unsupported operand type(s) for <<: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

        array_list = [self.array[..., i] for i in range(self.dim)]
        array_list += [other.array[..., i] for i in range(other.dim)]
        return self.__class__(self.mesh, dim=len(array_list),
                              value=np.stack(array_list, axis=3))

    def __rlshift__(self, other):
        if isinstance(other, numbers.Real):
            return self.__class__(self.mesh, dim=1, value=other) << self
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self.__class__(self.mesh, dim=len(other),
                                  value=other) << self
        else:
            msg = (f'Unsupported operand type(s) for <<: '
                   f'{type(self)=} and {type(other)=}.')
            raise TypeError(msg)

    def pad(self, pad_width, mode, **kwargs):
        """Field padding.

        This method pads the field by adding more cells in chosen direction and
        assigning to them the values as specified by the ``mode`` argument.
        The way in which the field is going to padded is defined by passing
        ``pad_width`` dictionary. The keys of the dictionary are the directions
        (axes), e.g. ``'x'``, ``'y'``, or ``'z'``, whereas the values are the
        tuples of length 2. The first integer in the tuple is the number of
        cells added in the negative direction, and the second integer is the
        number of cells added in the positive direction.

        This method accepts any other arguments allowed by ``numpy.pad``
        function.

        Parameters
        ----------
        pad_width : dict

            The keys of the dictionary are the directions (axes), e.g. ``'x'``,
            ``'y'``, or ``'z'``, whereas the values are the tuples of length 2.
            The first integer in the tuple is the number of cells added in the
            negative direction, and the second integer is the number of cells
            added in the positive direction.

        mode: str

            Padding mode as defined in ``numpy.pad``.

        Returns
        -------
        discretisedfield.Field

            Padded field.

        Examples
        --------
        1. Padding a field in the x direction by 1 cell with ``constant`` mode.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 1, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh, dim=1, value=1)
        ...
        >>> # Two cells with value 1
        >>> pf = field.pad({'x': (1, 1)}, mode='constant')  # zeros padded
        >>> pf.average
        0.5

        """
        d = {}
        for key, value in pad_width.items():
            d[dfu.axesdict[key]] = value
        padding_sequence = dfu.assemble_index((0, 0), len(self.array.shape), d)

        padded_array = np.pad(self.array, padding_sequence,
                              mode=mode, **kwargs)
        padded_mesh = self.mesh.pad(pad_width)

        return self.__class__(padded_mesh, dim=self.dim, value=padded_array)

    def derivative(self, direction, n=1):
        """Directional derivative.

        This method computes a directional derivative of the field and returns
        a field. The direction in which the derivative is computed is passed
        via ``direction`` argument, which can be ``'x'``, ``'y'``, or ``'z'``.
        The order of the computed derivative can be 1 or 2 and it is specified
        using argument ``n`` and it defaults to 1.

        Directional derivative cannot be computed if only one discretisation
        cell exists in a specified direction. In that case, a zero field is
        returned. More precisely, it is assumed that the field does not change
        in that direction. Computing of the directional derivative depends
        strongly on the boundary condition specified in the mesh on which the
        field is defined on. More precisely, the values of the derivatives at
        the boundary are different for periodic, Neumann, or no boundary
        conditions. For details on boundary conditions, please refer to the
        ``disretisedfield.Mesh`` class. The derivatives are computed using
        central differences inside the sample and using forward/backward
        differences at the boundaries.

        Parameters
        ----------
        direction : str

            The direction in which the derivative is computed. It can be
            ``'x'``, ``'y'``, or ``'z'``.

        n : int

            The order of the derivative. It can be 1 or 2 and it defaults to 1.

        Returns
        -------
        discretisedfield.Field

            Directional derivative.

        Raises
        ------
        NotImplementedError

            If order ``n`` higher than 2 is asked for.

        Example
        -------
        1. Compute the first-order directional derivative of a scalar field in
        the y-direction of a spatially varying field. For the field we choose
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
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     return 2*x + 3*y + -5*z
        ...
        >>> f = df.Field(mesh, dim=1, value=value_fun)
        >>> f.derivative('y').average  # first-order derivative by default
        3.0

        2. Try to compute the second-order directional derivative of the vector
        field which has only one discretisation cell in the z-direction. For
        the field we choose :math:`f(x, y, z) = (2x, 3y, -5z)`. Accordingly, we
        expect the directional derivatives to be: :math:`df/dx = (2, 0, 0)`,
        :math:`df/dy=(0, 3, 0)`, :math:`df/dz = (0, 0, -5)`. However, because
        there is only one discretisation cell in the z-direction, the
        derivative cannot be computed and a zero field is returned. Similarly,
        second-order derivatives in all directions are expected to be zero.

        >>> def value_fun(point):
        ...     x, y, z = point
        ...     return (2*x, 3*y, -5*z)
        ...
        >>> f = df.Field(mesh, dim=3, value=value_fun)
        >>> f.derivative('x', n=1).average
        (2.0, 0.0, 0.0)
        >>> f.derivative('y', n=1).average
        (0.0, 3.0, 0.0)
        >>> f.derivative('z', n=1).average  # derivative cannot be calculated
        (0.0, 0.0, 0.0)
        >>> # second-order derivatives

        """
        direction = dfu.axesdict[direction]

        # If there are no neighbouring cells in the specified direction, zero
        # field is returned.
        if self.mesh.n[direction] == 1:
            return self.zero

        # Preparation (padding) for computing the derivative, depending on the
        # boundary conditions (PBC, Neumann, or no BC). Depending on the BC,
        # the field array is padded.
        if dfu.raxesdict[direction] in self.mesh.bc:  # PBC
            pad_width = {dfu.raxesdict[direction]: (1, 1)}
            padding_mode = 'wrap'
        elif self.mesh.bc == 'neumann':
            pad_width = {dfu.raxesdict[direction]: (1, 1)}
            padding_mode = 'edge'
        else:  # No BC - no padding
            pad_width = {}
            padding_mode = 'constant'

        padded_array = self.pad(pad_width, mode=padding_mode).array

        if n not in (1, 2):
            msg = f'Derivative of the n={n} order is not implemented.'
            raise NotImplementedError(msg)

        elif n == 1:
            if self.dim == 1:
                derivative_array = np.gradient(padded_array[..., 0],
                                               self.mesh.cell[direction],
                                               axis=direction)[..., np.newaxis]
            else:
                derivative_array = np.gradient(padded_array,
                                               self.mesh.cell[direction],
                                               axis=direction)

        elif n == 2:
            derivative_array = np.zeros_like(padded_array)
            for i in range(padded_array.shape[direction]):
                if i == 0:
                    i1, i2, i3 = i+2, i+1, i
                elif i == padded_array.shape[direction] - 1:
                    i1, i2, i3 = i, i-1, i-2
                else:
                    i1, i2, i3 = i+1, i, i-1
                index1 = dfu.assemble_index(slice(None), 4, {direction: i1})
                index2 = dfu.assemble_index(slice(None), 4, {direction: i2})
                index3 = dfu.assemble_index(slice(None), 4, {direction: i3})
                index = dfu.assemble_index(slice(None), 4, {direction: i})
                derivative_array[index] = ((padded_array[index1] -
                                           2*padded_array[index2] +
                                           padded_array[index3]) /
                                           self.mesh.cell[direction]**2)

        # Remove padded values (if any).
        if derivative_array.shape != self.array.shape:
            derivative_array = np.delete(derivative_array,
                                         (0, self.mesh.n[direction]+1),
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

        >>> def value_fun(point):
        ...     x, y, z = point
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

        return (self.derivative('x') <<
                self.derivative('y') <<
                self.derivative('z'))

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
        >>> def value_fun(point):
        ...     x, y, z = point
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
        >>> def value_fun(point):
        ...     x, y, z = point
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

        return curl_x << curl_y << curl_z

    @property
    def laplace(self):
        """Laplace operator.

        This method computes the laplacian of a scalar (``dim=1``) or a vector
        (``dim=3``) field and returns a resulting field:

        .. math::

            \\nabla^2 f = \\frac{\\partial^{2} f}{\\partial x^{2}} +
                          \\frac{\\partial^{2} f}{\\partial y^{2}} +
                          \\frac{\\partial^{2} f}{\\partial z^{2}}

        .. math::

            \\nabla^2 \\mathbf{f} = (\\nabla^2 f_{x},
                                     \\nabla^2 f_{y},
                                     \\nabla^2 f_{z})

        Directional derivative cannot be computed if only one discretisation
        cell exists in a certain direction. In that case, a zero field is
        considered to be that directional derivative. More precisely, it is
        assumed that the field does not change in that direction.

        Returns
        -------
        discretisedfield.Field

            Resulting field.

        Example
        -------
        1. Compute Laplacian of a contant scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> cell = (2e-9, 2e-9, 2e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=1, value=5)
        >>> f.laplace.average
        0.0

        2. Compute Laplacian of a spatially varying field. For a field we
        choose :math:`f(x, y, z) = 2x^{2} + 3y - 5z`. Accordingly, we expect
        the Laplacian to be a constant vector field :math:`\\nabla f = (4, 0,
        0)`.

        >>> def value_fun(point):
        ...     x, y, z = point
        ...     return 2*x**2 + 3*y - 5*z
        ...
        >>> f = df.Field(mesh, dim=1, value=value_fun)
        >>> assert abs(f.laplace.average - 4) < 1e-3

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.dim == 1:
            return (self.derivative('x', n=2) +
                    self.derivative('y', n=2) +
                    self.derivative('z', n=2))
        else:
            return self.x.laplace << self.y.laplace << self.z.laplace

    def integral(self, direction='xyz', improper=False):
        """Integral.

        This method integrates the field over the mesh along the specified
        direction(s), which can ce specified using ``direction``. Field must be
        explicitly multiplied by an infinitesimal value (``DValue``) before
        integration. Improper integral can be computed by passing
        ``improper=True`` and by specifying a single direction.

        Parameters
        ----------
        direction : str, optional

            Direction(s) along which the field is integrated. Defaults to
            ``'xyz'``.

        improper : bool, optional

            If ``True``, an improper (cumulative) integral is computed.
            Defaults to ``False``.

        Returns
        -------
        discretisedfield.Field, numbers.Real, or (3,) array_like

            Integration result. If the field is integrated in all directions,
            ``numbers.Real`` or ``array_like`` value is returned depending on
            the dimension of the field.

        Raises
        ------
        ValueError

            If ``improper=True`` and more than one integration direction is
            specified.

        Example
        -------
        1. Volume integral of a scalar field.

        .. math::

            \\int_\\mathrm{V} f(\\mathbf{r}) \\mathrm{d}V

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=1, value=5)
        >>> (f * df.dV).integral()
        5000.0

        2. Volume integral of a vector field.

        .. math::

            \\int_\\mathrm{V} \\mathbf{f}(\\mathbf{r}) \\mathrm{d}V

        >>> f = df.Field(mesh, dim=3, value=(-1, -2, -3))
        >>> (f * df.dV).integral()
        (-1000.0, -2000.0, -3000.0)

        3. Surface integral of a scalar field.

        .. math::

            \\int_\\mathrm{S} f(\\mathbf{r}) |\\mathrm{d}\\mathbf{S}|

        >>> f = df.Field(mesh, dim=1, value=5)
        >>> f_plane = f.plane('z')
        >>> (f_plane * abs(df.dS)).integral()
        500.0

        4. Surface integral of a vector field (flux).

        .. math::

            \\int_\\mathrm{S} \\mathbf{f}(\\mathbf{r}) \\cdot
            \\mathrm{d}\\mathbf{S}

        >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
        >>> f_plane = f.plane('z')
        >>> (f_plane @ df.dS).integral()
        300.0

        5. Integral along x-direction.

        .. math::

            \\int_{x_\\mathrm{min}}^{x_\\mathrm{max}} \\mathbf{f}(\\mathbf{r})
            \\mathrm{d}x

        >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
        >>> f_plane = f.plane('z')
        >>> (f_plane * df.dx).integral(direction='x').average
        (10.0, 20.0, 30.0)

        6. Improper integral along x-direction.

        .. math::

            \\int_{x_\\mathrm{min}}^{x} \\mathbf{f}(\\mathbf{r})
            \\mathrm{d}x'

        >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
        >>> f_plane = f.plane('z')
        >>> (f_plane * df.dx).integral(direction='x', improper=True)
        Field(...)

        """
        if improper and len(direction) > 1:
            msg = 'Cannot compute improper integral along multiple directions.'
            raise ValueError(msg)

        mesh = self.mesh

        if not improper:
            for i in direction:
                mesh = mesh.plane(i)
            axes = [dfu.axesdict[i] for i in direction]
            res_array = np.sum(self.array, axis=tuple(axes), keepdims=True)
        else:
            res_array = np.cumsum(self.array, axis=dfu.axesdict[direction])

        res = self.__class__(mesh, dim=self.dim, value=res_array)

        if len(direction) == 3:
            return dfu.array2tuple(res.array.squeeze())
        else:
            return res

    def line(self, p1, p2, n=100):
        """Sampling the field along the line.

        Given two points :math:`p_{1}` and :math:`p_{2}`, :math:`n` position
        coordinates are generated and the corresponding field values.

        .. math::

           \\mathbf{r}_{i} = i\\frac{\\mathbf{p}_{2} -
           \\mathbf{p}_{1}}{n-1}

        Parameters
        ----------
        p1, p2 : (3,) array_like

            Two points between which the line is generated.

        n : int, optional

            Number of points on the line. Defaults to 100.

        Returns
        -------
        discretisedfield.Line

            Line object.

        Raises
        ------
        ValueError

            If ``p1`` or ``p2`` is outside the mesh domain.

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
        ...
        >>> line = field.line(p1=(0, 0, 0), p2=(2, 0, 0), n=5)

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

    def __getitem__(self, item):
        """Extracts the field on a subregion.

        If subregions were defined by passing ``subregions`` dictionary when
        the mesh was created, this method returns a field in a subregion
        ``subregions[item]``. Alternatively, a ``discretisedfield.Region``
        object can be passed and a minimum-sized field containing it will be
        returned. The resulting mesh has the same discretisation cell as the
        original field's mesh.

        Parameters
        ----------
        item : str, discretisedfield.Region

            The key of a subregion in ``subregions`` dictionary or a region
            object.

        Returns
        -------
        disretisedfield.Field

            Field on a subregion.

        Example
        -------
        1. Extract field on the subregion by passing a key.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (10, 10, 10)
        >>> subregions = {'r1': df.Region(p1=(0, 0, 0), p2=(50, 100, 100)),
        ...               'r2': df.Region(p1=(50, 0, 0), p2=(100, 100, 100))}
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions=subregions)
        >>> def value_fun(point):
        ...     x, y, z = point
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

        2. Extracting a subfield by passing a region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -25e-9, 0)
        >>> p2 = (50e-9, 25e-9, 5e-9)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, cell=cell)
        >>> field = df.Field(mesh=mesh, dim=1, value=5)
        ...
        >>> subregion = df.Region(p1=(-9e-9, -1e-9, 1e-9),
        ...                       p2=(9e-9, 14e-9, 4e-9))
        >>> subfield = field[subregion]
        >>> subfield.array.shape
        (4, 4, 1, 1)

        """
        submesh = self.mesh[item]

        index_min = self.mesh.point2index(submesh.index2point((0, 0, 0)))
        index_max = np.add(index_min, submesh.n)
        slices = [slice(i, j) for i, j in zip(index_min, index_max)]
        return self.__class__(submesh, dim=self.dim,
                              value=self.array[tuple(slices)])

    def project(self, direction):
        """Projects the field along one direction and averages it out along
        that direction.

        One of the axes (``'x'``, ``'y'``, or ``'z'``) is passed and the field
        is projected (averaged) along that direction. For example
        ``project('z')`` would average the field in the z-direction and return
        the field which has only one discretisation cell in the z-direction.

        Parameters
        ----------
        direction : str

            Direction along which the field is projected (``'x'``, ``'y'``, or
            ``'z'``).

        Returns
        ------
        discretisedfield.Field

            A projected field.

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
        n_cells = self.mesh.n[dfu.axesdict[direction]]
        return self.integral(direction=direction) / n_cells

    @property
    def angle(self):
        """In-plane angle of the vector field.

        This method can be applied only on sliced fields, when a plane is
        defined. This method then returns a scalar field which is an angle
        between the in-plane compoenent of the vector field and the horizontal
        axis. The angle is computed in radians and all values are in :math:`(0,
        2\\pi)` range.

        Returns
        -------
        discretisedfield.Field

            Angle scalar field.

        Raises
        ------
        ValueError

            If the field is not sliced.

        Example
        -------
        1. Computing the angle of the field in yz-plane.

        >>> import discretisedfield as df
        >>> import numpy as np
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(0, 1, 0))
        ...
        >>> abs(field.plane('z').angle.average - np.pi/2) < 1e-3
        True

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before angle can be computed.'
            raise ValueError(msg)

        angle_array = np.arctan2(self.array[..., self.mesh.info['axis2']],
                                 self.array[..., self.mesh.info['axis1']])

        # Place all values in [0, 2pi] range
        angle_array[angle_array < 0] += 2 * np.pi

        return self.__class__(self.mesh, dim=1,
                              value=angle_array[..., np.newaxis])

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
        """Write the field to an OVF2.0 file.

        Data representation (``'bin4'``, ``'bin8'``, or ``'txt'``) is passed
        using ``representation`` argument. If ``extend_scalar=True``, a scalar
        field will be saved as a vector field. More precisely, if the value at
        a cell is X, that cell will be saved as (X, 0, 0).

        Parameters
        ----------
        filename : str

            Name with an extension of the file written.

        representation : str, optional

            Representation; ``'bin4'``, ``'bin8'``, or ``'txt'``. Defaults to
            ``'txt'``.

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
        >>> value_fun = lambda point: (point[0], point[1], point[2])
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
                  'Title: Field',
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
                  'valueunits: None None None',
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

        # Write data.
        if representation in binary_reps:
            # Reopen with binary write, appending to the end of the file.
            with open(filename, 'ab') as f:
                # Add the binary checksum.
                packarray = [binary_reps[representation][0]]

                # Write data to the ovf file.
                for point, value in self:
                    if self.dim == 3:
                        v = value
                    else:
                        if extend_scalar:
                            v = [value, 0.0, 0.0]
                        else:
                            v = [value]
                    for vi in v:
                        packarray.append(vi)

                format = binary_reps[representation][1]*len(packarray)
                f.write(struct.pack(format, *packarray))
        else:
            # Reopen with txt representation, appending to the end of the file.
            with open(filename, 'a') as f:
                for point, value in self:
                    if self.dim == 3:
                        v = value
                    else:
                        if extend_scalar:
                            v = [value, 0.0, 0.0]
                        else:
                            v = [value]
                    for vi in v:
                        f.write(f' {str(vi)}')
                    f.write('\n')

        # Write footer lines to OOMMF file.
        with open(filename, 'a') as f:
            f.write(''.join(map(lambda line: f'# {line}\n', footer)))

    def _writevtk(self, filename):
        """Write the field to a VTK file.

        The data is saved as a ``RECTILINEAR_GRID`` dataset. Scalar field
        (``dim=1``) is saved as ``SCALARS``. On the other hand, vector field
        (``dim=3``) is saved as both ``VECTORS`` as well as ``SCALARS`` for all
        three components to enable easy colouring of vectors in some
        visualisation packages.

        The saved VTK file can be opened with `Paraview
        <https://www.paraview.org/>`_ or `Mayavi
        <https://docs.enthought.com/mayavi/mayavi/>`_.

        Parameters
        ----------
        filename : str

            File name with an extension.

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
        >>> value_fun = lambda point: (point[0], point[1], point[2])
        >>> field = df.Field(mesh, dim=3, value=value_fun)
        ...
        >>> filename = 'mytestfile.vtk'
        >>> field._writevtk(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> os.remove(filename)  # delete the file

        """
        header = ['# vtk DataFile Version 3.0',
                  'Field',
                  'ASCII',
                  'DATASET RECTILINEAR_GRID',
                  'DIMENSIONS {} {} {}'.format(*self.mesh.n),
                  f'X_COORDINATES {self.mesh.n[0]} float',
                  ' '.join(map(str, self.mesh.axis_points('x'))),
                  f'Y_COORDINATES {self.mesh.n[1]} float',
                  ' '.join(map(str, self.mesh.axis_points('y'))),
                  f'Z_COORDINATES {self.mesh.n[2]} float',
                  ' '.join(map(str, self.mesh.axis_points('z'))),
                  f'POINT_DATA {len(self.mesh)}']

        if self.dim == 1:
            data = dfu.vtk_scalar_data(self, 'field')
        elif self.dim == 3:
            data = dfu.vtk_scalar_data(self.x, 'x-component')
            data += dfu.vtk_scalar_data(self.y, 'y-component')
            data += dfu.vtk_scalar_data(self.z, 'z-component')
            data += dfu.vtk_vector_data(self, 'field')

        with open(filename, 'w') as f:
            f.write('\n'.join(header+data))

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
        >>> value_fun = lambda point: (point[0], point[1], point[2])
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
        """Read the field from an OVF (1.0 or 2.0), VTK, or HDF5 file.

        The extension of the ``filename`` should correspond to either:
            - OVF (``.ovf``, ``.omf``, ``.ohf``, ``.oef``)
            - VTK (``.vtk``), or
            - HDF5 (``.hdf5`` or ``.h5``).

        This is a ``classmethod`` and should be called as, for instance,
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
        1. Read the field from an OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> dirname = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'oommf-ovf2-bin4.omf')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(mesh=...)

        2. Read a field from the VTK file.

        >>> filename = os.path.join(dirname, 'vtk-file.vtk')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(mesh=...)

        3. Read a field from the HDF5 file.

        >>> filename = os.path.join(dirname, 'hdf5-file.hdf5')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(mesh=...)

        .. seealso:: :py:func:`~discretisedfield.Field._fromovf`
        .. seealso:: :py:func:`~discretisedfield.Field._fromhdf5`
        .. seealso:: :py:func:`~discretisedfield.Field._fromhdf5`
        .. seealso:: :py:func:`~discretisedfield.Field.write`

        """
        if any([filename.endswith(ext) for ext in ['.omf', '.ovf',
                                                   '.ohf', '.oef']]):
            return cls._fromovf(filename)
        elif any([filename.endswith(ext) for ext in ['.vtk']]):
            return cls._fromvtk(filename)
        elif any([filename.endswith(ext) for ext in ['.hdf5', '.h5']]):
            return cls._fromhdf5(filename)
        else:
            msg = (f'Reading file with extension {filename.split(".")[-1]} '
                   f'not supported.')
            raise ValueError(msg)

    @classmethod
    def _fromovf(cls, filename):
        """Read the field from an OVF file.

        Data representation (``txt``, ``bin4``, or ``bin8``) as well as the OVF
        version (OVF1.0 or OVF2.0) are extracted from the file itself.

        This is a ``classmethod`` and should be called as, for instance,
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
        >>> filename = os.path.join(dirname, 'oommf-ovf2-bin8.omf')
        >>> field = df.Field._fromovf(filename)
        >>> field
        Field(mesh=...)

        .. seealso:: :py:func:`~discretisedfield.Field._writeovf`

        """
        # valuedim is not in OVF1 metadata and has to be extracted
        # from the data itself.
        mdatalist = ['xmin', 'ymin', 'zmin',
                     'xmax', 'ymax', 'zmax',
                     'xstepsize', 'ystepsize', 'zstepsize',
                     'valuedim']
        mdatadict = dict()

        try:
            # Encoding in open is important on Windows.
            with open(filename, 'r', encoding='utf-8') as ovffile:
                lines = ovffile.readlines()

            mdatalines = list(filter(lambda s: s.startswith('#'), lines))
            datalines = np.loadtxt(filter(lambda s: not s.startswith('#'),
                                          lines))

            if '1.0' in mdatalines[0]:
                # valuedim is not in OVF1 file.
                mdatadict['valuedim'] = datalines.shape[-1]

            for line in mdatalines:
                for mdatum in mdatalist:
                    if mdatum in line:
                        mdatadict[mdatum] = float(line.split()[-1])
                        break

        except UnicodeDecodeError:
            with open(filename, 'rb') as ovffile:
                f = ovffile.read()
                lines = f.split(b'\n')

            mdatalines = list(filter(lambda s: s.startswith(bytes('#',
                                                                  'utf-8')),
                                     lines))

            if bytes('2.0', 'utf-8') in mdatalines[0]:
                endian = '<'  # little-endian
            elif bytes('1.0', 'utf-8') in mdatalines[0]:
                endian = '>'  # big-endian

            for line in mdatalines:
                for mdatum in mdatalist:
                    if bytes(mdatum, 'utf-8') in line:
                        mdatadict[mdatum] = float(line.split()[-1])
                        break

            header = b'# Begin: Data Binary '
            data_start = f.find(header)
            header = f[data_start:(data_start + len(header) + 1)]

            data_start += len(header)
            data_end = f.find(b'# End: Data Binary ')

            if b'4' in header:
                nbytes = 4
                formatstr = endian + 'f'
                checkvalue = 1234567.0
            elif b'8' in header:
                nbytes = 8
                formatstr = endian + 'd'
                checkvalue = 123456789012345.0

            newlines = [b'\n\r', b'\r\n', b'\n']  # ordered by length
            for nl in newlines:
                if f.startswith(nl, data_start):
                    data_start += len(nl)
                    # There is a difference between files written by OOMMF and
                    # mumax3. OOMMF has a newline character before the "end
                    # metadata line', whereas mumax3 does not. Therefore if the
                    # length of the data stream is not a multiple of nbytes, we
                    # have to subtract the length of newline character from
                    # data_end.
                    if (data_end - data_start) % nbytes != 0:
                        data_end -= len(nl)
                    break

            listdata = list(struct.iter_unpack(formatstr,
                                               f[data_start:data_end]))
            datalines = np.array(listdata)

            if datalines[0] != checkvalue:
                # These two lines cannot be accessed via tests. Therefore, they
                # are excluded from coverage.
                msg = 'Error in checksum comparison.'  # pragma: no cover
                raise AssertionError(msg)  # pragma: no cover

            datalines = datalines[1:]

        p1 = (mdatadict[key] for key in ['xmin', 'ymin', 'zmin'])
        p2 = (mdatadict[key] for key in ['xmax', 'ymax', 'zmax'])
        cell = (mdatadict[key] for key in ['xstepsize', 'ystepsize',
                                           'zstepsize'])

        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

        # valuedim is not in OVF1 file and for binary data it has to be
        # extracted here.
        if 'valuedim' not in mdatadict.keys():
            mdatadict['valuedim'] = len(datalines) / len(mesh)

        r_tuple = (*tuple(reversed(mesh.n)), int(mdatadict['valuedim']))
        t_tuple = (*tuple(reversed(range(3))), 3)

        return cls(mesh, dim=int(mdatadict['valuedim']),
                   value=datalines.reshape(r_tuple).transpose(t_tuple))

    @classmethod
    def _fromvtk(cls, filename):
        """Read the field from a VTK file.

        This method reads the field from a VTK file defined on
        STRUCTURED_POINTS written by ``discretisedfield._writevtk``.

        This is a ``classmethod`` and should be called as, for instance,
        ``discretisedfield.Field._fromvtk('myfile.vtk')``.

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
        1. Read a field from the VTK file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> dirname = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'vtk-file.vtk')
        >>> field = df.Field._fromvtk(filename)
        >>> field
        Field(mesh=...)

        .. seealso:: :py:func:`~discretisedfield.Field._writevtk`

        """
        with open(filename, 'r') as f:
            content = f.read()
        lines = content.split('\n')

        # Determine the dimension of the field.
        if 'VECTORS' in content:
            dim = 3
            data_marker = 'VECTORS'
            skip = 0  # after how many lines data starts after marker
        else:
            dim = 1
            data_marker = 'SCALARS'
            skip = 1

        # Extract the metadata
        mdatalist = ['X_COORDINATES', 'Y_COORDINATES', 'Z_COORDINATES']
        n = []
        cell = []
        origin = []
        for i, line in enumerate(lines):
            for mdatum in mdatalist:
                if mdatum in line:
                    n.append(int(line.split()[1]))
                    coordinates = list(map(float, lines[i+1].split()))
                    origin.append(coordinates[0])
                    if len(coordinates) > 1:
                        cell.append(coordinates[1] - coordinates[0])
                    else:
                        # If only one cell exists, 1nm cell is used by default.
                        cell.append(1e-9)

        # Create objects from metadata info
        p1 = np.subtract(origin, np.multiply(cell, 0.5))
        p2 = np.add(p1, np.multiply(n, cell))
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        field = cls(mesh, dim=dim)

        # Find where data starts.
        for i, line in enumerate(lines):
            if line.startswith(data_marker):
                start_index = i
                break

        # Extract data.
        for i, line in zip(mesh.indices, lines[start_index+skip+1:]):
            if not line[0].isalpha():
                field.array[i] = list(map(float, line.split()))

        return field

    @classmethod
    def _fromhdf5(cls, filename):
        """Read the field from an HDF5 file.

        This method reads the field from an HDF5 file defined on written by
        ``discretisedfield._writevtk``.

        This is a ``classmethod`` and should be called as, for instance,
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
        >>> filename = os.path.join(dirname, 'hdf5-file.hdf5')
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
            mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
            return cls(mesh, dim=dim, value=array[:])

    def mpl_scalar(self, *, ax=None, figsize=None, filter_field=None,
                   lightness_field=None, colorbar=True, colorbar_label=None,
                   multiplier=None, filename=None, **kwargs):
        """Plots the scalar field on a plane.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be a scalar field
        (``dim=1``). Otherwise, ``ValueError`` is raised. ``mpl_scalar`` adds
        the plot to ``matplotlib.axes.Axes`` passed via ``ax`` argument. If
        ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. By passing ``filter_field`` the points at which the pixels
        are not coloured can be determined. More precisely, only those
        discretisation cells where ``filter_field != 0`` are plotted. By
        passing a scalar field as ``lightness_field``, ligtness component is
        added to HSL colormap. In this case, colormap cannot be passed using
        ``kwargs``. Colorbar is shown by default and it can be removed from the
        plot by passing ``colorbar=False``. The label for the colorbar can be
        defined by passing ``colorbar_label`` as a string.

        It is often the case that the region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3  (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally. The plot can be saved as a PDF when ``filename``
        is passed.

        This method plots the field using ``matplotlib.pyplot.imshow``
        function, so any keyword arguments accepted by it can be passed (for
        instance, ``cmap`` - colormap, ``clim`` - colorbar limits, etc.).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the field plot is added. Defaults to ``None`` - axes
            are created internally.

        figsize : (2,) tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        filter_field : discretisedfield.Field, optional

            A scalar field used for determining whether certain discretisation
            cells are coloured. More precisely, only those discretisation cells
            where ``filter_field != 0`` are plotted. Defaults to ``None``.

        lightness_field : discretisedfield.Field, optional

            A scalar field used for adding lightness to the color. Field values
            are hue. Defaults to ``None``.

        colorbar : bool, optional

            If ``True``, colorbar is shown and it is hidden when ``False``.
            Defaults to ``True``.

        colorbar_label : str, optional

            Colorbar label. Defaults to ``None``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the field has not been sliced, its dimension is not 1, or the
            dimension of ``filter_field`` is not 1.

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
        >>> field.plane('y').mpl_scalar()

        .. seealso:: :py:func:`~discretisedfield.Field.mpl_vector`

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        if self.dim != 1:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        unit = f' ({uu.rsi_prefixes[multiplier]}m)'

        points, values = map(list, zip(*list(self)))

        if filter_field is not None:
            if filter_field.dim != 1:
                msg = f'Cannot use {filter_field.dim=} filter_field.'
                raise ValueError(msg)

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

        if lightness_field is not None:
            if lightness_field.dim != 1:
                msg = f'Cannot use {lightness_field.dim=} lightness_field.'
                raise ValueError(msg)

            _, lightness = map(list,
                               zip(*list(lightness_field[self.mesh.region])))

            rgb = dfu.hls2rgb(hue=values,
                              lightness=lightness,
                              saturation=None).reshape((*n, 3))

            kwargs['cmap'] = 'hsv'  # only hsv cmap allowed
            cp = ax.imshow(rgb, origin='lower', extent=extent, **kwargs)

        else:
            cp = ax.imshow(np.array(values).reshape(n),
                           origin='lower', extent=extent, **kwargs)

        if colorbar:
            cbar = plt.colorbar(cp)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        ax.set_xlabel(dfu.raxesdict[self.mesh.info['axis1']] + unit)
        ax.set_ylabel(dfu.raxesdict[self.mesh.info['axis2']] + unit)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def mpl_vector(self, ax=None, figsize=None, color=True, color_field=None,
                   colorbar=True, colorbar_label=None, multiplier=None,
                   filename=None, **kwargs):
        """Plots the vector field on a plane.

        Before the field can be plotted, it must be sliced with a plane (e.g.
        ``field.plane('z')``). In addition, field must be a vector field
        (``dim=3``). Otherwise, ``ValueError`` is raised. ``mpl_vector`` adds
        the plot to ``matplotlib.axes.Axes`` passed via ``ax`` argument. If
        ``ax`` is not passed, ``matplotlib.axes.Axes`` object is created
        automatically and the size of a figure can be specified using
        ``figsize``. By default, plotted vectors are coloured according to the
        out-of-plane component of the vectors. This can be changed by passing
        ``color_field`` with ``dim=1``. To disable colouring of the plot,
        ``color=False`` can be passed. Colorbar is shown by default and it can
        be removed from the plot by passing ``colorbar=False``. The label for
        the colorbar can be defined by passing ``colorbar_label`` as a string.
        It is often the case that the region size is small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3  (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all mesh points will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally. The plot can be saved as a PDF when ``filename``
        is passed.

        This method plots the field using ``matplotlib.pyplot.quiver``
        function, so any keyword arguments accepted by it can be passed (for
        instance, ``cmap`` - colormap, ``clim`` - colorbar limits, etc.). In
        particular, there are cases when ``matplotlib`` fails to find optimal
        scale for plotting vectors. More precisely, sometimes vectors appear
        too large in the plot. This can be resolved by passing ``scale``
        argument, which scales all vectors in the plot. In other words, larger
        ``scale``, smaller the vectors and vice versa. Please note that scale
        can be in a very large range (e.g. 1e20).

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional

            Axes to which the field plot is added. Defaults to ``None`` - axes
            are created internally.

        figsize : tuple, optional

            The size of a created figure if ``ax`` is not passed. Defaults to
            ``None``.

        color_field : discretisedfield.Field, optional

            A scalar field used for colouring the vectors. Defaults to ``None``
            and vectors are coloured according to their out-of-plane
            components.

        colorbar : bool, optional

            If ``True``, colorbar is shown and it is hidden when ``False``.
            Defaults to ``True``.

        colorbar_label : str, optional

            Colorbar label. Defaults to ``None``.

        multiplier : numbers.Real, optional

            ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is
            a multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that
            value, the axes will be scaled and appropriate units shown. For
            instance, if ``multiplier=1e-9`` is passed, the mesh points will be
            divided by :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will
            be used as axis labels. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the field has not been sliced, its dimension is not 3, or the
            dimension of ``color_field`` is not 1.

        Example
        -------
        1. Visualising the vector field using ``matplotlib``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1.1, 2.1, 3.1))
        ...
        >>> field.plane('y').mpl_vector()

        .. seealso:: :py:func:`~discretisedfield.Field.mpl_scalar`

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        if self.dim != 3:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        unit = f' ({uu.rsi_prefixes[multiplier]}m)'

        points, values = map(list, zip(*list(self)))

        # Remove points and values where norm is 0.
        points = [p for p, v in zip(points, values)
                  if not np.equal(v, 0).all()]
        values = [v for v in values if not np.equal(v, 0).all()]

        if color:
            if color_field is None:
                planeaxis = dfu.raxesdict[self.mesh.info['planeaxis']]
                color_field = getattr(self, planeaxis)

            colors = [color_field(p) for p in points]

        # "Unpack" values inside arrays and convert to np.ndarray.
        points = np.array(list(zip(*points)))
        values = np.array(list(zip(*values)))

        points = np.divide(points, multiplier)

        if color:
            cp = ax.quiver(points[self.mesh.info['axis1']],
                           points[self.mesh.info['axis2']],
                           values[self.mesh.info['axis1']],
                           values[self.mesh.info['axis2']],
                           colors, pivot='mid', **kwargs)
        else:
            ax.quiver(points[self.mesh.info['axis1']],
                      points[self.mesh.info['axis2']],
                      values[self.mesh.info['axis1']],
                      values[self.mesh.info['axis2']],
                      pivot='mid', **kwargs)

        if colorbar and color:
            cbar = plt.colorbar(cp)
            if colorbar_label is not None:
                cbar.ax.set_ylabel(colorbar_label)

        ax.set_xlabel(dfu.raxesdict[self.mesh.info['axis1']] + unit)
        ax.set_ylabel(dfu.raxesdict[self.mesh.info['axis2']] + unit)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    def mpl(self, ax=None, figsize=None, scalar_field=None,
            scalar_filter_field=None, scalar_lightness_field=None,
            scalar_cmap='viridis', scalar_clim=None, scalar_colorbar=True,
            scalar_colorbar_label=None, vector_field=None, vector_color=False,
            vector_color_field=None, vector_cmap='cividis', vector_clim=None,
            vector_colorbar=False, vector_colorbar_label=None,
            vector_scale=None, multiplier=None, filename=None):
        """Plots the field on a plane.

        This is a convenience method used for quick plotting, which combines
        ``discretisedfield.Field.mpl_scalar`` and
        ``discretisedfield.Field.mpl_vector`` methods. Depending on the
        dimensionality of the field, it determines what plot is going to be
        shown. For a scalar field only ``discretisedfield.Field.mpl_scalar`` is
        used, whereas for a vector field, both
        ``discretisedfield.Field.mpl_scalar`` and
        ``discretisedfield.Field.mpl_vector`` plots are shown, where vector
        plot shows the in-plane components of the vector and scalar plot
        encodes the out-of-plane component.

        All the default values can be changed by passing arguments, which are
        then used in subplots. The way parameters of this function are used to
        create plots can be understood with the following code snippet.

        .. code-block::

            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)

            scalar_field.mpl_scalar(ax=ax, filter_field=scalar_filter_field,
                                    lightness_field=scalar_lightness_field,
                                    colorbar=scalar_colorbar,
                                    colorbar_label=scalar_colorbar_label,
                                    multiplier=multiplier, cmap=scalar_cmap,
                                    clim=scalar_clim,)

            vector_field.mpl_vector(ax=ax, color=vector_color,
                                    color_field=vector_color_field,
                                    colorbar=vector_colorbar,
                                    colorbar_label=vector_colorbar_label,
                                    multiplier=multiplier, scale=vector_scale,
                                    cmap=vector_cmap, clim=vector_clim,)

            if filename is not None:
                plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)
            ```

            Therefore, to understand the meaning of the arguments which can be
            passed to this method, please refer to
            ``discretisedfield.Field.mpl_scalar`` and
            ``discretisedfield.Field.mpl_vector`` documentation.

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

            :py:func:`~discretisedfield.Field.mpl_scalar`
            :py:func:`~discretisedfield.Field.mpl_vector`

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        unit = f' ({uu.rsi_prefixes[multiplier]}m)'

        planeaxis = dfu.raxesdict[self.mesh.info['planeaxis']]

        # Set up default values.
        if self.dim == 1:
            if scalar_field is None:
                scalar_field = self
            else:
                scalar_field = self.__class__(self.mesh, dim=1,
                                              value=scalar_field)
            if vector_field is not None:
                vector_field = self.__class__(self.mesh, dim=3,
                                              value=vector_field)
        if self.dim == 3:
            if vector_field is None:
                vector_field = self
            else:
                vector_field = self.__class__(self.mesh, dim=3,
                                              value=vector_field)
            if scalar_field is None:
                scalar_field = getattr(self, planeaxis)
                scalar_colorbar_label = f'{planeaxis}-component'
            else:
                scalar_field = self.__class__(self.mesh, dim=1,
                                              value=scalar_field)
            if scalar_filter_field is None:
                scalar_filter_field = self.norm
            else:
                scalar_filter_field = self.__class__(self.mesh, dim=1,
                                                     value=scalar_filter_field)

        if scalar_field is not None:
            scalar_field.mpl_scalar(ax=ax, filter_field=scalar_filter_field,
                                    lightness_field=scalar_lightness_field,
                                    colorbar=scalar_colorbar,
                                    colorbar_label=scalar_colorbar_label,
                                    multiplier=multiplier, cmap=scalar_cmap,
                                    clim=scalar_clim,)
        if vector_field is not None:
            vector_field.mpl_vector(ax=ax, color=vector_color,
                                    color_field=vector_color_field,
                                    colorbar=vector_colorbar,
                                    colorbar_label=vector_colorbar_label,
                                    multiplier=multiplier, scale=vector_scale,
                                    cmap=vector_cmap, clim=vector_clim,)

        ax.set_xlabel(dfu.raxesdict[self.mesh.info['axis1']] + unit)
        ax.set_ylabel(dfu.raxesdict[self.mesh.info['axis2']] + unit)

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight', pad_inches=0.02)

    def k3d_nonzero(self, plot=None, color=dfu.cp_int[0], multiplier=None,
                    interactive_field=None, **kwargs):
        """``k3d`` plot of non-zero discretisation cells.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The colour of the non-zero discretisation cells can be
        specified using ``color`` argument.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        For interactive plots the field itself, before being sliced with the
        field, must be passed as ``interactive_field``. For example, if
        ``field.x.plane('z')`` is plotted, ``interactive_field=field`` must be
        passed. In addition, ``k3d.plot`` object cannot be created internally
        and it must be passed and displayed by the user.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally. This is not true in the case of an interactive
            plot, when ``plot`` must be created externally.

        color : int, optional

            Colour of the non-zero discretisation cells. Defaults to the
            default color palette.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        interactive_field : discretisedfield.Field, optional

            The whole field object (before any slices) used for interactive
            plots. Defaults to ``None``.

        Raises
        ------
        ValueError

            If the dimension of the field is not 1.

        Examples
        --------
        1. Visualising non-zero discretisation cells using ``k3d``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -50e-9, -50e-9)
        >>> p2 = (50e-9, 50e-9, 50e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> def normfun(point):
        ...     x, y, z = point
        ...     if x**2 + y**2 < 30**2:
        ...         return 1
        ...     else:
        ...         return 0
        >>> field.norm = normfun
        ...
        >>> field.norm.k3d_nonzero()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Field.k3d_voxels`

        """
        if self.dim != 1:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        unit = f'({uu.rsi_prefixes[multiplier]}m)'

        if interactive_field is not None:
            plot.camera_auto_fit = False

            objects_to_be_removed = []
            for i in plot.objects:
                if i.name != 'total_region':
                    objects_to_be_removed.append(i)
            for i in objects_to_be_removed:
                plot -= i

            if not any([o.name == 'total_region' for o in plot.objects]):
                interactive_field.mesh.region.k3d(plot=plot,
                                                  multiplier=multiplier,
                                                  name='total_region',
                                                  opacity=0.025)

        plot_array = np.ones_like(self.array)  # all voxels have the same color
        plot_array[self.array == 0] = 0  # remove voxels where field is zero
        plot_array = plot_array[..., 0]  # remove an empty dimension
        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)
        plot_array = plot_array.astype(np.uint8)  # to avoid k3d warning

        bounds = [i for sublist in
                  zip(np.divide(self.mesh.region.pmin, multiplier),
                      np.divide(self.mesh.region.pmax, multiplier))
                  for i in sublist]

        plot += k3d.voxels(plot_array, color_map=color, bounds=bounds,
                           outlines=False, **kwargs)

        plot.axes = [i + r'\,\text{{{}}}'.format(unit)
                     for i in dfu.axesdict.keys()]

    def k3d_scalar(self, plot=None, filter_field=None, cmap='cividis',
                   multiplier=None, interactive_field=None, **kwargs):
        """``k3d`` plot of a scalar field.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. The colormap can be specified using ``cmap`` argument.
        By passing ``filter_field`` the points at which the voxels are not
        shown can be determined. More precisely, only those discretisation
        cells where ``filter_field != 0`` are plotted.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        For interactive plots the field itself, before being sliced with the
        field, must be passed as ``interactive_field``. For example, if
        ``field.x.plane('z')`` is plotted, ``interactive_field=field`` must be
        passed. In addition, ``k3d.plot`` object cannot be created internally
        and it must be passed and displayed by the user.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally. This is not true in the case of an interactive
            plot, when ``plot`` must be created externally.

        filter_field : discretisedfield.Field, optional

            Scalar field. Only discretisation cells where ``filter_field != 0``
            are shown. Defaults to ``None``.

        cmap : str, optional

            Colormap.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        interactive_field : discretisedfield.Field, optional

            The whole field object (before any slices) used for interactive
            plots. Defaults to ``None``.

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
        >>> field.k3d_scalar()
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Field.k3d_vector`

        """
        if self.dim != 1:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if filter_field is not None:
            if filter_field.dim != 1:
                msg = f'Cannot use dim={self.dim} filter_field.'
                raise ValueError(msg)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        unit = f'({uu.rsi_prefixes[multiplier]}m)'

        if interactive_field is not None:
            plot.camera_auto_fit = False

            objects_to_be_removed = []
            for i in plot.objects:
                if i.name != 'total_region':
                    objects_to_be_removed.append(i)
            for i in objects_to_be_removed:
                plot -= i

            if not any([o.name == 'total_region' for o in plot.objects]):
                interactive_field.mesh.region.k3d(plot=plot,
                                                  multiplier=multiplier,
                                                  name='total_region',
                                                  opacity=0.025)

        plot_array = np.copy(self.array)  # make a deep copy
        plot_array = plot_array[..., 0]  # remove an empty dimension

        # All values must be in (1, 255) -> (1, n-1), for n=256 range, with
        # maximum n=256. This is the limitation of k3d.voxels(). Voxels where
        # values are zero, are invisible.
        plot_array = dfu.normalise_to_range(plot_array, (1, 255))
        # Remove voxels where filter_field = 0.
        if filter_field is not None:
            for i in self.mesh.indices:
                if filter_field(self.mesh.index2point(i)) == 0:
                    plot_array[i] = 0
        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)
        plot_array = plot_array.astype(np.uint8)  # to avoid k3d warning

        cmap = matplotlib.cm.get_cmap(cmap, 256)
        cmap_int = []
        for i in range(cmap.N):
            rgb = cmap(i)[:3]
            cmap_int.append(int(matplotlib.colors.rgb2hex(rgb)[1:], 16))

        bounds = [i for sublist in
                  zip(np.divide(self.mesh.region.pmin, multiplier),
                      np.divide(self.mesh.region.pmax, multiplier))
                  for i in sublist]

        plot += k3d.voxels(plot_array, color_map=cmap_int, bounds=bounds,
                           outlines=False, **kwargs)

        plot.axes = [i + r'\,\text{{{}}}'.format(unit)
                     for i in dfu.axesdict.keys()]

    def k3d_vector(self, plot=None, color_field=None, cmap='cividis',
                   head_size=1, points=True, point_size=None,
                   vector_multiplier=None, multiplier=None,
                   interactive_field=None, **kwargs):
        """``k3d`` plot of a vector field.

        If ``plot`` is not passed, ``k3d.Plot`` object is created
        automatically. By passing ``color_field`` vectors are coloured
        according to the values of that field. The colormap can be specified
        using ``cmap`` argument. The head size of vectors can be changed using
        ``head_size``. The size of the plotted vectors is computed
        automatically in order to fit the plot. However, it can be adjusted
        using ``vector_multiplier``.

        By default both vectors and points, corresponding to discretisation
        cell coordinates, are plotted. They can be removed from the plot by
        passing ``points=False``. The size of the points are calculated
        automatically, but it can be adjusted with ``point_size``.

        It is often the case that the object size is either small (e.g. on a
        nanoscale) or very large (e.g. in units of kilometers). Accordingly,
        ``multiplier`` can be passed as :math:`10^{n}`, where :math:`n` is a
        multiple of 3 (..., -6, -3, 0, 3, 6,...). According to that value, the
        axes will be scaled and appropriate units shown. For instance, if
        ``multiplier=1e-9`` is passed, all axes will be divided by
        :math:`1\\,\\text{nm}` and :math:`\\text{nm}` units will be used as
        axis labels. If ``multiplier`` is not passed, the best one is
        calculated internally.

        For interactive plots the field itself, before being sliced with the
        field, must be passed as ``interactive_field``. For example, if
        ``field.plane('z')`` is plotted, ``interactive_field=field`` must be
        passed. In addition, ``k3d.plot`` object cannot be created internally
        and it must be passed and displayed by the user.

        This method is based on ``k3d.voxels``, so any keyword arguments
        accepted by it can be passed (e.g. ``wireframe``).

        Parameters
        ----------
        plot : k3d.Plot, optional

            Plot to which the plot is added. Defaults to ``None`` - plot is
            created internally. This is not true in the case of an interactive
            plot, when ``plot`` must be created externally.

        color_field : discretisedfield.Field, optional

            Scalar field. Vectors are coloured according to the values of
            ``color_field``. Defaults to ``None``.

        cmap : str, optional

            Colormap.

        head_size : int, optional

            The size of vector heads. Defaults to ``None``.

        points : bool, optional

            If ``True``, points are shown together with vectors. Defaults to
            ``True``.

        point_size : int, optional

            The size of the points if shown in the plot. Defaults to ``None``.

        vector_multiplier : numbers.Real, optional

            All vectors are divided by this value before being plotted.
            Defaults to ``None``.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        interactive_field : discretisedfield.Field, optional

            The whole field object (before any slices) used for interactive
            plots. Defaults to ``None``.

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
        >>> field = df.Field(mesh, dim=3, value=(0, 0, 1))
        ...
        >>> field.k3d_vector()
        Plot(...)

        """
        if self.dim != 3:
            msg = f'Cannot plot dim={self.dim} field.'
            raise ValueError(msg)

        if plot is None:
            plot = k3d.plot()
            plot.display()

        if color_field is not None:
            if color_field.dim != 1:
                msg = f'Cannot use dim={self.dim} color_field.'
                raise ValueError(msg)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        unit = f'({uu.rsi_prefixes[multiplier]}m)'

        if interactive_field is not None:
            plot.camera_auto_fit = False

            objects_to_be_removed = []
            for i in plot.objects:
                if i.name != 'total_region':
                    objects_to_be_removed.append(i)
            for i in objects_to_be_removed:
                plot -= i

            if not any([o.name == 'total_region' for o in plot.objects]):
                interactive_field.mesh.region.k3d(plot=plot,
                                                  multiplier=multiplier,
                                                  name='total_region',
                                                  opacity=0.025)

        coordinates, vectors, color_values = [], [], []
        norm_field = self.norm  # assigned to be computed only once
        for point, value in self:
            if norm_field(point) != 0:
                coordinates.append(point)
                vectors.append(value)
                if color_field is not None:
                    color_values.append(color_field(point))

        if color_field is not None:
            color_values = dfu.normalise_to_range(color_values, (0, 255))

            # Generate double pairs (body, head) for colouring vectors.
            cmap = matplotlib.cm.get_cmap(cmap, 256)
            cmap_int = []
            for i in range(cmap.N):
                rgb = cmap(i)[:3]
                cmap_int.append(int(matplotlib.colors.rgb2hex(rgb)[1:], 16))

            colors = []
            for cval in color_values:
                colors.append(2*(cmap_int[cval],))
        else:
            # Uniform colour.
            colors = (len(vectors) * ([2*(dfu.cp_int[1],)]))

        coordinates = np.array(coordinates)
        vectors = np.array(vectors)

        if vector_multiplier is None:
            vector_multiplier = (vectors.max() /
                                 np.divide(self.mesh.cell, multiplier).min())

        coordinates = np.divide(coordinates, multiplier)
        vectors = np.divide(vectors, vector_multiplier)

        coordinates = coordinates.astype(np.float32)
        vectors = vectors.astype(np.float32)

        plot += k3d.vectors(coordinates-0.5*vectors, vectors, colors=colors,
                            head_size=head_size, **kwargs)

        if points:
            if point_size is None:
                # If undefined, the size of the point is 1/4 of the smallest
                # cell dimension.
                point_size = np.divide(self.mesh.cell, multiplier).min() / 4

            plot += k3d.points(coordinates, color=dfu.cp_int[0],
                               point_size=point_size)

        plot.axes = [i + r'\,\text{{{}}}'.format(unit)
                     for i in dfu.axesdict.keys()]
