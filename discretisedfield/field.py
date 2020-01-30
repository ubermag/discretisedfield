import k3d
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

# TODO: Laplacian, h5, vtk, tutorials, check rtd requirements, line object,
# plotting, plotting small samples, refactor plotting, pycodestyle, coverage,
# plotting line, remove numbers from tutorials, add more random numbers in tests
# add math equations in doc strings, check doc string consistency,
# do only test-coverage instead of twice,

@ts.typesystem(mesh=ts.Typed(expected_type=df.Mesh),
               dim=ts.Scalar(expected_type=int, unsigned=True, const=True))
class Field:
    """Finite difference field.

    This class defines a finite difference field and enables certain
    operations for its analysis and visualisation. The field is
    defined on a finite difference mesh (`discretisedfield.Mesh`).

    Parameters
    ----------
    mesh : discretisedfield.Mesh
        Finite difference rectangular mesh.
    dim : int, optional
        Dimension of the field value. For instance, if `dim=3` the
        field is a three-dimensional vector field and for `dim=1`
        the field is a scalar field. Defaults to `dim=3`.
    value : array_like, callable, optional
        Please refer to the `value` property:
        :py:func:`~discretisedfield.Field.value`. Defaults to 0,
        meaning that if the value is not provided in the
        initialisation process, "zero-field" will be defined.
    norm : numbers.Real, callable, optional
        Please refer to the `norm` property:
        :py:func:`~discretisedfield.Field.norm`. Defaults to `None`
        (`norm=None` defines no norm).

    Examples
    --------
    1. Creating a uniform three-dimensional vector field on a
    nano-sized thin film.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> cell = (1e-9, 1e-9, 0.1e-9)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    ...
    >>> dim = 3
    >>> value = (0, 0, 1)
    >>> field = df.Field(mesh=mesh, dim=dim, value=value)
    >>> field
    Field(mesh=...)

    2. Creating a scalar field.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-10, -10, -10)
    >>> p2 = (10, 10, 10)
    >>> n = (1, 1, 1)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> dim = 1
    >>> value = 3.14
    >>> field = df.Field(mesh=mesh, dim=dim, value=value)
    >>> field
    Field(mesh=...)

    .. seealso:: :py:func:`~discretisedfield.Mesh`

    """
    def __init__(self, mesh, dim=3, value=0, norm=None):
        self.mesh = mesh
        self.dim = dim
        self.value = value
        self.norm = norm

    @property
    def value(self):
        """Field value representation.

        This propertry returns a representation of the field value if
        it exists. Otherwise, the `numpy.ndarray` containing all
        values from the field is returned.

        Parameters
        ----------
        value : 0, array_like, callable
            For scalar fields (`dim=1`) `numbers.Real` values are
            allowed. In the case of vector fields, "array_like" (list,
            tuple, numpy.ndarray) value with length equal to `dim`
            should be used. Finally, the value can also be a callable
            (e.g. Python function or another field), which for every
            coordinate in the mesh returns a valid value. If
            `value=0`, all values in the field will be set to zero
            independent of the field dimension.

        Returns
        -------
        array_like, callable, numbers.Real
            The value used (representation) for setting the field is
            returned. However, if the actual value of the field does
            not correspond to the initially used value anymore, a
            `numpy.ndarray` is returned containing all field values.

        Raises
        ------
        ValueError
            If unsupported type is passed

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
        """Numpy array of a field value.

        `array` has shape of `(self.mesh.n[0], self.mesh.n[1],
        self.mesh.n[2], dim)`.

        Parameters
        ----------
        array : numpy.ndarray
            Numpy array with dimensions `(self.mesh.n[0],
            self.mesh.n[1], self.mesh.n[2], dim)`

        Returns
        -------
        numpy.ndarray
            Field values array.

        Raises
        ------
        ValueError
            If setting the array with wrong type, shape, or value.

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
        >>> field = df.Field(mesh=mesh, dim=3, value=value)
        >>> field.array
        array(...)
        >>> field.array.shape
        (2, 1, 1, 3)
        >>> field.array = np.ones(field.array.shape)
        >>> field.array
        array(...)

        .. seealso:: :py:func:`~discretisedfield.Field.value`

        """
        return self._array

    @array.setter
    def array(self, val):
        if isinstance(val, np.ndarray) and \
        val.shape == (*self.mesh.n, self.dim):
            self._array = val
        else:
            msg = f'Unsupported {type(val)} or invalid value shape.'
            raise ValueError(msg)

    @property
    def norm(self):
        """Norm of a field.

        This property computes the norm of the field and returns it as
        a `discretisedfield.Field` object with `dim=1`. Norm of a
        scalar field cannot be set and `ValueError` is raised.

        Parameters
        ----------
        numbers.Real, numpy.ndarray
            Norm value

        Returns
        -------
        discretisedfield.Field
            Scalar field with norm values.

        Raises
        ------
        ValueError
            If setting the norm with wrong type, shape, or value. In
            addition, if the field is scalar (dim=1) or it contains
            zero vector values.

        Examples
        --------
        1. Manipulating the field norm.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (1, 1, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field.norm
        Field(...)
        >>> field.norm = 2
        >>> field.average
        (0.0, 0.0, 2.0)
        >>> field.value = (1, 0, 0)
        >>> field.norm.average
        (1.0,)

        """
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

    @property
    def orientation(self):
        """Orientation field.

        This method computes the orientation (direction) field of a
        vector (`dim=3`) field and returns a `discretisedfield.Field`
        object with `dim=3`. More precisely, at every discretisation
        cell, the vector is divided by its norm, so that a unit vector
        is obtained. However, if the vector at a discretisation cell
        is a zero-vector, it remains unchanged. In the case of a
        scalar (`dim=1`) field, `ValueError` is raised.

        Returns
        -------
        discretisedfield.Field
            Orientation field.

        Raises
        ------
        ValueError
            If the field is not a `dim=3` field.

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
        (1.0,)

        """
        if self.dim == 1:
            msg = (f'Cannot compute orientation field for a '
                   f'field with dim={self.dim}.')
            raise ValueError(msg)

        orientation_array = np.divide(self.array,
                                      self.norm.array,
                                      out=np.zeros_like(self.array),
                                      where=(self.norm.array!=0))
        return self.__class__(self.mesh, dim=self.dim, value=orientation_array)

    @property
    def average(self):
        """Field average.

        It computes the average of the field over the entire volume of
        the mesh.

        Returns
        -------
        tuple
            Field average tuple whose length equals to the field's
            dimension.

        Examples
        --------
        1. Computing the vector field average.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field1 = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field1.average
        (0.0, 0.0, 1.0)
        >>> field2 = df.Field(mesh=mesh, dim=1, value=55)
        >>> field2.average
        (55.0,)

        """
        return dfu.array2tuple(self.array.mean(axis=(0, 1, 2)))

    def __repr__(self):
        """Field representation string.

        This method returns the string that can ideally be copied in
        another Python script so that exactly the same field object
        could be defined. However, this is usually not the case due to
        complex values used.

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
        >>> field = df.Field(mesh, dim=1, value=1)
        >>> repr(field)
        'Field(mesh=..., dim=1)'

        """
        return f'Field(mesh={repr(self.mesh)}, dim={self.dim})'

    def __call__(self, point):
        """Sample the field at `point`.

        It returns the value of the discreatisation cell `point`
        belongs to. It always returns a tuple, whose length is the
        same as the dimension of the field.

        Parameters
        ----------
        point : (3,) array_like
            The mesh point coordinate :math:`(p_{x}, p_{y}, p_{z})`.

        Returns
        -------
        tuple
            A tuple, whose length is the same as the dimension of the
            field.

        Example
        -------
        1. Sampling the field value

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (20, 20, 20)
        >>> n = (20, 20, 20)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 3, 4))
        >>> point = (10, 2, 3)
        >>> field(point)
        (1.0, 3.0, 4.0)

        """
        value = self.array[self.mesh.point2index(point)]
        if self.dim > 1:
            value = dfu.array2tuple(value)
        return value

    def __getattr__(self, attr):
        """Extracting the component of the vector field.

        If `'x'`, `'y'`, or `'z'` is accessed, a new scalar field of
        that component will be returned. This method is effective for
        vector fields with dimension 2 or 3.

        Parameters
        ----------
        attr : str
            Vector field component (`'x'`, `'y'`, or `'z'`)

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
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field.x
        Field(...)
        >>> field.y
        Field(...)
        >>> field.z
        Field(...)
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
        """Extension of the tab-completion list.

        Adds `'x'`, `'y'`, and `'z'`, depending on the dimension of
        the field, to the tab-completion list. This is effective in
        IPython or Jupyter notebook environment.

        """
        if self.dim in (2, 3):
            extension = list(dfu.axesdict.keys())[:self.dim]
        else:
            extension = []
        return dir(self.__class__) + extension

    def __iter__(self):
        """Generator yielding coordinates and values of all field cells.

        The discretisation cell coordinate corresponds to the cell
        centre point.

        Yields
        ------
        tuple (2,)
            The first value is the mesh cell coordinates (`px`, `py`,
            `pz`), whereas the second one is the field value.

        Examples
        --------
        1. Iterating through the field coordinates and values

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
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
        """Determine whether two fields are equal.

        Two fields are considered to be equal if:

          1. They are defined on the same mesh.

          2. They have the same dimension (`dim`).

          3. They both contain the same values in `array` attributes.

        Parameters
        ----------
        other : discretisedfield.Field
            Field object, which is compared to `self`.

        Returns
        -------
        bool

        Examples
        --------
        1. Check if fields are equal.

        >>> import discretisedfield as df
        ...
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        ...
        >>> f1 = df.Field(mesh, dim=1, value=3)
        >>> f2 = df.Field(mesh, dim=1, value=4-1)
        >>> f3 = df.Field(mesh, dim=3, value=(1, 4, 3))
        >>> f1 == f2
        True
        >>> f1 == f3
        False
        >>> f2 == f3
        False
        >>> f1 == 'a'
        False
        >>> f2 == 5
        False

        .. seealso:: :py:func:`~discretisedfield.Field.__ne__`

        """
        if not isinstance(other, self.__class__):
            return False
        elif self.mesh == other.mesh and self.dim == other.dim and \
             np.array_equal(self.array, other.array):
            return True
        else:
            return False

    def __ne__(self, other):
        """Determine whether two fields are not equal.

        This method returns `not (self == other)`. For details, refer
        to `discretisedfield.Field.__eq__` method.

        Parameters
        ----------
        other : discretisedfield.Field
            Field object, which is compared to `self`.

        Returns
        -------
        bool

        Examples
        --------
        1. Check if fields are not equal.

        >>> import discretisedfield as df
        ...
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        ...
        >>> f1 = df.Field(mesh, dim=1, value=3)
        >>> f2 = df.Field(mesh, dim=1, value=4-1)
        >>> f3 = df.Field(mesh, dim=3, value=(1, 4, 3))
        >>> f1 != f2
        False
        >>> f1 != f3
        True
        >>> f2 != f3
        True

        .. seealso:: :py:func:`~discretisedfield.Field.__eq__`

        """
        return not self == other

    def __abs__(self):
        """Field norm.

        This method implements abs() built-in method. If the field is a scalar
        field (`dim=1`), absolute value is returned. On the other hand, if the
        field is a vector field (`dim=3`), norm of the field is returned. For
        more details, refer to `discretisedfield.Field.norm()` method.

        Returns
        -------
        discretisedfield.Field
            Absolute value or norm of the field (`self.norm`).

        Examples
        --------
        1. Applying abs() built-in method on field.

        >>> import discretisedfield as df
        ...
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(5, 5, 5), cell=(1, 1, 1))
        ...
        >>> f1 = df.Field(mesh, dim=1, value=-3)
        >>> f2 = df.Field(mesh, dim=3, value=(-6, 8, 0))
        >>> abs(f1)
        Field(...)
        >>> abs(f1).average
        (3.0,)
        >>> abs(f2)
        Field(...)
        >>> abs(f2).average
        (10.0,)

        .. seealso:: :py:func:`~discretisedfield.Field.norm`

        """
        return self.norm

    def __pos__(self):
        """Unary + operator.

        This method defines the unary operator `+`. It returns the field itself.

        Returns
        -------
        discretisedfield.Field

        Example
        -------
        1. Applying unary + operator on a field.

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

        """
        return self

    def __neg__(self):
        """Unary - operator.

        This method defines the unary operator `-`. It negates the
        value of each discretisation cell. It is equivalent to
        multiplication with -1.

        Returns
        -------
        discretisedfield.Field

        Example
        -------
        1. Applying unary negation operator on a scalar field.

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
        (-3.1,)
        >>> f == -(-f)
        True

        2. Applying unary negation operator on a vector field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5e-9, 5e-9, 5e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> f = df.Field(mesh, dim=3, value=(0, -1000, -3))
        >>> res = -f
        >>> res.average
        (0.0, 1000.0, 3.0)

        """
        return -1 * self

    def __pow__(self, other):
        """Unary power operator.

        This method defines the unary operator `**` for scalar
        (`dim=1`) fields. This operator is not defined for vector
        (`dim=3`) fields, and in that case, ValueError is raised.

        Parameters
        ----------
        other : numbers.Real

        Returns
        -------
        discretisedfield.Field

        Example
        -------
        1. Applying unary power operator on a scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-25e-3, -25e-3, -25e-3)
        >>> p2 = (25e-3, 25e-3, 25e-3)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> f = df.Field(mesh, dim=1, value=2)
        >>> res = f**(-1)
        >>> res
        Field(...)
        >>> res.average
        (0.5,)
        >>> res = f**2
        >>> res.average
        (4.0,)
        >>> f**f  # the power must be numbers.Real
        Traceback (most recent call last):
        ...
        TypeError: ...

        2. Attempt to apply power operator on a vector field.

        >>> import discretisedfield as df
        ...
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
            msg = f'Cannot apply power operator on dim={self.dim} field.'
            raise ValueError(msg)
        if not isinstance(other, numbers.Real):
            msg = (f'Unsupported operand type(s) for **: '
                   f'{type(self)} and {type(other)}.')
            raise TypeError(msg)

        return self.__class__(self.mesh, dim=1,
                              value=np.power(self.array, other))

    def __add__(self, other):
        """Addition operator.

        This method defines the binary operator `+`, which can be
        applied only between two `discretisedfield.Field`
        objects. Both `discretisedfield.Field` objects must be defined
        on the same mesh and have the same dimensions.

        Parameters
        ----------
        other : discretisedfield.Field
            Second operand

        Returns
        -------
        discretisedfield.Field

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
        """Subtraction operator.

        This method defines the binary operator `-`, which can be
        applied only between two `discretisedfield.Field`
        objects. Both `discretisedfield.Field` objects must be defined
        on the same mesh and have the same dimensions.

        Parameters
        ----------
        other : discretisedfield.Field
            Second operand

        Returns
        -------
        discretisedfield.Field

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
        >>> f1 = df.Field(mesh, dim=3, value=(0, 1, 6))
        >>> f2 = df.Field(mesh, dim=3, value=(0, 1, 3))
        >>> res = f1 - f2
        >>> res.average
        (0.0, 0.0, 3.0)
        >>> f1 - 3.14
        Traceback (most recent call last):
        ...
        TypeError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.__add__`

        """
        return self + (-other)

    def __mul__(self, other):
        """Multiplication operator.

        This method defines the binary operator `*`, which can be
        applied between:

        1. Two scalar (`dim=1`) fields,

        2. A field of any dimension and `numbers.Real`, or

        3. A field of any dimension and a scalar (`dim=1`) field.

        If both operands are `discretisedfield.Field` objects, they
        must be defined on the same mesh.

        Parameters
        ----------
        other : discretisedfield.Field, numbers.Real
            Second operand

        Returns
        -------
        discretisedfield.Field

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
        (45.0,)
        >>> f1 * f2 == f2 * f1
        True

        2. Multiply vector field with a scalar.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 3, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=3, value=(0, 2, 5))
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
                msg = f'Cannot multiply dim={self.dim} and dim={other.dim} fields.'
                raise ValueError(msg)
            res_array = np.multiply(self.array, other.array)
        if isinstance(other, numbers.Real):
            res_array = np.multiply(self.array, other)

        return self.__class__(self.mesh, dim=res_array.shape[-1],
                              value=res_array)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Division operator.

        This method defines the binary operator `/`, which can be
        applied between:

        1. Two scalar (`dim=1`) fields,

        2. A field of any dimension and `numbers.Real`, or

        3. A field of any dimension and a scalar (`dim=1`) field.

        If both operands are `discretisedfield.Field` objects, they
        must be defined on the same mesh.

        Parameters
        ----------
        other : discretisedfield.Field, numbers.Real
            Second operand

        Returns
        -------
        discretisedfield.Field

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
        (5.0,)
        >>> f1 / f2 == (f2 / f1)**(-1)
        True

        2. Divide vector field by a scalar.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 3, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
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
        """Dot product operator (@).

        This function computes the dot product between two fields. Both
        fields must be three-dimensional and defined on the same mesh. If
        any of the fields is not of dimension 3, `ValueError` is raised.

        Parameter
        ---------
        other : discretisedfield.Field
            Three-dimensional fields

        Returns
        -------
        discretisedfield.Field

        Raises
        ------
        ValueError
            If the dimension of any of the fields is not 3.

        Example
        -------
        1. Compute the dot product of two vector fields.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f1 = df.Field(mesh, dim=3, value=(1, 3, 6))
        >>> f2 = df.Field(mesh, dim=3, value=(-1, -2, 2))
        >>> (f1@f2).average
        (5.0,)

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

        This method computes a directional derivative of the field and
        returns a field as a result. The dimensionality of the output
        field is the same as the dimensionality of the input
        field. The direction in which the derivative is computed is
        passed via `direction` argument, which can be `'x'`, `'y'`, or
        `'z'`. Alternatively, 0, 1, or 2 can be passed, respectively.

        Directional derivative cannot be computed if only one
        discretisation cell exists in a specified direction. In that
        case, a zero field is returned. More precisely, it is assumed
        that the field does not change in that direction.

        Parameters
        ----------
        direction : str, int
            The direction in which the derivative is computed. It can
            be `'x'`, `'y'`, or `'z'` (alternatively, 0, 1, or 2,
            respectively).

        Returns
        -------
        discretisedfield.Field

        Example
        -------
        1. Compute directional derivative of a scalar field in the
        y-direction of a spatially varying field. For the field we
        choose f(x, y, z) = 2*x + 3*y - 5*z. Accordingly, we expect
        the derivative in the y-direction to be to be a constant
        scalar field df/dy = 3.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 10e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> def value_fun(pos):
        ...     x, y, z = pos
        ...     return 2*x + 3*y - 5*z
        ...
        >>> f = df.Field(mesh, dim=1, value=value_fun)
        >>> f.derivative('y').average
        (3.0,)

        2. Try to compute directional derivatives of the vector field
        which has only one cell in the z-direction. For the field we
        choose f(x, y, z) = (2*x, 3*y, -5*z). Accordingly, we expect
        the directional derivatives to be: df/dx = (2, 0, 0), df/dy =
        (0, 3, 0), df/dz = (0, 0, -5). However, because there is only
        one discretisation cell in the z-direction, the derivative
        cannot be computed and a zero field is returned.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 10e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
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

        This method computes the gradient of a scalar (`dim=1`) field
        and returns a vector field. If the field is not of dimension
        1, `ValueError` is raised.

        Directional derivative cannot be computed if only one
        discretisation cell exists in a certain direction. In that
        case, a zero field is considered to be that directional
        derivative. More precisely, it is assumed that the field does
        not change in that direction.

        Returns
        -------
        discretisedfield.Field

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
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=1, value=5)
        >>> f.grad.average
        (0.0, 0.0, 0.0)

        2. Compute gradient of a spatially varying field. For a field
        we choose f(x, y, z) = 2*x + 3*y - 5*z. Accordingly, we expect
        the gradient to be a constant vector field
        grad(f) = (2, 3, -5).

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 100e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> def value_fun(pos):
        ...     x, y, z = pos
        ...     return 2*x + 3*y - 5*z
        ...
        >>> f = df.Field(mesh, dim=1, value=value_fun)
        >>> f.grad.average
        (2.0, 3.0, -5.0)

        2. Attempt to compute the gradient of a vector field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 100e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
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

        This method computes the divergence of a vector field
        (`dim=3`) and returns a scalar field (`dim=1`) as a result. If
        the field is not of dimension 3, `ValueError` is raised.

        Directional derivative cannot be computed if only one
        discretisation cell exists in a certain direction. In that
        case, a zero field is considered to be that directional
        derivative. More precisely, it is assumed that the field does
        not change in that direction.

        Returns
        -------
        discretisedfield.Field

        Raises
        ------
        ValueError
            If the dimension of the field is not 3.

        Example
        -------
        1. Compute the divergence of a vector field. For a field we
        choose f(x, y, z) = (2*x, -2*y, 5*z). Accordingly, we expect
        the divergence to be to be a constant scalar field div(f) = 5.

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
        (5.0,)

        2. Attempt to compute the divergence of a scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 100e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
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

        This method computes the curl of a vector field (`dim=3`) and
        returns a vector field (`dim=3`) as a result. If the field is
        not of dimension 3, `ValueError` is raised.

        Directional derivative cannot be computed if only one
        discretisation cell exists in a certain direction. In that
        case, a zero field is considered to be that directional
        derivative. More precisely, it is assumed that the field does
        not change in that direction.

        Returns
        -------
        discretisedfield.Field

        Raises
        ------
        ValueError
            If the dimension of the field is not 3.

        Example
        -------
        1. Compute curl of a vector field. For a field we choose
        f(x, y, z) = (2*x*y, -2*y, 5*x*z). Accordingly, we expect
        the curl to be to be a constant vector field
        curl(f) = (0, -5*z, -2*x).

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

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100e-9, 100e-9, 100e-9)
        >>> cell = (10e-9, 10e-9, 10e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
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

        This method computes the volume integral of the field and
        returns a single (scalar or vector) value as tuple. This value
        can be understood as the product of field's average value and
        the mesh volume, because the volume of all discretisation
        cells is the same.

        Returns
        -------
        tuple

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
        (5000.0,)

        2. Compute the volume integral of a vector field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (0.5, 0.5, 0.5)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=3, value=(-1, -2, -3))
        >>> f.integral
        (-8.0, -16.0, -24.0)

        .. seealso:: :py:func:`~discretisedfield.Field.average`

        """
        cell_volume = self.mesh.region.volume / len(self.mesh)
        field_sum = np.sum(self.array, axis=(0, 1, 2))
        return dfu.array2tuple(field_sum * cell_volume)

    @property
    def topological_charge_density(self):
        """Topological charge density.

        This method computes the topological charge density for the
        vector field (`dim=3`). Topological charge is defined on
        two-dimensional samples only. Therefore, the field must be "sliced"
        using the `discretisedfield.Field.plane` method. If the field
        is not three-dimensional or the field is not sliced,
        `ValueError` is raised.

        Returns
        -------
        discretisedfield.Field

        Raises
        ------
        ValueError
            If the field is not three-dimensional or the field is not
            sliced

        Example
        -------
        1. Compute the topological charge density of a spatially
        constant vector field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=3, value=(1, 1, -1))
        >>> f.plane('z').topological_charge_density.average
        (0.0,)

        2. Attempt to compute the topological charge density of a
        scalar field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=1, value=12)
        >>> f.plane('z').topological_charge_density
        Traceback (most recent call last):
        ...
        ValueError: ...

        3. Attempt to compute the topological charge density of a
        vector field, which is not sliced.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> cell = (2e-9, 2e-9, 2e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
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
    def _bergluescher(self):
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

        This method computes the topological charge for the vector
        field (`dim=3`). There are two possible methods:

        1. `continuous`: Topological charge density is integrated.

        2. `berg-luescher`: Topological charge is computed on a
        discrete lattice, as described in Berg and Luescher, Nuclear
        Physics, Section B, Volume 190, Issue 2, p. 412-424.

        Topological charge is defined on two-dimensional
        samples. Therefore, the field must be "sliced" using
        `discretisedfield.Field.plane` method. If the field is not
        three-dimensional or the field is not sliced, `ValueError` is
        raised.

        Parameters
        ----------
        method : str, optional
            Method how the topological charge is computed. It can be
            `continuous` or `berg-luescher`. Default is `continuous`.

        Returns
        -------
        float
            Topological charge

        Raises
        ------
        ValueError
            If the field does not have `dim=3` or the field is not
            sliced.

        Example
        -------
        1. Compute the topological charge of a spatially constant
        vector field. Zero value is expected.

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

        2. Attempt to compute the topological charge of a scalar
        field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=1, value=12)
        >>> f.plane('z').topological_charge()
        Traceback (most recent call last):
        ...
        ValueError: ...

        3. Attempt to compute the topological charge of a vector
        field, which is not sliced.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> cell = (2e-9, 2e-9, 2e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, dim=3, value=(1, 2, 3))
        >>> f.topological_charge_density()
        Traceback (most recent call last):
        ...
        ValueError: ...

        """
        if method == 'continuous':
            return self.topological_charge_density.integral[0]
        elif method == 'berg-luescher':
            return self._bergluescher
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
        >>> for coord, value in field.line(p1=(0, 0, 0), p2=(2, 0, 0), n=3):
        ...     print(coord, value)
        (0.0, 0.0, 0.0) (0.0, 3.0)
        (1.0, 0.0, 0.0) (0.0, 3.0)
        (2.0, 0.0, 0.0) (0.0, 3.0)

        """
        for point in self.mesh.line(p1=p1, p2=p2, n=n):
            yield point, self.__call__(point)

    def plane(self, *args, n=None, **kwargs):
        """Slices the field with a plane.

        If one of the axes (`'x'`, `'y'`, or `'z'`) is passed as a
        string, a plane perpendicular to that axis is generated which
        intersects the field at its centre. Alternatively, if a keyword
        argument is passed (e.g. `x=1`), a plane perpendicular to the
        x-axis and intersecting it at x=1 is generated. The number of
        points in two dimensions on the plane can be defined using `n`
        (e.g. `n=(10, 15)`). Using the generated plane, a new
        "two-dimensional" field is created and returned.

        Parameters
        ----------
        n : tuple of length 2
            The number of points on the plane in two dimensions

        Returns
        ------
        discretisedfield.Field
            A field obtained as an intersection of mesh and the plane.

        Example
        -------
        1. Intersecting the field with a plane.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh, dim=3)
        >>> field.plane(y=1)
        Field(mesh=...)

        """
        plane_mesh = self.mesh.plane(*args, n=n, **kwargs)
        return self.__class__(plane_mesh, dim=self.dim, value=self)

    def __getitem__(self, key):
        """Extract field of a subregion defined in mesh.

        If subregions are defined in mesh, this method returns a field on a
        subregion `mesh.subregions[key]` with the same discretisation cell as
        the parent mesh.

        Parameters
        ----------
        key : str
            The key associated to the region in `self.mesh.subregions`

        Returns
        -------
        disretisedfield.Field
            Field of a subregion

        Example
        -------
        1. Extract subregion field.

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
        >>> f['r1'].average
        (1.0, 2.0, 3.0)
        >>> f['r2'].average
        (-1.0, -2.0, -3.0)

        """
        return self.__class__(self.mesh[key], dim=self.dim, value=self)

    def project(self, *args, n=None):
        """Projects the field along one direction and averages it out along
        that direction.

        One of the axes (`'x'`, `'y'`, or `'z'`) is passed and the
        field is projected (averaged) along that direction. For example
        `project('z')` would average the field in the z-direction and
        return the field which has only one discretisation cell in the
        z-direction. The number of points in two dimensions on the
        plane can be defined using `n` (e.g. `n=(10, 15)`). The
        resulting field has the same dimension as the field itself.

        Parameters
        ----------
        n : (2,) tuple
            The number of points on the plane in two dimensions

        Returns
        -------
        discretisedfield.Field
            A field projected along a certain direction.

        Example
        -------
        1. Projecting the field along a certain direction.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 3))
        >>> field.project('z')
        Field(...)
        >>> field.project('z').array.shape
        (2, 2, 1, 3)

        """
        plane_mesh = self.mesh.plane(*args, n=n)
        plane_array = self.array.mean(axis=plane_mesh.info['planeaxis'],
                                      keepdims=True)
        return self.__class__(plane_mesh, dim=self.dim, value=plane_array)

    def write(self, filename, representation='txt', extend_scalar=False):
        """Write the field in .ovf, .omf, .ohf, or vtk format.

        If the extension of the `filename` is `.vtk`, a VTK file is
        written
        (:py:func:`~discretisedfield.Field._writevtk`). Otherwise, for
        `.ovf`, `.omf`, or `.ohf` extensions, an OOMMF file is written
        (:py:func:`~discretisedfield.Field._writeovf`). The
        representation (`bin4`, 'bin8', or 'txt') is passed using
        `representation` argument.

        Parameters
        ----------
        filename : str
            Name of the file written. It depends on its extension the
            format it is going to be written as.
        representation : str
            In the case of OOMMF files (`.ovf`, `.omf`, or `.ohf`),
            representation can be specified (`bin4`, `bin8`, or
            `txt`). Defaults to 'txt'.
        extend_scalar : bool
            If True, a scalar field will be saved as a vector
            field. More precisely, if the value at a cell is 3, that
            cell will be saved as (3, 0, 0). This is valid only for
            the OVF file formats.

        Example
        -------
        1. Write an .omf file and delete it from the disk

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5)
        >>> p2 = (5, 15, 15)
        >>> n = (5, 15, 20)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, value=(5, 6, 7))
        >>> filename = 'mytestfile.omf'
        >>> field.write(filename)  # write the file
        >>> os.remove(filename)  # delete the file

        .. seealso:: :py:func:`~discretisedfield.Field.fromfile`

        """
        if any([filename.endswith(ext) for ext in ['.omf', '.ovf', '.ohf']]):
            self._writeovf(filename, representation=representation,
                           extend_scalar=extend_scalar)
        elif filename.endswith('.vtk'):
            self._writevtk(filename)
        else:
            msg = ('Allowed file extensions for are .omf, .ovf, '
                   '.ohf, and .vtk.')
            raise ValueError(msg)

    def _writeovf(self, filename, representation='txt', extend_scalar=False):
        """Write the field in .ovf, .omf, or .ohf format.

        The extension of the `filename` should be `.ovf`, `.omf`, or
        `.ohf`. The representation (`bin4`, 'bin8', or 'txt') is
        passed using `representation` argument.

        Parameters
        ----------
        filename : str
            Name of the file written.
        representation : str
            Representation of the file (`bin4`, `bin8`, or
            `txt`). Defaults to 'txt'.
        extend_scalar : bool
            If True, a scalar field will be saved as a vector
            field. More precisely, if the value at a cell is 3, that
            cell will be saved as (3, 0, 0).

        Example
        -------
        1. Write an .omf file and delete it from the disk

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5)
        >>> p2 = (5, 15, 15)
        >>> n = (5, 15, 20)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, value=(5, 6, 7))
        >>> filename = 'mytestfile.omf'
        >>> field._writeovf(filename)  # write the file
        >>> os.remove(filename)  # delete the file

        """
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
                  # TODO: fix names
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

        # Write header lines to the ovf file.
        f = open(filename, 'w')
        f.write(''.join(map(lambda line: f'# {line}\n', header)))
        f.close()

        binary_reps = {'bin4': (1234567.0, 'f'),
                       'bin8': (123456789012345.0, 'd')}

        if representation in binary_reps:
            # Reopen the file with binary write, appending to the end
            # of the file.
            f = open(filename, 'ab')

            # Add the 8 bit binary check value that OOMMF uses.
            packarray = [binary_reps[representation][0]]

            # Write data to the ovf file.
            for i in self.mesh.indices:
                for vi in self.array[i]:
                    packarray.append(vi)

            v_bin = struct.pack(binary_reps[representation][1]*len(packarray),
                                *packarray)
            f.write(v_bin)
            f.close()

        else:
            # Reopen the file for txt representation, appending to the
            # file.
            f = open(filename, 'a')
            for i in self.mesh.indices:
                if self.dim == 3:
                    v = [vi for vi in self.array[i]]
                elif self.dim == 1:
                    if extend_scalar:
                        v = [self.array[i][0], 0.0, 0.0]
                    else:
                        v = [self.array[i][0]]
                else:
                    msg = (f'Cannot write dim={self.dim} field.')
                    raise TypeError(msg)
                for vi in v:
                    f.write(' ' + str(vi))
                f.write('\n')
            f.close()

        # Write footer lines to OOMMF file.
        f = open(filename, 'a')
        f.write(''.join(map(lambda line: f'# {line}\n', footer)))
        f.close()

    def _writevtk(self, filename):
        """Write the field in the VTK format.

        The extension of the `filename` should be `.vtk`.

        Parameters
        ----------
        filename : str
            Name of the file written.

        Example
        -------
        1. Write a .vtk file and delete it from the disk

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5)
        >>> p2 = (5, 15, 15)
        >>> n = (5, 15, 20)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, value=(5, 6, 7))
        >>> filename = 'mytestfile.vtk'
        >>> field._writevtk(filename)  # write the file
        >>> os.remove(filename)  # delete the file

        """
        grid = [pmini + np.linspace(0, li, ni+1) for pmini, li, ni in
                zip(self.mesh.region.pmin,
                    self.mesh.region.edges,
                    self.mesh.n)]

        structure = pyvtk.RectilinearGrid(*grid)
        vtkdata = pyvtk.VtkData(structure)

        vectors = [self.__call__(coord) for coord in self.mesh.coordinates]
        vtkdata.cell_data.append(pyvtk.Vectors(vectors, 'field'))
        for i, component in enumerate(dfu.axesdict.keys()):
            name = f'field_{component}'
            vtkdata.cell_data.append(pyvtk.Scalars(list(zip(*vectors))[i],
                                                   name))

        vtkdata.tofile(filename)

    @classmethod
    def fromfile(cls, filename, norm=None):
        """Read the field from .ovf, .omf, or .ohf file.

        The extension of the `filename` should be `.ovf`, `.omf`, or
        `.ohf`. If the field should be normalised, `norm` argument can
        be passed. The `name` of the field defaults to `'field'`. This
        is a `classmethod` and should be called as
        `discretisedfield.Field.fromfile('myfile.omf')`.

        Parameters
        ----------
        filename : str
            Name of the file to be read.
        norm : numbers.Real, numpy.ndarray, callable
            For details, refer to :py:func:`~discretisedfield.Field.value`.

        Returns
        -------
        discretisedfield.Field

        Example
        -------
        1. Read a field from the .ovf file

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> ovffile = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample',
        ...                        'mumax-output-linux.ovf')
        >>> field = df.Field.fromfile(ovffile)
        >>> field
        Field(mesh=...)

        .. seealso:: :py:func:`~discretisedfield.Field.write`

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
                # These two lines cannot be accessed via
                # tests. Therefore, they are excluded from coverage.
                msg = 'Binary Data cannot be read.'  # pragma: no cover
                raise AssertionError(msg)  # pragma: no cover

            datalines = datalines[1:]  # check value removal

        p1 = (mdatadict[key] for key in ['xmin', 'ymin', 'zmin'])
        p2 = (mdatadict[key] for key in ['xmax', 'ymax', 'zmax'])
        cell = (mdatadict[key] for key in ['xstepsize', 'ystepsize',
                                           'zstepsize'])
        dim = int(mdatadict['valuedim'])

        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        field = df.Field(mesh, dim=dim)

        r_tuple = tuple(reversed(field.mesh.n)) + (int(mdatadict['valuedim']),)
        t_tuple = tuple(reversed(range(3))) + (3,)
        field.array = datalines.reshape(r_tuple).transpose(t_tuple)
        field.norm = norm  # Normalise if norm is passed

        return field

    def mpl(self, figsize=None):
        """Plots a field plane using matplotlib.

        Before the field can be plotted, it must be sliced with a
        plane (e.g. `field.plane(`z`)`). Otherwise, ValueError is
        raised. For vector fields, this method plots both `quiver`
        (vector) and `imshow` (scalar) plots. The `imshow` plot
        represents the value of the out-of-plane vector component and
        the `quiver` plot is not coloured. On the other hand, only
        `imshow` is plotted for scalar fields. Where the norm of the
        field is zero, no vectors are shown and those `imshow` pixels
        are not coloured. In order to use this function inside Jupyter
        notebook `%matplotlib inline` must be activated after
        `discretisedfield` is imported.

        Parameters
        ----------
        figsize : tuple, optional
            Length-2 tuple passed to the `matplotlib.figure` function.

        Raises
        ------
        ValueError
            If the field has not been sliced with a plane.

        Example
        -------
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> field.plane(z=50, n=(5, 5)).mpl()

        .. seealso:: :py:func:`~discretisedfield.Field.k3d_vectors`

        """
        if not hasattr(self.mesh, 'info'):
            msg = 'The field must be sliced before it can be plotted.'
            raise ValueError(msg)

        sns.set(style='whitegrid')
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        planeaxis = dfu.raxesdict[self.mesh.info['planeaxis']]

        if self.dim > 1:
            # Vector field has both quiver and imshow plots.
            self.quiver(ax=ax, headwidth=5)
            scalar_field = getattr(self, planeaxis)
            coloredplot = scalar_field.imshow(ax=ax, filter_field=self.norm,
                                              cmap='cividis')
        else:
            # Scalar field has only imshow.
            coloredplot = self.imshow(ax=ax, filter_field=None)

        cbar = self.colorbar(ax, coloredplot)

        # Add labels.
        ax.set_xlabel(dfu.raxesdict[self.mesh.info['axis1']])
        ax.set_ylabel(dfu.raxesdict[self.mesh.info['axis2']])
        if self.dim > 1:
            cbar.ax.set_ylabel(planeaxis + ' component')

        ax.figure.tight_layout()

    def imshow(self, ax, filter_field=None, **kwargs):
        """Plots a scalar field plane using `matplotlib.pyplot.imshow`.

        Before the field can be plotted, it must be sliced with a
        plane (e.g. `field.plane(`y`)`) and field must be of dimension
        1 (scalar field). Otherwise, ValueError is raised. `imshow`
        adds the plot to `matplotlib.axes.Axes` passed via `ax`
        argument. If the scalar field plotted is extracted from a
        vector field, which has coordinates where the norm of the
        field is zero, the norm of that vector field can be passed
        using `norm_field` argument, so that pixels at those
        coordinates are not coloured. All other parameters accepted by
        `matplotlib.pyplot.imshow` can be passed. In order to use this
        function inside Jupyter notebook `%matplotlib inline` must be
        activated after `discretisedfield` is imported.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to which the scalar plot will be added.
        norm_field : discretisedfield.Field, optional
            A (scalar) norm field used for determining whether certain
            pixels should be coloured.

        Returns
        -------
        matplotlib.image.AxesImage object

        Raises
        ------
        ValueError
            If the field has not been sliced with a plane or its
            dimension is not 1.

        Example
        -------
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=1, value=2)
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
                    values[i] = np.array([np.nan])

        # "Unpack" values inside arrays.
        values =  list(zip(*values))

        extent = [self.mesh.region.pmin[self.mesh.info['axis1']],
                  self.mesh.region.pmax[self.mesh.info['axis1']],
                  self.mesh.region.pmin[self.mesh.info['axis2']],
                  self.mesh.region.pmax[self.mesh.info['axis2']]]
        n = (self.mesh.n[self.mesh.info['axis2']],
             self.mesh.n[self.mesh.info['axis1']])

        imax = ax.imshow(np.array(values).reshape(n), origin='lower',
                         extent=extent, **kwargs)
        return imax

    def quiver(self, ax=None, color_field=None, **kwargs):
        """Plots a vector field plane using `matplotlib.pyplot.quiver`.

        Before the field can be plotted, it must be sliced with a
        plane (e.g. `field.plane(`y`)`) and field must be of dimension
        3 (vector field). Otherwise, ValueError is raised. `quiver`
        adds the plot to `matplotlib.axes.Axes` passed via `ax`
        argument. If there are coordinates where the norm of the field
        is zero, vectors are not plotted at those coordinates. By
        default, plot is not coloured, but by passing a
        `discretisedfield.Field` object of dimension 1 as
        `color_field`, quiver plot will be coloured based on the
        values from the field. All other parameters accepted by
        `matplotlib.pyplot.quiver` can be passed. In order to use this
        function inside Jupyter notebook `%matplotlib inline` must be
        activated after `discretisedfield` is imported.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to which the quiver plot will be added.
        color_field : discretisedfield.Field, optional
            A (scalar) field used for determining the colour of the
            quiver plot.

        Returns
        -------
        matplotlib.quiver.Quiver object

        Raises
        ------
        ValueError
            If the field has not been sliced with a plane or its
            dimension is not 3.

        Example
        -------
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> field.plane(z=50).quiver(ax=ax, color_field=field.z)
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
            colors = list(zip(*colors))

        # "Unpack" values inside arrays.
        points, values = list(zip(*points)), list(zip(*values))

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
        """Adds a colorbar to the axes using `matplotlib.pyplot.colorbar`.

        Axes to which the colorbar should be added is passed via `ax`
        argument. If the colorbar axes are made before the method is
        called, they should be passed as `cax`. The plot to which the
        colorbar should correspond to is passed via `coloredplot`. All
        other parameters accepted by `matplotlib.pyplot.colorbar` can
        be passed. In order to use this function inside Jupyter
        notebook `%matplotlib inline` must be activated after
        `discretisedfield` is imported.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object to which the colorbar will be added.
        coloredplot : matplotlib.quiver.Quiver, matplotlib.image.AxesImage
            A plot to which the colorbar should correspond
        cax : matplotlib.axes.Axes, optional
            Colorbar axes.

        Returns
        -------
        matplotlib.colorbar.Colorbar

        Example
        -------
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> fig = plt.figure()
        >>> ax = fig.add_subplot(111)
        >>> coloredplot = field.plane(z=50).quiver(ax=ax, color_field=field.z)
        >>> field.colorbar(ax=ax, coloredplot=coloredplot)
        <matplotlib.colorbar.Colorbar object at ...>

        """
        if cax is None:
            divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.1)

        cbar = plt.colorbar(coloredplot, cax=cax, **kwargs)

        return cbar

    def k3d_nonzero(self, plot=None, multiplier=None,
                    color=dfu.cp_int_cat[0], **kwargs):
        """Plots the voxels where the value of a scalar field is nonzero.

        All mesh cells where the value of the field is not zero will
        be marked using the same color. Only scalar fields can be
        plotted. Otherwise, ValueError is raised. Different colour of
        voxels can be passed in the RGB format using `color`
        parameter. This function is often used to look at the defined
        sample in the finite difference mesh, by inspecting its norm
        (`field.norm.k3d_nonzero`). If `plot` is passed as a
        `k3d.plot.Plot`, plot is added to it. Otherwise, a new k3d
        plot is created. All arguments allowed in `k3d.voxels()` can
        be passed. This function is to be called in Jupyter notebook.

        Parameters
        ----------
        color : int/hex, optional
            Voxel color in hexadecimal format.
        plot : k3d.plot.Plot, optional
            If this argument is passed, plot is added to
            it. Otherwise, a new k3d plot is created.

        Example
        -------
        >>> import discretisedfield as df
        ...
        >>> p1 = (-50, -50, -50)
        >>> p2 = (50, 50, 50)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
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

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        dfu.voxels(plot_array, pmin=self.mesh.region.pmin,
                   pmax=self.mesh.region.pmax,
                   color_palette=color, multiplier=multiplier,
                   plot=plot, **kwargs)

    def k3d_voxels(self, plot=None, filter_field=None, multiplier=None,
                   cmap='viridis', n=256, **kwargs):
        """Plots the scalar field as a coloured `k3d.voxels()` plot.

        At all mesh cells, a voxel will be plotted anc coloured
        according to its value. If the scalar field plotted is
        extracted from a vector field, which has coordinates where the
        norm of the field is zero, the norm of that vector field can
        be passed using `norm_field` argument, so that voxels at those
        coordinates are not showed. Only scalar fields can be
        plotted. Otherwise, ValueError is raised. If `plot` is passed
        as a `k3d.plot.Plot`, plot is added to it. Otherwise, a new
        k3d plot is created. All arguments allowed in `k3d.voxels()`
        can be passed. This function is to be called in Jupyter
        notebook.

        Parameters
        ----------
        norm_field : discretisedfield.Field, optional
            A (scalar) norm field used for determining whether certain
            voxels should be plotted.
        plot : k3d.plot.Plot, optional
            If this argument is passed, plot is added to
            it. Otherwise, a new k3d plot is created.

        Example
        -------
        >>> import discretisedfield as df
        ...
        >>> p1 = (-50, -50, -50)
        >>> p2 = (50, 50, 50)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> def normfun(pos):
        ...     x, y, z = pos
        ...     if x**2 + y**2 < 30**2:
        ...         return 1
        ...     else:
        ...         return 0
        >>> field.norm = normfun
        >>> field.x.k3d_voxels(norm_field=field.norm)
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

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

        dfu.voxels(plot_array, pmin=self.mesh.region.pmin,
                   pmax=self.mesh.region.pmax, color_palette=color_palette,
                   multiplier=multiplier, plot=plot, **kwargs)

    def k3d_vectors(self, plot=None, color_field=None, points=True,
                    point_color=dfu.cp_int_cat[0], point_size=None,
                    multiplier=None, vector_multiplier=None,
                    cmap='viridis', n=256, **kwargs):
        """Plots the vector field as a `k3d.vectors()` plot.

        At all mesh cells, a vector will be plotted if its norm is not
        zero. Vectors can be coloured according to the values of the
        scalar field passed as `color_field`. Only vector fields can
        be plotted. Otherwise, ValueError is raised. Points at the
        discretisation cell centres can be added by setting
        `points=True`. If `plot` is passed as a `k3d.plot.Plot`, plot
        is added to it. Otherwise, a new k3d plot is created. All
        arguments allowed in `k3d.vectors()` can be passed. This
        function is to be called in Jupyter notebook.

        Parameters
        ----------
        color_field : discretisedfield.Field, optional
            A (scalar) field used for determining the colours of
            vectors.
        points : bool, optional
            If `True`, points will be added to the discretisation cell
            centres.
        plot : k3d.plot.Plot, optional
            If this argument is passed, plot is added to
            it. Otherwise, a new k3d plot is created.

        Example
        -------
        1. Plotting an entire vector field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50, -50, -50)
        >>> p2 = (50, 50, 50)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> field.k3d_vectors(color_field=field.x)
        Plot(...)

        2. Plotting the slice of a vector field.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50, -50, -50)
        >>> p2 = (50, 50, 50)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
        >>> field.plane('x').k3d_vectors(color_field=field.x)
        Plot(...)

        .. seealso:: :py:func:`~discretisedfield.Field.k3d_voxels`

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
                    color_values.append(color_field(coord)[0])
        coordinates, vectors = np.array(coordinates), np.array(vectors)

        if multiplier is None:
            multiplier = uu.si_max_multiplier(self.mesh.region.edges)

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
            colors = len(vectors)*([2*(dfu.cp_int_cat[1],)])  # uniform colour

        if plot is None:
            plot = k3d.plot()
            plot.display()

        dfu.vectors(coordinates, vectors, colors=colors,
                    multiplier=multiplier, vector_multiplier=vector_multiplier,
                    plot=plot, **kwargs)

        if points:
            if point_size is None:
                # If undefined, the size of the point is 1/4 of the smallest
                # cell dimension.
                point_size = np.divide(self.mesh.cell, multiplier).min() / 4

            dfu.points(coordinates, color=point_color, point_size=point_size,
                       multiplier=multiplier, plot=plot)
