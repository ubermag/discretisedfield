import collections
import functools
import math
import numbers
import re
import struct
import warnings

import h5py
import numpy as np
import pandas as pd
import ubermagutil.typesystem as ts
import xarray as xr
from vtkmodules.util import numpy_support as vns
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid
from vtkmodules.vtkIOLegacy import vtkRectilinearGridReader, vtkRectilinearGridWriter
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader, vtkXMLRectilinearGridWriter

import discretisedfield as df
import discretisedfield.plotting as dfp
import discretisedfield.util as dfu

from . import html
from .mesh import Mesh

# TODO: tutorials, line operations


@ts.typesystem(
    mesh=ts.Typed(expected_type=Mesh, const=True),
    dim=ts.Scalar(expected_type=int, positive=True, const=True),
    units=ts.Typed(expected_type=str, allow_none=True),
)
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

        Finite-difference rectangular mesh.

    dim : int

        Dimension of the field's value. For instance, if `dim=3` the field is a
        three-dimensional vector field and for `dim=1` the field is a scalar
        field.

    value : array_like, callable, dict, optional

        Please refer to ``discretisedfield.Field.value`` property. Defaults to
        0, meaning that if the value is not provided in the initialisation,
        "zero-field" will be defined.

    norm : numbers.Real, callable, optional

        Please refer to ``discretisedfield.Field.norm`` property. Defaults to
        ``None`` (``norm=None`` defines no norm).

    dtype : str, type, np.dtype, optional

        Data type of the underlying numpy array. If not specified the best data
        type is automatically determined if ``value`` is  array_like, for
        callable and dict ``value`` the numpy default (currently
        ``float64``) is used. Defaults to ``None``.

    units : str, optional

        Physical unit of the field.

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
    Field(...)
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
    Field(...)
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
    Field(...)
    >>> field.average
    (0.0, 0.0, 1.0)

    .. seealso:: :py:func:`~discretisedfield.Mesh`

    """

    def __init__(
        self, mesh, dim, value=0.0, norm=None, components=None, dtype=None, units=None
    ):
        self.mesh = mesh
        self.dim = dim
        self.dtype = dtype
        self.units = units

        self.value = value
        self.norm = norm

        self._components = None  # required in here for correct initialisation
        self.components = components

    @classmethod
    def coordinate_field(cls, mesh):
        """Create a field whose values are the mesh coordinates.

        This method can be used to create a 3d vector field with values equal to the
        coordinates of the cell midpoints. The result is equivalent to a mesh created
        with the following code:

        .. code-block::

            mesh = df.Mesh(...)
            df.Field(mesh, dim=3, value=lambda point: point)

        This class method should be preferred over the manual creation with a callable
        because it provides much better performance.

        Parameters
        ----------
        mesh : discretisedfield.Mesh

            Finite-difference rectangular mesh.

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
        >>> cfield = df.Field.coordinate_field(mesh)
        >>> cfield
        Field(...)

        2. Extract its value at position (0.5, 0.5, 0.5)

        >>> cfield((0.5, 0.5, 0.5))
        (0.5, 0.5, 0.5)

        3. Compare with manually created coordinate field

        >>> manually = df.Field(mesh, dim=3, value=lambda point: point)
        >>> cfield.allclose(manually)
        True

        """
        nx, ny, nz = mesh.n
        field = cls(mesh, dim=3)
        field.array[..., 0] = mesh.midpoints.x.reshape((nx, 1, 1))
        field.array[..., 1] = mesh.midpoints.y.reshape((1, ny, 1))
        field.array[..., 2] = mesh.midpoints.z.reshape((1, 1, nz))
        return field

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
        value : numbers.Real, array_like, callable, dict

            For scalar fields (``dim=1``) ``numbers.Real`` values are allowed.
            In the case of vector fields, ``array_like`` (list, tuple,
            numpy.ndarray) value with length equal to `dim` should be used.
            Finally, the value can also be a callable (e.g. Python function or
            another field), which for every coordinate in the mesh returns a
            valid value. If ``value=0``, all values in the field will be set to
            zero independent of the field dimension.

            If subregions are defined value can be initialised with a dict.
            Allowed keys are names of all subregions and ``default``. Items
            must be either ``numbers.Real`` for ``dim=1`` or ``array_like``
            for ``dim=3``. If subregion names are missing, the value of
            ``default`` is used if given. If parts of the region are not
            contained within one subregion ``default`` is used if specified,
            else these values are set to 0.

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
        0.0
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

        2. Field with subregions in mesh
        >>> import discretisedfield as df
        ...
        >>> p1 = (0,0,0)
        >>> p2 = (2,2,2)
        >>> cell = (1,1,1)
        >>> sub1 = df.Region(p1=(0,0,0), p2=(2,2,1))
        >>> sub2 = df.Region(p1=(0,0,1), p2=(2,2,2))
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell,\
                           subregions={'s1': sub1, 's2': sub2})
        >>> field = df.Field(mesh, dim=1, value={'s1': 1, 's2': 1})
        >>> (field.array == 1).all()
        True
        >>> field = df.Field(mesh, dim=1, value={'s1': 1})
        Traceback (most recent call last):
        ...
        KeyError: ...
        >>> field = df.Field(mesh, dim=1, value={'s1': 2, 'default': 1})
        >>> (field.array == 1).all()
        False
        >>> (field.array == 0).any()
        False
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions={'s': sub1})
        >>> field = df.Field(mesh, dim=1, value={'s': 1})
        Traceback (most recent call last):
        ...
        KeyError: ...
        >>> field = df.Field(mesh, dim=1, value={'default': 1})
        >>> (field.array == 1).all()
        True

        .. seealso:: :py:func:`~discretisedfield.Field.array`

        """
        value_array = _as_array(self._value, self.mesh, self.dim, dtype=self.dtype)
        if np.array_equal(self.array, value_array):
            return self._value
        else:
            return self.array

    @value.setter
    def value(self, val):
        self._value = val
        self.array = _as_array(val, self.mesh, self.dim, dtype=self.dtype)

    @property
    def components(self):
        """Vector components of the field."""
        return self._components

    @components.setter
    def components(self, components):
        if components is not None:
            if len(components) != self.dim:
                raise ValueError(f"Number of components does not match {self.dim=}.")
            if len(components) != len(set(components)):
                raise ValueError("Components must be unique.")
            for c in components:
                if hasattr(self, c):
                    # redefining component labels is okay.
                    if self._components is None or c not in self._components:
                        raise ValueError(
                            f"Component name {c} is already "
                            "used by a different method/property."
                        )
            self._components = list(components)
        else:
            if 2 <= self.dim <= 3:
                components = ["x", "y", "z"][: self.dim]
            elif self.dim > 3:
                warnings.warn(
                    "Component labels must be specified for "
                    f"{self.dim=} fields to get access to individual"
                    " vector components."
                )
            self._components = components

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
        self._array = _as_array(val, self.mesh, self.dim, dtype=self.dtype)

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

        return self.__class__(self.mesh, dim=1, value=res, units=self.units)

    @norm.setter
    def norm(self, val):
        if val is not None:
            if self.dim == 1:
                msg = f"Cannot set norm for field with dim={self.dim}."
                raise ValueError(msg)

            if not np.all(self.norm.array):
                msg = "Cannot normalise field with zero values."
                raise ValueError(msg)

            self.array /= self.norm.array  # normalise to 1
            self.array *= _as_array(val, self.mesh, dim=1, dtype=None)

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
        return self.__class__(
            self.mesh,
            dim=self.dim,
            value=0,
            components=self.components,
            units=self.units,
        )

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
            msg = f"Cannot compute orientation field for a dim={self.dim} field."
            raise ValueError(msg)

        orientation_array = np.divide(
            self.array, self.norm.array, where=(self.norm.array != 0)
        )
        return self.__class__(
            self.mesh, dim=self.dim, value=orientation_array, components=self.components
        )

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
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh, dim=1, value=1)
        >>> field
        Field(...)

        """
        return html.strip_tags(self._repr_html_())

    def _repr_html_(self):
        """Show HTML-based representation in Jupyter notebook."""
        return html.get_template("field").render(field=self)

    def __call__(self, point):
        r"""Sample the field value at ``point``.

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
        """Extract the component of the vector field.

        This method provides access to individual field components for fields
        with dimension > 1. Component labels are defined in the ``components``
        attribute. For dimension 2 and 3 default values ``'x'``, ``'y'``, and
        ``'z'`` are used if no custom component labels are provided. For fields
        with ``dim>3`` component labels must be specified manually to get
        access to individual vector components.

        Parameters
        ----------
        attr : str

            Vector field component defined in ``components``.

        Returns
        -------
        discretisedfield.Field

            Scalar field with vector field component values.

        Examples
        --------
        1. Accessing the default vector field components.

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

        2. Accessing custom vector field components.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1),
        ...                  components=['mx', 'my', 'mz'])
        >>> field.mx
        Field(...)
        >>> field.mx.average
        0.0
        >>> field.my
        Field(...)
        >>> field.my.average
        0.0
        >>> field.mz
        Field(...)
        >>> field.mz.average
        1.0
        >>> field.mz.dim
        1

        """
        if self.components is not None and attr in self.components:
            attr_array = self.array[..., self.components.index(attr), np.newaxis]
            return self.__class__(
                mesh=self.mesh, dim=1, value=attr_array, units=self.units
            )
        else:
            msg = f"Object has no attribute {attr}."
            raise AttributeError(msg)

    def __dir__(self):
        """Extension of the ``dir(self)`` list.

        Adds component labels to the ``dir(self)`` list. Similarly, adds or
        removes methods (``grad``, ``div``,...) depending on the dimension of
        the field.

        Returns
        -------
        list

            Avalilable attributes.

        """
        dirlist = dir(self.__class__)

        if self.components is not None:
            dirlist += self.components
        if self.dim == 1:
            need_removing = ["div", "curl", "orientation"]
        if self.dim == 2:
            need_removing = ["grad", "curl", "k3d"]
        if self.dim == 3:
            need_removing = ["grad"]

        for attr in need_removing:
            dirlist.remove(attr)

        return dirlist

    def __iter__(self):
        r"""Generator yielding coordinates and values of all mesh
        discretisation cells.

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
        elif (
            self.mesh == other.mesh
            and self.dim == other.dim
            and np.array_equal(self.array, other.array)
        ):
            return True
        else:
            return False

    # TODO The mesh comparison has no tolerance.
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
            msg = (
                "Cannot apply allclose method between "
                f"{type(self)=} and {type(other)=} objects."
            )
            raise TypeError(msg)

        if self.mesh == other.mesh and self.dim == other.dim:
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
        r"""Unary ``-`` operator.

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
            msg = f"Cannot apply ** operator on {self.dim=} field."
            raise ValueError(msg)
        if not isinstance(other, numbers.Real):
            msg = (
                f"Unsupported operand type(s) for **: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        return self.__class__(
            self.mesh,
            dim=1,
            value=np.power(self.array, other),
            components=self.components,
        )

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
                msg = f"Cannot apply operator + on {self.dim=} and {other.dim=} fields."
                raise ValueError(msg)
            if self.mesh != other.mesh:
                msg = "Cannot apply operator + on fields defined on different meshes."
                raise ValueError(msg)
        elif self.dim == 1 and isinstance(other, numbers.Complex):
            return self + self.__class__(self.mesh, dim=self.dim, value=other)
        elif self.dim == 3 and isinstance(other, (tuple, list, np.ndarray)):
            return self + self.__class__(self.mesh, dim=self.dim, value=other)
        else:
            msg = (
                f"Unsupported operand type(s) for +: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        return self.__class__(
            self.mesh,
            dim=self.dim,
            value=self.array + other.array,
            components=self.components,
        )

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
                msg = f"Cannot apply operator * on {self.dim=} and {other.dim=} fields."
                raise ValueError(msg)
            if self.mesh != other.mesh:
                msg = "Cannot apply operator * on fields defined on different meshes."
                raise ValueError(msg)
        elif isinstance(other, numbers.Complex):
            return self * self.__class__(self.mesh, dim=1, value=other)
        elif self.dim == 1 and isinstance(other, (tuple, list, np.ndarray)):
            return self * self.__class__(
                self.mesh, dim=np.array(other).shape[-1], value=other
            )
        elif isinstance(other, df.DValue):
            return self * other(self)
        else:
            msg = (
                f"Unsupported operand type(s) for *: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        res_array = np.multiply(self.array, other.array)
        components = self.components if self.dim == res_array.shape[-1] else None
        return self.__class__(
            self.mesh,
            dim=res_array.shape[-1],
            value=res_array,
            components=components,
        )

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
        return self * other ** (-1)

    def __rtruediv__(self, other):
        return self ** (-1) * other

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
                msg = "Cannot apply operator @ on fields defined on different meshes."
                raise ValueError(msg)
            if self.dim != 3 or other.dim != 3:
                msg = f"Cannot apply operator @ on {self.dim=} and {other.dim=} fields."
                raise ValueError(msg)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self @ self.__class__(
                self.mesh, dim=3, value=other, components=self.components
            )
        elif isinstance(other, df.DValue):
            return self @ other(self)
        else:
            msg = (
                f"Unsupported operand type(s) for @: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        res_array = np.einsum("ijkl,ijkl->ijk", self.array, other.array)
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
                msg = "Cannot apply operator & on fields defined on different meshes."
                raise ValueError(msg)
            if self.dim != 3 or other.dim != 3:
                msg = f"Cannot apply operator & on {self.dim=} and {other.dim=} fields."
                raise ValueError(msg)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self & self.__class__(
                self.mesh, dim=3, value=other, components=self.components
            )
        else:
            msg = (
                f"Unsupported operand type(s) for &: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        res_array = np.cross(self.array, other.array)
        return self.__class__(
            self.mesh,
            dim=3,
            value=res_array,
            components=self.components,
        )

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
                msg = "Cannot apply operator << on fields defined on different meshes."
                raise ValueError(msg)
        elif isinstance(other, numbers.Complex):
            return self << self.__class__(self.mesh, dim=1, value=other)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self << self.__class__(self.mesh, dim=len(other), value=other)
        else:
            msg = (
                f"Unsupported operand type(s) for <<: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        array_list = [self.array[..., i] for i in range(self.dim)]
        array_list += [other.array[..., i] for i in range(other.dim)]

        if self.components is None or other.components is None:
            components = None
        else:
            components = self.components + other.components
            if len(components) != len(set(components)):
                # Component name duplicated; could happen e.g. for lshift with
                # a number -> choose labels automatically
                components = None

        return self.__class__(
            self.mesh,
            dim=len(array_list),
            value=np.stack(array_list, axis=3),
            components=components,
        )

    def __rlshift__(self, other):
        if isinstance(other, numbers.Complex):
            return self.__class__(self.mesh, dim=1, value=other) << self
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self.__class__(self.mesh, dim=len(other), value=other) << self
        else:
            msg = (
                f"Unsupported operand type(s) for <<: {type(self)=} and {type(other)=}."
            )
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

        padded_array = np.pad(self.array, padding_sequence, mode=mode, **kwargs)
        padded_mesh = self.mesh.pad(pad_width)

        return self.__class__(
            padded_mesh,
            dim=self.dim,
            value=padded_array,
            components=self.components,
            units=self.units,
        )

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
            padding_mode = "wrap"
        elif self.mesh.bc == "neumann":
            pad_width = {dfu.raxesdict[direction]: (1, 1)}
            padding_mode = "edge"
        else:  # No BC - no padding
            pad_width = {}
            padding_mode = "constant"

        padded_array = self.pad(pad_width, mode=padding_mode).array

        if n not in (1, 2):
            msg = f"Derivative of the n={n} order is not implemented."
            raise NotImplementedError(msg)

        elif n == 1:
            if self.dim == 1:
                derivative_array = np.gradient(
                    padded_array[..., 0], self.mesh.cell[direction], axis=direction
                )[..., np.newaxis]
            else:
                derivative_array = np.gradient(
                    padded_array, self.mesh.cell[direction], axis=direction
                )

        elif n == 2:
            derivative_array = np.zeros_like(padded_array)
            for i in range(padded_array.shape[direction]):
                if i == 0:
                    i1, i2, i3 = i + 2, i + 1, i
                elif i == padded_array.shape[direction] - 1:
                    i1, i2, i3 = i, i - 1, i - 2
                else:
                    i1, i2, i3 = i + 1, i, i - 1
                index1 = dfu.assemble_index(slice(None), 4, {direction: i1})
                index2 = dfu.assemble_index(slice(None), 4, {direction: i2})
                index3 = dfu.assemble_index(slice(None), 4, {direction: i3})
                index = dfu.assemble_index(slice(None), 4, {direction: i})
                derivative_array[index] = (
                    padded_array[index1]
                    - 2 * padded_array[index2]
                    + padded_array[index3]
                ) / self.mesh.cell[direction] ** 2

        # Remove padded values (if any).
        if derivative_array.shape != self.array.shape:
            derivative_array = np.delete(
                derivative_array, (0, self.mesh.n[direction] + 1), axis=direction
            )

        return self.__class__(
            self.mesh, dim=self.dim, value=derivative_array, components=self.components
        )

    @property
    def grad(self):
        r"""Gradient.

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
            msg = f"Cannot compute gradient for dim={self.dim} field."
            raise ValueError(msg)

        return self.derivative("x") << self.derivative("y") << self.derivative("z")

    @property
    def div(self):
        r"""Divergence.

        This method computes the divergence of a vector (``dim=2`` or
        ``dim=3``) field and returns a scalar (``dim=1``) field as a result.

        .. math::

            \\nabla\\cdot\\mathbf{v} = \\sum_i\\frac{\\partial v_{i}}
            {\\partial i}

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

            If the dimension of the field is 1.

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
        if self.dim not in [2, 3]:
            msg = f"Cannot compute divergence for dim={self.dim} field."
            raise ValueError(msg)

        return sum(
            [
                getattr(self, self.components[i]).derivative(dfu.raxesdict[i])
                for i in range(self.dim)
            ]
        )

    @property
    def curl(self):
        r"""Curl.

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
            msg = f"Cannot compute curl for dim={self.dim} field."
            raise ValueError(msg)

        x, y, z = self.components
        curl_x = getattr(self, z).derivative("y") - getattr(self, y).derivative("z")
        curl_y = getattr(self, x).derivative("z") - getattr(self, z).derivative("x")
        curl_z = getattr(self, y).derivative("x") - getattr(self, x).derivative("y")

        return curl_x << curl_y << curl_z

    @property
    def laplace(self):
        r"""Laplace operator.

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
        if self.dim not in [1, 3]:
            raise ValueError(f"Cannot compute laplace for dim={self.dim} field.")
        if self.dim == 1:
            return (
                self.derivative("x", n=2)
                + self.derivative("y", n=2)
                + self.derivative("z", n=2)
            )
        else:
            x, y, z = self.components
            return (
                getattr(self, x).laplace
                << getattr(self, y).laplace
                << getattr(self, z).laplace
            )

    def integral(self, direction="xyz", improper=False):
        r"""Integral.

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
            msg = "Cannot compute improper integral along multiple directions."
            raise ValueError(msg)

        mesh = self.mesh

        if not improper:
            for i in direction:
                mesh = mesh.plane(i)
            axes = [dfu.axesdict[i] for i in direction]
            res_array = np.sum(self.array, axis=tuple(axes), keepdims=True)
        else:
            res_array = np.cumsum(self.array, axis=dfu.axesdict[direction])

        res = self.__class__(
            mesh, dim=self.dim, value=res_array, components=self.components
        )

        if len(direction) == 3:
            return dfu.array2tuple(res.array.squeeze())
        else:
            return res

    def line(self, p1, p2, n=100):
        r"""Sample the field along the line.

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
        -------
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
        if n is not None:
            value = self
        else:
            p_axis = plane_mesh.attributes["planeaxis"]
            plane_idx = self.mesh.point2index(plane_mesh.region.centre)[p_axis]
            slices = tuple(
                slice(plane_idx, plane_idx + 1) if i == p_axis else slice(0, axis_len)
                for i, axis_len in enumerate(self.array.shape)
            )
            value = self.array[slices]
        return self.__class__(
            plane_mesh,
            dim=self.dim,
            value=value,
            components=self.components,
            units=self.units,
        )

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
        return self.__class__(
            submesh,
            dim=self.dim,
            value=self.array[tuple(slices)],
            components=self.components,
            units=self.units,
        )

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
        -------
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
        r"""In-plane angle of the vector field.

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
        if not self.mesh.attributes["isplane"]:
            msg = "The field must be sliced before angle can be computed."
            raise ValueError(msg)

        angle_array = np.arctan2(
            self.array[..., self.mesh.attributes["axis2"]],
            self.array[..., self.mesh.attributes["axis1"]],
        )

        # Place all values in [0, 2pi] range
        angle_array[angle_array < 0] += 2 * np.pi

        return self.__class__(self.mesh, dim=1, value=angle_array[..., np.newaxis])

    def write(self, filename, representation="bin8", extend_scalar=False):
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

            Only supported for OVF and VTK files. In the case of OVF files
            (``.ovf``, ``.omf``, or ``.ohf``) the representation can be
            ``'bin4'``, ``'bin8'``, or ``'txt'``. For VTK files (``.vtk``) the
            representation can be ``bin``, ``xml``, or ``txt``. Defaults to
            ``'bin8'`` (interpreted as ``bin`` for VTK files).

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
        if filename.endswith((".omf", ".ovf", ".ohf")):
            self._writeovf(
                filename, representation=representation, extend_scalar=extend_scalar
            )
        elif filename.endswith((".hdf5", ".h5")):
            self._writehdf5(filename)
        elif filename.endswith(".vtk"):
            self._writevtk(filename, representation=representation)
        else:
            msg = (
                f'Writing file with extension {filename.split(".")[-1]} not supported.'
            )
            raise ValueError(msg)

    def _writeovf(self, filename, representation="bin8", extend_scalar=False):
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
            ``'bin8'``.

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
        write_dim = 3 if extend_scalar and self.dim == 1 else self.dim
        valueunits = " ".join([str(self.units) if self.units else "None"] * write_dim)
        if write_dim == 1:
            valuelabels = "field_x"
        elif extend_scalar:
            valuelabels = " ".join(["field_x"] * write_dim)
        else:
            valuelabels = " ".join(f"field_{c}" for c in self.components)

        if representation == "bin4":
            repr_string = "Binary 4"
        elif representation == "bin8":
            repr_string = "Binary 8"
        elif representation == "txt":
            repr_string = "Text"
        else:
            raise ValueError(f"Unknown {representation=}.")

        bheader = "".join(
            f"# {line}\n"
            for line in [
                "OOMMF OVF 2.0",
                "",
                "Segment count: 1",
                "",
                "Begin: Segment",
                "Begin: Header",
                "",
                "Title: Field",
                "Desc: File generated by Field class",
                f'meshunit: {self.mesh.attributes["unit"]}',
                "meshtype: rectangular",
                f"xbase: {self.mesh.region.pmin[0] + self.mesh.cell[0]/2}",
                f"ybase: {self.mesh.region.pmin[1] + self.mesh.cell[1]/2}",
                f"zbase: {self.mesh.region.pmin[2] + self.mesh.cell[2]/2}",
                f"xnodes: {self.mesh.n[0]}",
                f"ynodes: {self.mesh.n[1]}",
                f"znodes: {self.mesh.n[2]}",
                f"xstepsize: {self.mesh.cell[0]}",
                f"ystepsize: {self.mesh.cell[1]}",
                f"zstepsize: {self.mesh.cell[2]}",
                f"xmin: {self.mesh.region.pmin[0]}",
                f"ymin: {self.mesh.region.pmin[1]}",
                f"zmin: {self.mesh.region.pmin[2]}",
                f"xmax: {self.mesh.region.pmax[0]}",
                f"ymax: {self.mesh.region.pmax[1]}",
                f"zmax: {self.mesh.region.pmax[2]}",
                f"valuedim: {write_dim}",
                f"valuelabels: {valuelabels}",
                f"valueunits: {valueunits}",
                "",
                "End: Header",
                "",
                f"Begin: Data {repr_string}",
            ]
        ).encode("utf-8")

        bfooter = "".join(
            f"# {line}\n" for line in [f"End: Data {repr_string}", "End: Segment"]
        ).encode("utf-8")

        reordered = self.array.transpose((2, 1, 0, 3))  # ovf ordering

        bin_rep = {"bin4": ("<f", 1234567.0), "bin8": ("<d", 123456789012345.0)}

        with open(filename, "wb") as f:
            f.write(bheader)

            if representation in bin_rep:
                # Add the binary checksum.
                f.write(struct.pack(*bin_rep[representation]))

                if extend_scalar:
                    # remove scalar vector dimension
                    reordered = reordered.reshape(list(reversed(self.mesh.n)))
                    reordered = np.stack(
                        (reordered, np.zeros_like(reordered), np.zeros_like(reordered)),
                        axis=-1,
                    )

                # ndarray.tofile seems to be ~20% slower
                f.write(
                    np.asarray(reordered, dtype=bin_rep[representation][0]).tobytes()
                )
                f.write(b"\n")
            else:
                data = pd.DataFrame(reordered.reshape((-1, self.dim)))
                data.insert(loc=0, column="leading_space", value="")

                if extend_scalar:
                    data.insert(loc=2, column="y", value=0.0)
                    data.insert(loc=3, column="z", value=0.0)

                data.to_csv(f, sep=" ", header=False, index=False)

            f.write(bfooter)

    def to_vtk(self):
        """Convert field to vtk rectilinear grid.

        This method convers at `discretisedfield.Field` into a
        `vtk.vtkRectilinearGrid`. The field data (``field.array``) is stored as
        ``CELL_DATA`` of the ``RECTILINEAR_GRID``. Scalar fields (``dim=1``)
        contain one VTK array called ``field``. Vector fields (``dim>1``)
        contain one VTK array called ``field`` containing vector data and
        scalar VTK arrays for each field component (called
        ``<component-name>-component``).

        Returns
        -------
        vtk.vtkRectilinearGrid

            VTK representation of the field.

        Raises
        ------
        AttributeError

            If the field has ``dim>1`` and component labels are missing.

        Examples
        --------
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(1, 1, 1))
        >>> f = df.Field(mesh, dim=3, value=(0, 0, 1))
        >>> f_vtk = f.to_vtk()
        >>> print(f_vtk)
        vtkRectilinearGrid (...)
        ...
        >>> f_vtk.GetNumberOfCells()
        1000

        """
        if self.dim > 1 and self.components is None:
            raise AttributeError(
                "Field components must be assigned before converting to vtk."
            )
        rgrid = vtkRectilinearGrid()
        rgrid.SetDimensions(*(n + 1 for n in self.mesh.n))

        rgrid.SetXCoordinates(
            vns.numpy_to_vtk(np.fromiter(self.mesh.vertices.x, float))
        )
        rgrid.SetYCoordinates(
            vns.numpy_to_vtk(np.fromiter(self.mesh.vertices.y, float))
        )
        rgrid.SetZCoordinates(
            vns.numpy_to_vtk(np.fromiter(self.mesh.vertices.z, float))
        )

        cell_data = rgrid.GetCellData()
        field_norm = vns.numpy_to_vtk(
            self.norm.array.transpose((2, 1, 0, 3)).reshape(-1)
        )
        field_norm.SetName("norm")
        cell_data.AddArray(field_norm)
        if self.dim > 1:
            # For some visualisation packages it is an advantage to have direct
            # access to the individual field components, e.g. for colouring.
            for comp in self.components:
                component_array = vns.numpy_to_vtk(
                    getattr(self, comp).array.transpose((2, 1, 0, 3)).reshape((-1))
                )
                component_array.SetName(f"{comp}-component")
                cell_data.AddArray(component_array)
        field_array = vns.numpy_to_vtk(
            self.array.transpose((2, 1, 0, 3)).reshape((-1, self.dim))
        )
        field_array.SetName("field")
        cell_data.AddArray(field_array)

        if self.dim == 3:
            cell_data.SetActiveVectors("field")
        elif self.dim == 1:
            cell_data.SetActiveScalars("field")
        return rgrid

    def _writevtk(self, filename, representation="bin"):
        """Write the field to a VTK file.

        The data is saved as a ``RECTILINEAR_GRID`` dataset. Scalar field
        (``dim=1``) is saved as ``SCALARS``. On the other hand, vector field
        (``dim=3``) is saved as both ``VECTORS`` as well as ``SCALARS`` for all
        three components to enable easy colouring of vectors in some
        visualisation packages. The data is stored as ``CELL_DATA``.

        The saved VTK file can be opened with `Paraview
        <https://www.paraview.org/>`_ or `Mayavi
        <https://docs.enthought.com/mayavi/mayavi/>`_. To show contour lines in
        Paraview one has to first convert Cell Data to Point Data using a
        filter.

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
        if representation == "xml":
            writer = vtkXMLRectilinearGridWriter()
        elif representation in ["bin", "bin8", "txt"]:
            # Allow bin8 for convenience as this is the default for omf.
            # This does not affect the actual datatype used in vtk files.
            writer = vtkRectilinearGridWriter()
        else:
            raise ValueError(f"Unknown {representation=}.")

        if representation == "txt":
            writer.SetFileTypeToASCII()
        elif representation in ["bin", "bin8"]:
            writer.SetFileTypeToBinary()
        # xml has no distinction between ascii and binary

        writer.SetFileName(filename)
        writer.SetInputData(self.to_vtk())
        writer.Write()

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
        with h5py.File(filename, "w") as f:
            # Set up the file structure
            gfield = f.create_group("field")
            gmesh = gfield.create_group("mesh")
            gregion = gmesh.create_group("region")

            # Save everything as datasets
            gregion.create_dataset("p1", data=self.mesh.region.p1)
            gregion.create_dataset("p2", data=self.mesh.region.p2)
            gmesh.create_dataset("n", dtype="i4", data=self.mesh.n)
            gfield.create_dataset("dim", dtype="i4", data=self.dim)
            gfield.create_dataset("array", data=self.array)

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
        Field(...)

        2. Read a field from the VTK file.

        >>> filename = os.path.join(dirname, 'vtk-file.vtk')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(...)

        3. Read a field from the HDF5 file.

        >>> filename = os.path.join(dirname, 'hdf5-file.hdf5')
        >>> field = df.Field.fromfile(filename)
        >>> field
        Field(...)

        .. seealso:: :py:func:`~discretisedfield.Field._fromovf`
        .. seealso:: :py:func:`~discretisedfield.Field._fromhdf5`
        .. seealso:: :py:func:`~discretisedfield.Field._fromhdf5`
        .. seealso:: :py:func:`~discretisedfield.Field.write`

        """
        if filename.endswith((".omf", ".ovf", ".ohf", ".oef")):
            return cls._fromovf(filename)
        elif filename.endswith(".vtk"):
            return cls._fromvtk(filename)
        elif filename.endswith((".hdf5", ".h5")):
            return cls._fromhdf5(filename)
        else:
            msg = (
                f'Reading file with extension {filename.split(".")[-1]} not supported.'
            )
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
        Field(...)

        .. seealso:: :py:func:`~discretisedfield.Field._writeovf`

        """
        header = {}
        with open(filename, "rb") as f:
            # >>> READ HEADER <<<
            ovf_v2 = b"2.0" in next(f)
            for line in f:
                line = line.decode("utf-8")
                if line.startswith("# Begin: Data"):
                    mode = line.split()[3]
                    if mode == "Binary":
                        nbytes = int(line.split()[-1])
                    break
                information = line[1:].split(":")  # remove leading `#`
                if len(information) > 1:
                    key = information[0].strip()
                    header[key] = information[1].strip()

            # valuedim is fixed to 3 and not in the header for OVF 1.0
            header["valuedim"] = int(header["valuedim"]) if ovf_v2 else 3

            # >>> MESH <<<
            p1 = (float(header[f"{key}min"]) for key in "xyz")
            p2 = (float(header[f"{key}max"]) for key in "xyz")
            cell = (float(header[f"{key}stepsize"]) for key in "xyz")
            mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)

            nodes = math.prod(int(header[f"{key}nodes"]) for key in "xyz")

            # >>> READ DATA <<<
            if mode == "Binary":
                # OVF2 uses little-endian and OVF1 uses big-endian
                format = f'{"<" if ovf_v2 else ">"}{"d" if nbytes == 8 else "f"}'

                test_value = struct.unpack(format, f.read(nbytes))[0]
                check = {4: 1234567.0, 8: 123456789012345.0}
                if nbytes not in (4, 8) or test_value != check[nbytes]:
                    raise ValueError(  # pragma: no cover
                        f"Cannot read file {filename}. The file seems to be in"
                        f" binary format ({nbytes} bytes) but the check value"
                        f" is not correct: Expected {check[nbytes]}, got"
                        f" {test_value}."
                    )

                array = np.fromfile(
                    f, count=int(nodes * header["valuedim"]), dtype=format
                ).reshape((-1, header["valuedim"]))
            else:
                array = pd.read_csv(
                    f,
                    sep=" ",
                    header=None,
                    dtype=np.float64,
                    skipinitialspace=True,
                    nrows=nodes,
                    comment="#",
                ).to_numpy()

        r_tuple = (*reversed(mesh.n), header["valuedim"])
        t_tuple = (2, 1, 0, 3)

        try:
            # multi-word components are surrounded by {}
            components = re.findall(r"(\w+|{[\w ]+})", header["valuelabels"])
        except KeyError:
            components = None
        else:

            def convert(comp):
                # Magnetization_x -> x
                # {Total field_x} -> x
                # {Total energy density} -> Total_energy_density
                comp = comp.split("_")[1] if "_" in comp else comp
                comp = comp.replace("{", "").replace("}", "")
                return "_".join(comp.split())

            components = [convert(c) for c in components]
            if len(components) != len(set(components)):  # components are not unique
                components = None

        try:
            unit_list = header["valueunits"].split()
        except KeyError:
            units = None
        else:
            if len(unit_list) == 0:
                units = None  # no unit in the file
            elif len(set(unit_list)) != 1:
                warnings.warn(
                    f"File {filename} contains multiple units for the individual"
                    f" components: {unit_list=}. This is not supported by"
                    " discretisedfield. Units are set to None."
                )
                units = None
            else:
                units = unit_list[0]

        return cls(
            mesh,
            dim=header["valuedim"],
            value=array.reshape(r_tuple).transpose(t_tuple),
            components=components,
            units=units,
        )

    @classmethod
    def _fromvtk(cls, filename):
        """Read the field from a VTK file.

        This method reads the field from a VTK file defined on RECTILINEAR GRID
        written by ``discretisedfield._writevtk``. It expects the data do be
        specified as cell data and one (vector) field with the name ``field``.
        A vector field should also contain data for the individual components.
        The individual component names are used as ``components`` for the new
        field. They must appear in the form ``<componentname>-component``.

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
        Field(...)

        .. seealso:: :py:func:`~discretisedfield.Field._writevtk`

        """
        with open(filename, "rb") as f:
            xml = "xml" in f.readline().decode("utf8")
        if xml:
            reader = vtkXMLRectilinearGridReader()
        else:
            reader = vtkRectilinearGridReader()
            reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
        reader.SetFileName(filename)
        reader.Update()

        output = reader.GetOutput()
        p1 = output.GetBounds()[::2]
        p2 = output.GetBounds()[1::2]
        n = [i - 1 for i in output.GetDimensions()]

        cell_data = output.GetCellData()

        if cell_data.GetNumberOfArrays() == 0:
            # Old writing routine did write to points instead of cells.
            return dfu.fromvtk_legacy(filename)

        components = []
        for i in range(cell_data.GetNumberOfArrays()):
            name = cell_data.GetArrayName(i)
            if name == "field":
                field_idx = i
            elif name.endswith("-component"):
                components.append(name[: -len("-component")])
        array = cell_data.GetArray(field_idx)
        dim = array.GetNumberOfComponents()

        if len(components) != dim:
            components = None

        value = vns.vtk_to_numpy(array).reshape(*reversed(n), dim)
        value = value.transpose((2, 1, 0, 3))

        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        return cls(mesh, dim=dim, value=value, components=components)

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
        Field(...)

        .. seealso:: :py:func:`~discretisedfield.Field._writehdf5`

        """
        with h5py.File(filename, "r") as f:
            # Read data from the file.
            p1 = f["field/mesh/region/p1"]
            p2 = f["field/mesh/region/p2"]
            n = np.array(f["field/mesh/n"]).tolist()
            dim = np.array(f["field/dim"]).tolist()
            array = f["field/array"]

            # Create field.
            mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
            return cls(mesh, dim=dim, value=array[:])

    @property
    def mpl(self):
        """Plot interface, matplotlib based.

        This property provides access to the different plotting methods. It is
        also callable to quickly generate plots. For more details and the
        available methods refer to the documentation linked below.

        .. seealso::

            :py:func:`~discretisedfield.plotting.Mpl.__call__`
            :py:func:`~discretisedfield.plotting.Mpl.scalar`
            :py:func:`~discretisedfield.plotting.Mpl.vector`
            :py:func:`~discretisedfield.plotting.Mpl.lightness`
            :py:func:`~discretisedfield.plotting.Mpl.contour`

        Examples
        --------
        .. plot:: :context: close-figs

            1. Visualising the field using ``matplotlib``.

            >>> import discretisedfield as df
            ...
            >>> p1 = (0, 0, 0)
            >>> p2 = (100, 100, 100)
            >>> n = (10, 10, 10)
            >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
            >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
            >>> field.plane(z=50, n=(5, 5)).mpl()

        """
        return dfp.MplField(self)

    @property
    def k3d(self):
        """Plot interface, k3d based."""
        return dfp.K3dField(self)

    @property
    def fftn(self):
        """Fourier transform.

        Computes 3D FFT for "normal" fields, 2D FFT if the field is sliced.

        Returns
        -------
        discretisedfield.Field
        """
        mesh = self._fft_mesh()

        values = []
        for idx in range(self.dim):
            ft = np.fft.fftshift(np.fft.fftn(self.array[..., idx].squeeze()))
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            dim=len(values),
            value=np.stack(values, axis=3),
            components=self.components,
        )

    @property
    def ifftn(self):
        """Inverse Fourier transform.

        Returns
        -------
        discretisedfield.Field
        """
        mesh = self.mesh.attributes["realspace_mesh"]
        if self.mesh.attributes["isplane"] and not mesh.attributes["isplane"]:
            mesh = mesh.plane(dfu.raxesdict[self.mesh.attributes["planeaxis"]])

        values = []
        for idx in range(self.dim):
            ft = np.fft.ifftn(np.fft.ifftshift(self.array[..., idx].squeeze()))
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            dim=len(values),
            value=np.stack(values, axis=3),
            components=self.components,
        )

    @property
    def rfftn(self):
        """Real Fourier transform.

        Returns
        -------
        discretisedfield.Field
        """
        mesh = self._fft_mesh(rfft=True)

        values = []
        for idx in range(self.dim):
            array = self.array[..., idx].squeeze()
            # no shifting for the last axis
            ft = np.fft.fftshift(np.fft.rfftn(array), axes=range(len(array.shape) - 1))
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            dim=len(values),
            value=np.stack(values, axis=3),
            components=self.components,
        )

    @property
    def irfftn(self):
        """Inverse real Fourier transform.

        Returns
        -------
        discretisedfield.Field
        """
        mesh = self.mesh.attributes["realspace_mesh"]
        if self.mesh.attributes["isplane"] and not mesh.attributes["isplane"]:
            mesh = mesh.plane(dfu.raxesdict[self.mesh.attributes["planeaxis"]])

        values = []
        for idx in range(self.dim):
            array = self.array[..., idx].squeeze()
            ft = np.fft.irfftn(
                np.fft.ifftshift(array, axes=range(len(array.shape) - 1)),
                s=[i for i in mesh.n if i > 1],
            )
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            dim=len(values),
            value=np.stack(values, axis=3),
            components=self.components,
        )

    def _fft_mesh(self, rfft=False):
        """FFT can be one of fftfreq, rfftfreq."""
        p1 = []
        p2 = []
        n = []
        for i in range(3):
            if self.mesh.n[i] == 1:
                p1.append(0)
                p2.append(1 / self.mesh.cell[i])
                n.append(1)
            else:
                freqs = np.fft.fftshift(
                    np.fft.fftfreq(self.mesh.n[i], self.mesh.cell[i])
                )
                # Shift the region boundaries to get the correct coordinates of
                # mesh cells.
                dfreq = (freqs[1] - freqs[0]) / 2
                p1.append(min(freqs) - dfreq)
                p2.append(max(freqs) + dfreq)
                n.append(len(freqs))

        if rfft:
            # last frequency is different for rfft
            for i in [2, 1, 0]:
                if self.mesh.n[i] > 1:
                    freqs = np.fft.rfftfreq(self.mesh.n[i], self.mesh.cell[i])
                    dfreq = (freqs[1] - freqs[0]) / 2
                    p1[i] = min(freqs) - dfreq
                    p2[i] = max(freqs) + dfreq
                    n[i] = len(freqs)
                    break

        # TODO: Using PlaneMesh will simplify the code a lot here.
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        if self.mesh.attributes["isplane"]:
            mesh = mesh.plane(dfu.raxesdict[self.mesh.attributes["planeaxis"]])

        mesh.attributes["realspace_mesh"] = self.mesh
        mesh.attributes["fourierspace"] = True
        mesh.attributes["unit"] = rf'({mesh.attributes["unit"]})$^{{-1}}$'
        return mesh

    @property
    def real(self):
        """Real part of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.dim,
            value=self.array.real,
            components=self.components,
            units=self.units,
        )

    @property
    def imag(self):
        """Imaginary part of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.dim,
            value=self.array.imag,
            components=self.components,
            units=self.units,
        )

    @property
    def phase(self):
        """Phase of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.dim,
            value=np.angle(self.array),
            components=self.components,
        )

    @property
    def conjugate(self):
        """Complex conjugate of complex field."""
        return self.__class__(
            self.mesh,
            dim=self.dim,
            value=self.array.conjugate(),
            components=self.components,
            units=self.units,
        )

    # TODO check and write tests
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Field class support for numpy ``ufuncs``."""
        # See reference implementation at:
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
        for x in inputs:
            if not isinstance(x, (Field, np.ndarray, numbers.Number)):
                return NotImplemented
        out = kwargs.get("out", ())
        if out:
            for x in out:
                if not isinstance(x, Field):
                    return NotImplemented

        mesh = [x.mesh for x in inputs if isinstance(x, Field)]
        inputs = tuple(x.array if isinstance(x, Field) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.array for x in out)

        result = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(result, tuple):
            if len(result) != len(mesh):
                raise ValueError("wrong number of Field objects")
            return tuple(
                self.__class__(
                    m,
                    dim=x.shape[-1],
                    value=x,
                    components=self.components,
                )
                for x, m in zip(result, mesh)
            )
        elif method == "at":
            return None
        else:
            return self.__class__(
                mesh[0],
                dim=result.shape[-1],
                value=result,
                components=self.components,
            )

    def to_xarray(self, name="field", units=None):
        """Field value as ``xarray.DataArray``.

        The function returns an ``xarray.DataArray`` with dimensions ``x``,
        ``y``, ``z``, and ``comp`` (``only if field.dim > 1``). The coordinates
        of the geometric dimensions are derived from ``self.mesh.midpoints``,
        and for vector field components from ``self.components``. Addtionally,
        the values of ``self.mesh.cell``, ``self.mesh.region.p1``, and
        ``self.mesh.region.p2`` are stored as ``cell``, ``p1``, and ``p2``
        attributes of the DataArray. The ``units`` attribute of geometric
        dimensions is set to ``self.mesh.attributes['unit']``.

        The name and units of the field ``DataArray`` can be set by passing
        ``name`` and ``units``. If the type of value passed to any of the two
        arguments is not ``str``, then a ``TypeError`` is raised.

        Parameters
        ----------
        name : str, optional

            String to set name of the field ``DataArray``.

        units : str, optional

            String to set units of the field ``DataArray``.

        Returns
        -------
        xarray.DataArray

            Field values DataArray.

        Raises
        ------
        TypeError

            If either ``name`` or ``units`` argument is not a string.

        Examples
        --------
        1. Create a field

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh=mesh, dim=3, value=(1, 0, 0), norm=1.)
        ...
        >>> field
        Field(...)

        2. Create `xarray.DataArray` from field

        >>> xa = field.to_xarray()
        >>> xa
        <xarray.DataArray 'field' (x: 10, y: 10, z: 10, comp: 3)>
        ...

        3. Select values of `x` component

        >>> xa.sel(comp='x')
        <xarray.DataArray 'field' (x: 10, y: 10, z: 10)>
        ...

        """
        if not isinstance(name, str):
            msg = "Name argument must be a string."
            raise TypeError(msg)

        if units is not None and not isinstance(units, str):
            msg = "Units argument must be a string."
            raise TypeError(msg)

        axes = ["x", "y", "z"]

        data_array_coords = {axis: getattr(self.mesh.midpoints, axis) for axis in axes}

        if "unit" in self.mesh.attributes:
            geo_units_dict = dict.fromkeys(axes, self.mesh.attributes["unit"])
        else:
            geo_units_dict = dict.fromkeys(axes, "m")

        if self.dim > 1:
            data_array_dims = axes + ["comp"]
            if self.components is not None:
                data_array_coords["comp"] = self.components
            field_array = self.array
        else:
            data_array_dims = axes
            field_array = np.squeeze(self.array, axis=-1)

        data_array = xr.DataArray(
            field_array,
            dims=data_array_dims,
            coords=data_array_coords,
            name=name,
            attrs=dict(
                units=units or self.units,
                cell=self.mesh.cell,
                p1=self.mesh.region.p1,
                p2=self.mesh.region.p2,
            ),
        )

        for dim in geo_units_dict:
            data_array[dim].attrs["units"] = geo_units_dict[dim]

        return data_array

    @classmethod
    def from_xarray(cls, xa):
        """Create ``discretisedfield.Field`` from ``xarray.DataArray``

        The class method accepts an ``xarray.DataArray`` as an argument to
        return a ``discretisedfield.Field`` object. The DataArray must have
        either three (``x``, ``y``, and ``z`` for a scalar field) or four
        (additionally ``comp`` for a vector field) dimensions corresponding to
        geometric axes and components of the field, respectively. The
        coordinates of the ``x``, ``y``, and ``z`` dimensions represent the
        discretisation along the respective axis and must have equally spaced
        values. The coordinates of ``comp`` represent the field components
        (e.g. ['x', 'y', 'z'] for a 3D vector field).

        The ``DataArray`` is expected to have ``cell``, ``p1``, and ``p2``
        attributes for creating ``discretisedfield.Mesh`` required by the
        ``discretisedfield.Field`` object. However, in the absence of these
        attributes, the coordinates of ``x``, ``y``, and ``z`` dimensions are
        utilized. It should be noted that ``cell`` attribute is required if
        any of the geometric directions has only a single cell.

        Parameters
        ----------
        xa : xarray.DataArray

            DataArray to create Field.

        Returns
        -------
        discretisedfield.Field

            Field created from DataArray.

        Raises
        ------
        TypeError

            If argument is not ``xarray.DataArray``.

        KeyError

            If at least one of the geometric dimension coordinates has a single
            value and ``cell`` attribute is missing.

        ValueError

            - If ``DataArray.ndim`` is not 3 or 4.
            - If ``DataArray.dims`` are not either ``['x', 'y', 'z']`` or
              ``['x', 'y', 'z', 'comp']``
            - If coordinates of ``x``, ``y``, or ``z`` are not equally
              spaced

        Examples
        --------
        1. Create a DataArray

        >>> import xarray as xr
        >>> import numpy as np
        ...
        >>> xa = xr.DataArray(np.ones((20, 20, 20, 3), dtype=float),
        ...                   dims = ['x', 'y', 'z', 'comp'],
        ...                   coords = dict(x=np.arange(0, 20),
        ...                                 y=np.arange(0, 20),
        ...                                 z=np.arange(0, 20),
        ...                                 comp=['x', 'y', 'z']),
        ...                   name = 'mag',
        ...                   attrs = dict(cell=[1., 1., 1.],
        ...                                p1=[1., 1., 1.],
        ...                                p2=[21., 21., 21.]))
        >>> xa
        <xarray.DataArray 'mag' (x: 20, y: 20, z: 20, comp: 3)>
        ...

        2. Create Field from DataArray

        >>> import discretisedfield as df
        ...
        >>> field = df.Field.from_xarray(xa)
        >>> field
        Field(...)
        >>> field.average
        (1.0, 1.0, 1.0)

        """
        if not isinstance(xa, xr.DataArray):
            raise TypeError("Argument must be a xr.DataArray.")

        if xa.ndim not in [3, 4]:
            raise ValueError(
                "DataArray dimensions must be 3 for a scalar and 4 for a vector field."
            )

        if xa.ndim == 3 and sorted(xa.dims) != ["x", "y", "z"]:
            raise ValueError("The dimensions must be 'x', 'y', and 'z'.")
        elif xa.ndim == 4 and sorted(xa.dims) != ["comp", "x", "y", "z"]:
            raise ValueError("The dimensions must be 'x', 'y', 'z',and 'comp'.")

        for i in "xyz":
            if xa[i].values.size > 1 and not np.allclose(
                np.diff(xa[i].values), np.diff(xa[i].values).mean()
            ):
                raise ValueError(f"Coordinates of {i} must be equally spaced.")

        try:
            cell = xa.attrs["cell"]
        except KeyError:
            if any(len_ == 1 for len_ in xa.values.shape[:3]):
                raise KeyError(
                    "DataArray must have a 'cell' attribute if any "
                    "of the geometric directions has a single cell."
                ) from None
            cell = [np.diff(xa[i].values).mean() for i in "xyz"]

        p1 = (
            xa.attrs["p1"]
            if "p1" in xa.attrs
            else [xa[i].values[0] - c / 2 for i, c in zip("xyz", cell)]
        )
        p2 = (
            xa.attrs["p2"]
            if "p2" in xa.attrs
            else [xa[i].values[-1] + c / 2 for i, c in zip("xyz", cell)]
        )

        if any("units" not in xa[i].attrs for i in "xyz"):
            mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        else:
            mesh = df.Mesh(
                p1=p1, p2=p2, cell=cell, attributes={"unit": xa["z"].attrs["units"]}
            )

        comp = xa.comp.values if "comp" in xa.coords else None
        val = np.expand_dims(xa.values, axis=-1) if xa.ndim == 3 else xa.values
        dim = 1 if xa.ndim == 3 else val.shape[-1]
        return cls(
            mesh=mesh, dim=dim, value=val, components=comp, dtype=xa.values.dtype
        )


@functools.singledispatch
def _as_array(val, mesh, dim, dtype):
    raise TypeError("Unsupported type {type(val)}.")


# to avoid str being interpreted as iterable
@_as_array.register(str)
def _(val, mesh, dim, dtype):
    raise TypeError("Unsupported type {type(val)}.")


@_as_array.register(numbers.Complex)
@_as_array.register(collections.abc.Iterable)
def _(val, mesh, dim, dtype):
    if isinstance(val, numbers.Complex) and dim > 1 and val != 0:
        raise ValueError(
            f"Wrong dimension 1 provided for value; expected dimension is {dim}"
        )
    dtype = dtype or max(np.asarray(val).dtype, np.float64)
    return np.full((*mesh.n, dim), val, dtype=dtype)


@_as_array.register(collections.abc.Callable)
def _(val, mesh, dim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    array = np.empty((*mesh.n, dim), dtype=dtype)
    for index, point in zip(mesh.indices, mesh):
        array[index] = val(point)
    return array


@_as_array.register(Field)
def _(val, mesh, dim, dtype):
    if mesh.region not in val.mesh.region:
        raise ValueError(
            f"{val.mesh.region} of the provided field does not "
            f"contain {mesh.region} of the field that is being "
            "created."
        )
    value = (
        val.to_xarray()
        .sel(
            x=mesh.midpoints.x, y=mesh.midpoints.y, z=mesh.midpoints.z, method="nearest"
        )
        .data
    )
    if dim == 1:
        # xarray dataarrays for scalar data are three dimensional
        return value.reshape(mesh.n + (-1,))
    return value


@_as_array.register(dict)
def _(val, mesh, dim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    dtype = dtype or np.float64
    fill_value = (
        val["default"] if "default" in val and not callable(val["default"]) else np.nan
    )
    array = np.full((*mesh.n, dim), fill_value, dtype=dtype)

    for subregion in reversed(mesh.subregions.keys()):
        # subregions can overlap, first subregion takes precedence
        try:
            submesh = mesh[subregion]
            subval = val[subregion]
        except KeyError:
            continue
        else:
            slices = mesh.region2slices(submesh.region)
            array[slices] = _as_array(subval, submesh, dim, dtype)

    if np.any(np.isnan(array)):
        # not all subregion keys specified and 'default' is missing or callable
        if "default" not in val:
            raise KeyError(
                "Key 'default' required if not all subregion keys are specified."
            )
        subval = val["default"]
        for ix, iy, iz in np.argwhere(np.isnan(array[..., 0])):
            # only spatial indices required -> array[..., 0]
            array[ix, iy, iz] = subval(mesh.index2point((ix, iy, iz)))

    return array
