import collections
import functools
import numbers
import pathlib
import warnings

import findiff as fd
import numpy as np
import xarray as xr
from vtkmodules.util import numpy_support as vns
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid

import discretisedfield as df
import discretisedfield.plotting as dfp
import discretisedfield.util as dfu
from discretisedfield.plotting.util import hv_key_dim

from . import html, io

# TODO: tutorials, line operations


class Field:
    """Finite-difference field.

    This class specifies a finite-difference field and defines operations for
    its analysis and visualisation. The field is defined on a finite-difference
    mesh (`discretisedfield.Mesh`) passed using ``mesh``. Another value that
    must be passed is the dimension of the field's value using ``nvdim``. For
    instance, for a scalar field, ``nvdim=1`` and for a three-dimensional vector
    field ``nvdim=3`` must be passed. The value of the field can be set by
    passing ``value``. For details on how the value can be defined, refer to
    ``discretisedfield.Field.value``. Similarly, if the field has ``nvdim>1``,
    the field can be normalised by passing ``norm``. For details on setting the
    norm, please refer to ``discretisedfield.Field.norm``.

    Parameters
    ----------
    mesh : discretisedfield.Mesh

        Finite-difference rectangular mesh.

    nvdim : int

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

    unit : str, optional

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
    >>> nvdim = 3
    >>> value = (0, 0, 1)
    ...
    >>> field = df.Field(mesh=mesh, nvdim=nvdim, value=value)
    >>> field
    Field(...)
    >>> field.mean()
    array([0., 0., 1.])

    2. Defining a scalar field.

    >>> p1 = (-10, -10, -10)
    >>> p2 = (10, 10, 10)
    >>> n = (1, 1, 1)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> nvdim = 1
    >>> value = 3.14
    ...
    >>> field = df.Field(mesh=mesh, nvdim=nvdim, value=value)
    >>> field
    Field(...)
    >>> field.mean()
    array([3.14])

    3. Defining a uniform three-dimensional normalised vector field.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e9, -25e9, 0)
    >>> p2 = (50e9, 25e9, 5e9)
    >>> cell = (1e9, 1e9, 0.1e9)
    >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), cell=cell)
    >>> nvdim = 3
    >>> value = (0, 0, 8)
    >>> norm = 1
    ...
    >>> field = df.Field(mesh=mesh, nvdim=nvdim, value=value, norm=norm)
    >>> field
    Field(...)
    >>> field.mean()
    array([0., 0., 1.])

    .. seealso:: :py:func:`~discretisedfield.Mesh`

    """

    def __init__(
        self,
        mesh,
        nvdim=None,
        value=0.0,
        norm=None,
        vdims=None,
        dtype=None,
        unit=None,
        **kwargs,
    ):
        if not isinstance(mesh, df.Mesh):
            raise TypeError("'mesh' must be of class discretisedfield.Mesh.")
        self._mesh = mesh

        if nvdim is None and "dim" in kwargs:
            nvdim = kwargs.pop("dim")
            warnings.warn(
                "Use of 'dim' is deprecated; use 'nvdim' instead.", DeprecationWarning
            )

        if not isinstance(nvdim, numbers.Integral):
            raise TypeError("'dim' must be of type int.")
        elif nvdim < 1:
            raise ValueError("'dim' must be greater than zero.")
        self._nvdim = nvdim

        self.dtype = dtype

        self.unit = unit

        self.update_field_values(value)
        self.norm = norm

        self._vdims = None  # required in here for correct initialisation
        self.vdims = vdims

    @property
    def mesh(self):
        """The mesh on which the field is defined.

        Returns
        -------
        discretisedfield.Mesh

            The finite-difference rectangular mesh on which the field is defined.
        """
        return self._mesh

    @property
    def nvdim(self):
        """Number of value dimensions.

        Returns
        -------
        int

            Scalar fields have dimension 1, vector fields can have any dimension greater
            than 1.

        """
        return self._nvdim

    @property
    def unit(self):
        """Unit of the field.

        Returns
        -------
        str

            The unit of the field.
        """
        return self._unit

    @unit.setter
    def unit(self, unit):
        if unit is not None and not isinstance(unit, str):
            raise TypeError("'unit' must be of type str.")
        self._unit = unit

    def update_field_values(self, value):
        """Set field value representation.

        The value of the field can be set using a scalar value for ``nvdim=1``
        fields (e.g. ``field.update_field_values(3)``) or ``array_like`` value
        for ``nvdim>1`` fields (e.g. ``field.update_field_values((1, 2, 3))``).
        Alternatively, the value can be defined
        using a callable object, which takes a point tuple as an input argument
        and returns a value of appropriate dimension. Internally, callable
        object is called for every point in the mesh on which the field is
        defined. For instance, callable object can be a Python function or
        another ``discretisedfield.Field``. Finally, ``numpy.ndarray`` with
        shape ``(*self.mesh.n, nvdim)`` can be passed.

        Parameters
        ----------
        value : numbers.Real, array_like, callable, dict

            For scalar fields (``nvdim=1``) ``numbers.Real`` values are allowed.
            In the case of vector fields, ``array_like`` (list, tuple,
            numpy.ndarray) value with length equal to `nvdim` should be used.
            Finally, the value can also be a callable (e.g. Python function or
            another field), which for every coordinate in the mesh returns a
            valid value. If ``field.update_field_values(0)``, all values in the field
            will be set to zero independent of the field dimension.

            If subregions are defined value can be initialised with a dict.
            Allowed keys are names of all subregions and ``default``. Items
            must be either ``numbers.Real`` for ``nvdim=1`` or ``array_like``
            for ``nvdim=3``. If subregion names are missing, the value of
            ``default`` is used if given. If parts of the region are not
            contained within one subregion ``default`` is used if specified,
            else these values are set to 0.

        Raises
        ------
        ValueError

            If unsupported type is passed.

        Examples
        --------
        1. Different ways of setting the field value.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> value = (0, 0, 1)

        If value is not specified, zero-field is defined

        >>> field = df.Field(mesh=mesh, nvdim=3)
        >>> field.mean()
        array([0., 0., 0.])
        >>> field.update_field_values((0, 0, 1))
        >>> field.mean()
        array([0., 0., 1.])

        Setting the field value using a Python function (callable).

        >>> def value_function(point):
        ...     x, y, z = point
        ...     if x <= 1:
        ...         return (0, 0, 1)
        ...     else:
        ...         return (0, 0, -1)
        >>> field.update_field_values(value_function)
        >>> field((0.5, 1.5, 0.5))
        array([0., 0., 1.])
        >>> field((1.5, 1.5, 0.5))
        array([ 0.,  0., -1.])

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
        >>> field = df.Field(mesh, nvdim=1, value={'s1': 1, 's2': 1})
        >>> (field.array == 1).all()
        True
        >>> field = df.Field(mesh, nvdim=1, value={'s1': 1})
        Traceback (most recent call last):
        ...
        KeyError: ...
        >>> field = df.Field(mesh, nvdim=1, value={'s1': 2, 'default': 1})
        >>> (field.array == 1).all()
        False
        >>> (field.array == 0).any()
        False
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell, subregions={'s': sub1})
        >>> field = df.Field(mesh, nvdim=1, value={'s': 1})
        Traceback (most recent call last):
        ...
        KeyError: ...
        >>> field = df.Field(mesh, nvdim=1, value={'default': 1})
        >>> (field.array == 1).all()
        True

        .. seealso:: :py:func:`~discretisedfield.Field.array`

        """
        self.array = _as_array(value, self.mesh, self.nvdim, dtype=self.dtype)

    @property
    def vdims(self):
        """Vector components of the field."""
        return self._vdims

    @vdims.setter
    def vdims(self, vdims):
        if vdims is not None:
            if len(vdims) != self.nvdim:
                raise ValueError(f"Number of vdims does not match {self.nvdim=}.")
            if len(vdims) != len(set(vdims)):
                raise ValueError("'vdims' must be unique.")
            for c in vdims:
                if hasattr(self, c):
                    # redefining component labels is okay.
                    if self._vdims is None or c not in self._vdims:
                        raise ValueError(
                            f"Component name {c} is already "
                            "used by a different method/property."
                        )
            self._vdims = list(vdims)
        else:
            if 2 <= self.nvdim <= 3:
                vdims = ["x", "y", "z"][: self.nvdim]
            elif self.nvdim > 3:
                warnings.warn(
                    "Component labels must be specified for "
                    f"{self.nvdim=} fields to get access to individual"
                    " vector components."
                )
            self._vdims = vdims

    @property
    def array(self):
        """Field value as ``numpy.ndarray``.

        The shape of the array is ``(*mesh.n, nvdim)``.

        Parameters
        ----------
        array : numpy.ndarray

            Array with shape ``(*mesh.n, nvdim)``.

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
        >>> field = df.Field(mesh=mesh, nvdim=3, value=value)
        >>> field.array
        array(...)
        >>> field.mean()
        array([0., 0., 1.])
        >>> field.array.shape
        (2, 1, 1, 3)
        >>> field.array = np.ones_like(field.array)
        >>> field.array
        array(...)
        >>> field.mean()
        array([1., 1., 1.])

        """
        return self._array

    @array.setter
    def array(self, val):
        self._array = _as_array(val, self.mesh, self.nvdim, dtype=self.dtype)

    @property
    def norm(self):
        """Norm of the field.

        Computes the norm of the field and returns ``discretisedfield.Field``
        with ``nvdim=1``. Norm of a scalar field is interpreted as an absolute
        value of the field.

        The field norm can be set by passing ``numbers.Real``,
        ``numpy.ndarray``, or callable. If the field contains zero values, norm
        cannot be set and ``ValueError`` is raised.

        Parameters
        ----------
        numbers.Real, numpy.ndarray, callable

            Norm value.

        Returns
        -------
        discretisedfield.Field

            Norm of the field.

        Raises
        ------
        ValueError

            If the norm is set with wrong type, shape, or value. In addition,
            if the field contains zero values.

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
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(0, 0, 1))
        >>> field.norm
        Field(...)
        >>> field.norm.mean()
        array([1.])
        >>> field.norm = 2
        >>> field.mean()
        array([0., 0., 2.])
        >>> field.update_field_values((1, 0, 0))
        >>> field.norm.mean()
        array([1.])

        Set the norm for a zero field.
        >>> field.update_field_values(0)
        >>> field.mean()
        array([0., 0., 0.])
        >>> field.norm = 1
        >>> field.mean()
        array([0., 0., 0.])

        .. seealso:: :py:func:`~discretisedfield.Field.__abs__`

        """
        res = np.linalg.norm(self.array, axis=-1, keepdims=True)

        return self.__class__(self.mesh, nvdim=1, value=res, unit=self.unit)

    @norm.setter
    def norm(self, val):
        if val is not None:
            self.array = np.divide(
                self.array,
                self.norm.array,
                out=np.zeros_like(self.array),
                where=self.norm.array != 0.0,
            )
            self.array *= _as_array(val, self.mesh, nvdim=1, dtype=None)

    def __abs__(self):
        """Absolute value of the field.

        This is a convenience operator and it returns
        absolute value of the field.

        Returns
        -------
        discretisedfield.Field

            Absolute value of the field.

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
        >>> field = df.Field(mesh=mesh, nvdim=1, value=-5)
        >>> abs(field).mean()
        array([5.])

        .. seealso:: :py:func:`~discretisedfield.Field.norm`

        """
        return self.__class__(
            self.mesh, nvdim=self.nvdim, value=np.abs(self.array), unit=self.unit
        )

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
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(3, -1, 1))
        >>> zero_field = field.zero
        >>> zero_field.mean()
        array([0., 0., 0.])

        """
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=0,
            vdims=self.vdims,
            unit=self.unit,
        )

    @property
    def orientation(self):
        """Orientation field.

        This method computes the orientation (direction) of a vector field and
        returns ``discretisedfield.Field`` with the same dimension. More
        precisely, at every mesh discretisation cell, the vector is divided by
        its norm, so that a unit vector is obtained. However, if the vector at
        a discretisation cell is a zero-vector, it remains unchanged. In the
        case of a scalar (``nvdim=1``) field, ``ValueError`` is raised.

        Returns
        -------
        discretisedfield.Field

            Orientation field.

        Raises
        ------
        ValueError

            If the field is has ``nvdim=1``.

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
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(6, 0, 8))
        >>> field.orientation
        Field(...)
        >>> field.orientation.norm.mean()
        array([1.])

        """
        if self.nvdim == 1:
            msg = f"Cannot compute orientation field for a nvdim={self.nvdim} field."
            raise ValueError(msg)

        orientation_array = np.divide(
            self.array,
            self.norm.array,
            where=np.invert(np.isclose(self.norm.array, 0)),
            out=np.zeros_like(self.array),
        )
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=orientation_array,
            vdims=self.vdims,
        )

    def mean(self, direction=None):
        """Field mean.

        It computes the arithmetic mean along the specified direction of the field
        over the entire volume of the mesh. It returns a numpy array
        containing the mean values.


        Parameters
        ----------
        direction

            None or string or tuple of strings, optional.
            Directions along which
            the means are computed. The default is to
            compute the mean of the entire volume and return an array of the
            averaged vector components.


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
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(0, 0, 1))
        >>> field.mean()
        array([0., 0., 1.])

        2. Computing the scalar field average.

        >>> field = df.Field(mesh=mesh, nvdim=1, value=55)
        >>> field.mean()
        array([55.])

        """
        if direction is None:
            return self.array.mean(axis=tuple(range(self.mesh.region.ndim)))
        elif isinstance(direction, (tuple, list)):
            if not all(d in self.mesh.region.dims for d in direction):
                raise ValueError(
                    f"Invalid direction. Directions must be in {self.mesh.region.dims}."
                )
            elif len(direction) != len(set(direction)):
                raise ValueError("Duplicate directions are not allowed.")

            if sorted(direction) == sorted(self.mesh.region.dims):
                return self.array.mean(axis=tuple(range(self.mesh.region.ndim)))
            else:
                # NOTE: this is a temporary solution until mesh.sel is implemented.
                # Hence, this is not the most efficient way to do it.
                mesh = self.mesh  # Do we need a deepcopy here?
                array = self.array
                for d in direction:
                    mesh = mesh.plane(d)
                    # Keepdims is needed for the current 3D behaviour
                    array = array.mean(axis=mesh.region.dims.index(d), keepdims=True)
                return self.__class__(mesh, nvdim=self.nvdim, value=array)
        elif isinstance(direction, str):
            if direction not in self.mesh.region.dims:
                raise ValueError(
                    f"Invalid direction. Direction must be in {self.mesh.region.dims}."
                )
            axis = self.mesh.region.dims.index(direction)
            return self.__class__(
                self.mesh.plane(direction),
                nvdim=self.nvdim,
                value=self.array.mean(axis=axis, keepdims=True),
                vdims=self.vdims,
                units=self.units,
            )
        else:
            raise ValueError(
                "Direction must be None, string or tuple of strings, not"
                f" {type(direction)}."
            )

        return self.__class__(
            self.mesh.plane(direction),  # mesh sel method
            nvdim=self.nvdim,
            value=self.array.mean(axis=axis),
            vdims=self.vdims,
            units=self.units,
        )

    @property
    def average(self):
        warnings.warn(
            "The average property is deprecated. Please use the mean function.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.mean()

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
        >>> field = df.Field(mesh, nvdim=1, value=1)
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
        the dimension (``nvdim``) of the field.

        Parameters
        ----------
        point : (3,) array_like

            The mesh point coordinate :math:`\mathbf{p} = (p_{x}, p_{y},
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
        >>> field = df.Field(mesh, nvdim=3, value=(1, 3, 4))
        >>> point = (10, 2, 3)
        >>> field(point)
        array([1., 3., 4.])

        """
        return self.array[self.mesh.point2index(point)]

    def __getattr__(self, attr):
        """Extract the component of the vector field.

        This method provides access to individual field components for fields
        with dimension > 1. Component labels are defined in the ``vdims``
        attribute. For dimension 2 and 3 default values ``'x'``, ``'y'``, and
        ``'z'`` are used if no custom component labels are provided. For fields
        with ``nvdim>3`` vdims must be specified manually to get
        access to individual vector components.

        Parameters
        ----------
        attr : str

            Vector field component defined in ``vdims``.

        Returns
        -------
        discretisedfield.Field

            Scalar field with vector field component values.

        Examples
        --------
        1. Accessing the default vector field vdims.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(0, 0, 1))
        >>> field.x
        Field(...)
        >>> field.x.mean()
        array([0.])
        >>> field.y
        Field(...)
        >>> field.y.mean()
        array([0.])
        >>> field.z
        Field(...)
        >>> field.z.mean()
        array([1.])
        >>> field.z.nvdim
        1

        2. Accessing custom vector field vdims.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 2)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(0, 0, 1),
        ...                  vdims=['mx', 'my', 'mz'])
        >>> field.mx
        Field(...)
        >>> field.mx.mean()
        array([0.])
        >>> field.my
        Field(...)
        >>> field.my.mean()
        array([0.])
        >>> field.mz
        Field(...)
        >>> field.mz.mean()
        array([1.])
        >>> field.mz.nvdim
        1

        """
        if self.vdims is not None and attr in self.vdims:
            attr_array = self.array[..., self.vdims.index(attr), np.newaxis]
            return self.__class__(
                mesh=self.mesh, nvdim=1, value=attr_array, unit=self.unit
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

        if self.vdims is not None:
            dirlist += self.vdims
        if self.nvdim == 1:
            need_removing = ["div", "curl", "orientation"]
        if self.nvdim == 2:
            need_removing = ["grad", "curl", "k3d"]
        if self.nvdim == 3:
            need_removing = ["grad"]

        for attr in need_removing:
            dirlist.remove(attr)

        return dirlist

    def __iter__(self):
        r"""Generator yielding values of all discretisation cells.

        Yields
        ------
        np.ndarray

            The field value in one discretisation cell.

        Examples
        --------
        1. Iterating through the field values

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh, nvdim=3, value=(0, 0, 1))
        >>> for value in field:
        ...     print(value)
        [0. 0. 1.]
        [0. 0. 1.]
        [0. 0. 1.]
        [0. 0. 1.]

        2. Iterating through the mesh coordinates and field values

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh, nvdim=3, value=(0, 0, 1))
        >>> for coord, value in zip(field.mesh, field):
        ...     print(coord, value)
        (0.5, 0.5, 0.5) [0. 0. 1.]
        (1.5, 0.5, 0.5) [0. 0. 1.]
        (0.5, 1.5, 0.5) [0. 0. 1.]
        (1.5, 1.5, 0.5) [0. 0. 1.]

        See also
        --------
        :py:func:`~discretisedfield.Mesh.__iter__`
        :py:func:`~discretisedfield.Mesh.indices`

        """
        for point in self.mesh:
            yield self(point)

    def __eq__(self, other):
        """Relational operator ``==``.

        Two fields are considered to be equal if:

          1. They are defined on the same mesh.

          2. They have the same dimension (``nvdim``).

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
        >>> f1 = df.Field(mesh, nvdim=1, value=3)
        >>> f2 = df.Field(mesh, nvdim=1, value=4-1)
        >>> f3 = df.Field(mesh, nvdim=3, value=(1, 4, 3))
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
        return (
            self.mesh == other.mesh
            and self.nvdim == other.nvdim
            and np.array_equal(self.array, other.array)
        )

    # TODO The mesh comparison has no tolerance.
    def allclose(self, other, rtol=1e-5, atol=1e-8):
        """Allclose method.

        This method determines whether two fields are:

          1. Defined on the same mesh.

          2. Have the same dimension (``nvdim``).

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
        >>> f1 = df.Field(mesh, nvdim=1, value=3)
        >>> f2 = df.Field(mesh, nvdim=1, value=3+1e-9)
        >>> f3 = df.Field(mesh, nvdim=1, value=3.1)
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

        if self.mesh == other.mesh and self.nvdim == other.nvdim:
            return np.allclose(self.array, other.array, rtol=rtol, atol=atol)
        else:
            return False

    def is_same_vectorspace(self, other):  # TODO: check vdims
        if not isinstance(other, self.__class__):
            raise TypeError(f"Object of type {type(other)} not supported.")
        return self.nvdim == other.nvdim

    def _check_same_mesh_and_field_dim(self, other, ignore_scalar=False):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Object of type {type(other)} not supported.")

        if self.mesh != other.mesh:
            raise ValueError(
                "To perform this operation both fields must have the same mesh."
            )

        if ignore_scalar and (self.nvdim == 1 or other.nvdim == 1):
            return

        if not self.is_same_vectorspace(other):
            raise ValueError(
                "To perform this operation both fields must have the same"
                " number of vector components."
            )

    def _apply_operator(self, other, function, operator):
        if isinstance(other, self.__class__):
            self._check_same_mesh_and_field_dim(other, ignore_scalar=True)
            other = other.array
        elif isinstance(other, numbers.Complex):
            pass
        elif isinstance(other, (tuple, list, np.ndarray)):
            if not (
                self.array.shape == np.shape(other)
                or self.nvdim == len(other)
                or self.nvdim == 1
            ):
                raise TypeError(
                    f"Unsupported operand type(s) for {operator}: {type(self)} with"
                    f" {self.nvdim} vdims and {type(other)} with shape"
                    f" {np.shape(other)}."
                )
        else:
            msg = (
                f"Unsupported operand type(s) for {operator}: {type(self)=} and"
                f" {type(other)=}."
            )
            raise TypeError(msg)

        res_array = function(self.array, other)
        vdims = self.vdims if self.nvdim == res_array.shape[-1] else None
        return self.__class__(
            self.mesh,
            nvdim=res_array.shape[-1],
            value=res_array,
            vdims=vdims,
        )

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
        >>> f = df.Field(mesh, nvdim=3, value=(0, -1000, -3))
        >>> res = +f
        >>> res.mean()
        array([    0., -1000.,    -3.])
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

            -f(x, y, z) = -1 \cdot f(x, y, z)

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
        >>> f = df.Field(mesh, nvdim=1, value=3.1)
        >>> res = -f
        >>> res.mean()
        array([-3.1])
        >>> f == -(-f)
        True

        2. Applying unary negation operator on a vector field.

        >>> f = df.Field(mesh, nvdim=3, value=(0, -1000, -3))
        >>> res = -f
        >>> res.mean()
        array([   0., 1000.,    3.])

        """
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=-self.array,
            vdims=self.vdims,
        )

    def __pow__(self, other):
        """Unary ``**`` operator.

        This method defines the ``**`` operator for scalar (``nvdim=1``) fields
        only. This operator is not defined for vector (``nvdim>1``) fields, and
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
        >>> f = df.Field(mesh, nvdim=1, value=2)
        >>> res = f**(-1)
        >>> res
        Field(...)
        >>> res.mean()
        array([0.5])
        >>> res = f**2
        >>> res.mean()
        array([4.])
        >>> (f**f).mean()
        array([4.])

        2. Attempt to apply power operator on a vector field.

        >>> p1 = (0, 0, 0)
        >>> p2 = (5e-9, 5e-9, 5e-9)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        ...
        >>> f = df.Field(mesh, nvdim=3, value=(0, -1, -3))
        >>> (f**2).mean()
        array([0., 1., 9.])

        """
        return self._apply_operator(other, np.power, "**")

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
        >>> f1 = df.Field(mesh, nvdim=3, value=(0, -1, -3.1))
        >>> f2 = df.Field(mesh, nvdim=3, value=(0, 1, 3.1))
        >>> res = f1 + f2
        >>> res.mean()
        array([0., 0., 0.])
        >>> f1 + f2 == f2 + f1
        True
        >>> res = f1 + (1, 2, 3.1)
        >>> res.mean()
        array([1., 1., 0.])
        >>> (f1 + 5).mean()
        array([5. , 4. , 1.9])

        .. seealso:: :py:func:`~discretisedfield.Field.__sub__`

        """
        return self._apply_operator(other, np.add, "+")

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
        >>> f1 = df.Field(mesh, nvdim=3, value=(0, 1, 6))
        >>> f2 = df.Field(mesh, nvdim=3, value=(0, 1, 3))
        >>> res = f1 - f2
        >>> res.mean()
        array([0., 0., 3.])
        >>> f1 - f2 == -(f2 - f1)
        True
        >>> res = f1 - (0, 1, 0)
        >>> res.mean()
        array([0., 0., 6.])

        .. seealso:: :py:func:`~discretisedfield.Field.__add__`

        """
        return self._apply_operator(other, np.subtract, "-")

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        """Binary ``*`` operator.

        It can be applied between:

        1. Two fields with equal vector dimentions,

        2. A field of any dimension and ``numbers.Complex``,

        3. A field of any dimension and a scalar (``nvdim=1``) field.

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
        >>> f1 = df.Field(mesh, nvdim=1, value=5)
        >>> f2 = df.Field(mesh, nvdim=1, value=9)
        >>> res = f1 * f2
        >>> res.mean()
        array([45.])
        >>> f1 * f2 == f2 * f1
        True

        2. Multiply vector field with a scalar.

        >>> f1 = df.Field(mesh, nvdim=3, value=(0, 2, 5))
        ...
        >>> res = f1 * 5  # discretisedfield.Field.__mul__ is called
        >>> res.mean()
        array([ 0., 10., 25.])
        >>> res = 10 * f1  # discretisedfield.Field.__rmul__ is called
        >>> res.mean()
        array([ 0., 20., 50.])

        .. seealso:: :py:func:`~discretisedfield.Field.__truediv__`

        """
        return self._apply_operator(other, np.multiply, "*")

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """Binary ``/`` operator.

        It can be applied between:

        1. Two fields with equal vector dimentions,

        2. A field of any dimension and ``numbers.Complex``,

        3. A field of any dimension and a scalar (``nvdim=1``) field.

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
        >>> f1 = df.Field(mesh, nvdim=1, value=100)
        >>> f2 = df.Field(mesh, nvdim=1, value=20)
        >>> res = f1 / f2
        >>> res.mean()
        array([5.])
        >>> f1 / f2 == (f2 / f1)**(-1)
        True

        2. Divide vector field by a scalar.

        >>> f1 = df.Field(mesh, nvdim=3, value=(0, 10, 5))
        >>> res = f1 / 5  # discretisedfield.Field.__mul__ is called
        >>> res.mean()
        array([0., 2., 1.])
        >>> (10 / f1).mean()  # division by a vector is not allowed
        array([inf,  1.,  2.])

        .. seealso:: :py:func:`~discretisedfield.Field.__mul__`

        """
        return self._apply_operator(other, np.divide, "/")

    def __rtruediv__(self, other):
        # TODO: Fix error messages - wrong order
        return self._apply_operator(other, lambda x, y: np.divide(y, x), "/")

    def dot(self, other):
        """Dot product.

        This method computes the dot product between two fields. Both fields
        must have the same number of vector dimentions and defined on the same mesh

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
        >>> f1 = df.Field(mesh, nvdim=3, value=(1, 3, 6))
        >>> f2 = df.Field(mesh, nvdim=3, value=(-1, -2, 2))
        >>> f1.dot(f2).mean()
        array([5.])

        """
        if isinstance(other, self.__class__):
            self._check_same_mesh_and_field_dim(other)
            other = other.array
        elif not isinstance(other, (tuple, list, np.ndarray)):
            msg = (
                f"Unsupported operand type(s) for dot product: {type(self)=} and"
                f" {type(other)=}."
            )
            raise TypeError(msg)

        res_array = np.einsum("...l,...l->...", self.array, other)
        return df.Field(self.mesh, nvdim=1, value=res_array[..., np.newaxis])

    def __matmul__(self, other):
        return self.dot(other)

    def __rmatmul__(self, other):
        return self.dot(other)

    def cross(self, other):
        """Cross product.

        This method computes the cross product between two fields. Both fields
        must be three-dimensional (``nvdim=3``) and defined on the same mesh.

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
        >>> f1 = df.Field(mesh, nvdim=3, value=(1, 0, 0))
        >>> f2 = df.Field(mesh, nvdim=3, value=(0, 1, 0))
        >>> (f1.cross(f2)).mean()
        array([0., 0., 1.])
        >>> (f1.cross((0, 0, 1))).mean()
        array([ 0., -1.,  0.])

        """
        if isinstance(other, self.__class__):
            self._check_same_mesh_and_field_dim(other)
            if self.nvdim != 3 or other.nvdim != 3:
                msg = (
                    f"Cannot apply cross product on {self.nvdim=} and"
                    f" {other.nvdim=} fields."
                )
                raise ValueError(msg)
            other = other.array
        elif not isinstance(other, (tuple, list, np.ndarray)):
            msg = (
                f"Unsupported operand type(s) for cross product: {type(self)=} and"
                f" {type(other)=}."
            )
            raise TypeError(msg)

        return self.__class__(
            self.mesh,
            nvdim=3,
            value=np.cross(self.array, other),
            vdims=self.vdims,
        )

    def __and__(self, other):
        return self.cross(other)

    def __rand__(self, other):
        return -self.cross(other)

    def __lshift__(self, other):
        """Stacks multiple scalar fields in a single vector field.

        This method takes a list of scalar (``nvdim=1``) fields and returns a
        vector field, whose components are defined by the scalar fields passed.
        If any of the fields passed has ``nvdim!=1`` or they are not defined on
        the same mesh, an exception is raised. The dimension of the resulting
        field is equal to the length of the passed list.

        Parameters
        ----------
        fields : list

            List of ``discretisedfield.Field`` objects with ``nvdim=1``.

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
        >>> f1 = df.Field(mesh, nvdim=1, value=1)
        >>> f2 = df.Field(mesh, nvdim=1, value=5)
        >>> f3 = df.Field(mesh, nvdim=1, value=-3)
        ...
        >>> f = f1 << f2 << f3
        >>> f.mean()
        array([ 1.,  5., -3.])
        >>> f.nvdim
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
            return self << self.__class__(self.mesh, nvdim=1, value=other)
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self << self.__class__(self.mesh, nvdim=len(other), value=other)
        else:
            msg = (
                f"Unsupported operand type(s) for <<: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

        array_list = [self.array[..., i] for i in range(self.nvdim)]
        array_list += [other.array[..., i] for i in range(other.nvdim)]

        if self.vdims is None or other.vdims is None:
            vdims = None
        else:
            vdims = self.vdims + other.vdims
            if len(vdims) != len(set(vdims)):
                # Component name duplicated; could happen e.g. for lshift with
                # a number -> choose labels automatically
                vdims = None

        return self.__class__(
            self.mesh,
            nvdim=len(array_list),
            value=np.stack(array_list, axis=3),
            vdims=vdims,
        )

    def __rlshift__(self, other):
        if isinstance(other, numbers.Complex):
            return self.__class__(self.mesh, nvdim=1, value=other) << self
        elif isinstance(other, (tuple, list, np.ndarray)):
            return self.__class__(self.mesh, nvdim=len(other), value=other) << self
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
        >>> field = df.Field(mesh, nvdim=1, value=1)
        ...
        >>> # Two cells with value 1
        >>> pf = field.pad({'x': (1, 1)}, mode='constant')  # zeros padded
        >>> pf.mean()
        array([0.5])

        """
        d = {}
        for key, value in pad_width.items():
            d[dfu.axesdict[key]] = value
        padding_sequence = dfu.assemble_index((0, 0), len(self.array.shape), d)

        padded_array = np.pad(self.array, padding_sequence, mode=mode, **kwargs)
        padded_mesh = self.mesh.pad(pad_width)

        return self.__class__(
            padded_mesh,
            nvdim=self.nvdim,
            value=padded_array,
            vdims=self.vdims,
            unit=self.unit,
        )

    def diff(self, direction, order=1):
        """Directional derivative.

        This method computes a directional derivative of the field and returns
        a field. The direction in which the derivative is computed is passed
        via ``direction`` argument, which can be ``'x'``, ``'y'``, or ``'z'``.
        The order of the computed derivative can be 1 or 2 and it is specified
        using argument ``order`` and it defaults to 1.

        This method uses second order accurate finite difference stencils by default
        unless the field is defined on a mesh with too few cells in the differential
        direction. In this case the first order accurate finite difference stencils
        are used at the boundaries and the second order accurate finite difference
        stencils are used in the interior.

        Directional derivative cannot be computed if less or equal discretisation
        cells exists in a specified direction than the order.
        In that case, a zero field is
        returned. Computing of the directional derivative depends
        strongly on the boundary condition specified in the mesh on which the
        field is defined on. More precisely, the values of the derivatives at
        the boundary are different for periodic, Neumann, dirichlet, or no boundary
        conditions. For details on boundary conditions, please refer to the
        ``disretisedfield.Mesh`` class. The derivatives are computed using
        central differences inside the sample and using forward/backward
        differences at the boundaries.

        Parameters
        ----------
        direction : str

            The direction in which the derivative is computed. It can be
            ``'x'``, ``'y'``, or ``'z'``.

        order : int

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
        >>> f = df.Field(mesh, nvdim=1, value=value_fun)
        >>> f.diff('y').mean()  # first-order derivative by default
        array([3.])

        2. Try to compute the second-order directional derivative of the vector
        field which has only one discretisation cell in the z-direction. For
        the field we choose :math:`f(x, y, z) = (2x, 3y, -5z)`. Accordingly, we
        expect the directional derivatives to be: :math:`df/dx = (2, 0, 0)`,
        :math:`df/dy=(0, 3, 0)`, :math:`df/dz = (0, 0, -5)`. However, because
        there is only one discretisation cell in the z-direction, the
        derivative cannot be computed and a zero field is returned.

        >>> import numpy as np
        >>> def value_fun(point):
        ...     x, y, z = point
        ...     return (2*x, 3*y, -5*z)
        ...
        >>> f = df.Field(mesh, nvdim=3, value=value_fun)
        >>> np.allclose(f.diff('x', order=1).mean(), [2, 0, 0])
        True
        >>> np.allclose(f.diff('y', order=1).mean(), [0, 3, 0])
        True
        >>> f.diff('z', order=1).mean()  # derivative cannot be calculated
        array([0., 0., 0.])

        """
        if direction not in self.mesh.region.dims:
            raise ValueError(
                f"Direction {direction} is not valid. "
                "It must be one of the following: "
                f"{self.mesh.region.dims}."
            )
        direction_idx = self.mesh.region.dims.index(direction)

        # If there are no neighbouring cells in the specified direction, zero
        # field is returned.
        # Directional derivative cannot be computed if less or an equal number of
        # discretisation cells exists in a specified direction than the order.
        # In that case, a zero field is returned.
        if self.mesh.n[direction_idx] <= order:
            return self.zero

        # Preparation (padding) for computing the derivative, depending on the
        # boundary conditions (PBC, Neumann, or no BC). Depending on the BC,
        # the field array is padded.
        if direction in self.mesh.bc:  # PBC
            pad_width = {direction: (1, 1)}
            padding_mode = "wrap"
        elif self.mesh.bc == "neumann":
            pad_width = {direction: (1, 1)}
            padding_mode = "symmetric"
        elif self.mesh.bc == "dirichlet":
            pad_width = {direction: (1, 1)}
            padding_mode = "constant"
        else:  # No BC - no padding
            pad_width = {}
            padding_mode = "constant"

        padded_array = self.pad(pad_width, mode=padding_mode).array

        if order not in (1, 2):
            msg = f"Derivative of the {order} order is not implemented."
            raise NotImplementedError(msg)

        elif order == 1:
            if self.mesh.n[direction_idx] < 3:
                # The derivative is computed using the central difference
                # with forward/backward difference at the boundaries.
                derivative_array = np.gradient(
                    padded_array, self.mesh.cell[direction_idx], axis=direction_idx
                )
            else:
                # The derivative is computed using accuracy of 2 everywhere
                diff_fd = fd.FinDiff(direction_idx, self.mesh.cell[direction_idx], 1)
                derivative_array = diff_fd(padded_array)

        elif order == 2:
            if self.mesh.n[direction_idx] < 4:
                # The derivative is computed using the central difference
                # with forward/backward difference at the boundaries.
                if self.mesh.bc == "":
                    # Pad with specific values so that the same finite difference
                    # stencil can be used across the whole array
                    # central difference = forward difference
                    # f(1) + f(-1) - 2 f(0) = f(2) + f(0) - 2 f(1)
                    # f(-1) = f(2) - 3 f(1) + 3f(0)

                    def pad_fun(vector, pad_width, iaxis, kwargs):
                        if iaxis == direction_idx:
                            vector[0] = vector[3] - 3 * vector[2] + 3 * vector[1]
                            vector[-1] = vector[-4] - 3 * vector[-3] + 3 * vector[-2]

                    pad_width = [(0, 0)] * 4
                    pad_width[direction_idx] = (1, 1)
                    padded_array = np.pad(padded_array, pad_width, pad_fun)

                index_p1 = dfu.assemble_index(
                    slice(None), 4, {direction_idx: slice(2, None)}
                )
                index_0 = dfu.assemble_index(
                    slice(None), 4, {direction_idx: slice(1, -1)}
                )
                index_m1 = dfu.assemble_index(
                    slice(None), 4, {direction_idx: slice(None, -2)}
                )
                derivative_array = (
                    padded_array[index_p1]
                    - 2 * padded_array[index_0]
                    + padded_array[index_m1]
                ) / self.mesh.cell[direction_idx] ** 2
            else:
                # The derivative is computed using accuracy of 2 everywhere
                diff_fd = fd.FinDiff(direction_idx, self.mesh.cell[direction_idx], 2)
                derivative_array = diff_fd(padded_array)

        # Remove padded values (if any).
        if derivative_array.shape != self.array.shape:
            derivative_array = derivative_array[
                dfu.assemble_index(slice(None), 4, {direction_idx: slice(1, -1)})
            ]

        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=derivative_array,
            vdims=self.vdims,
        )

    @property
    def grad(self):
        r"""Gradient.

        This method computes the gradient of a scalar (``nvdim=1``) field and
        returns a vector field:

        .. math::

            \nabla f = (\frac{\partial f}{\partial x},
                         \frac{\partial f}{\partial y},
                         \frac{\partial f}{\partial z})

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
        >>> f = df.Field(mesh, nvdim=1, value=5)
        >>> f.grad.mean()
        array([0., 0., 0.])

        2. Compute gradient of a spatially varying field. For a field we choose
        :math:`f(x, y, z) = 2x + 3y - 5z`. Accordingly, we expect the gradient
        to be a constant vector field :math:`\nabla f = (2, 3, -5)`.

        >>> def value_fun(point):
        ...     x, y, z = point
        ...     return 2*x + 3*y - 5*z
        ...
        >>> f = df.Field(mesh, nvdim=1, value=value_fun)
        >>> f.grad.mean()
        array([ 2.,  3., -5.])

        3. Attempt to compute the gradient of a vector field.

        >>> f = df.Field(mesh, nvdim=3, value=(1, 2, -3))
        >>> f.grad
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.nvdim != 1:
            msg = f"Cannot compute gradient for nvdim={self.nvdim} field."
            raise ValueError(msg)

        return self.diff("x") << self.diff("y") << self.diff("z")

    @property
    def div(self):
        r"""Divergence.

        This method computes the divergence of a vector (``nvdim=2`` or
        ``nvdim=3``) field and returns a scalar (``nvdim=1``) field as a result.

        .. math::

            \nabla\cdot\mathbf{v} = \sum_i\frac{\partial v_{i}}
            {\partial i}

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
        :math:`\mathbf{v}(x, y, z) = (2x, -2y, 5z)`. Accordingly, we expect
        the divergence to be to be a constant scalar field :math:`\nabla\cdot
        \mathbf{v} = 5`.

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
        >>> f = df.Field(mesh, nvdim=3, value=value_fun)
        >>> f.div.mean()
        array([5.])

        2. Attempt to compute the divergence of a scalar field.

        >>> f = df.Field(mesh, nvdim=1, value=3.14)
        >>> f.div
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.nvdim not in [2, 3]:
            msg = f"Cannot compute divergence for nvdim={self.nvdim} field."
            raise ValueError(msg)

        return sum(getattr(self, vdim).diff(vdim) for vdim in self.vdims)

    @property
    def curl(self):
        r"""Curl.

        This method computes the curl of a vector (``nvdim=3``) field and returns
        a vector (``nvdim=3``) as a result:

        .. math::

            \nabla \times \mathbf{v} = \left(\frac{\partial
            v_{z}}{\partial y} - \frac{\partial v_{y}}{\partial z},
            \frac{\partial v_{x}}{\partial z} - \frac{\partial
            v_{z}}{\partial x}, \frac{\partial v_{y}}{\partial x} -
            \frac{\partial v_{x}}{\partial y},\right)

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
        :math:`\mathbf{v}(x, y, z) = (2xy, -2y, 5xz)`. Accordingly, we expect
        the curl to be to be a constant vector field :math:`\nabla\times
        \mathbf{v} = (0, -5z, -2x)`.

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
        >>> f = df.Field(mesh, nvdim=3, value=value_fun)
        >>> f.curl((1, 1, 1))
        array([ 0., -5., -2.])

        2. Attempt to compute the curl of a scalar field.

        >>> f = df.Field(mesh, nvdim=1, value=3.14)
        >>> f.curl
        Traceback (most recent call last):
        ...
        ValueError: ...

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.nvdim != 3:
            msg = f"Cannot compute curl for nvdim={self.nvdim} field."
            raise ValueError(msg)

        x, y, z = self.vdims
        curl_x = getattr(self, z).diff("y") - getattr(self, y).diff("z")
        curl_y = getattr(self, x).diff("z") - getattr(self, z).diff("x")
        curl_z = getattr(self, y).diff("x") - getattr(self, x).diff("y")

        return curl_x << curl_y << curl_z

    @property
    def laplace(self):
        r"""Laplace operator.

        This method computes the laplacian of a scalar (``nvdim=1``) or a vector
        (``nvdim=3``) field and returns a resulting field:

        .. math::

            \nabla^2 f = \frac{\partial^{2} f}{\partial x^{2}} +
                          \frac{\partial^{2} f}{\partial y^{2}} +
                          \frac{\partial^{2} f}{\partial z^{2}}

        .. math::

            \nabla^2 \mathbf{f} = (\nabla^2 f_{x},
                                     \nabla^2 f_{y},
                                     \nabla^2 f_{z})

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
        >>> f = df.Field(mesh, nvdim=1, value=5)
        >>> f.laplace.mean()
        array([0.])

        2. Compute Laplacian of a spatially varying field. For a field we
        choose :math:`f(x, y, z) = 2x^{2} + 3y - 5z`. Accordingly, we expect
        the Laplacian to be a constant vector field :math:`\nabla f = (4, 0,
        0)`.

        >>> def value_fun(point):
        ...     x, y, z = point
        ...     return 2*x**2 + 3*y - 5*z
        ...
        >>> f = df.Field(mesh, nvdim=1, value=value_fun)
        >>> assert abs(f.laplace.mean() - 4) < 1e-3

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.nvdim not in [1, 3]:
            raise ValueError(f"Cannot compute laplace for nvdim={self.nvdim} field.")
        if self.nvdim == 1:
            return (
                self.diff("x", order=2)
                + self.diff("y", order=2)
                + self.diff("z", order=2)
            )
        else:
            x, y, z = self.vdims
            return (
                getattr(self, x).laplace
                << getattr(self, y).laplace
                << getattr(self, z).laplace
            )

    def integral(self, direction="xyz", improper=False):
        raise AttributeError("This method has been renamed to 'integrate'.")

    def integrate(self, direction=None, cumulative=False):
        r"""Integral.

        This method integrates the field over the mesh along the specified direction,
        which can be specified using ``direction``. The field is internally multiplied
        with the cell size in that direction. If no direction is specified the integral
        is computed along all directions.

        To compute surface integrals, e.g. flux, the field must be multiplied with the
        surface normal vector prior to integration (see example 4).

        A cumulative integral can be computed by passing ``cumulative=True`` and by
        specifying a single direction. It resembles the following integral (here as an
        example in the x direction):

        .. math::

            F(x, y, z) = \int_{p_\mathrm{min}}^x f(x', y, z) \mathrm{d}x

        The method sums all cells up to (excluding) the cell that contains the point x.
        The cell containing x is added with a weight 1/2.

        Parameters
        ----------
        direction : str, optional

            Direction along which the field is integrated. The direction must be in
            ``field.mesh.region.dims``. Defaults to ``None``.

        cumulative : bool, optional

            If ``True``, an cumulative integral is computed. Defaults to ``False``.

        Returns
        -------
        discretisedfield.Field or np.ndarray

            Integration result. If the field is integrated in all directions, an
            ``np.ndarray`` is returned.

        Raises
        ------
        ValueError

            If ``cumulative=True`` and no integration direction is specified.

        Example
        -------
        1. Volume integral of a scalar field.

        .. math::

            \int_\mathrm{V} f(\mathbf{r}) \mathrm{d}V

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (2, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, nvdim=1, value=5)
        >>> f.integrate()
        array([5000.])

        2. Volume integral of a vector field.

        .. math::

            \int_\mathrm{V} \mathbf{f}(\mathbf{r}) \mathrm{d}V

        >>> f = df.Field(mesh, nvdim=3, value=(-1, -2, -3))
        >>> f.integrate()
        array([-1000., -2000., -3000.])

        3. Surface integral of a scalar field.

        .. math::

            \int_\mathrm{S} f(\mathbf{r}) |\mathrm{d}\mathbf{S}|

        >>> f = df.Field(mesh, nvdim=1, value=5)
        >>> f_plane = f.plane('z')
        >>> f_plane.integrate()
        array([500.])

        4. Surface integral of a vector field (flux). The dot product with the surface
        normal vector must be calculated manually.

        .. math::

            \int_\mathrm{S} \mathbf{f}(\mathbf{r}) \cdot \mathrm{d}\mathbf{S}

        >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> f_plane = f.plane('z')
        >>> e_z = [0, 0, 1]
        >>> f_plane.dot(e_z).integrate()
        array([300.])

        5. Integral along x-direction.

        .. math::

            \int_{x_\mathrm{min}}^{x_\mathrm{max}} \mathbf{f}(\mathbf{r}) \mathrm{d}x

        >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> f_plane = f.plane('z')
        >>> f_plane.integrate(direction='x').mean()
        array([10., 20., 30.])

        6. Cumulative integral along x-direction.

        .. math::

            \int_{x_\mathrm{min}}^{x} \mathbf{f}(\mathbf{r}) \mathrm{d}x'

        >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> f_plane = f.plane('z')
        >>> f_plane.integrate(direction='x', cumulative=True)
        Field(...)

        """
        if direction is None:
            if cumulative:
                raise ValueError(
                    "A cumulative integral can only computed along one direction."
                )
            sum_ = np.sum(self.array, axis=tuple(range(self.mesh.region.ndim)))
            dV = np.prod(self.mesh.cell)
            # NOTE the next 3 lines can be removed when the mesh is n dimensional
            if self.mesh.attributes["isplane"]:
                dV = (
                    self.mesh.cell[self.mesh.attributes["axis1"]]
                    * self.mesh.cell[self.mesh.attributes["axis2"]]
                )
            return sum_ * dV
        elif not isinstance(direction, str):
            raise TypeError("'direction' must be of type str.")
        elif direction not in self.mesh.region.dims:
            raise ValueError(f"{direction=} not in {self.mesh.region.dims}.")

        mesh = self.mesh

        axis = mesh.region.dims.index(direction)
        if cumulative:
            tmp_array = self.array / 2
            left_cells = dfu.assemble_index(slice(None), 3, {axis: slice(None, -1)})
            right_cells = dfu.assemble_index(slice(None), 3, {axis: slice(1, None)})
            tmp_array[right_cells] += np.cumsum(self.array, axis=axis)[left_cells]
            res_array = tmp_array * mesh.cell[axis]
        else:
            # NOTE reduce dimension n -> n-1:
            # - remove keepdims
            # - replace mesh.plane
            #   - either mesh.sel
            #   - or manually
            res_array = np.sum(self.array, axis=axis, keepdims=True) * mesh.cell[axis]
            mesh = mesh.plane(direction)

        # NOTE what should this method return for ndim == 0?
        # if mesh.region.ndim == 0:
        #     return res_array

        return self.__class__(mesh, nvdim=self.nvdim, value=res_array, vdims=self.vdims)

    def line(self, p1, p2, n=100):
        r"""Sample the field along the line.

        Given two points :math:`p_{1}` and :math:`p_{2}`, :math:`n` position
        coordinates are generated and the corresponding field values.

        .. math::

           \mathbf{r}_{i} = i\frac{\mathbf{p}_{2} -
           \mathbf{p}_{1}}{n-1}

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
        >>> field = df.Field(mesh, nvdim=2, value=(0, 3))
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
        >>> f = df.Field(mesh, nvdim=3, value=(0, 0, 1))
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
            plane_idx = self.mesh.point2index(plane_mesh.region.center)[p_axis]
            slices = tuple(
                slice(plane_idx, plane_idx + 1) if i == p_axis else slice(0, axis_len)
                for i, axis_len in enumerate(self.array.shape)
            )
            value = self.array[slices]
        return self.__class__(
            plane_mesh,
            nvdim=self.nvdim,
            value=value,
            vdims=self.vdims,
            unit=self.unit,
        )

    def resample(self, n):
        """Resample field.

        This method computes the field on a new mesh with ``n`` cells. The boundaries
        ``pmin`` and ``pmax`` stay unchanged. The values of the new cells are taken from
        the nearest old cell, no interpolation is performed.

        Parameters
        ----------
        n : array_like

            Number of cells in each direction. The number of elements must match
            field.mesh.region.ndim.

        Returns
        -------
        discretisedfield.Field

            The resampled field.

        Examples
        --------
        1. Decrease the number of cells.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> f = df.Field(mesh, dim=1, value=1)
        >>> f.mesh.n
        array([10, 10, 10])
        >>> down_sampled = f.resample((5, 5, 5))
        >>> down_sampled.mesh.n
        array([5, 5, 5])

        2. Increase the number of cells.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> cell = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> f = df.Field(mesh, dim=1, value=1)
        >>> f.mesh.n
        array([10, 10, 10])
        >>> up_sampled = f.resample((10, 15, 20))
        >>> up_sampled.mesh.n
        array([10, 15, 20])

        """
        mesh = df.Mesh(region=self.mesh.region, n=n)
        return self.__class__(
            mesh,
            nvdim=self.nvdim,
            value=self,
            vdims=self.vdims,
            unit=self.unit,
            dtype=self.dtype,
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
        >>> f = df.Field(mesh, nvdim=3, value=value_fun)
        >>> f.mean()
        array([0., 0., 0.])
        >>> f['r1']
        Field(...)
        >>> f['r1'].mean()
        array([1., 2., 3.])
        >>> f['r2'].mean()
        array([-1., -2., -3.])

        2. Extracting a subfield by passing a region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (-50e-9, -25e-9, 0)
        >>> p2 = (50e-9, 25e-9, 5e-9)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, cell=cell)
        >>> field = df.Field(mesh=mesh, nvdim=1, value=5)
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
            nvdim=self.nvdim,
            value=self.array[tuple(slices)],
            vdims=self.vdims,
            unit=self.unit,
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
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        ...
        >>> field.project('z')
        Field(...)
        >>> field.project('z').mean()
        array([1., 2., 3.])
        >>> field.project('z').array.shape
        (2, 2, 1, 3)

        """
        n_cells = self.mesh.n[dfu.axesdict[direction]]
        return self.integrate(direction=direction) / n_cells

    def angle(self, vector):
        r"""Angle between two vectors.

        It can be applied between two ``discretisedfield.Field`` objects.
        For a vector field, the second operand can be a vector in the form of
        an iterable, such as ``tuple``, ``list``,
        or ``numpy.ndarray``. If the second operand
        is a ``discretisedfield.Field`` object, both must be defined on the
        same mesh and have the same dimensions.
        This method then returns a scalar field which is an angle
        between the component of the vector field and a vector.
        The angle is computed in radians and all values are in :math:`(0,
        2\pi)` range.

        Parameters
        ----------
        other : discretisedfield.Field, numbers.Real, tuple, list, np.ndarray
            Second operand.

        Returns
        -------
        discretisedfield.Field

            Angle scalar field.

        Raises
        ------
        ValueError, TypeError

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
        >>> field = df.Field(mesh, nvdim=3, value=(0, 1, 0))
        ...
        >>> field.angle((1, 0, 0)).mean()
        array([1.57079633])

        """
        if isinstance(vector, self.__class__):
            self._check_same_mesh_and_field_dim(vector)
        elif self.nvdim == 1 and isinstance(vector, numbers.Complex):
            vector = self.__class__(self.mesh, nvdim=self.nvdim, value=vector)
        elif self.nvdim != 1 and isinstance(vector, (tuple, list, np.ndarray)):
            vector = self.__class__(self.mesh, nvdim=self.nvdim, value=vector)
        else:
            msg = (
                f"Unsupported operand type(s) for angle: {type(self)=} and"
                f" {type(vector)=}."
            )
            raise TypeError(msg)

        angle_array = np.arccos((self.dot(vector) / (self.norm * vector.norm)).array)

        return self.__class__(self.mesh, nvdim=1, value=angle_array)

    def write(self, *args, **kwargs):
        raise AttributeError("This method has been renamed to 'to_file'.")

    def to_file(
        self, filename, representation="bin8", extend_scalar=False, save_subregions=True
    ):
        """Write the field to OVF, HDF5, or VTK file.

        If the extension of ``filename`` is ``.vtk``, a VTK file is written
        (:py:func:`~discretisedfield.io.field_to_vtk`). The representation of the data
        (``'bin'`` [``'bin8'`` as equivalent], ``'txt'``, or ``'xml'``) is passed as
        ``'representation'``.

        For ``.ovf``, ``.omf``, or ``.ohf`` extensions, the field is saved to
        OVF file (:py:func:`~discretisedfield.io.field_to_ovf`). In that case,
        the representation of data (``'bin4'``, ``'bin8'``, or ``'txt'``) is
        passed as ``representation`` and if ``extend_scalar=True``, a scalar
        field will be saved as a vector field. More precisely, if the value at
        a cell is X, that cell will be saved as (X, 0, 0).

        Finally, if the extension of ``filename`` is ``.hdf5``, HDF5 file will
        be written (:py:func:`~discretisedfield.io.field_to_hdf5`).

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

        save_subregions : bool, optional

            If ``True`` and subregions are defined for the mesh the subregions will be
            saved to a json file. Defaults to ``True``.

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
        >>> field = df.Field(mesh, nvdim=3, value=(5, 6, 7))
        ...
        >>> filename = 'mytestfile.omf'
        >>> field.to_file(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.from_file(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        2. Write field to the VTK file.

        >>> filename = 'mytestfile.vtk'
        >>> field.to_file(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> os.remove(filename)  # delete the file

        3. Write field to the HDF5 file.

        >>> filename = 'mytestfile.hdf5'
        >>> field.to_file(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.from_file(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        See also
        --------
        ~discretisedfield.Field.from_file
        ~discretisedfield.io.field_to_ovf
        ~discretisedfield.io.field_to_vtk
        ~discretisedfield.io.field_to_hdf5

        """
        filename = pathlib.Path(filename)
        if filename.suffix in [".omf", ".ovf", ".ohf"]:
            io.field_to_ovf(
                self,
                filename,
                representation=representation,
                extend_scalar=extend_scalar,
                save_subregions=save_subregions,
            )
        elif filename.suffix in [".hdf5", ".h5"]:
            io.field_to_hdf5(self, filename, save_subregions=save_subregions)
        elif filename.suffix == ".vtk":
            io.field_to_vtk(
                self,
                filename,
                representation=representation,
                save_subregions=save_subregions,
            )
        else:
            raise ValueError(
                f"Writing file with extension {filename.suffix} not supported."
            )

    def to_vtk(self):
        """Convert field to vtk rectilinear grid.

        This method convers at `discretisedfield.Field` into a
        `vtk.vtkRectilinearGrid`. The field data (``field.array``) is stored as
        ``CELL_DATA`` of the ``RECTILINEAR_GRID``. Scalar fields (``nvdim=1``)
        contain one VTK array called ``field``. Vector fields (``nvdim>1``)
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

            If the field has ``nvdim>1`` and component labels are missing.

        Examples
        --------
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(1, 1, 1))
        >>> f = df.Field(mesh, nvdim=3, value=(0, 0, 1))
        >>> f_vtk = f.to_vtk()
        >>> print(f_vtk)
        vtkRectilinearGrid (...)
        ...
        >>> f_vtk.GetNumberOfCells()
        1000

        """
        if self.nvdim > 1 and self.vdims is None:
            raise AttributeError(
                "Field vdims must be assigned before converting to vtk."
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
        if self.nvdim > 1:
            # For some visualisation packages it is an advantage to have direct
            # access to the individual field components, e.g. for colouring.
            for comp in self.vdims:
                component_array = vns.numpy_to_vtk(
                    getattr(self, comp).array.transpose((2, 1, 0, 3)).reshape((-1))
                )
                component_array.SetName(f"{comp}-component")
                cell_data.AddArray(component_array)
        field_array = vns.numpy_to_vtk(
            self.array.transpose((2, 1, 0, 3)).reshape((-1, self.nvdim))
        )
        field_array.SetName("field")
        cell_data.AddArray(field_array)

        if self.nvdim == 3:
            cell_data.SetActiveVectors("field")
        elif self.nvdim == 1:
            cell_data.SetActiveScalars("field")
        return rgrid

    @classmethod
    def fromfile(cls, filename):
        raise AttributeError("This method has been renamed to 'from_file'.")

    @classmethod
    def from_file(cls, filename):
        """Read the field from an OVF (1.0 or 2.0), VTK, or HDF5 file.

        The extension of the ``filename`` should correspond to either:
            - OVF (``.ovf``, ``.omf``, ``.ohf``, ``.oef``)
            - VTK (``.vtk``), or
            - HDF5 (``.hdf5`` or ``.h5``).

        This method automatically determines the file type based on the file name
        extension.

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
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        2. Read a field from the VTK file.

        >>> filename = os.path.join(dirname, 'vtk-file.vtk')
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        3. Read a field from the HDF5 file.

        >>> filename = os.path.join(dirname, 'hdf5-file.hdf5')
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        See also
        --------
        ~discretisedfield.Field.to_file
        ~discretisedfield.io.field_from_ovf
        ~discretisedfield.io.field_from_vtk
        ~discretisedfield.io.field_from_hdf5

        """
        filename = pathlib.Path(filename)
        if filename.suffix in [".omf", ".ovf", ".ohf", ".oef"]:
            return io.field_from_ovf(filename)
        elif filename.suffix == ".vtk":
            return io.field_from_vtk(filename)
        elif filename.suffix in [".hdf5", ".h5"]:
            return io.field_from_hdf5(filename)
        else:
            raise ValueError(
                f"Reading file with extension {filename.suffix} not supported."
            )

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
            >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 0))
            >>> field.plane(z=50, n=(5, 5)).mpl()

        """
        return dfp.MplField(self)

    @property
    def k3d(self):
        """Plot interface, k3d based."""
        return dfp.K3dField(self)

    @property
    def hv(self):
        """Plot interface, Holoviews/hvplot based.

        This property provides access to the different plotting methods. It is
        also callable to quickly generate plots. For more details and the
        available methods refer to the documentation linked below.

        .. seealso::

            :py:func:`~discretisedfield.plotting.Hv.__call__`
            :py:func:`~discretisedfield.plotting.Hv.scalar`
            :py:func:`~discretisedfield.plotting.Hv.vector`
            :py:func:`~discretisedfield.plotting.Hv.contour`

        Examples
        --------

        1. Visualising the field using ``hv``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 0))
        >>> field.hv(kdims=['x', 'y'])
        :DynamicMap...

        """
        return dfp.Hv(self._hv_key_dims, self._hv_data_selection, self._hv_vdims_guess)

    def _hv_data_selection(self, **kwargs):
        """Select field part as specified by the input arguments."""
        comp = kwargs.pop("comp") if "comp" in kwargs else None
        res = self.to_xarray().sel(**kwargs, method="nearest")
        if comp:
            res = res.sel(comp=comp)
        return res

    def _hv_vdims_guess(self, kdims):
        """Try to find vector components matching the given kdims."""
        mesh_dims = "xyz"
        if len(mesh_dims) != self.nvdim:
            return None
        vdims = []
        for dim in kdims:
            if dim not in mesh_dims:  # hard-coded names in Mesh
                return None
            vdims.append(self.vdims[mesh_dims.index(dim)])
        return vdims

    @property
    def _hv_key_dims(self):
        """Dict of key dimensions of the field.

        Keys are the field dimensions (domain and vector space, e.g. x, y, z, comp) that
        have length > 1. Values are named_tuples ``hv_key_dim(data, unit)`` that contain
        the data (which has to fulfill len(data) > 1, typically as a numpy array or
        list) and the unit of a string (empty string if there is no unit).

        """
        mesh_dims = "xyz"
        key_dims = {
            dim: hv_key_dim(coords, "m")
            for dim in mesh_dims
            if len(coords := getattr(self.mesh.points, dim)) > 1
        }
        if self.nvdim > 1:
            key_dims["comp"] = hv_key_dim(self.vdims, "")
        return key_dims

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
        for idx in range(self.nvdim):
            ft = np.fft.fftshift(np.fft.fftn(self.array[..., idx].squeeze()))
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            nvdim=len(values),
            value=np.stack(values, axis=3),
            vdims=self.vdims,
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
        for idx in range(self.nvdim):
            ft = np.fft.ifftn(np.fft.ifftshift(self.array[..., idx].squeeze()))
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            nvdim=len(values),
            value=np.stack(values, axis=3),
            vdims=self.vdims,
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
        for idx in range(self.nvdim):
            array = self.array[..., idx].squeeze()
            # no shifting for the last axis
            ft = np.fft.fftshift(np.fft.rfftn(array), axes=range(len(array.shape) - 1))
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            nvdim=len(values),
            value=np.stack(values, axis=3),
            vdims=self.vdims,
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
        for idx in range(self.nvdim):
            array = self.array[..., idx].squeeze()
            ft = np.fft.irfftn(
                np.fft.ifftshift(array, axes=range(len(array.shape) - 1)),
                s=[i for i in mesh.n if i > 1],
            )
            values.append(ft.reshape(mesh.n))

        return self.__class__(
            mesh,
            nvdim=len(values),
            value=np.stack(values, axis=3),
            vdims=self.vdims,
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
            nvdim=self.nvdim,
            value=self.array.real,
            vdims=self.vdims,
            unit=self.unit,
        )

    @property
    def imag(self):
        """Imaginary part of complex field."""
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=self.array.imag,
            vdims=self.vdims,
            unit=self.unit,
        )

    @property
    def phase(self):
        """Phase of complex field."""
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=np.angle(self.array),
            vdims=self.vdims,
        )

    @property
    def abs(self):
        """Absolute value of complex field."""
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=np.abs(self.array),
            vdims=self.vdims,
        )

    @property
    def conjugate(self):
        """Complex conjugate of complex field."""
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=self.array.conjugate(),
            vdims=self.vdims,
            unit=self.unit,
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
                    nvdim=x.shape[-1],
                    value=x,
                    vdims=self.vdims,
                )
                for x, m in zip(result, mesh)
            )
        elif method == "at":
            return None
        else:
            return self.__class__(
                mesh[0],
                nvdim=result.shape[-1],
                value=result,
                vdims=self.vdims,
            )

    def to_xarray(self, name="field", unit=None):
        """Field value as ``xarray.DataArray``.

        The function returns an ``xarray.DataArray`` with dimensions ``x``,
        ``y``, ``z``, and ``comp`` (only if ``field.nvdim > 1``). The coordinates
        of the geometric dimensions are derived from ``self.mesh.points``,
        and for vector field components from ``self.vdims``. Addtionally,
        the values of ``self.mesh.cell``, ``self.mesh.region.pmin``, and
        ``self.mesh.region.pmax`` are stored as ``cell``, ``pmin``, and ``pmax``
        attributes of the DataArray. The ``unit`` attribute of geometric
        dimensions is set to the respective strings in ``self.mesh.region.units``.

        The name and unit of the field ``DataArray`` can be set by passing
        ``name`` and ``unit``. If the type of value passed to any of the two
        arguments is not ``str``, then a ``TypeError`` is raised.

        Parameters
        ----------
        name : str, optional

            String to set name of the field ``DataArray``.

        unit : str, optional

            String to set units of the field ``DataArray``.

        Returns
        -------
        xarray.DataArray

            Field values DataArray.

        Raises
        ------
        TypeError

            If either ``name`` or ``unit`` argument is not a string.

        Examples
        --------
        1. Create a field

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 10, 10)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(1, 0, 0), norm=1.)
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

        if unit is not None and not isinstance(unit, str):
            msg = "Unit argument must be a string."
            raise TypeError(msg)

        axes = ["x", "y", "z"]

        data_array_coords = {axis: getattr(self.mesh.points, axis) for axis in axes}

        geo_units_dict = dict(zip(axes, self.mesh.region.units))

        if self.nvdim > 1:
            data_array_dims = axes + ["comp"]
            if self.vdims is not None:
                data_array_coords["comp"] = self.vdims
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
                units=unit or self.unit,
                cell=self.mesh.cell,
                pmin=self.mesh.region.pmin,
                pmax=self.mesh.region.pmax,
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
        >>> field.mean()
        array([1., 1., 1.])

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
            xa.attrs["pmin"]
            if "pmin" in xa.attrs
            else [xa[i].values[0] - c / 2 for i, c in zip("xyz", cell)]
        )
        p2 = (
            xa.attrs["pmax"]
            if "pmax" in xa.attrs
            else [xa[i].values[-1] + c / 2 for i, c in zip("xyz", cell)]
        )

        if any("units" not in xa[i].attrs for i in "xyz"):
            mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        else:
            print(xa["x"].units)
            region = df.Region(p1=p1, p2=p2, units=[xa[i].units for i in "xyz"])
            mesh = df.Mesh(region=region, cell=cell)

        comp = xa.comp.values if "comp" in xa.coords else None
        val = np.expand_dims(xa.values, axis=-1) if xa.ndim == 3 else xa.values
        nvdim = 1 if xa.ndim == 3 else val.shape[-1]
        return cls(mesh=mesh, nvdim=nvdim, value=val, vdims=comp, dtype=xa.values.dtype)


@functools.singledispatch
def _as_array(val, mesh, nvdim, dtype):
    raise TypeError("Unsupported type {type(val)}.")


# to avoid str being interpreted as iterable
@_as_array.register(str)
def _(val, mesh, nvdim, dtype):
    raise TypeError("Unsupported type {type(val)}.")


@_as_array.register(numbers.Complex)
@_as_array.register(collections.abc.Iterable)
def _(val, mesh, nvdim, dtype):
    if isinstance(val, numbers.Complex) and nvdim > 1 and val != 0:
        raise ValueError(
            f"Wrong dimension 1 provided for value; expected dimension is {nvdim}"
        )
    dtype = dtype or max(np.asarray(val).dtype, np.float64)
    return np.full((*mesh.n, nvdim), val, dtype=dtype)


@_as_array.register(collections.abc.Callable)
def _(val, mesh, nvdim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    array = np.empty((*mesh.n, nvdim), dtype=dtype)
    for index, point in zip(mesh.indices, mesh):
        array[index] = val(point)
    return array


@_as_array.register(Field)
def _(val, mesh, nvdim, dtype):
    if mesh.region not in val.mesh.region:
        raise ValueError(
            f"{val.mesh.region} of the provided field does not "
            f"contain {mesh.region} of the field that is being "
            "created."
        )
    value = (
        val.to_xarray()
        .sel(x=mesh.points.x, y=mesh.points.y, z=mesh.points.z, method="nearest")
        .data
    )
    if nvdim == 1:
        # xarray dataarrays for scalar data are three dimensional
        return value.reshape(*mesh.n, -1)
    return value


@_as_array.register(dict)
def _(val, mesh, nvdim, dtype):
    # will only be called on user input
    # dtype must be specified by the user for complex values
    dtype = dtype or np.float64
    fill_value = (
        val["default"] if "default" in val and not callable(val["default"]) else np.nan
    )
    array = np.full((*mesh.n, nvdim), fill_value, dtype=dtype)

    for subregion in reversed(mesh.subregions.keys()):
        # subregions can overlap, first subregion takes precedence
        try:
            submesh = mesh[subregion]
            subval = val[subregion]
        except KeyError:
            continue
        else:
            slices = mesh.region2slices(submesh.region)
            array[slices] = _as_array(subval, submesh, nvdim, dtype)

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
