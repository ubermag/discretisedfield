import collections
import functools
import numbers

import numpy as np
import scipy.fft as spfft
import xarray as xr
from vtkmodules.util import numpy_support as vns
from vtkmodules.vtkCommonDataModel import vtkRectilinearGrid

import discretisedfield as df
import discretisedfield.plotting as dfp
import discretisedfield.util as dfu
from discretisedfield.operators import _split_diff_combine
from discretisedfield.plotting.util import hv_key_dim

from . import html
from .io import _FieldIO

# TODO: tutorials, line operations


class Field(_FieldIO):
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

        Number of Value dimensions of the field. For instance, if `nvdim=3` the field is
        a three-dimensional vector field and for `nvdim=1` the field is a scalar field.

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

    valid : numpy.ndarray, str, optional

        Property used to mask invalid values of the field. Please refer to
        ``discretisedfield.Field.valid``. Defaults to ``True``.

    vdim_mapping : dict, optional

        Dictionary that maps value dimensions to the spatial dimensions. It defaults to
        the same as spatial dimensions if the number of value and spatial dimensions are
        the same, else it is empty.

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

    __slots__ = [
        "_array",
        "_mesh",
        "_nvdim",
        "_unit",
        "_valid",
        "_vdim_mapping",
        "_vdims",
        "dtype",
    ]

    # removed attribute: new method/property
    # implemented in __getattr__
    # to exclude methods from tap completion and documentation
    _removed_attributes = {
        "average": "mean",
        "integral": "integrate",
        "project": "mean",
        "value": "update_field_values",
        "write": "to_file",  # method is in io.__init__
    }

    def __init__(
        self,
        mesh,
        nvdim=None,
        value=0.0,
        norm=None,
        vdims=None,
        dtype=None,
        unit=None,
        valid=True,
        vdim_mapping=None,
        **kwargs,
    ):
        if not isinstance(mesh, df.Mesh):
            raise TypeError("'mesh' must be of class discretisedfield.Mesh.")
        self._mesh = mesh

        if not isinstance(nvdim, numbers.Integral):
            raise TypeError("'nvdim' must be of type int.")
        elif nvdim < 1:
            raise ValueError("'nvdim' must be greater than zero.")
        self._nvdim = nvdim

        self.dtype = dtype

        self.unit = unit

        # This is required for correct initialisation when also using a
        # norm. The norm setter requires the norm property
        # (which requires valid). However, valid cannot be set
        # before the norm is set as the valid setter has the option
        # to set valid based on the norm.
        self.valid = True
        self.update_field_values(value)
        self.norm = norm
        self.valid = valid

        # required in here for correct initialisation:
        self._vdims = None  # the vdims setter reads self.vdims
        self._vdim_mapping = {}  # the vdims setter reads self.vdim_mapping
        self.vdims = vdims

        self.vdim_mapping = vdim_mapping

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
        self.array = self._as_array(value, self.mesh, self.nvdim, dtype=self.dtype)

    @property
    def vdims(self):
        """Vector components of the field."""
        return self._vdims

    @vdims.setter
    def vdims(self, vdims):
        if vdims is None:
            if 2 <= self.nvdim <= 3:
                vdims = ["x", "y", "z"][: self.nvdim]
            elif self.nvdim > 3:
                vdims = [f"v{i}" for i in range(self.nvdim)]
        elif not isinstance(vdims, (list, tuple, np.ndarray)) or any(
            not isinstance(vdim, str) for vdim in vdims
        ):
            raise TypeError(f"vdims must be a sequence of strings, not {type(vdims)=}.")
        elif len(vdims) == 0:
            vdims = None
        else:
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
            vdims = list(vdims)

        # setting vdim_mapping reads self.vdims -> self.vdims has to be updated before
        # updating self.vdim_mapping
        old_vdims = self._vdims
        self._vdims = vdims

        if len(self.vdim_mapping) > 0 and vdims is not None and old_vdims is not None:
            # update vdim mapping with new vdim names
            self.vdim_mapping = {
                new_vdim: self.vdim_mapping[old_vdim]
                for new_vdim, old_vdim in zip(vdims, old_vdims)
            }

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
        self._array = self._as_array(val, self.mesh, self.nvdim, dtype=self.dtype)

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

        return self.__class__(
            self.mesh, nvdim=1, value=res, unit=self.unit, valid=self.valid
        )

    @norm.setter
    def norm(self, val):
        if val is not None:
            self.array = np.divide(
                self.array,
                self.norm.array,
                out=np.zeros_like(self.array),
                where=self.norm.array != 0.0,
            )
            self.array *= self._as_array(val, self.mesh, nvdim=1, dtype=None)

    @property
    def valid(self):
        """Valid field values.

        This property is used to mask invalid field values.
        This can be achieved by passing ``numpy.ndarray`` of
        the same shape as the field array with boolean values,
        the string ``"norm"`` (which masks zero values), or
        None (which sets all values to True).

        """
        return self._valid

    @valid.setter
    def valid(self, valid):
        if valid is not None:
            if isinstance(valid, str) and valid == "norm":
                valid = ~np.isclose(self.norm.array, 0)
        else:
            valid = True
        # Using self._as_array creates an array with shape (*mesh.n, 1).
        # We only want a shape of mesh.n so we can directly use it
        # to index field.array i.e. field.array[field.valid].
        self._valid = self._as_array(valid, self.mesh, nvdim=1, dtype=bool)[..., 0]

    @property
    def _valid_as_field(self):
        return self.__class__(self.mesh, nvdim=1, value=self.valid, dtype=bool)

    @property
    def vdim_mapping(self):
        """Map vdims to dims."""
        return self._vdim_mapping

    @vdim_mapping.setter
    def vdim_mapping(self, vdim_mapping):
        if vdim_mapping is None:
            if self.nvdim == 1:
                vdim_mapping = {}
            elif self.nvdim == self.mesh.region.ndim:
                vdim_mapping = dict(zip(self.vdims, self.mesh.region.dims))
            else:
                vdim_mapping = {}
        elif not isinstance(vdim_mapping, dict):
            raise TypeError(f"Invalid {type(vdim_mapping)=}; must be of type 'dict'.")
        elif len(vdim_mapping) == 1 and self.nvdim == 1 and self.vdims is None:
            # no mapping for scalar fields unless vdims is set manually
            # (there is no default vdims for scalar fields)
            vdim_mapping = {}
        elif len(vdim_mapping) > 0 and sorted(vdim_mapping) != sorted(self.vdims):
            raise ValueError(
                f"Invalid {vdim_mapping.keys()=}; keys must be {self.vdims}."
            )

        self._vdim_mapping = vdim_mapping

    @property
    def _r_dim_mapping(self):
        """Map dims to vdims."""
        reversed_mapping = {val: key for key, val in self.vdim_mapping.items()}
        return {dim: reversed_mapping.get(dim, None) for dim in self.mesh.region.dims}

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
            self.mesh,
            nvdim=self.nvdim,
            value=np.abs(self.array),
            unit=self.unit,
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
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
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
        )

    def mean(self, direction=None):
        """Field mean.

        It computes the arithmetic mean along the specified direction of the field.

        It returns a numpy array containing the mean value if all the geometrical
        directions or none are selected. If one or more than one directions (and less
        than ``region.dims``) are selected, the method returns a field of appropriate
        geometric dimensions with the calculated mean value.


        Parameters
        ----------
        direction : None, string or tuple of strings, optional.

            Directions along which
            the mean is computed. The default is to
            compute the mean of the entire volume and return an array of the
            averaged vector components.


        Returns
        -------
        numpy.ndarray

            Field average along all the geometrical directions combined.

        discretisedfield.Field

            Field of reduced geometrical dimensions holding the mean value along the
            selected direction(s).

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

        3. Computing the vector field mean along x direction.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(0, 0, 1))
        >>> field.mean(direction='x')((0.5, 0.5))
        array([0., 0., 1.])

        4. Computing the vector field mean along x and y directions.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> field = df.Field(mesh=mesh, nvdim=3, value=(0, 0, 1))
        >>> field.mean(direction=['x', 'y'])(0.5)
        array([0., 0., 1.])

        """
        # Mean over all directions implicitly.
        if direction is None:
            return self.array.mean(axis=tuple(range(self.mesh.region.ndim)))
        elif isinstance(direction, (tuple, list)):
            if len(direction) != len(set(direction)):
                raise ValueError("Duplicate directions are not allowed.")
            # Mean over all directions explicitly.
            if sorted(direction) == sorted(self.mesh.region.dims):
                return self.array.mean(axis=tuple(range(self.mesh.region.ndim)))
            else:
                # Multiple directions mean
                mesh = self.mesh
                axis = np.zeros(len(direction), dtype=int)
                for i, d in enumerate(direction):
                    mesh = mesh.sel(d)
                    axis[i] = self.mesh.region._dim2index(d)
                array = self.array.mean(axis=tuple(axis))
                return self.__class__(
                    mesh,
                    nvdim=self.nvdim,
                    value=array,
                    unit=self.unit,
                    vdims=self.vdims,
                    vdim_mapping=self.vdim_mapping,
                )
        elif isinstance(direction, str):
            axis = self.mesh.region._dim2index(direction)
            return self.__class__(
                self.mesh.sel(direction),
                nvdim=self.nvdim,
                value=self.array.mean(axis=axis),
                vdims=self.vdims,
                unit=self.unit,
                vdim_mapping=self.vdim_mapping,
            )
        else:
            raise ValueError(
                "Direction must be None, string or tuple of strings, not"
                f" {type(direction)}."
            )

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
        point : array_like

            For example, in three dimensions, the mesh point coordinate
            :math:`\mathbf{p} = (p_{x}, p_{y}, p_{z})`.

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
        if attr in self._removed_attributes:
            raise AttributeError(
                f"'{attr}' has been removed; use '{self._removed_attributes[attr]}'"
                " instead."
            )
        if self.vdims is not None and attr in self.vdims:
            attr_array = self.array[..., self.vdims.index(attr), np.newaxis]
            try:
                vdim_mapping = {attr: self.vdim_mapping[attr]}
            except KeyError:
                vdim_mapping = {}
            return self.__class__(
                mesh=self.mesh,
                nvdim=1,
                value=attr_array,
                unit=self.unit,
                valid=self.valid,
                vdim_mapping=vdim_mapping,
            )
        else:
            raise AttributeError(f"Object has no attribute {attr}.")

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

        return dirlist

    def __iter__(self):
        r"""Generator yielding field values of all discretisation cells.

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
        [0.5 0.5 0.5] [0. 0. 1.]
        [1.5 0.5 0.5] [0. 0. 1.]
        [0.5 1.5 0.5] [0. 0. 1.]
        [1.5 1.5 0.5] [0. 0. 1.]

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

          2. They have the same number of value dimensions (``nvdim``).

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

    def allclose(self, other, rtol=1e-5, atol=1e-8):
        """Allclose method.

        This method determines whether two fields are:

          1. Defined on the same mesh.

          2. Have the same number of value dimension (``nvdim``).

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

        if self.mesh.allclose(other.mesh) and self.nvdim == other.nvdim:
            # Order matters absolute(a - b) <= (atol + rtol * absolute(b))
            # The above equation is not symmetric in a and b, so that allclose(a, b)
            # might be different from allclose(b, a)
            # We want it relative to the original array
            return np.allclose(other.array, self.array, rtol=rtol, atol=atol)
        else:
            return False

    def is_same_vectorspace(self, other):  # TODO: check vdims
        if not isinstance(other, self.__class__):
            raise TypeError(f"Object of type {type(other)} not supported.")
        return self.nvdim == other.nvdim

    def _check_same_mesh_and_field_dim(self, other, ignore_scalar=False):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Object of type {type(other)} not supported.")

        if not self.mesh.allclose(other.mesh):
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
        valid = self.valid
        if isinstance(other, self.__class__):
            self._check_same_mesh_and_field_dim(other, ignore_scalar=True)
            valid = np.logical_and(valid, other.valid)
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
            valid=valid,
            vdim_mapping=self.vdim_mapping,
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
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
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

        1. Two fields with equal vector dimensions ``nvdim``,

        2. A field of any dimension ``nvdim`` and ``numbers.Complex``,

        3. A field of any dimension ``nvdim`` and a scalar (``nvdim=1``) field.

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

        1. Two fields with equal vector dimensions ``nvdim``,

        2. A field of any dimension ``nvdim`` and ``numbers.Complex``,

        3. A field of any dimension ``nvdim`` and a scalar (``nvdim=1``) field.

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
        >>> (f1 / f2).allclose((f2 / f1)**(-1))
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
        valid = self.valid
        if isinstance(other, self.__class__):
            self._check_same_mesh_and_field_dim(other)
            valid = np.logical_and(valid, other.valid)
            other = other.array
        elif not isinstance(other, (tuple, list, np.ndarray)):
            msg = (
                f"Unsupported operand type(s) for dot product: {type(self)=} and"
                f" {type(other)=}."
            )
            raise TypeError(msg)

        res_array = np.einsum("...l,...l->...", self.array, other)
        return self.__class__(
            self.mesh, nvdim=1, value=res_array[..., np.newaxis], valid=valid
        )

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
        valid = self.valid
        if isinstance(other, self.__class__):
            self._check_same_mesh_and_field_dim(other)
            if self.nvdim != 3 or other.nvdim != 3:
                msg = (
                    f"Cannot apply cross product on {self.nvdim=} and"
                    f" {other.nvdim=} fields."
                )
                raise ValueError(msg)
            valid = np.logical_and(valid, other.valid)
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
            valid=valid,
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

            List of ``discretisedfield.Field`` objects, each with ``nvdim=1``.

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
        valid = self.valid
        if isinstance(other, self.__class__):
            if self.mesh != other.mesh:
                msg = "Cannot apply operator << on fields defined on different meshes."
                raise ValueError(msg)
            valid = np.logical_and(valid, other.valid)
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

        vdim_mapping = self.vdim_mapping.copy()
        vdim_mapping.update(other.vdim_mapping)
        if len(vdim_mapping) != len(array_list):
            # keys are missing or not unique -> the user has to set the mapping manually
            vdim_mapping = None

        return self.__class__(
            self.mesh,
            nvdim=len(array_list),
            value=np.stack(array_list, axis=-1),
            vdims=vdims,
            valid=valid,
            vdim_mapping=vdim_mapping,
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
            d[self.mesh.region._dim2index(key)] = value
        padding_sequence = dfu.assemble_index((0, 0), len(self.array.shape), d)
        padded_array = np.pad(self.array, padding_sequence, mode=mode, **kwargs)

        padding_sequence = dfu.assemble_index((0, 0), len(self.valid.shape), d)
        padded_valid = np.pad(self.valid, padding_sequence, mode=mode, **kwargs)
        padded_mesh = self.mesh.pad(pad_width)

        return self.__class__(
            padded_mesh,
            nvdim=self.nvdim,
            value=padded_array,
            vdims=self.vdims,
            unit=self.unit,
            vdim_mapping=self.vdim_mapping,
            valid=padded_valid,
        )

    def _diff_old(self, direction, order=1):
        """Dreprecated directional derivative.

        This method uses second order accurate finite difference stencils by default
        unless the field is defined on a mesh with too few cells in the differential
        direction. In this case the first order accurate finite difference stencils
        are used at the boundaries and the second order accurate finite difference
        stencils are used in the interior.

        Computing of the directional derivative depends
        strongly on the boundary condition specified in the mesh on which the
        field is defined on. More precisely, the values of the derivatives at
        the boundary are different for periodic, Neumann, dirichlet, or no boundary
        conditions. For details on boundary conditions, please refer to the
        ``disretisedfield.Mesh`` class.

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

        """
        # We tried using FinDiff for calculating the derivatives, but there
        # was a problem with the boundary conditions. We have implemented
        # differential operators using slicing. This is not the most efficient
        # way, and we should consider using ndimage.convolve in the future.

        direction_idx = self.mesh.region._dim2index(direction)

        # If there are no neighbouring cells in the specified direction, zero
        # field is returned.
        # Directional derivative cannot be computed if less or an equal number of
        # discretisation cells exists in a specified direction than the order.
        # In that case, a zero field is returned.
        if self.mesh.n[direction_idx] <= order:
            return self.__class__(
                self.mesh,
                nvdim=self.nvdim,
                vdims=self.vdims,
                unit=self.unit,
                valid=self.valid,
                vdim_mapping=self.vdim_mapping,
            )

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
                # The derivative is computed using forward/backward difference
                # as the array is too small to use central difference.
                # This is first order accurate.
                derivative_array = np.gradient(
                    padded_array, self.mesh.cell[direction_idx], axis=direction_idx
                )
            else:
                if self.mesh.bc == "":
                    # If there is no boundary condition, the derivative is
                    # computed using central difference in the interior and
                    # forward/backward difference at the boundaries.
                    # All of these finite difference methods are second-order
                    # accurate.
                    # These stencil coefficients are taken from FinDiff.
                    # https://findiff.readthedocs.io/en/latest/
                    # Pad with specific values so that the same finite difference
                    # stencil can be used across the whole array
                    # The example for forward difference is shown below with the
                    # 0 index for the cell at the start of the array and the
                    # p index for the for the padded cell which is positioned
                    # at before index 0.
                    #
                    # central difference = forward difference
                    # 0.5 * f(1) - 0.5 * f(p) = - 0.5 f(2) + 2 f(1) - 1.5 f(0)
                    # f(p) = f(2) - 3 f(1) + 3 f(0)
                    def pad_fun(vector, pad_width, iaxis, kwargs):
                        # The incides of the padded array is
                        # Original array -> Padded array
                        # f(p) -> f(0)
                        # f(0) -> f(1)
                        # f(1) -> f(2)
                        # f(2) -> f(3)

                        if iaxis == direction_idx:
                            vector[0] = vector[3] - 3 * vector[2] + 3 * vector[1]
                            vector[-1] = vector[-4] - 3 * vector[-3] + 3 * vector[-2]

                    pad_width = [(0, 0)] * 4
                    pad_width[direction_idx] = (1, 1)
                    padded_array = np.pad(padded_array, pad_width, pad_fun)

                index_p1 = dfu.assemble_index(
                    slice(None), 4, {direction_idx: slice(2, None)}
                )
                index_m1 = dfu.assemble_index(
                    slice(None), 4, {direction_idx: slice(None, -2)}
                )
                derivative_array = (
                    0.5 * padded_array[index_p1] - 0.5 * padded_array[index_m1]
                ) / self.mesh.cell[direction_idx]

        elif order == 2:
            # The derivative is computed using the central difference
            # with forward/backward difference at the boundaries.
            if self.mesh.bc == "":
                if self.mesh.n[direction_idx] < 4:
                    # Pad with specific values so that the same finite difference
                    # stencil can be used across the whole array
                    # See comments in order == 1 for more details.
                    # central difference = forward difference
                    # f(1) + f(p) - 2 f(0) = f(2) + f(0) - 2 f(1)
                    # f(p) = f(2) - 3 f(1) + 3f(0)

                    def pad_fun(vector, pad_width, iaxis, kwargs):
                        if iaxis == direction_idx:
                            vector[0] = vector[3] - 3 * vector[2] + 3 * vector[1]
                            vector[-1] = vector[-4] - 3 * vector[-3] + 3 * vector[-2]

                else:
                    # The derivative is computed using accuracy of 2 everywhere
                    # Pad with specific values so that the same finite difference
                    # stencil can be used across the whole array
                    # See comments in order == 1 for more details.
                    # central difference = forward difference
                    # f(1) + f(p) - 2 f(0) =  - f(3) + 4 f(2) - 5 f(1) + 2 f(0)
                    # f(p) = - f(3) + 4 f(2) - 6 f(1) + 4 f(0)

                    def pad_fun(vector, pad_width, iaxis, kwargs):
                        if iaxis == direction_idx:
                            vector[0] = (
                                -vector[4]
                                + 4 * vector[3]
                                - 6 * vector[2]
                                + 4 * vector[1]
                            )
                            vector[-1] = (
                                -vector[-5]
                                + 4 * vector[-4]
                                - 6 * vector[-3]
                                + 4 * vector[-2]
                            )

                pad_width = [(0, 0)] * 4
                pad_width[direction_idx] = (1, 1)
                padded_array = np.pad(padded_array, pad_width, pad_fun)

            index_p1 = dfu.assemble_index(
                slice(None), 4, {direction_idx: slice(2, None)}
            )
            index_0 = dfu.assemble_index(slice(None), 4, {direction_idx: slice(1, -1)})
            index_m1 = dfu.assemble_index(
                slice(None), 4, {direction_idx: slice(None, -2)}
            )
            derivative_array = (
                padded_array[index_p1]
                - 2 * padded_array[index_0]
                + padded_array[index_m1]
            ) / self.mesh.cell[direction_idx] ** 2

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
            unit=self.unit,
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
        )

    def diff(self, direction, order=1, restrict2valid=True):
        """Directional derivative.

        This method computes a directional derivative of the field and returns
        a field. The direction in which the derivative is computed is passed
        via ``direction`` argument, which can be any element of ``region.dims``.
        The order of the computed derivative can be 1 or 2 and it is specified
        using argument ``order`` and it defaults to 1.

        This method uses second order accurate finite difference stencils by default
        unless the field is defined on a mesh with too few cells in the differential
        direction. In this case the first order accurate finite difference stencils
        are used at the boundaries and the second order accurate finite difference
        stencils are used in the interior.

        Directional derivative cannot be computed if less or equal discretisation
        cells exists in a specified direction than the order.
        In that case, a zero field is returned.

        By default, the directional derivative is computed only across contiguous
        areas of the field where the field is valid. This behaviour can be
        changed to compute the directional derivative across the whole field
        by setting ``restrict2valid`` to ``False``.

        Computing of the directional derivative depends
        strongly on the boundary condition specified. In this method,
        only periodic boundary conditions at the edges of the region
        are supported. To enable periodic boundary conditions, set ``mesh.bc``.

        Parameters
        ----------
        direction : str

            The spatial direction in which the derivative is computed.

        order : int

            The order of the derivative. It can be 1 or 2 and it defaults to 1.

        restrict2valid : bool

            If ``True``, the directional derivative is computed only across
            contiguous areas of the field where the field is valid. If ``False``,
            the directional derivative is computed across the whole field.
            The default value is ``True``.


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

        3. Compute the second-order directional derivative of a scalar field with
        which is only valid below x=5.

        >>> def value_fun(point):
        ...     x = point
        ...     return x
        ...
        >>> def valid_fun(point):
        ...     x = point
        ...     return x < 5
        ...
        >>> mesh = df.Mesh(p1=0, p2=10, cell=1)
        >>> f = df.Field(mesh, nvdim=1, value=value_fun, valid=valid_fun)
        >>> f.diff('x', order=1).array.tolist()
        [[1.0], [1.0], [1.0], [1.0], [1.0], [0.0], [0.0], [0.0], [0.0], [0.0]]

        """
        # Check order of derivative
        if order not in (1, 2):
            raise NotImplementedError(f"Derivative of {order=} is not implemented.")

        direction_idx = self.mesh.region._dim2index(direction)

        # If direction is periodic pad the field
        if direction in self.mesh.bc:
            field = self.pad({direction: (1, 1)}, mode="wrap")
        else:
            field = self

        # Use only valid values for the derivative if restrict2valid is True
        # or use all values if restrict2valid is False
        valid = field.valid if restrict2valid else np.ones_like(field.valid, dtype=bool)

        out = np.zeros_like(field.array)

        # Loop over all cells in the plane perpendicular to the direction of the
        # derivative. Use sel method in a way that keeps the dimensionality but
        # only returns a single layer in the direction of the derivative
        point = field.mesh.region.pmin[direction_idx]
        for idx in field.mesh.sel(**{direction: (point, point)}).indices:
            idx = list(idx)
            idx[direction_idx] = slice(None)
            valid_arr = valid[tuple(idx)]
            # Loop over all value dimensions of the vector field and
            # compute the derivative for each value dimension.
            for dim in range(field.nvdim):
                out[tuple([*idx, dim])] = _split_diff_combine(
                    field.array[tuple([*idx, dim])],
                    valid_arr,
                    order,
                    field.mesh.cell[direction_idx],
                )

        # Remove the padding if periodic is True
        if direction in self.mesh.bc:
            slices = field.mesh.region2slices(self.mesh.region)
            out = out[slices]

        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=out,
            vdims=self.vdims,
            unit=self.unit,
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
        )

    @property
    def grad(self):
        r"""Gradient.

        This method computes the gradient of a scalar (``nvdim=1``) field and
        returns a vector field with the same number of dimensions as the space
        on which the scalar field is defined (``ndim``):

        .. math::

            \nabla f = (\frac{\partial f}{\partial x_1},
                        ...
                        \frac{\partial f}{\partial x_ndim}

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
        >>> import numpy as np
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (10e-9, 10e-9, 10e-9)
        >>> cell = (2e-9, 2e-9, 2e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        ...
        >>> f = df.Field(mesh, nvdim=1, value=5)
        >>> np.allclose(f.grad.mean(), 0, atol=1e-06)
        True

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

        # Create a list of derivatives for each dimension
        derivatives = [self.diff(dim) for dim in self.mesh.region.dims]

        result = derivatives[0]
        for derivative in derivatives[1:]:
            result = result << derivative

        return result

    @property
    def div(self):
        r"""Compute the divergence of a field.

        This method calculates the divergence of a field of dimension `nvdim`
        and returns a scalar (``nvdim=1``) field as a result.

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

            If the field and the mesh don't have the same dimentionality or
            they are not mapped correctly.

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

        .. seealso:: :py:func:`~discretisedfield.Field.derivative`

        """
        if self.nvdim != self.mesh.region.ndim:
            raise ValueError(
                f"Cannot compute divergence for field with a differnt {self.nvdim=} and"
                f" {self.mesh.region.ndim}."
            )

        for vdim in self.vdims:
            if vdim not in self.vdim_mapping:
                raise ValueError(
                    f"Cannot compute divergence for field as {vdim} is not present in"
                    f"{self.vdim_mapping=}."
                )
            elif self.vdim_mapping[vdim] not in self.mesh.region.dims:
                raise ValueError(
                    f"Cannot compute divergence for field as {self.vdim_mapping[vdim]}"
                    f"is not present in {self.mesh.region.dims=}."
                )

        return sum(
            getattr(self, vdim).diff(self.vdim_mapping[vdim]) for vdim in self.vdims
        )

    @property
    def curl(self):
        r"""Curl.

        This method computes the curl of a three dimensional vector (``nvdim=3``)
        field in three spatial dimensions (``ndim=3``) and returns
        a three dimensional vector (``nvdim=3``) field in three spatial
        dimensions (``ndim=3``)

        .. math::

            \nabla \times \mathbf{v} = \left(\frac{\partial
            v_{2}}{\partial x_{1}} - \frac{\partial v_{1}}{\partial x_{2}},
            \frac{\partial v_{0}}{\partial x_{2}} - \frac{\partial
            v_{2}}{\partial x_{0}}, \frac{\partial v_{1}}{\partial x_{0}} -
            \frac{\partial v_{0}}{\partial x_{1}},\right)

        Directional derivative cannot be computed if only one discretisation
        cell exists in a certain direction. In that case, a zero field is
        considered to be that directional derivative. More precisely, it is
        assumed that the field does not change in that direction.
        ``vdim_mapping`` needs to be set in order to relate the vector and
        spatial dimensions.

        Returns
        -------
        discretisedfield.Field

            Curl of the field.

        Raises
        ------
        ValueError

            If the ``ndim`` or ``nvdim`` of the field is not 3.
            The ``vdims`` are not correctly mapped to the ``dims``.

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
        if self.nvdim != 3 or self.mesh.region.ndim != 3:
            raise ValueError(
                "Curl can only be computed for a field with nvdim=3 and ndim=3,"
                f"Not {self.nvdim=} and {self.mesh.region.ndim}"
            )

        for vdim in self.vdims:
            if vdim not in self.vdim_mapping:
                raise ValueError(
                    f"Cannot compute curl of the field as {vdim} is not present in"
                    f" {self.vdim_mapping=}."
                )
            elif self.vdim_mapping[vdim] not in self.mesh.region.dims:
                raise ValueError(
                    f"Cannot compute curl of the field as {self.vdim_mapping[vdim]}"
                    f" is not present in {self.mesh.region.dims=}."
                )

        # Use dims order instead of vdims
        x, y, z = self.mesh.region.dims
        curl_x = getattr(self, self._r_dim_mapping[z]).diff(y) - getattr(
            self, self._r_dim_mapping[y]
        ).diff(z)
        curl_y = getattr(self, self._r_dim_mapping[x]).diff(z) - getattr(
            self, self._r_dim_mapping[z]
        ).diff(x)
        curl_z = getattr(self, self._r_dim_mapping[y]).diff(x) - getattr(
            self, self._r_dim_mapping[x]
        ).diff(y)

        return curl_x << curl_y << curl_z

    @property
    def laplace(self):
        r"""Laplace operator.

        This method computes the laplacian for any field:

        .. math::

            \nabla^2 f = \sum_{i=0}^\mathrm{ndim}
                                \frac{\partial^{2} f}{\partial x_i^{2}}

        .. math::

            \nabla^2 \mathbf{f} = (\nabla^2 f_{0},
                                     ...
                                     \nabla^2 f_mathrm{nvdim})

        Directional derivative cannot be computed if only one discretisation
        cell exists in a certain direction. In that case, a zero field is
        considered to be that directional derivative. More precisely, it is
        assumed that the field does not change in that direction.

        Returns
        -------
        discretisedfield.Field

            Laplacian of the field.

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

        # Create a list of derivatives for each dimension
        if self.nvdim == 1:
            derivatives = [
                sum(self.diff(dim, order=2) for dim in self.mesh.region.dims)
            ]
        else:
            derivatives = [
                sum(
                    getattr(self, vdim).diff(dim, order=2)
                    for dim in self.mesh.region.dims
                )
                for vdim in self.vdims
            ]

        result = derivatives[0]
        for derivative in derivatives[1:]:
            result = result << derivative

        return result

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
        >>> f_plane = f.sel('z')
        >>> f_plane.integrate()
        array([500.])

        4. Surface integral of a vector field (flux). The dot product with the surface
        normal vector must be calculated manually.

        .. math::

            \int_\mathrm{S} \mathbf{f}(\mathbf{r}) \cdot \mathrm{d}\mathbf{S}

        >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> f_plane = f.sel('z')
        >>> e_z = [0, 0, 1]
        >>> f_plane.dot(e_z).integrate()
        array([300.])

        5. Integral along x-direction.

        .. math::

            \int_{x_\mathrm{min}}^{x_\mathrm{max}} \mathbf{f}(\mathbf{r}) \mathrm{d}x

        >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> f_plane = f.sel('z')
        >>> f_plane.integrate(direction='x').mean()
        array([10., 20., 30.])

        6. Cumulative integral along x-direction.

        .. math::

            \int_{x_\mathrm{min}}^{x} \mathbf{f}(\mathbf{r}) \mathrm{d}x'

        >>> f = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> f_plane = f.sel('z')
        >>> f_plane.integrate(direction='x', cumulative=True)
        Field(...)

        """
        if direction is None:
            if cumulative:
                raise ValueError(
                    "A cumulative integral can only computed along one direction."
                )
            sum_ = np.sum(self.array, axis=tuple(range(self.mesh.region.ndim)))
            return sum_ * self.mesh.dV
        elif not isinstance(direction, str):
            raise TypeError("'direction' must be of type str.")

        axis = self.mesh.region._dim2index(direction)

        if cumulative:
            # Sum all cell values up to (excuding) point x and add half the cell value
            # of the cell containing point x then multiply by the cell size.
            tmp_array = self.array / 2
            ndim = self.mesh.region.ndim
            left_cells = dfu.assemble_index(slice(None), ndim, {axis: slice(None, -1)})
            right_cells = dfu.assemble_index(slice(None), ndim, {axis: slice(1, None)})
            tmp_array[right_cells] += np.cumsum(self.array, axis=axis)[left_cells]
            res_array = tmp_array * self.mesh.cell[axis]
        else:
            res_array = np.sum(self.array, axis=axis) * self.mesh.cell[axis]

        if self.mesh.region.ndim == 1 and not cumulative:
            # no 0-dimensional region and mesh
            return res_array

        mesh = self.mesh if cumulative else self.mesh.sel(direction)
        return self.__class__(
            mesh,
            nvdim=self.nvdim,
            value=res_array,
            vdims=self.vdims,
            vdim_mapping=self.vdim_mapping,
        )

    def line(self, p1, p2, n=100):
        r"""Sample the field along the line.

        Given two points :math:`p_{1}` and :math:`p_{2}`, :math:`n` position
        coordinates are generated and the corresponding field values.

        .. math::

           \mathbf{r}_{i} = i\frac{\mathbf{p}_{2} -
           \mathbf{p}_{1}}{n-1}

        Parameters
        ----------
        p1, p2 : array_like

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

        return df.Line(
            points=points,
            values=values,
            point_columns=self.mesh.region.dims,
            value_columns=[f"v{dim}" for dim in self.vdims]
            if self.vdims is not None
            else "v",
        )  # TODO scalar fields have no vdim

    def sel(self, *args, **kwargs):
        """Select a part of the field.

        If one of the axis from ``region.dims`` is passed as a string, a field of a
        reduced dimension along the axis and perpendicular to it is extracted,
        intersecting the axis at its center. Alternatively, if a keyword (representing
        the axis) argument is passed with a real number value (e.g. ``x=1e-9``), a field
        of reduced dimensions intersects the axis at a point 'nearest' to the provided
        value is returned. If the mesh is already 1 dimentional a numpy array of the
        field values is returned.

        If instead a tuple, list or a numpy array of length 2 is
        passed as a value containing two real numbers (e.g. ``x=(1e-9, 7e-9)``), a sub
        field is returned with minimum and maximum points along the selected axis,
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
        discretisedfield.Field or numpy.ndarray

            An extracted field.

        Examples
        --------
        1. Extracting the mesh at a specific point (``y=1``).

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (5, 5, 5)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> plane_field = field.sel(y=1)
        >>> plane_field.mesh.region.ndim
        2
        >>> plane_field.mesh.region.dims
        ('x', 'z')
        >>> plane_field.mean()
        array([1., 2., 3.])

        2. Extracting the xy-plane mesh at the mesh region center.

        >>> plane_field = field.sel('z')
        >>> plane_field.mesh.region.ndim
        2
        >>> plane_field.mesh.region.dims
        ('x', 'y')
        >>> plane_field.mean()
        array([1., 2., 3.])

        3. Specifying a range of points along axis ``x`` to be selected from mesh.

        >>> selected_field = field.sel(x=(2, 4))
        >>> selected_field.mesh.region.ndim
        3
        >>> selected_field.mesh.region.dims
        ('x', 'y', 'z')

        4. Extracting the mesh at a specific point on a 1D mesh.
        >>> mesh = df.Mesh(p1=0, p2=5, cell=1)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> field.sel('x')
        array([1., 2., 3.])

        """
        dim, dim_index, _, sel_index = self.mesh._sel_convert_input(*args, **kwargs)

        slices = dfu.assemble_index(
            slice(None), self.mesh.region.ndim + 1, {dim_index: sel_index}
        )
        array = self.array[slices]

        valid = self.valid[slices[:-1]]

        try:
            mesh = self.mesh.sel(*args, **kwargs)
        except ValueError as e:
            if "p1 and p2 must not be empty" not in str(e):
                raise
            return array  # 1 dim case
        else:  # n dim case
            return self.__class__(
                mesh,
                nvdim=self.nvdim,
                value=array,
                vdims=self.vdims,
                unit=self.unit,
                valid=valid,
                vdim_mapping=self.vdim_mapping,
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
        >>> f = df.Field(mesh, nvdim=1, value=1)
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
        >>> f = df.Field(mesh, nvdim=1, value=1)
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
            valid=self.__class__(self.mesh, nvdim=1, value=self.valid, dtype=bool),
            vdim_mapping=self.vdim_mapping,
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

        index_min = self.mesh.point2index(
            submesh.index2point((0,) * submesh.region.ndim)
        )
        index_max = np.add(index_min, submesh.n)
        slices = [slice(i, j) for i, j in zip(index_min, index_max)]
        return self.__class__(
            submesh,
            nvdim=self.nvdim,
            value=self.array[tuple(slices)],
            vdims=self.vdims,
            unit=self.unit,
            valid=self.valid[tuple(slices)],
            vdim_mapping=self.vdim_mapping,
        )

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
        valid = self.valid
        if isinstance(vector, self.__class__):
            self._check_same_mesh_and_field_dim(vector)
            valid = np.logical_and(valid, vector.valid)
        elif self.nvdim == 1 and isinstance(vector, numbers.Complex):
            vector = self.__class__(self.mesh, nvdim=self.nvdim, value=vector)
        elif isinstance(vector, (tuple, list, np.ndarray)):
            vector = self.__class__(self.mesh, nvdim=self.nvdim, value=vector)
        else:
            msg = (
                f"Unsupported operand type(s) for angle: {type(self)=} and"
                f" {type(vector)=}."
            )
            raise TypeError(msg)

        angle_array = np.arccos((self.dot(vector) / (self.norm * vector.norm)).array)
        return self.__class__(
            self.mesh, nvdim=1, value=angle_array, unit="rad", valid=valid
        )

    def rotate90(self, ax1, ax2, k=1, reference_point=None, inplace=False):
        """Rotate field and underlying mesh by 90°.

        Rotate the field ``k`` times by 90 degrees in the plane defined by ``ax1`` and
        ``ax2``. The rotation direction is from ``ax1`` to ``ax2``, the two must be
        different.

        For vector fields (``nvdim>1``) the components of the vector pointing along
        ``ax1`` and ``ax2`` are determined from ``vdim_mapping``. Rotation is only
        possible if this mapping defines vector components along both directions ``ax1``
        and ``ax2``.

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
            centre point of the field is used.

        inplace : bool, optional

            If ``True``, the rotation is applied in-place. Defaults to ``False``.

        Returns
        -------
        discretisedfield.Field

            The rotated field object. Either a new object or a reference to the
            existing field for ``inplace=True``.

        Raises
        ------

        RuntimeError

            If a vector field (``nvdim>1``) does not provide the required mapping
            between spatial directions and vector components in ``vdim_mapping``.

        Examples
        --------

        >>> import discretisedfield as df
        >>> import numpy as np
        >>> p1 = (0, 0, 0)
        >>> p2 = (10, 8, 6)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=(10, 4, 6))
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> rotated = field.rotate90('x', 'y')
        >>> rotated.mesh.region.pmin
        array([ 1., -1.,  0.])
        >>> rotated.mesh.region.pmax
        array([9., 9., 6.])
        >>> rotated.mesh.n
        array([ 4, 10,  6])
        >>> rotated.mean()
        array([-2.,  1.,  3.])

        See also
        --------
        :py:func:`~discretisedfield.Region.rotate90`
        :py:func:`~discretisedfield.Mesh.rotate90`

        """
        # all checks are performed when rotating the mesh
        mesh = self.mesh.rotate90(
            ax1=ax1, ax2=ax2, k=k, reference_point=reference_point, inplace=inplace
        )

        idx1 = self.mesh.region._dim2index(ax1)
        idx2 = self.mesh.region._dim2index(ax2)
        value = np.rot90(self.array.copy(), k=k, axes=(idx1, idx2))
        valid = np.rot90(self.valid.copy(), k=k, axes=(idx1, idx2))

        if self.nvdim > 1:
            # rotate the vector, i.e. the relevant in-plane components
            try:
                vdim1 = self.vdims.index(self._r_dim_mapping[ax1])
                vdim2 = self.vdims.index(self._r_dim_mapping[ax2])
            except ValueError:
                raise RuntimeError(
                    "Missing information about vector orientation in"
                    f" {self.vdim_mapping=}. Manual update of the relation between"
                    " vector dimensions and spatial dimensions required."
                ) from None

            value1 = value[..., vdim1].copy()
            value2 = value[..., vdim2].copy()

            theta = k * np.pi / 2
            value[..., vdim1] = np.cos(theta) * value1 - np.sin(theta) * value2
            value[..., vdim2] = np.sin(theta) * value1 + np.cos(theta) * value2

        if inplace:
            self.update_field_values(value)
            self.valid = valid
            return self
        else:
            return self.__class__(
                mesh,
                nvdim=self.nvdim,
                value=value,
                vdims=self.vdims,
                dtype=self.dtype,
                unit=self.unit,
                valid=valid,
                vdim_mapping=self.vdim_mapping,
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
        if self.mesh.region.ndim != 3:
            raise RuntimeError(
                "Conversion to VTK RectilinearGrid is only possible for 'ndim=3', not"
                f" {self.mesh.region.ndim=}"
            )
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
            >>> field.sel(z=50).resample(n=(5, 5)).mpl()

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

        Data shown in the plot is automatically filtered using the `valid` property of
        the field.

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
        vdims = kwargs.pop("vdims") if "vdims" in kwargs else None
        xrfield = self.to_xarray().copy()
        # we create copy to avoid changing field values
        # using np.where instead did cause issues with broadcasting
        xrfield.data[~self.valid] = np.nan
        res = xrfield.sel(**kwargs, method="nearest")
        if vdims:
            res = res.sel(vdims=vdims)
        return res

    def _hv_vdims_guess(self, kdims):
        """Try to find vector components matching the given kdims."""
        vdims = []
        for dim in kdims:
            vdims.append(self._r_dim_mapping[dim])
        # the hv class expects two valid vdims or None
        return None if None in vdims else vdims

    @property
    def _hv_key_dims(self):
        """Dict of key dimensions of the field.

        Keys are the field dimensions (domain and vector space, e.g. x, y, z, vdims)
        that have length > 1. Values are named_tuples ``hv_key_dim(data, unit)`` that
        contain the data (which has to fulfil len(data) > 1, typically as a numpy array
        or list) and the unit of a string (empty string if there is no unit).

        """
        key_dims = {
            dim: hv_key_dim(coords, unit)
            for dim, unit in zip(self.mesh.region.dims, self.mesh.region.units)
            if len(coords := getattr(self.mesh.cells, dim)) > 1
        }
        if self.nvdim > 1:
            key_dims["vdims"] = hv_key_dim(self.vdims, "")
        return key_dims

    def fftn(self, **kwargs):
        """Performs an N-dimensional discrete Fast Fourier Transform (FFT)
        on the field.

        This method applies an FFT to the field and the underying mesh, transforming
        it from a spatial domain into a frequency domain. During this process, any
        information about subregions within the field is lost.


        Parameters
        ----------
        **kwargs

            Keyword arguments passed directly to the FFT function provided by
            SciPy's fftpack :py:func:`scipy.fft.fftn`.

        Returns
        -------
        discretisedfield.Field

            A field representing the Fourier transform of the original field.
            This returned field has value dimensions labeled with frequency
            (``ft_`` in front of each vdim) and values corresponding to frequencies
            in the frequency domain.


        Examples
        --------
        1. Create a mesh and perform an FFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=0, p2=10, cell=2)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> fft_field = field.fftn()
        >>> fft_field.nvdim
        3
        >>> fft_field.vdims
        ['ft_x', 'ft_y', 'ft_z']

        2. Create a 3D mesh and perform an FFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(2, 2, 2))
        >>> field = df.Field(mesh, nvdim=4, value=(1, 2, 3, 4))
        >>> fft_field = field.fftn()
        >>> fft_field.nvdim
        4
        >>> fft_field.vdims
        ['ft_v0', 'ft_v1', 'ft_v2', 'ft_v3']

        See also
        --------
        :py:func:`~discretisedfield.Field.ifftn`
        :py:func:`~discretisedfield.Field.rfftn`
        :py:func:`~discretisedfield.Field.irfftn`
        :py:func:`~discretisedfield.Mesh.fftn`
        :py:func:`~discretisedfield.Mesh.ifftn`

        """
        mesh = self.mesh.fftn()

        # Use scipy as faster than numpy
        axes = range(self.mesh.region.ndim)
        ft = spfft.fftshift(
            spfft.fftn(self.array, axes=axes, **kwargs),
            axes=axes,
        )

        return self._fftn(mesh=mesh, array=ft, ifftn=False)

    def ifftn(self, **kwargs):
        """Performs an N-dimensional discrete inverse Fast Fourier Transform (iFFT)
        on the field.

        This method applies an iFFT to the field and the underying mesh, transforming
        it from a frequency domain into a spatial domain. During this process, any
        information about subregions within the field is lost.

        Parameters
        ----------
        **kwargs

            Keyword arguments passed directly to the iFFT function provided by
            SciPy's fftpack :py:func:`scipy.fft.ifftn`.

        Returns
        -------
        discretisedfield.Field

            A field representing the inverse Fourier transform of the original field.
            This returned field is in the spatial domain and has value dimensions
            labeled with the ``ft_`` removed if the vdims of the original field
            started with ``ft_``.

        Examples
        --------
        1. Create a mesh and perform an iFFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=0, p2=10, cell=2)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> ifft_field = field.fftn().ifftn()
        >>> ifft_field.nvdim
        3
        >>> ifft_field.vdims
        ['x', 'y', 'z']

        2. Create a 3D mesh and perform an iFFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(2, 2, 2))
        >>> field = df.Field(mesh, nvdim=4, value=(1, 2, 3, 4))
        >>> fft_field = field.fftn().ifftn()
        >>> fft_field.nvdim
        4
        >>> fft_field.vdims
        ['v0', 'v1', 'v2', 'v3']

        See also
        --------
        :py:func:`~discretisedfield.Field.fftn`
        :py:func:`~discretisedfield.Field.rfftn`
        :py:func:`~discretisedfield.Field.irfftn`
        :py:func:`~discretisedfield.Mesh.fftn`
        :py:func:`~discretisedfield.Mesh.ifftn`

        """
        mesh = self.mesh.ifftn()

        axes = range(self.mesh.region.ndim)
        ft = spfft.ifftn(
            spfft.ifftshift(self.array, axes=axes),
            axes=axes,
            **kwargs,
        )

        return self._fftn(mesh=mesh, array=ft, ifftn=True)

    def rfftn(self, **kwargs):
        """Performs an N-dimensional discrete real Fast Fourier Transform (rFFT)
        on the field.

        This method applies an rFFT to the field and the underying mesh, transforming
        it from a spatial domain into a frequency domain. During this process, any
        information about subregions within the field is lost.


        Parameters
        ----------
        **kwargs

            Keyword arguments passed directly to the rFFT function provided by
            SciPy's fftpack :py:func:`scipy.fft.rfftn`.

        Returns
        -------
        discretisedfield.Field

            A field representing the Fourier transform of the original field.
            This returned field has value dimensions labeled with frequency
            (``ft_`` in front of each vdim) and values corresponding to frequencies
            in the frequency domain.


        Examples
        --------
        1. Create a mesh and perform an rFFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=0, p2=10, cell=2)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> fft_field = field.rfftn()
        >>> fft_field.nvdim
        3
        >>> fft_field.vdims
        ['ft_x', 'ft_y', 'ft_z']

        2. Create a 3D mesh and perform an rFFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(2, 2, 2))
        >>> field = df.Field(mesh, nvdim=4, value=(1, 2, 3, 4))
        >>> fft_field = field.rfftn()
        >>> fft_field.nvdim
        4
        >>> fft_field.vdims
        ['ft_v0', 'ft_v1', 'ft_v2', 'ft_v3']

        See also
        --------
        :py:func:`~discretisedfield.Field.fftn`
        :py:func:`~discretisedfield.Field.ifftn`
        :py:func:`~discretisedfield.Field.irfftn`
        :py:func:`~discretisedfield.Mesh.fftn`
        :py:func:`~discretisedfield.Mesh.ifftn`

        """
        mesh = self.mesh.fftn(rfft=True)

        axes = range(self.mesh.region.ndim)
        ft = spfft.fftshift(
            spfft.rfftn(self.array, axes=axes, **kwargs),
            axes=axes[:-1],
        )

        return self._fftn(mesh=mesh, array=ft, ifftn=False)

    def irfftn(self, shape=None, **kwargs):
        """Performs an N-dimensional discrete inverse real Fast Fourier Transform
        (irFFT) on the field.

        This method applies an irFFT to the field and the underying mesh, transforming
        it from a frequency domain into a spatial domain. During this process, any
        information about subregions within the field is lost.

        Parameters
        ----------
        **kwargs

            Keyword arguments passed directly to the irFFT function provided by
            SciPy's fftpack :py:func:`scipy.fft.irfftn`.

        Returns
        -------
        discretisedfield.Field

            A field representing the inverse Fourier transform of the original field.
            This returned field is in the spatial domain and has value dimensions
            labeled with the ``ft_`` removed if the vdims of the original field
            started with ``ft_``.

        Examples
        --------
        1. Create a mesh and perform an irFFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=0, p2=10, cell=2)
        >>> field = df.Field(mesh, nvdim=3, value=(1, 2, 3))
        >>> ifft_field = field.fftn().irfftn()
        >>> ifft_field.nvdim
        3
        >>> ifft_field.vdims
        ['x', 'y', 'z']

        2. Create a 3D mesh and perform an irFFT.
        >>> import discretisedfield as df
        >>> mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(2, 2, 2))
        >>> field = df.Field(mesh, nvdim=4, value=(1, 2, 3, 4))
        >>> fft_field = field.fftn().irfftn()
        >>> fft_field.nvdim
        4
        >>> fft_field.vdims
        ['v0', 'v1', 'v2', 'v3']

        See also
        --------
        :py:func:`~discretisedfield.Field.fftn`
        :py:func:`~discretisedfield.Field.ifftn`
        :py:func:`~discretisedfield.Field.rfftn`
        :py:func:`~discretisedfield.Mesh.fftn`
        :py:func:`~discretisedfield.Mesh.ifftn`

        """
        mesh = self.mesh.ifftn(rfft=True, shape=shape)

        axes = range(self.mesh.region.ndim)
        ft = spfft.irfftn(
            spfft.ifftshift(self.array, axes=axes[:-1]),
            axes=axes,
            s=shape,
            **kwargs,
        )

        return self._fftn(mesh=mesh, array=ft, ifftn=True)

    def _fftn(self, mesh, array, ifftn=False):
        if self.vdims is None:
            new_vdims = None
            new_vdim_mapping = None
        else:
            # vdims
            if ifftn:
                new_vdims = [
                    vdim[3:] if vdim.startswith("ft_") else vdim for vdim in self.vdims
                ]
            else:
                new_vdims = [f"ft_{vdim}" for vdim in self.vdims]
            # vdim mapping
            new_vdim_mapping = {}
            for vdim, new_vdim in zip(self.vdims, new_vdims):
                if vdim in self.vdim_mapping:
                    if ifftn:
                        new_vdim_mapping[new_vdim] = (
                            self.vdim_mapping[vdim][2:]
                            if self.vdim_mapping[vdim].startswith("k_")
                            else self.vdim_mapping[vdim]
                        )
                    else:
                        new_vdim_mapping[new_vdim] = f"k_{self.vdim_mapping[vdim]}"

        return self.__class__(
            mesh,
            nvdim=self.nvdim,
            value=array,
            vdims=new_vdims,
            unit=self.unit,
            vdim_mapping=new_vdim_mapping,
        )

    @property
    def real(self):
        """Real part of complex field."""
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=self.array.real,
            vdims=self.vdims,
            unit=self.unit,
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
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
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
        )

    @property
    def phase(self):
        """Phase of complex field."""
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=np.angle(self.array),
            vdims=self.vdims,
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
        )

    @property
    def abs(self):
        """Absolute value of complex field."""
        return self.__class__(
            self.mesh,
            nvdim=self.nvdim,
            value=np.abs(self.array),
            vdims=self.vdims,
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
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
            valid=self.valid,
            vdim_mapping=self.vdim_mapping,
        )

    # TODO check and write tests
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Field class support for numpy ``ufuncs``."""
        # See reference implementation at:
        # https://numpy.org/doc/stable/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html#numpy.lib.mixins.NDArrayOperatorsMixin
        for x in inputs:
            if not isinstance(x, (Field, np.ndarray, numbers.Number)):
                raise NotImplementedError()
        out = kwargs.get("out", ())
        if out:
            for x in out:
                if not isinstance(x, Field):
                    raise NotImplementedError()

        mesh = [x.mesh for x in inputs if isinstance(x, Field)]
        inputs = tuple(x.array if isinstance(x, Field) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(x.array for x in out)

        result = getattr(ufunc, method)(*inputs, **kwargs)
        if isinstance(result, tuple):
            if len(result) != len(mesh):
                raise NotImplementedError()
            try:
                return tuple(
                    self.__class__(
                        m,
                        nvdim=x.shape[-1],
                        value=x,
                        vdims=self.vdims,
                        vdim_mapping=self.vdim_mapping,
                    )
                    for x, m in zip(result, mesh)
                )
            except Exception:
                raise NotImplementedError()
        elif method == "at":
            return None
        else:
            if not np.array_equal(result.shape[:-1], self.mesh.n):
                raise NotImplementedError()
            try:
                return self.__class__(
                    self.mesh,
                    nvdim=result.shape[-1],
                    value=result,
                    vdims=self.vdims,
                    vdim_mapping=self.vdim_mapping,
                )
            except Exception:
                raise NotImplementedError()

    def to_xarray(self, name="field", unit=None):
        """Field value as ``xarray.DataArray``.

        The function returns an ``xarray.DataArray`` with the dimensions
        ``self.mesh.region.dims`` and ``vdims`` (only if ``field.nvdim > 1``). The
        coordinates of the geometric dimensions are derived from ``self.mesh.points``,
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
        <xarray.DataArray 'field' (x: 10, y: 10, z: 10, vdims: 3)>
        ...

        3. Select values of `x` component

        >>> xa.sel(vdims='x')
        <xarray.DataArray 'field' (x: 10, y: 10, z: 10)>
        ...

        """
        if not isinstance(name, str):
            msg = "Name argument must be a string."
            raise TypeError(msg)

        if unit is not None and not isinstance(unit, str):
            msg = "Unit argument must be a string."
            raise TypeError(msg)

        axes = self.mesh.region.dims

        data_array_coords = {axis: getattr(self.mesh.cells, axis) for axis in axes}

        geo_units_dict = dict(zip(axes, self.mesh.region.units))

        if self.nvdim > 1:
            data_array_dims = axes + ("vdims",)
            if self.vdims is not None:
                data_array_coords["vdims"] = self.vdims
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
                nvdim=self.nvdim,
                tolerance_factor=self.mesh.region.tolerance_factor,
            ),
        )

        # TODO save vdim_mapping

        for dim in geo_units_dict:
            data_array[dim].attrs["units"] = geo_units_dict[dim]

        return data_array

    @classmethod
    def from_xarray(cls, xa):
        """Create ``discretisedfield.Field`` from ``xarray.DataArray``

        The class method accepts an ``xarray.DataArray`` as an argument to
        return a ``discretisedfield.Field`` object. The first n (or n-1) dimensions of
        the DataArray are considered geometric dimensions of a scalar (or vector) field.
        In case of a vector field, the last dimension must be named ``vdims``. The
        DataArray attribute ``nvdim`` determines whether it is a scalar or a vector
        field (i.e. ``nvdim = 1`` is a scalar field and ``nvdim >= 1`` is a vector
        field). Hence, ``nvdim`` attribute must be present, greater than or equal to
        one, and of an integer type.

        The DataArray coordinates corresponding to the geometric dimensions represent
        the discretisation along the respective dimension and must have equally spaced
        values. The coordinates of ``vdims`` represent the name of field components
        (e.g. ['x', 'y', 'z'] for a 3D vector field).

        Additionally, it is expected to have ``cell``, ``p1``, and ``p2`` attributes for
        creating the right mesh for the field; however, in the absence of these, the
        coordinates of the geometric axes dimensions are utilized. It should be noted
        that ``cell`` attribute is required if any of the geometric directions has only
        a single cell.

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

            - If argument is not ``xarray.DataArray``.
            - If ``nvdim`` attribute in not an integer.

        KeyError

            - If at least one of the geometric dimension coordinates has a single
              value and ``cell`` attribute is missing.
            - If ``nvdim`` attribute is absent.

        ValueError

            - If DataArray does not have a dimension ``vdims`` when attribute ``nvdim``
              is grater than one.
            - If coordinates of geometrical dimensions are not equally spaced.

        Examples
        --------
        1. Create a DataArray

        >>> import xarray as xr
        >>> import numpy as np
        ...
        >>> xa = xr.DataArray(np.ones((20, 20, 20, 3), dtype=float),
        ...                   dims = ['x', 'y', 'z', 'vdims'],
        ...                   coords = dict(x=np.arange(0, 20),
        ...                                 y=np.arange(0, 20),
        ...                                 z=np.arange(0, 20),
        ...                                 vdims=['x', 'y', 'z']),
        ...                   name = 'mag',
        ...                   attrs = dict(cell=[1., 1., 1.],
        ...                                p1=[1., 1., 1.],
        ...                                p2=[21., 21., 21.],
        ...                                nvdim=3),)
        >>> xa
        <xarray.DataArray 'mag' (x: 20, y: 20, z: 20, vdims: 3)>
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
            raise TypeError("Argument must be a xarray.DataArray.")

        if "nvdim" not in xa.attrs:
            raise KeyError(
                'The DataArray must have an attribute "nvdim" to identify a scalar or'
                " a vector field."
            )

        if xa.attrs["nvdim"] < 1:
            raise ValueError('"nvdim" attribute must be greater or equal to 1.')
        elif not isinstance(xa.attrs["nvdim"], int):
            raise TypeError("The value of nvdim must be an integer.")

        if xa.attrs["nvdim"] > 1 and "vdims" not in xa.dims:
            raise ValueError(
                'The DataArray must have a dimension "vdims" when "nvdim" attribute is'
                " greater than 1."
            )

        dims_list = [dim for dim in xa.dims if dim != "vdims"]

        for i in dims_list:
            if xa[i].values.size > 1 and not np.allclose(
                np.diff(xa[i].values), np.diff(xa[i].values).mean()
            ):
                raise ValueError(f"Coordinates of {i} must be equally spaced.")

        try:
            cell = xa.attrs["cell"]
        except KeyError:
            if any(len_ == 1 for len_ in xa.values.shape[:-1]):
                raise KeyError(
                    "DataArray must have a 'cell' attribute if any "
                    "of the geometric directions has a single cell."
                ) from None
            cell = [np.diff(xa[i].values).mean() for i in dims_list]

        p1 = (
            xa.attrs["pmin"]
            if "pmin" in xa.attrs
            else [xa[i].values[0] - c / 2 for i, c in zip(dims_list, cell)]
        )
        p2 = (
            xa.attrs["pmax"]
            if "pmax" in xa.attrs
            else [xa[i].values[-1] + c / 2 for i, c in zip(dims_list, cell)]
        )

        if any("units" not in xa[i].attrs for i in dims_list):
            region = df.Region(p1=p1, p2=p2, dims=dims_list)
            mesh = df.Mesh(region=region, cell=cell)
        else:
            region = df.Region(
                p1=p1, p2=p2, dims=dims_list, units=[xa[i].units for i in dims_list]
            )
            mesh = df.Mesh(region=region, cell=cell)

        if "tolerance_factor" in xa.attrs:
            mesh.region.tolerance_factor = xa.attrs["tolerance_factor"]

        vdims = xa.vdims.values if "vdims" in xa.coords else None
        nvdim = xa.attrs["nvdim"]
        val = np.expand_dims(xa.values, axis=-1) if nvdim == 1 else xa.values
        # print(val.shape)
        # TODO load vdim_mapping
        return cls(
            mesh=mesh, nvdim=nvdim, value=val, vdims=vdims, dtype=xa.values.dtype
        )

    @functools.singledispatchmethod
    def _as_array(self, val, mesh, nvdim, dtype):
        raise TypeError(f"Unsupported type {type(val)}.")

    # to avoid str being interpreted as iterable
    @_as_array.register(str)
    def _(self, val, mesh, nvdim, dtype):
        raise TypeError(f"Unsupported type {type(val)}.")

    @_as_array.register(numbers.Complex)
    @_as_array.register(collections.abc.Iterable)
    def _(self, val, mesh, nvdim, dtype):
        if isinstance(val, numbers.Complex) and nvdim > 1 and val != 0:
            raise ValueError(
                f"Wrong dimension 1 provided for value; expected dimension is {nvdim}"
            )

        if isinstance(val, collections.abc.Iterable):
            if nvdim == 1 and np.array_equal(np.shape(val), mesh.n):
                return np.expand_dims(val, axis=-1)
            elif np.shape(val)[-1] != nvdim:
                raise ValueError(
                    f"Wrong dimension {len(val)} provided for value; expected dimension"
                    f" is {nvdim}."
                )
        dtype = dtype or max(np.asarray(val).dtype, np.float64)
        return np.full((*mesh.n, nvdim), val, dtype=dtype)

    @_as_array.register(collections.abc.Callable)
    def _(self, val, mesh, nvdim, dtype):
        # will only be called on user input
        # dtype must be specified by the user for complex values
        array = np.empty((*mesh.n, nvdim), dtype=dtype)
        for index, point in zip(mesh.indices, mesh):
            # Conversion to array and reshaping is required for numpy >= 1.24
            # and for certain inputs, e.g. a tuple of numpy arrays which can e.g. occur
            # for 1d vector fields.
            array[index] = np.asarray(val(point)).reshape(nvdim)
        return array

    @_as_array.register(dict)
    def _(self, val, mesh, nvdim, dtype):
        # will only be called on user input
        # dtype must be specified by the user for complex values
        dtype = dtype or np.float64
        fill_value = (
            val["default"]
            if "default" in val and not callable(val["default"])
            else np.nan
        )
        array = np.full((*mesh.n, nvdim), fill_value, dtype=dtype)

        for subregion in reversed(mesh.subregions.keys()):
            # subregions can overlap, first subregion takes precedence
            try:
                submesh = mesh[subregion]
                subval = val[subregion]
            except KeyError:
                continue  # subregion not in val when implicitly set via "default"
            else:
                slices = mesh.region2slices(submesh.region)
                array[slices] = self._as_array(subval, submesh, nvdim, dtype)

        if np.any(np.isnan(array)):
            # not all subregion keys specified and 'default' is missing or callable
            if "default" not in val:
                raise KeyError(
                    "Key 'default' required if not all subregion keys are specified."
                )
            subval = val["default"]
            for idx in np.argwhere(np.isnan(array[..., 0])):
                # only spatial indices required -> array[..., 0]
                # conversion to array and reshaping similar to "callable" implementation
                array[idx] = np.asarray(subval(mesh.index2point(idx))).reshape(nvdim)

        return array


# We cannot register to self (or df.Field) inside the class
@Field._as_array.register(Field)
def _(self, val, mesh, nvdim, dtype):
    if mesh.region not in val.mesh.region:
        raise ValueError(
            f"{val.mesh.region} of the provided field does not "
            f"contain {mesh.region} of the field that is being "
            "created."
        )
    value = (
        val.to_xarray()
        .sel(
            **{dim: getattr(mesh.cells, dim) for dim in mesh.region.dims},
            method="nearest",
        )
        .data
    )
    if nvdim == 1:
        # xarray dataarrays for scalar data are three dimensional
        return value.reshape(*mesh.n, -1)
    return value
