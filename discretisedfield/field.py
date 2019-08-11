import pyvtk
import struct
import matplotlib
import numpy as np
import mpl_toolkits.axes_grid1
import discretisedfield as df
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu
import matplotlib.pyplot as plt


@ts.typesystem(mesh=ts.Typed(expected_type=df.Mesh),
               dim=ts.Scalar(expected_type=int, unsigned=True, const=True),
               name=ts.Name(const=True))
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
    name : str, optional
        Field name (defaults to `'field'`). The field name must be a
        valid Python variable name string. More specifically, it must
        not contain spaces, or start with underscore or numeric
        character.

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
    def __init__(self, mesh, dim=3, value=0, norm=None, name='field'):
        self.mesh = mesh
        self.dim = dim
        self.value = value
        self.norm = norm
        self.name = name

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
           val.shape == self.mesh.n + (self.dim,):
            self._array = val
        else:
            msg = (f'Unsupported type(val)={type(val)} '
                   'or invalid value dimensions.')
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
        1. Manipulating the field norm

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (1, 1, 1)
        >>> cell = (1, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> field = df.Field(mesh=mesh, dim=3, value=(0, 0, 1))
        >>> field.norm
        Field(...)
        >>> field.norm = 2
        >>> field.norm
        Field(...)
        >>> field.value = (1, 0, 0)
        >>> field.norm.array
        array([[[[1.]]]])

        """
        current_norm = np.linalg.norm(self.array, axis=-1)[..., None]
        return Field(self.mesh, dim=1, value=current_norm, name='norm')

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
        return tuple(self.array.mean(axis=(0, 1, 2)))

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
        "Field(mesh=...)"

        """
        return (f'Field(mesh={repr(self.mesh)}, '
                f'dim={self.dim}, name=\'{self.name}\')')

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
            value = tuple(value)
        return value

    def __getattr__(self, name):
        """Extracting the component of the vector field.

        If `'x'`, `'y'`, or `'z'` is accessed, a new scalar field of
        that component will be returned. This method is effective for
        vector fields with dimension 2 or 3.

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
        if name in list(dfu.axesdict.keys())[:self.dim] and 1 < self.dim <= 3:
            # Components x, y, and z make sense only for vector fields
            # with typical dimensions 2 and 3.
            component_array = self.array[..., dfu.axesdict[name]][..., None]
            fieldname = f'{self.name}-{name}'.format(self.name, name)
            return Field(mesh=self.mesh, dim=1,
                         value=component_array, name=fieldname)
        else:
            msg = f'{type(self).__name__} object has no attribute {name}.'
            raise AttributeError(msg.format(type(self).__name__, name))

    def __dir__(self):
        """Extension of the tab-completion list.

        Adds `'x'`, `'y'`, and `'z'`, depending on the dimension of
        the field, to the tab-completion list. This is effective in
        IPython or Jupyter notebook environment.

        """
        if 1 < self.dim <= 3:
            extension = list(dfu.axesdict.keys())[:self.dim]
        else:
            extension = []
        return list(self.__dict__.keys()) + extension

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
        for point in self.mesh.coordinates:
            yield point, self.__call__(point)

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

    def write(self, filename, representation='txt'):
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
            self._writeovf(filename, representation=representation)
        elif filename.endswith('.vtk'):
            self._writevtk(filename)
        else:
            msg = ('Allowed extensions for writing the field are '
                   '.omf, .ovf, .ohf, and .vtk.')
            raise ValueError(msg)

    def _writeovf(self, filename, representation='txt'):
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
                  f'xbase: {self.mesh.pmin[0] + self.mesh.cell[0]/2}',
                  f'ybase: {self.mesh.pmin[1] + self.mesh.cell[1]/2}',
                  f'zbase: {self.mesh.pmin[2] + self.mesh.cell[2]/2}',
                  f'xnodes: {self.mesh.n[0]}',
                  f'ynodes: {self.mesh.n[1]}',
                  f'znodes: {self.mesh.n[2]}',
                  f'xstepsize: {self.mesh.cell[0]}',
                  f'ystepsize: {self.mesh.cell[1]}',
                  f'zstepsize: {self.mesh.cell[2]}',
                  f'xmin: {self.mesh.pmin[0]}',
                  f'ymin: {self.mesh.pmin[1]}',
                  f'zmin: {self.mesh.pmin[2]}',
                  f'xmax: {self.mesh.pmax[0]}',
                  f'ymax: {self.mesh.pmax[1]}',
                  f'zmax: {self.mesh.pmax[2]}',
                  f'valuedim: {self.dim}',
                  f'valuelabels: {self.name}_x {self.name}_y {self.name}_z',
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
                zip(self.mesh.pmin, self.mesh.l, self.mesh.n)]

        structure = pyvtk.RectilinearGrid(*grid)
        vtkdata = pyvtk.VtkData(structure)

        vectors = [self.__call__(coord) for coord in self.mesh.coordinates]
        vtkdata.cell_data.append(pyvtk.Vectors(vectors, self.name))
        for i, component in enumerate(dfu.axesdict.keys()):
            name = f'{self.name}_{component}'
            vtkdata.cell_data.append(pyvtk.Scalars(list(zip(*vectors))[i],
                                                   name))

        vtkdata.tofile(filename)

    @classmethod
    def fromfile(cls, filename, norm=None, name='field'):
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
        name : str
            Name of the field read.

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

        field = df.Field(mesh, dim=dim, name=name)

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
            msg = ('Only sliced field can be plotted using mpl. '
                   'For instance, field.plane(\'x\').mpl().')
            raise ValueError(msg)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        planeaxis = dfu.raxesdict[self.mesh.info['planeaxis']]

        if self.dim > 1:
            # Vector field has both quiver and imshow plots.
            self.quiver(ax=ax, headwidth=5)
            scfield = getattr(self, planeaxis)
            coloredplot = scfield.imshow(ax=ax, norm_field=self.norm)
        else:
            # Scalar field has only imshow.
            scfield = self
            coloredplot = scfield.imshow(ax=ax, norm_field=None)

        # Add colorbar to imshow plot.
        cbar = self.colorbar(ax, coloredplot)

        # Add labels.
        ax.set_xlabel(dfu.raxesdict[self.mesh.info['axis1']])
        ax.set_ylabel(dfu.raxesdict[self.mesh.info['axis2']])
        if self.dim > 1:
            cbar.ax.set_ylabel(planeaxis + ' component')

    def imshow(self, ax, norm_field=None, **kwargs):
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
            msg = ('Only sliced field can be plotted using imshow. '
                   'For instance, field.plane(\'x\').imshow(ax=ax).')
            raise ValueError(msg)
        if self.dim > 1:
            msg = ('Only scalar (dim=1) fields can be plotted. Consider '
                   'plotting one component, e.g. field.x.imshow(ax=ax) '
                   'or norm field.norm.imshow(ax=ax).')
            raise ValueError(msg)

        points, values = list(zip(*list(self)))

        # If norm_field is passed, set values where norm=0 to np.nan,
        # so that they are not plotted.
        if norm_field is not None:
            values = list(values)  # make values mutable
            for i, point in enumerate(points):
                if norm_field(point) == 0:
                    values[i] = np.nan

            # "Unpack" values inside arrays.
            values = [v[0] if not np.isnan(v) else v for v in values]
        else:
            # "Unpack" values inside arrays.
            values = list(zip(*values))

        points = list(zip(*points))

        extent = [self.mesh.pmin[self.mesh.info['axis1']],
                  self.mesh.pmax[self.mesh.info['axis1']],
                  self.mesh.pmin[self.mesh.info['axis2']],
                  self.mesh.pmax[self.mesh.info['axis2']]]
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
            msg = ('Only sliced field can be plotted using quiver. '
                   'For instance, field.plane(\'x\').quiver(ax=ax).')
            raise ValueError(msg)
        if self.dim != 3:
            msg = 'Only three-dimensional (dim=3) fields can be plotted.'
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
            # quiver plot is not coloured.
            qvax = ax.quiver(points[self.mesh.info['axis1']],
                             points[self.mesh.info['axis2']],
                             values[self.mesh.info['axis1']],
                             values[self.mesh.info['axis2']],
                             **kwargs)

        else:
            # quiver plot is coloured.
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

    def k3d_nonzero(self, color=dfu.colormap[0], plot=None, **kwargs):
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
        if self.dim > 1:
            msg = ('Only scalar (dim=1) fields can be plotted. Consider '
                   'plotting one component, e.g. field.x.k3d_nonzero() '
                   'or norm field.norm.k3d_nonzero().')
            raise ValueError(msg)
        plot_array = np.copy(self.array)  # make a deep copy
        plot_array = np.squeeze(plot_array)  # remove an empty dimension
        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)
        plot_array[plot_array != 0] = 1  # all cells have the same colour

        # In the case of nano-sized samples, fix the order of
        # magnitude of the plot extent to avoid freezing the k3d plot.
        if np.any(np.divide(self.mesh.cell, 1e-9) < 1e3):
            pmin = np.divide(self.mesh.pmin, 1e-9)
            pmax = np.divide(self.mesh.pmax, 1e-9)
        else:
            pmin = self.mesh.pmin
            pmax = self.mesh.pmax

        dfu.voxels(plot_array, pmin, pmax, colormap=color,
                   plot=plot, **kwargs)

    def k3d_voxels(self, norm_field=None, plot=None, **kwargs):
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
        if self.dim > 1:
            msg = ('Only scalar (dim=1) fields can be plotted. Consider '
                   'plotting one component, e.g. field.x.k3d_nonzero() '
                   'or norm field.norm.k3d_nonzero().')
            raise ValueError(msg)

        plot_array = np.copy(self.array)  # make a deep copy
        plot_array = plot_array[..., 0]  # remove an empty dimension

        plot_array -= plot_array.min()
        # In the case of uniform fields, division by zero can be
        # encountered.
        if plot_array.max() != 0:
            plot_array /= plot_array.max()
        plot_array *= 254
        plot_array += 1
        plot_array = plot_array.round()
        plot_array = plot_array.astype(int)

        if norm_field is not None:
            for index in self.mesh.indices:
                if norm_field(self.mesh.index2point(index)) == 0:
                    plot_array[index] = 0

        plot_array = np.swapaxes(plot_array, 0, 2)  # k3d: arrays are (z, y, x)

        cmap = matplotlib.cm.get_cmap('viridis', 256)
        colormap = [dfu.num2hexcolor(i, cmap) for i in range(cmap.N)]

        # In the case of nano-sized samples, fix the order of
        # magnitude of the plot extent to avoid freezing the k3d plot.
        if np.any(np.divide(self.mesh.cell, 1e-9) < 1e3):
            pmin = np.divide(self.mesh.pmin, 1e-9)
            pmax = np.divide(self.mesh.pmax, 1e-9)
        else:
            pmin = self.mesh.pmin
            pmax = self.mesh.pmax

        dfu.voxels(plot_array, pmin, pmax, colormap=colormap,
                   plot=plot, **kwargs)

    def k3d_vectors(self, color_field=None, points=True, plot=None, **kwargs):
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
            msg = 'Only three-dimensional (dim=3) fields can be plotted.'
            raise ValueError(msg)

        coordinates, vectors, color_values = [], [], []
        norm = self.norm  # assigned to be computed only once
        for coord, value in self:
            if norm(coord) > 0:
                coordinates.append(coord)
                vectors.append(value)
                if color_field is not None:
                    color_values.append(color_field(coord)[0])

        coordinates, vectors = np.array(coordinates), np.array(vectors)

        # In the case of nano-sized samples, fix the order of
        # magnitude of the coordinates to avoid freezing the k3d plot.
        if np.any(np.divide(self.mesh.cell, 1e-9) < 1e3):
            coordinates /= 1e-9
            cell = np.divide(self.mesh.cell, 1e-9)
        else:
            cell = self.mesh.cell

        # Scale the vectors to correspond to the size of cells.
        vectors /= vectors.max()
        vectors *= 0.8*np.array(cell)

        # Middle of the arrow is at the cell centre.
        coordinates -= 0.5 * vectors

        if color_field is not None:
            color_values = np.array(color_values)
            color_values -= color_values.min()
            # In the case of uniform fields, division by zero can be
            # encountered.
            if color_values.max() != 0:
                color_values /= color_values.max()
            color_values *= 256
            color_values = color_values.round()
            color_values = color_values.astype(int)

            cmap = matplotlib.cm.get_cmap('viridis', 256)
            colors = []
            for c in color_values:
                color = dfu.num2hexcolor(c, cmap)
                colors.append((color, color))
        else:
            colors = []

        plot = dfu.vectors(coordinates, vectors, colors=colors,
                           plot=plot, **kwargs)

        if points:
            dfu.points(coordinates + 0.5 * vectors, plot=plot)
