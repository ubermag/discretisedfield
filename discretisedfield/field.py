import pyvtk
import struct
import matplotlib
import numpy as np
import discretisedfield as df
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


representations = ['txt', 'bin4', 'bin8']


@ts.typesystem(mesh=ts.Typed(expected_type=df.Mesh),
               dim=ts.Scalar(expected_type=int, unsigned=True, const=True),
               name=ts.Name(const=True))
class Field:
    """Finite difference field.

    This class defines a finite difference field and enables certain
    operations for the analysis and visualisation. The field is
    defined on a finite difference mesh (`discretisedfield.Mesh`).

    Parameters
    ----------
    mesh : discretisedfield.Mesh
        Finite difference rectangular mesh.
    dim : int, optional
        Dimension of the field value. For instance, if `dim=3` the
        field is a three-dimensional vector field; and for ``dim=1``
        the field is a scalar field. Defaults to `dim=3`.
    value : array_like, callable, optional
        Please refer to the `value` property:
        :py:func:`~discretisedfield.Field.value`. Defaults to 0,
        meaning that if the value is not provided in the
        initialisation pricess, "zero-field" will be defined.
    norm : numbers.Real, callable, optional
        Please refer to the `norm` property:
        :py:func:`~discretisedfield.Field.norm`. Defaults to `None`
        (`norm=None` defines no norm).
    name : str, optional
        Field name (defaults to "field"). The field name must be a valid
        Python variable name string. More specifically, it must not
        contain spaces, or start with underscore or numeric character.

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
    >>> dim = 3
    >>> value = (0, 0, 1)
    >>> field = df.Field(mesh=mesh, dim=dim, value=value)

    .. seealso:: :py:func:`~discretisedfield.Mesh`

    """
    def __init__(self, mesh, dim=3, value=0, norm=None, name="field"):
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
            allowed. In the case of vector fields, array_like value
            with length equal to `dim` should be used. Finally, the
            value can also be a callable (e.g. Python function), which
            for every coordinate in the mesh returns a valid value. If
            `value=0`, all values in the field will be set to zero
            independent of the field dimension.

        Returns
        -------
        array_like, callable, numbers.Real
            The value used (representation) for setting the field is
            returned. However, if the actual value of the field does
            not correspond to the initially used value anymore, a
            `numpy.ndarray` is returned containing all field values.

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

        Parameters
        ----------
        numpy.ndarray
            Numpy array with dimensions `(field.mesh.n[0],
            field.mesh.n[1], field.mesh.n[2], dim)`

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
        if isinstance(val, np.ndarray) and val.shape == self.mesh.n + (self.dim,):
            self._array = val
        else:
            msg = f'Unsupported type(val)={type(val)} or invalid value dimensions.'
            raise ValueError(msg)

    @property
    def norm(self):
        """Norm of a field.

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
            If setting the norm with wrong type, shape, or value.

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
        <Field(...)>
        >>> field.norm = 2
        >>> field.norm.array
        array([[[[2.]]]])
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

        Returns
        -------
        tuple
            Field average tuple whose length equals to the field's
            dimension.

        Examples
        --------
        1. Getting the field average.

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
        """Representation method."""
        return f'<Field(mesh={repr(self.mesh)}, dim={self.dim}, name=\'{self.name}\')>'

    def __call__(self, point):
        """Sample the field at `point`.

        Parameters
        ----------
        p (tuple): point coordinate at which the field is sampled

        Returns:
          Field value in cell containing point p

        """
        value = self.array[self.mesh.point2index(point)]
        if self.dim > 1:
            value = tuple(value)
        return value

    def __getattr__(self, name):
        if name in list(dfu.axesdict.keys())[:self.dim] and 1 < self.dim <= 3:
            # Components x, y, and z make sense only for vector fields
            # with typical dimensions 2 and 3.
            component_array = self.array[..., dfu.axesdict[name]][..., None]
            fieldname = f'{self.name}-{name}'.format(self.name, name)
            return Field(mesh=self.mesh, dim=1, value=component_array, name=fieldname)
        else:
            msg = f'{type(self).__name__} object has no attribute {name}.'
            raise AttributeError(msg.format(type(self).__name__, name))

    def __dir__(self):
        if 1 < self.dim <= 3:
            extension = list(dfu.axesdict.keys())[:self.dim]
        else:
            extension = []
        return list(self.__dict__.keys()) + extension

    def __iter__(self):
        for point in self.mesh.coordinates:
            yield point, self.__call__(point)

    def line(self, p1, p2, n=100):
        for point in self.mesh.line(p1=p1, p2=p2, n=n):
            yield point, self.__call__(point)

    def plane(self, *args, **kwargs):
        for point in self.mesh.plane(*args, **kwargs):
            yield point, self.__call__(point)

    def plot_plane(self, *args, x=None, y=None, z=None, n=None, ax=None, figsize=None):
        info, points, values, n, ax = self._plot_data(*args, x=x, y=y, z=z,
                                                      n=n, ax=ax, figsize=figsize)

        if self.dim > 1:
            self.quiver(*args, x=x, y=y, z=z, n=n, ax=ax)
            scfield = getattr(self, list(dfu.axesdict.keys())[info["planeaxis"]])
        else:
            scfield = self

        colouredplot = scfield.imshow(*args, x=x, y=y, z=z, n=n, ax=ax)
        self.colorbar(ax, colouredplot)

        ax.set_xlabel(list(dfu.axesdict.keys())[info["axis1"]])
        ax.set_ylabel(list(dfu.axesdict.keys())[info["axis2"]])

    def _plot_data(self, *args, x=None, y=None, z=None, n=None, ax=None, figsize=None):
        info = dfu.plane_info(*args, x=x, y=y, z=z)
        data = list(self.plane(*args, x=x, y=y, z=z, n=n))
        ps, vs = list(zip(*data))
        points = list(zip(*ps))
        values = list(zip(*vs))

        if n is None:
            n = (self.mesh.n[info["axis1"]], self.mesh.n[info["axis2"]])

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

        return info, points, values, n, ax

    def imshow(self, *args, x=None, y=None, z=None, n=None, ax=None, **kwargs):
        info, points, values, n, ax = self._plot_data(*args, x=x, y=y, z=z,
                                                      n=n, ax=ax)

        extent = [self.mesh.pmin[info["axis1"]], self.mesh.pmax[info["axis1"]],
                  self.mesh.pmin[info["axis2"]], self.mesh.pmax[info["axis2"]]]
        imax = ax.imshow(np.array(values).reshape(n).T, origin="lower",
                         extent=extent, **kwargs)

        return imax

    def quiver(self, *args, x=None, y=None, z=None, n=None, ax=None,
               colour=None, **kwargs):
        info, points, values, n, ax = self._plot_data(*args, x=x, y=y, z=z,
                                                      n=n, ax=ax)

        if not any(values[info["axis1"]] + values[info["axis2"]]):
            kwargs["scale"] = 1

        if colour is None:
            qvax = ax.quiver(points[info["axis1"]], points[info["axis2"]],
                             values[info["axis1"]], values[info["axis2"]],
                             pivot='mid', **kwargs)
        elif colour in dfu.axesdict.keys():
            qvax = ax.quiver(points[info["axis1"]], points[info["axis2"]],
                             values[info["axis1"]], values[info["axis2"]],
                             values[dfu.axesdict[colour]],
                             pivot='mid', **kwargs)

        return qvax

    def colorbar(self, ax, colouredplot, cax=None, **kwargs):
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

        plt.colorbar(colouredplot, cax=cax, **kwargs)

    def write(self, filename, **kwargs):
        if any([filename.endswith(ext) for ext in [".omf", ".ovf", ".ohf"]]):
            self._writeovf(filename, **kwargs)
        elif filename.endswith(".vtk"):
            self._writevtk(filename)

    def _writevtk(self, filename):
        grid = [pmini + np.linspace(0, li, ni+1) for pmini, li, ni in
                zip(self.mesh.pmin, self.mesh.l, self.mesh.n)]

        structure = pyvtk.RectilinearGrid(*grid)
        vtkdata = pyvtk.VtkData(structure)

        vectors = [self.__call__(i) for i in self.mesh.coordinates]
        vtkdata.cell_data.append(pyvtk.Vectors(vectors, self.name))
        for i, component in enumerate(dfu.axesdict.keys()):
            name = "{}_{}".format(self.name, component)
            vtkdata.cell_data.append(pyvtk.Scalars(list(zip(*vectors))[i],
                                                   name))

        vtkdata.tofile(filename)

    def _writeovf(self, filename, **kwargs):
        representation = kwargs['representation'] if 'representation' in kwargs else 'txt'
        dim = kwargs['dim'] if 'dim' in kwargs else self.dim
        header = ["OOMMF OVF 2.0",
                  "",
                  "Segment count: 1",
                  "",
                  "Begin: Segment",
                  "Begin: Header",
                  "",
                  "Title: Field generated omf file",
                  "Desc: File generated by Field class",
                  "meshunit: m",
                  "meshtype: rectangular",
                  "xbase: {}".format(self.mesh.pmin[0] +
                                     self.mesh.cell[0]/2),
                  "ybase: {}".format(self.mesh.pmin[1] +
                                     self.mesh.cell[1]/2),
                  "zbase: {}".format(self.mesh.pmin[2] +
                                     self.mesh.cell[2]/2),
                  "xnodes: {}".format(self.mesh.n[0]),
                  "ynodes: {}".format(self.mesh.n[1]),
                  "znodes: {}".format(self.mesh.n[2]),
                  "xstepsize: {}".format(self.mesh.cell[0]),
                  "ystepsize: {}".format(self.mesh.cell[1]),
                  "zstepsize: {}".format(self.mesh.cell[2]),
                  "xmin: {}".format(self.mesh.pmin[0]),
                  "ymin: {}".format(self.mesh.pmin[1]),
                  "zmin: {}".format(self.mesh.pmin[2]),
                  "xmax: {}".format(self.mesh.pmax[0]),
                  "ymax: {}".format(self.mesh.pmax[1]),
                  "zmax: {}".format(self.mesh.pmax[2]),
                  "valuedim: {}".format(self.dim),
                  "valuelabels: {0}_x {0}_y {0}_z".format(self.name),
                  "valueunits: A/m A/m A/m",
                  "",
                  "End: Header",
                  ""]

        if representation == "bin4":
            header.append("Begin: Data Binary 4")
            footer = ["End: Data Binary 4",
                      "End: Segment"]
        elif representation == "bin8":
            header.append("Begin: Data Binary 8")
            footer = ["End: Data Binary 8",
                      "End: Segment"]
        elif representation == "txt":
            header.append("Begin: Data Text")
            footer = ["End: Data Text",
                      "End: Segment"]

        f = open(filename, "w")

        # Write header lines to OOMMF file.
        headerstr = "".join(map(lambda line: "# {}\n".format(line), header))
        f.write(headerstr)

        binary_reps = {'bin4': (1234567.0, 'f'),
            'bin8': (123456789012345.0, 'd')}

        if representation in binary_reps:
            # Close the file and reopen with binary write
            # appending to the end of file.
            f.close()
            f = open(filename, "ab")

            # Add the 8 bit binary check value that OOMMF uses
            packarray = [binary_reps[representation][0]]
            # Write data lines to OOMMF file.
            for i in self.mesh.indices:
                [packarray.append(vi) for vi in self.array[i]]

            v_binary = struct.pack(binary_reps[representation][1]*len(packarray), *packarray)
            f.write(v_binary)
            f.close()
            f = open(filename, "a")

        else:
            for i in self.mesh.indices:
                if self.dim == 3:
                    v = [vi for vi in self.array[i]]
                elif self.dim == 1:
                    v = [self.array[i][0]]
                else:
                    msg = ("Cannot write dim={} field to "
                           "omf file.".format(self.dim))
                    raise TypeError(msg)
                for vi in v:
                    f.write(" " + str(vi))
                f.write("\n")

        # Write footer lines to OOMMF file.
        footerstr = "".join(map(lambda line: "# {}\n".format(line), footer))
        f.write(footerstr)

        f.close()

    def plot3d_domain(self, k3d_plot=None, **kwargs):
        """Plots only an aria where norm is not zero
        (where the material is present).

        This function is called as a display function in Jupyter notebook.

        Parameters
        ----------
        k3d_plot : k3d.plot.Plot, optional
               We transfer a k3d.plot.Plot object to add the current 3d figure
               to the canvas(?).

        """
        plot_array = np.squeeze(self.x.array)
        plot_array = np.swapaxes(plot_array, 0, 2)  # in k3d, numpy arrays are (z, y, x)
        plot_array[plot_array != 0] = 1  # make all domain cells to have the same colour
        voxels(plot_array, self.mesh.pmin, self.mesh.pmax, k3d_plot=k3d_plot, **kwargs)

    def get_coord_and_vect(self, raw):
        # Get arrows only with norm > 0.
        data = [(i, self(i)) for i in raw
                if self.norm(i) > 0]
        coordinates, vectors = zip(*data)
        coordinates, vectors = np.array(coordinates, dtype=np.float32), \
                               np.array(vectors, dtype=np.float32)

        # Middle of the arrow at the cell centre.
        coordinates -= 0.5 * vectors

        return coordinates, vectors

    def plot3d_vectors(self, k3d_plot=None, points=False, **kwargs):
        """Plots the vector fields where norm is not zero
        (where the material is present). Shift the vector so that
        its center passes through the center of the cell.

        This function is called as a display function in Jupyter notebook.

        Parameters
        ----------
        k3d_plot : k3d.plot.Plot, optional
               We transfer a k3d.plot.Plot object to add the current 3d figure
               to the canvas(?).

        """
        coordinates, vectors = self.get_coord_and_vect(self.mesh.coordinates)

        k3d_vectors(coordinates, vectors, k3d_plot=k3d_plot, points=points,
                    **kwargs)

    def plot3d_vectors_slice(self, x=None, y=None, z=None,
                             k3d_plot=None, points=False, **kwargs):
        """Plots the slice of vector field by X,Y or Z plane, where norm
         is not zero (where the material is present). Shift the vector
         so that its center passes through the center of the cell.

        This function is called as a display function in Jupyter notebook.

        Parameters
        ----------
            x, y, z : float
                The coordinate of the axis along which the volume is cut.
            k3d_plot : k3d.plot.Plot, optional
                We transfer a k3d.plot.Plot object to add the current 3d figure
                to the canvas(?).

        """
        coordinates, vectors = self.get_coord_and_vect(self.mesh.plane(x=x, y=y, z=z))

        k3d_vectors(coordinates, vectors, k3d_plot=k3d_plot, points=points,
                    **kwargs)



    def plot3d_scalar(self, k3d_plot=None, **kwargs):
        """Plots the scalar fields.

        This function is called as a display function in Jupyter notebook.

        Parameters
        ----------
        k3d_plot : k3d.plot.Plot, optional
               We transfer a k3d.plot.Plot object to add the current 3d figure
               to the canvas(?).

        """
        field_array = self.array.copy()
        array_shape = self.array.shape  # TODO rewrite

        nx, ny, nz, _ = array_shape

        norm = np.linalg.norm(field_array, axis=3)[..., None]

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if norm[i, j, k] == 0:
                        field_array[i, j, k] = np.nan

        component = 0

        field_component = field_array[..., component]

        k3d_scalar(field_component, self.mesh, k3d_plot=k3d_plot,
                   **kwargs)


    def plot3d_isosurface(self, level, k3d_plot=None, **kwargs):
        """Plots isosurface where norm of field.array equal the `level`.

        This function is called as a display function in Jupyter notebook.

        Parameters
        ----------
            level : float
                The field surface value.
            k3d_plot : k3d.plot.Plot, optional
                We transfer a k3d.plot.Plot object to add the current 3d figure
                to the canvas(?).

        """
        k3d_isosurface(self.array, level, self.mesh, k3d_plot=k3d_plot,
                       **kwargs)
