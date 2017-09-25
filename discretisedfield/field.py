import pyvtk
import struct
import numpy as np
import joommfutil.typesystem as ts
import discretisedfield as df
import discretisedfield.util as dfu
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


@ts.typesystem(mesh=ts.Typed(expected_type=df.Mesh),
               dim=ts.UnsignedInt,
               name=ts.ObjectName)
class Field(dfu.Field):
    """Finite difference field

    This class defines a finite difference field and provides some
    basic operations. The field is defined on a finite difference mesh
    (`discretisedfield.Mesh`).

    Parameters
    ----------
    mesh : discretisedfield.Mesh
        Finite difference rectangular mesh on which the field is defined.
    dim : int, optional
        Dimension of the field value. For instance, if ``dim=3``
        the field is three-dimensional vector field; and for
        ``dim=1`` it is a scalar field.
    value : 0, array_like, callable, optional
        For more details, please refer to the `value` property.
    norm : numbers.Real, callable, optional
        For more details, please refer to the `norm` property.
    name : str, optional
        Field name (the default is "field"). The field name must be a valid
        Python variable name string. More specifically, it must not
        contain spaces, or start with underscore or numeric character.

    Examples
    --------
    Creating a uniform vector field on a nano-sized thin film

    >>> import discretisedfield as df
    >>> p1 = (-50e-9, -25e-9, 0)
    >>> p2 = (50e-9, 25e-9, 5e-9)
    >>> cell = (1e-9, 1e-9, 0.1e-9)
    >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
    >>> value = (0, 0, 1)
    >>> name = "uniform_field"
    >>> field = df.Field(mesh=mesh, dim=3, value=value, name=name)

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
        """Finite difference field value representation property.

        The getter of this propertry returns a field value
        representation if exists. Otherwise, it returns the
        numpy.ndarray.

        Parameters
        ----------
        value : numbers.Real, array_like, callable
            If the parameter for the setter of this property is 0, all
            values in the field will be set to zero (for vector fields
            ``dim > 1``, all componenets of the vector will be
            0). Input parameter can also be ``numbers.Real`` for
            scalar fields (``dim=1``). In the case of vector fields,
            ``numbers.Real`` is not allowed unless ``value=0``, but an
            array_like data with length equal to the ``dim`` should be
            used. Finally, the value can also be a callable
            (e.g. Python function), which for every coordinate in the
            mesh returns an appropriate value.

        Returns
        -------
        array_like, callable, numbers.Real
            The returned value in the case of vector field can be
            array_like (tuple, list, numpy.ndarray). In the case of
            scalar field, the returned value can be numbers.Real. If
            all components of a vector field are zero, 0 will be
            returned.

        Examples
        --------
        Different ways of setting and getting property `value`.

        >>> import discretisedfield as df
        >>> p1 = (-50e-9, -25e-9, 0)
        >>> p2 = (50e-9, 25e-9, 5e-9)
        >>> cell = (5e-9, 5e-9, 5e-9)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> value = (0, 0, 1)
        >>> name = "uniform_field"
        >>> field = df.Field(mesh=mesh, dim=3, value=value, name=name)
        >>> # Value getter
        >>> field.value
        (0, 0, 1)
        >>> field.value = 0
        >>> field.value
        0
        >>> def value_function(pos):
        ...     x, y, z = pos
        ...     if x <= 0:
        ...         return (0, 0, 1)
        ...     else:
        ...         return (0, 0, -1)
        >>> field.value = value_function
        >>> field((10e-9, 0, 0))
        (0.0, 0.0, -1.0)
        >>> field((-10e-9, 0, 0))
        (0.0, 0.0, 1.0)

        .. note::

           Please note this method is a property and should be called
           as ``field.value``, not ``field.value()``.

        .. seealso:: :py:func:`~discretisedfield.Field.array`

        """
        if np.array_equal(self.array,
                          dfu.as_array(self.mesh, self.dim, self._value)):
            return self._value
        else:
            return self.array

    @value.setter
    def value(self, val):
        self._value = val
        self.array = dfu.as_array(self.mesh, self.dim, val)

    @property
    def array(self):
        """Field value as a numpy array.

        Returns
        -------
        numpy.ndarray
            Field values array.

        Parameters
        ----------
        numpy.ndarray
            The dimensions must be ``(field.mesh.n[0],
            field.mesh.n[1], field.mesh.n[2], dim)``

        Examples
        --------
        Different ways of setting and getting property `value`.

        >>> import discretisedfield as df
        >>> p1 = (0, 0, 0)
        >>> p2 = (1, 1, 1)
        >>> cell = (0.5, 1, 1)
        >>> mesh = df.Mesh(p1=p1, p2=p2, cell=cell)
        >>> value = (0, 0, 1)
        >>> name = "uniform_field"
        >>> field = df.Field(mesh=mesh, dim=3, value=value, name=name)
        >>> # array getter
        >>> field.array.shape
        (2, 1, 1, 3)

        .. note::

           Please note this method is a property and should be called
           as ``field.array``, not ``field.array()``.

        .. seealso:: :py:func:`~discretisedfield.Field.value`

        """
        return self._array

    @array.setter
    def array(self, val):
        self._array = val

    @property
    def norm(self):
        current_norm = np.linalg.norm(self.array, axis=self.dim)[..., None]
        if self._norm:
            if np.array_equiv(current_norm, self._norm.array):
                return self._norm

        return Field(self.mesh, dim=1, value=current_norm, name="norm")

    @norm.setter
    def norm(self, val):
        if val:
            if self.dim == 1:
                msg = "Cannot normalise field with dim={}.".format(self.dim)
                raise ValueError(msg)

            if not np.any(self.array):
                msg = "Cannot normalise field with all zero values."
                raise ValueError(msg)

            self._norm = Field(mesh=self.mesh, dim=1, value=val, name="norm")
            self.array /= np.linalg.norm(self.array, axis=self.dim)[..., None]
            self.array *= self._norm.array
        else:
            self._norm = val

    @property
    def average(self):
        """Compute the finite difference field average.

        Returns:
          Finite difference field average.

        """
        return tuple(self.array.mean(axis=(0, 1, 2)))

    def __repr__(self):
        """Representation method."""
        rstr = "<Field(mesh={}, dim={}, name=\"{}\")>"
        return rstr.format(repr(self.mesh), self.dim, self.name)

    def __call__(self, point):
        """Sample the field at `point`.

        Args:
          p (tuple): point coordinate at which the field is sampled

        Returns:
          Field value in cell containing point p

        """
        field_value = self.array[self.mesh.point2index(point)]
        if len(field_value) == 1:
            return field_value
        else:
            return tuple(field_value)

    def __getattr__(self, name):
        if name in list(dfu.axesdict.keys())[:self.dim] and 1 < self.dim <= 3:
            val = self.array[..., dfu.axesdict[name]][..., None]
            fieldname = "{}_{}".format(self.name, name)
            return Field(mesh=self.mesh, dim=1, value=val, name=fieldname)
        else:
            msg = "{} object has no attribute {}."
            raise AttributeError(msg.format(type(self).__name__, name))

    def __dir__(self):
        if 1 < self.dim <= 3:
            extension = list(dfu.axesdict.keys())[:self.dim]
        else:
            extension = []
        return list(self.__dict__.keys()) + extension

    def plane(self, *args, x=None, y=None, z=None, n=None):
        for point in self.mesh.plane(*args, x=x, y=y, z=z, n=n):
            yield point, self.__call__(point)

    def _plot_data(self, *args, x=None, y=None, z=None, n=None, ax=None):
        info = dfu.plane_info(*args, x=x, y=y, z=z)
        data = list(self.plane(*args, x=x, y=y, z=z, n=n))
        ps, vs = list(zip(*data))
        points = list(zip(*ps))
        values = list(zip(*vs))

        if n is None:
            n = (self.mesh.n[info["haxis"]], self.mesh.n[info["vaxis"]])

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        return info, points, values, n, ax

    def imshow(self, *args, x=None, y=None, z=None, n=None, ax=None, **kwargs):
        info, points, values, n, ax = self._plot_data(*args, x=x, y=y, z=z,
                                                      n=n, ax=ax)

        extent = [self.mesh.pmin[info["haxis"]], self.mesh.pmax[info["haxis"]],
                  self.mesh.pmin[info["vaxis"]], self.mesh.pmax[info["vaxis"]]]
        imax = ax.imshow(np.array(values).reshape(n).T, origin="lower",
                         extent=extent, **kwargs)

        return imax

    def quiver(self, *args, x=None, y=None, z=None, n=None, ax=None,
               colour=None, **kwargs):
        info, points, values, n, ax = self._plot_data(*args, x=x, y=y, z=z,
                                                      n=n, ax=ax)

        if not any(values[info["haxis"]] + values[info["vaxis"]]):
            kwargs["scale"] = 1

        if colour is None:
            qvax = ax.quiver(points[info["haxis"]], points[info["vaxis"]],
                             values[info["haxis"]], values[info["vaxis"]],
                             pivot='mid', **kwargs)
        elif colour in dfu.axesdict.keys():
            qvax = ax.quiver(points[info["haxis"]], points[info["vaxis"]],
                             values[info["haxis"]], values[info["vaxis"]],
                             values[dfu.axesdict[colour]],
                             pivot='mid', **kwargs)

        return qvax

    def colorbar(self, ax, colouredplot, cax=None, **kwargs):
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)

        plt.colorbar(colouredplot, cax=cax, **kwargs)

    def plot_plane(self, *args, x=None, y=None, z=None, n=None, ax=None):
        info, points, values, n, ax = self._plot_data(*args, x=x, y=y, z=z,
                                                      n=n, ax=ax)

        if self.dim > 1:
            self.quiver(*args, x=x, y=y, z=z, n=n, ax=ax)
            scfield = getattr(self, list(dfu.axesdict.keys())[info["slice"]])
        else:
            scfield = self

        colouredplot = scfield.imshow(*args, x=x, y=y, z=z, n=n, ax=ax)
        self.colorbar(ax, colouredplot)

        ax.set_xlabel(list(dfu.axesdict.keys())[info["haxis"]])
        ax.set_ylabel(list(dfu.axesdict.keys())[info["vaxis"]])

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

    def _writeovf(self, filename, representation="text"):
        f = open(filename, "w")

        # Define header lines.
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
                  "ybase: {}".format(self.mesh.pmin[0] +
                                     self.mesh.cell[1]/2),
                  "zbase: {}".format(self.mesh.pmin[0] +
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
                  "valuedim: {}".format(3),
                  "valuelabels: {0}_x {0}_y {0}_z".format(self.name),
                  "valueunits: A/m A/m A/m",
                  "",
                  "End: Header",
                  ""]

        if representation == "binary":
            header.append("Begin: Data Binary 8")
            footer = ["End: Data Binary 8",
                      "End: Segment"]
        if representation == "text":
            header.append("Begin: Data Text")
            footer = ["End: Data Text",
                      "End: Segment"]

        # Write header lines to OOMMF file.
        for line in header:
            if not line:
                f.write("#\n")
            else:
                f.write("# " + line + "\n")
        if representation == "binary":
            # Close the file and reopen with binary write
            # appending to end of file.
            f.close()
            f = open(filename, "ab")
            # Add the 8 bit binary check value that OOMMF uses
            packarray = [123456789012345.0]
            # Write data lines to OOMMF file.
            for i in self.mesh.indices:
                [packarray.append(vi) for vi in self.array[i]]

            v_binary = struct.pack("d"*len(packarray), *packarray)
            f.write(v_binary)
            f.close()
            f = open(filename, "a")

        else:
            for i in self.mesh.indices:
                if self.dim == 3:
                    v = [vi for vi in self.array[i]]
                elif self.dim == 1:
                    v = [self.array[i][0], 0, 0]
                else:
                    msg = ("Cannot write dim={} field to "
                           "omf file.".format(self.dim))
                    raise TypeError(msg)
                for vi in v:
                    f.write(" " + str(vi))
                f.write("\n")

        # Write footer lines to OOMMF file.
        for line in footer:
            f.write("# " + line + "\n")

        # Close the file.
        f.close()
