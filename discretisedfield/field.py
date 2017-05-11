import pyvtk
import struct
import itertools
import numpy as np
import joommfutil.typesystem as ts
import discretisedfield as df
import discretisedfield.util as dfu
import matplotlib.pyplot as plt


@ts.typesystem(mesh=ts.TypedAttribute(expected_type=df.Mesh),
               dim=ts.UnsignedInt,
               name=ts.ObjectName)
class Field(dfu.Field):
    """Finite Difference field

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
    name : str, optional, optional
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
        if np.array_equiv(current_norm, self._norm.array):
            return self._norm
        else:
            return Field(self.mesh, dim=1, value=current_norm, name="norm")

    @norm.setter
    def norm(self, val):
        if val is not None:
            if self.dim == 1:
                msg = "Cannot normalise field with dim={}.".format(self.dim)
                raise ValueError(msg)

            if not np.any(self.array):
                msg = "Cannot normalise field with all zero values."
                raise ValueError(msg)

            self._norm = Field(mesh=self.mesh, dim=1, value=val, name="norm")
            self._normalise()
        else:
            self._norm = val

    def _normalise(self):
        self.array /= np.linalg.norm(self.array, axis=self.dim)[..., None]
        self.array *= self._norm.array

    @property
    def average(self):
        """Compute the finite difference field average.

        Returns:
          Finite difference field average.

        """
        return tuple(self.array.mean(axis=(0, 1, 2)))

    def __repr__(self):
        """Representation method."""
        rstr = ("Field(mesh={}, dim={}, "
                "name=\"{}\")").format(repr(self.mesh), self.dim, self.name)
        return rstr

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

    def slice_field(self, axis, point):
        """Returns the field slice.

        This method returns the values of a finite difference field
        on the plane perpendicular to the "axis" at "point".

        Args:
          axis (str): An axis to which the sampling plane is perpendicular to.
          point (int/float): The coordinate on axis at which the field is
            sampled.

        Returns:
          A 4 element tuple containing:

            - Axis 1 coodinates
            - Axis 2 coodinates
            - np.ndarray of field values on the plane
            - coordinate system details

        """
        if axis == "x":
            slice_num = 0
            axes = (1, 2)
        elif axis == "y":
            slice_num = 1
            axes = (0, 2)
        elif axis == "z":
            slice_num = 2
            axes = (0, 1)
        else:
            raise ValueError("Axis not properly defined.")

        if self.mesh.pmin[slice_num] <= point <= self.mesh.pmax[slice_num]:
            axis1_indices = np.arange(0, self.mesh.n[axes[0]])
            axis2_indices = np.arange(0, self.mesh.n[axes[1]])

            axis1_coords = np.zeros(len(axis1_indices))
            axis2_coords = np.zeros(len(axis2_indices))

            sample_centre = list(self.mesh.centre)
            sample_centre[slice_num] = point
            sample_centre = tuple(sample_centre)

            slice_index = self.mesh.point2index(sample_centre)[slice_num]

            field_slice = np.zeros([self.mesh.n[axes[0]],
                                    self.mesh.n[axes[1]],
                                    self.dim])
            for j in axis1_indices:
                for k in axis2_indices:
                    i = [0, 0, 0]
                    i[slice_num] = slice_index
                    i[axes[0]] = j
                    i[axes[1]] = k
                    i = tuple(i)

                    coord = self.mesh.index2point(i)

                    axis1_coords[j] = coord[axes[0]]
                    axis2_coords[k] = coord[axes[1]]

                    field_slice[j, k, :] = self.array[i]
            coord_system = (axes[0], axes[1], slice_num)

        else:
            raise ValueError("Point {} outside the domain.".format(point))

        return axis1_coords, axis2_coords, field_slice, coord_system

    def plot_slice(self, axis, point, xsize=10, axes=True, grid=True):
        """Plot the field slice.

        This method plots the field slice that is obtained
        using slice_field method.

        Args:
          axis (str): An axis to which the sampling plane is perpendicular to.
          point (int/float): The coordinate axis at which the field
            is sampled.
          xsize (Optional[int/float]): The horizontal size of a plot.
          grid (Optional[bool]): If True, grid is shown in the plot.

        Returns:
          matplotlib figure.

        """
        a1, a2, field_slice, coord_system = self.slice_field(axis, point)

        # Vector field
        if self.dim == 3:
            pm = self._prepare_for_quiver(a1, a2, field_slice, coord_system)

            if np.allclose(pm[:, 2], 0) and np.allclose(pm[:, 3], 0):
                raise ValueError("Vector plane components are zero.")
            ysize = xsize*(self.mesh.l[coord_system[1]] /
                           self.mesh.l[coord_system[0]])
            fig = plt.figure(figsize=(xsize, ysize))
            plt.quiver(pm[:, 0], pm[:, 1], pm[:, 2], pm[:, 3], pm[:, 4])
        elif self.dim == 1:
            ysize = xsize*(self.mesh.l[coord_system[1]] /
                           self.mesh.l[coord_system[0]])
            fig = plt.figure(figsize=(xsize, ysize))
            extent = [self.mesh.pmin[coord_system[0]],
                      self.mesh.pmax[coord_system[0]],
                      self.mesh.pmin[coord_system[1]],
                      self.mesh.pmax[coord_system[1]]]
            plt.imshow(field_slice[..., 0], extent=extent)
        else:
            raise TypeError(("Cannot plot slice of field with "
                             "dim={}".format(self.dim)))

        plt.xlim([self.mesh.pmin[coord_system[0]],
                  self.mesh.pmax[coord_system[0]]])
        plt.ylim([self.mesh.pmin[coord_system[1]],
                  self.mesh.pmax[coord_system[1]]])
        if axes:
            plt.xlabel("xyz"[coord_system[0]] + " (m)")
            plt.ylabel("xyz"[coord_system[1]] + " (m)")
            plt.title("xyz"[coord_system[2]] + " slice")
        if not axes:
            plt.axis("off")
        if grid:
            plt.grid()

        return fig

    def _prepare_for_quiver(self, a1, a2, field_slice, coord_system):
        """Generate arrays for plotting quiver plot."""
        nel = self.mesh.n[coord_system[0]]*self.mesh.n[coord_system[1]]
        plot_matrix = np.zeros([nel, 5])

        counter = 0
        for j in range(self.mesh.n[coord_system[0]]):
            for k in range(self.mesh.n[coord_system[1]]):
                entry = [a1[j], a2[k],
                         field_slice[j, k, coord_system[0]],
                         field_slice[j, k, coord_system[1]],
                         field_slice[j, k, coord_system[2]]]
                plot_matrix[counter, :] = np.array(entry)
                counter += 1

        return plot_matrix

    def tovtk(self, filename):
        grid = [pmini + np.linspace(0, li, ni+1) for pmini, li, ni in
                zip(self.mesh.pmin, self.mesh.l, self.mesh.n)]

        structure = pyvtk.RectilinearGrid(*grid)
        vtkdata = pyvtk.VtkData(structure)

        vectors = [self.__call__(i) for i in self.mesh.coordinates]
        vtkdata.cell_data.append(pyvtk.Vectors(vectors, "m"))
        vtkdata.cell_data.append(pyvtk.Scalars(zip(*vectors)[0], "mx"))
        vtkdata.cell_data.append(pyvtk.Scalars(zip(*vectors)[1], "my"))
        vtkdata.cell_data.append(pyvtk.Scalars(zip(*vectors)[2], "mz"))

        vtkdata.tofile(filename)

    def write_oommf_file(self, filename, datatype="text"):
        """Write the FD field to the OOMMF (omf, ohf) file.
        This method writes all necessary data to the omf or ohf file,
        so that it can be read by OOMMF.
        Args:
          filename (str): filename including extension
          type(str): Either "text" or "binary"

        """
        oommf_file = open(filename, "w")

        # Define header lines.
        header_lines = ["OOMMF OVF 2.0",
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
                        ("valuelabels: Magnetization_x "
                         "Magnetization_y Magnetization_z"),
                        "valueunits: A/m A/m A/m",
                        "",
                        "End: Header",
                        ""]

        if datatype == "binary":
            header_lines.append("Begin: Data Binary 8")
            footer_lines = ["End: Data Binary 8",
                            "End: Segment"]
        if datatype == "text":
            header_lines.append("Begin: Data Text")
            footer_lines = ["End: Data Text",
                            "End: Segment"]

        # Write header lines to OOMMF file.
        for line in header_lines:
            if line == "":
                oommf_file.write("#\n")
            else:
                oommf_file.write("# " + line + "\n")
        if datatype == "binary":
            # Close the file and reopen with binary write
            # appending to end of file.
            oommf_file.close()
            oommf_file = open(filename, "ab")
            # Add the 8 bit binary check value that OOMMF uses
            packarray = [123456789012345.0]
            # Write data lines to OOMMF file.
            for i in self.mesh.indices:
                [packarray.append(vi) for vi in self.array[i]]

            v_binary = struct.pack("d"*len(packarray), *packarray)
            oommf_file.write(v_binary)
            oommf_file.close()
            oommf_file = open(filename, "a")

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
                    oommf_file.write(" " + str(vi))
                oommf_file.write("\n")

        # Write footer lines to OOMMF file.
        for line in footer_lines:
            oommf_file.write("# " + line + "\n")

        # Close the file.
        oommf_file.close()
