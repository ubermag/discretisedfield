"""A Python package for analysing and manipulating finite difference fields.

This module is a Python package that provides:

- Analysing vector fields, such as sampling, averaging, plotting, etc.

discretisedfield is a member of JOOMMF project - a part of OpenDreamKit
Horizon 2020 European Research Infrastructure project

"""

import random
import numpy as np
import matplotlib.pyplot as plt


class Field(object):
    def __init__(self, cmin, cmax, d, dim=3, value=None, name='unnamed'):
        """Class for analysing, manipulating, and writing finite diffrenece fields.

        This class provides the functionality for:
          - Creating FD vector and scalar fields.
          - Plotting FD fields.
          - Computing common values characterising FD fields.

        Args:
          cmin (tuple): The minimum coordinate range.
            cmin tuple is of length 3 and defines the minimum x, y, and z
            coordinates of the finite difference domain: (xmin, ymin, zmax)
          cmax (tuple): The maximum coordinate range.
            cmax tuple is of length 3 and defines the maximum x, y, and z
            coordinates of the finite difference domain: (xmin, ymin, zmax)
          d (tuple): discretisation
            d is a discretisation tuple of length 3 and defines the
            discretisation steps in x, y, and z directions: (dx, dy, dz)
          dim (Optional[int]): The value dimensionality. Defaults to 3.
            If dim=1, a scalar field is initialised. On the other hand, if
            dim=3, a three dimensional vector field is created.
          value (Optional): Finite difference field values. Defaults to None.
            For the possible types of value argument, refer to set method.
            If no value argument is provided, a zero field is initialised.
          name (Optional[str]): Field name.

        Attributes:
          cmin (tuple): The minimum coordinate range. Defined in Args.

          cmax (tuple): The maximum coordinate range. Defined in Args.

          d (tuple): Discretisation. Defined in Args.

          dim (int): The value dimensionality. Defined in Args.

          name (str): Field name.

          l (tuple): length of domain x, y, and z edges (lx, ly, lz):

            lx = xmax - xmin

            ly = ymax - ymin

            lz = zmax - zmin

          n (tuple): The number of cells in all three dimensions (nx, ny, nz):

            nx = lx/dx

            ny = ly/dy

            nz = lz/dz

          f (np.ndarray): A field value four-dimensional numpy array.

        Example:

          .. code-block:: python

            >>> from discretisedfield import Field
            >>> cmin = (0, 0, 0)
            >>> cmax = (10, 5, 3)
            >>> d = (1, 0.5, 0.1)
            >>> value = (0.5, -0.3, 6)
            >>> field = Field(cmin, cmax, d, value=value, name='fdfield')

        """
        # Raise exceptions if invalid arguments are provided.
        if not isinstance(cmin, tuple) or \
           not all(isinstance(i, (float, int)) for i in cmin) or \
           len(cmin) != 3:
            raise TypeError("""cmin must be a 3-element tuple of
                            int or float values.""")
        if not isinstance(cmax, tuple) or \
           not all(isinstance(i, (float, int)) for i in cmax) or \
           len(cmax) != 3:
            raise TypeError("""cmax must be a 3-element tuple of
                            int or float values.""")
        if not isinstance(d, tuple) or \
           any(i <= 0 for i in d) or \
           len(d) != 3:
            raise TypeError("""d must be a 3-element tuple of positive
                            int or float values.""")
        if not isinstance(dim, int) or \
           dim <= 0:
            raise TypeError("""dim must be a positive int value.""")
        if not isinstance(name, str):
            raise TypeError("""name must be a string.""")

        # Copy arguments to attributes.
        self.cmin = cmin
        self.cmax = cmax
        self.d = d
        self.dim = dim
        self.name = name

        # Compute domain edge lengths.
        self.l = (self.cmax[0]-self.cmin[0],
                  self.cmax[1]-self.cmin[1],
                  self.cmax[2]-self.cmin[2])

        # Check if domain edge lengths are multiples of d.
        tol = 1e-12
        if self.d[0] - tol > self.l[0] % self.d[0] > tol or \
           self.d[1] - tol > self.l[1] % self.d[1] > tol or \
           self.d[2] - tol > self.l[2] % self.d[2] > tol:
            raise ValueError('Domain is not a multiple of {}.'.format(self.d))

        # Compute the number of cells in x, y, and z directions.
        self.n = (int(round(self.l[0]/self.d[0])),
                  int(round(self.l[1]/self.d[1])),
                  int(round(self.l[2]/self.d[2])))

        # Create an empty 3d vector field.
        self.f = np.zeros([self.n[0], self.n[1], self.n[2], dim])

        # Set the Field value if not None.
        if value is not None:
            self.set(value)

    def __call__(self, c):
        """Sample the field at coordinate c.

        Args:
          c (tuple): coordinate at which the field is sampled.

        Returns:
          Field value in cell containing coordinate c.

        Example:

        .. code-block:: python

          >>> from discretisedfield import Field
          >>> cmin = (0, 0, 0)
          >>> cmax = (10, 10, 10)
          >>> d = (1, 1, 1)
          >>> value = (1, 0, -5)
          >>> field = Field(cmin, cmax, d, value=value)
          >>> c = (5.5, 0.5, 3.5)
          >>> field(c)
          array([ 1.,  0., -5.])

        """
        return self.sample(c)

    def domain_centre(self):
        """Compute and return the domain centre coordinate.

        Returns:
          A domain centre coordinate tuple.

        Example:

        .. code-block:: python

          >>> from discretisedfield import Field
          >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

          >>> field.domain_centre()
          (2.5, 2.0, 1.5)

        """
        c = (self.cmin[0] + 0.5*self.l[0],
             self.cmin[1] + 0.5*self.l[1],
             self.cmin[2] + 0.5*self.l[2])

        return c

    def random_coord(self):
        """Generate a random coordinate in the domain.

        Returns:
          A random domain coordinate.

        Example:

        .. code-block:: python

           from discretisedfield import Field
           field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           field.random_coord()

        """
        c = (self.cmin[0] + random.random()*self.l[0],
             self.cmin[1] + random.random()*self.l[1],
             self.cmin[2] + random.random()*self.l[2])

        return c

    def index2coord(self, i):
        """Convert the cell index to its coordinate.

        The finite difference domain is disretised in x, y, and z directions
        in steps dx, dy, and dz steps, respectively. Accordingly, there are
        nx, ny, and nz discretisation steps. This method converts the cell
        index (ix, iy, iz) to the cell's centre coordinate.

        This method raises ValueError if the index is out of range.

        Args:
          i (tuple): A length 3 tuple of integers (ix, iy, iz)

        Returns:
          A length 3 tuple of x, y, and z coodinates.

        Example:

        .. code-block:: python

           >>> from discretisedfield import Field
           >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           >>> i = (2, 2, 1)
           >>> field.index2coord(i)
           (2.5, 2.5, 1.5)

        """
        if i[0] < 0 or i[0] > self.n[0]-1 or \
           i[1] < 0 or i[1] > self.n[1]-1 or \
           i[2] < 0 or i[2] > self.n[2]-1:
            raise ValueError('Index {} out of range.'.format(i))

        else:
            c = (self.cmin[0] + (i[0] + 0.5)*self.d[0],
                 self.cmin[1] + (i[1] + 0.5)*self.d[1],
                 self.cmin[2] + (i[2] + 0.5)*self.d[2])

        return c

    def coord2index(self, c):
        """Convert the cell's coordinate to its index.

        This method is an inverse function of index2coord method.
        (For details on index, please refer to the index2coord method.)
        More precisely, this method return the index of a cell containing
        the coordinate c.

        This method raises ValueError if the index is out of range.

        Args:
          c (tuple): A length 3 tuple of integers/floats (cx, cy, cz)

        Returns:
          A length 3 tuple of cell's indices (ix, iy, iz).

        Example:

        .. code-block:: python

           >>> from discretisedfield import Field
           >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           >>> c = (2.3, 2.1, 0.8)
           >>> field.coord2index(c)
           (2, 2, 0)

        """
        if c[0] < self.cmin[0] or c[0] > self.cmax[0] or \
           c[1] < self.cmin[1] or c[1] > self.cmax[1] or \
           c[2] < self.cmin[2] or c[2] > self.cmax[2]:
            raise ValueError('Coordinate {} out of domain.'. format(c))

        else:
            i = [int(round(float(c[0]-self.cmin[0])/self.d[0] - 0.5)),
                 int(round(float(c[1]-self.cmin[1])/self.d[1] - 0.5)),
                 int(round(float(c[2]-self.cmin[2])/self.d[2] - 0.5))]

            # If rounded to the out-of-range index.
            for j in range(3):
                if i[j] < 0:
                    i[j] = 0
                elif i[j] > self.n[j] - 1:
                    i[j] = self.n[j] - 1

        return tuple(i)

    def nearestcellcoord(self, c):
        """Find the cell coordinate nearest to c.

        This method computes the cell's centre coordinate containing
        the coodinate c.

        Args:
          c (tuple): A length 3 tuple of integers/floats.

        Returns:
          A length 3 tuple of integers/floats.

        Example:

        .. code-block:: python

          >>> from discretisedfield import Field
          >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

          >>> c = (2.3, 2.1, 0.8)
          >>> field.nearestcellcoord(c)
          (2.5, 2.5, 0.5)

        """
        return self.index2coord(self.coord2index(c))

    def sample(self, c):
        """Sample the Field at coordinate c.

        Compute the vector field value at the domain's coordinate c.
        Due to the finite difference discretisation, the value this method
        returns is the same for any coorinate in the sell.

        Args:
          c (tuple): A A length 3 tuple of integers/floats.

        Returns:
          The field value at coodinate c.

        Example:

        .. code-block:: python

           >>> from discretisedfield import Field
           >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           >>> c = (2.3, 2.1, 0.8)
           >>> field.sample(c)
           array([ 0.,  0.,  0.])

        """
        i = self.coord2index(c)
        return self.f[i[0], i[1], i[2]]

    def set(self, value, normalise=False):
        """Set the field value.

        This method sets the field values at all finite difference
        domain cells.

        Args:
          value: This argument can be an integer, float, tuple, list,
            np.ndarray, or Python function.

          normalise (bool): If True, the vector field value is
            normalised to 1.

        Example:

        .. code-block:: python

           >>> from discretisedfield import Field
           >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           >>> # Set the field value with int/float
           >>> value = 2.1
           >>> field.set(value)

           >>> # Set the field value with list/tuple/np.ndarray
           >>> value = [1, -0.2, 3.5]
           >>> field.set(value, normalise=True)

           >>> # Set the field value using Python function.
           >>> def m_init(pos):
           ...     x, y, z = pos
           ...     return (x+1, x**2+y, -z)

           >>> field.set(m_init)

        """
        # value is an int or float.
        # All components of the Field are set to value.
        if isinstance(value, (int, float)):
            self.f.fill(value)

        # value is a constant tuple, list or numpy array.
        elif isinstance(value, (tuple, list, np.ndarray)):
            for i in range(self.dim):
                self.f[:, :, :, i].fill(value[i])

        # value is a Python function.
        elif hasattr(value, '__call__'):
            for ix in range(self.n[0]):
                for iy in range(self.n[1]):
                    for iz in range(self.n[2]):
                        i = (ix, iy, iz)
                        coord = self.index2coord((ix, iy, iz))
                        self.f[ix, iy, iz, :] = value(coord)

        else:
            raise TypeError("Cannot set field using {}.".format(type(value)))

        # Normalise the vector field if required.
        if normalise:
            self.normalise()

    def set_at_index(self, i, value):
        """Set the field value at index i.

        This method sets the field value at a single index i.

        Args:
          i (tuple): A length 3 tuple of integers (ix, iy, iz)

        Example:

        .. code-block:: python

           >>> from discretisedfield import Field
           >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           >>> i = (2, 2, 1)
           >>> value = 5
           >>> field.set_at_index(i, value)
           >>> field.f[2, 2, 1]
           array([ 5.,  5.,  5.])
        """
        self.f[i[0], i[1], i[2], :] = value

    def average(self):
        """Compute the finite difference field average.

        Returns:
          Finite difference field average.

        Example:

        .. code-block:: python

           >>> from discretisedfield import Field
           >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           >>> field.set((1, 0, 5))
           >>> field.average()
           [1.0, 0.0, 5.0]

        """
        # Scalar field.
        if self.dim == 1:
            return np.mean(self.f)

        # Vector field.
        else:
            average = []
            for i in range(self.dim):
                average.append(np.mean(self.f[:, :, :, i]))

        return average

    def slice_field(self, axis, point):
        """Returns the slice of a finite difference field.

        This method returns the values of a finite difference field
        on the plane perpendicular to the axis at point.

        Args:
          axis (str): An axis to which the sampling plane is perpendicular to.
          point (int/float): The coorindta eon axis at which the field is
            sampled.

        Returns:
          A 4 element tuple containing:

            - Axis 1 coodinates
            - Axis 2 coodinates
            - np.ndarray of field values on the plane
            - coordinate system details

        Example:

        .. code-block:: python

           from discretisedfield import Field
           field = Field((0, 0, 0), (2, 2, 2), (1, 1, 1))

           field.set((1, 0, 5))
           field.slice_field('z', 0.5)

        """
        if axis == 'x':
            slice_num = 0
            axes = (1, 2)
        elif axis == 'y':
            slice_num = 1
            axes = (0, 2)
        elif axis == 'z':
            slice_num = 2
            axes = (0, 1)
        else:
            raise ValueError('Axis not properly defined.')

        if self.cmin[slice_num] <= point <= self.cmax[slice_num]:
            axis1_indices = np.arange(0, self.n[axes[0]])
            axis2_indices = np.arange(0, self.n[axes[1]])

            axis1_coords = np.zeros(len(axis1_indices))
            axis2_coords = np.zeros(len(axis2_indices))

            sample_centre = list(self.domain_centre())
            sample_centre[slice_num] = point
            sample_centre = tuple(sample_centre)

            slice_index = self.coord2index(sample_centre)[slice_num]

            field_slice = np.zeros([self.n[axes[0]],
                                    self.n[axes[1]],
                                    self.dim])
            for j in axis1_indices:
                for k in axis2_indices:
                    i = [0, 0, 0]
                    i[slice_num] = slice_index
                    i[axes[0]] = j
                    i[axes[1]] = k
                    i = tuple(i)

                    coord = self.index2coord(i)

                    axis1_coords[j] = coord[axes[0]]
                    axis2_coords[k] = coord[axes[1]]

                    field_slice[j, k, :] = self.f[i[0], i[1], i[2], :]
            coord_system = (axes[0], axes[1], slice_num)

        else:
            raise ValueError('Point {} outside the domain.'.format(point))

        return axis1_coords, axis2_coords, field_slice, coord_system

    def plot_slice(self, axis, point, xsize=10, axes=True, grid=True):
        """Plot the field slice.

        This method plots the field slice that is obtained
        using slice_field method.

        Args:
          axis (str): An axis to which the sampling plane is perpendicular to.
          point (int/float): The coorindta eon axis at which the field
            is sampled.
          xsize (Optional[int/float]): The horizonatl size of a plot.
          grid (Optional[bool]): If True, grid is shown in the plot.

        Returns:
          matplotlib figure.

        Example:

        .. code-block:: python

           from discretisedfield import Field
           field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           field.set((1, 0, 5))
           print field.plot_slice('z', 0.5)

        """
        a1, a2, field_slice, coord_system = self.slice_field(axis, point)

        # Vector field.
        if self.dim == 3:
            pm = self._prepare_for_quiver(a1, a2, field_slice, coord_system)

            if np.allclose(pm[:, 2], 0) and np.allclose(pm[:, 3], 0):
                raise ValueError('Vector plane components are zero.')
            else:
                ysize = xsize*(self.l[coord_system[1]]/self.l[coord_system[0]])
                fig = plt.figure(figsize=(xsize, ysize))
                plt.quiver(pm[:, 0], pm[:, 1], pm[:, 2], pm[:, 3], pm[:, 4])
                plt.xlim([self.cmin[coord_system[0]],
                          self.cmax[coord_system[0]]])
                plt.ylim([self.cmin[coord_system[1]],
                          self.cmax[coord_system[1]]])
                if axes:
                    plt.xlabel('xyz'[coord_system[0]] + ' (m)')
                    plt.ylabel('xyz'[coord_system[1]] + ' (m)')
                    plt.title('xyz'[coord_system[2]] + ' slice')
                if not axes:
                    plt.axis('off')
                if grid:
                    plt.grid()

        return fig

    def _prepare_for_quiver(self, a1, a2, field_slice, coord_system):
        """Generate arrays for plotting quiver plot."""
        nel = self.n[coord_system[0]]*self.n[coord_system[1]]
        plot_matrix = np.zeros([nel, 5])

        counter = 0
        for j in range(self.n[coord_system[0]]):
            for k in range(self.n[coord_system[1]]):
                entry = [a1[j], a2[k],
                         field_slice[j, k, coord_system[0]],
                         field_slice[j, k, coord_system[1]],
                         field_slice[j, k, coord_system[2]]]
                plot_matrix[counter, :] = np.array(entry)
                counter += 1

        return plot_matrix

    def normalise(self, norm=1):
        """Normalise the finite difference vector field.

        If the finite difference field is multidimensional (vector),
        its value is normalised so that at all points.

        Args:
          norm (int/float): Norm value at all finite difference cells.

        Example:

        .. code-block:: python

           >>> from discretisedfield import Field
           >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

           >>> field.set((1, 0, 5))
           >>> field.normalise(5)

        """
        # Scalar field.
        if self.dim == 1:
            raise NotImplementedError("""Normalisation of scalar
                                      fields is not implemented.""")

        # Vector field.
        else:
            # Compute norm.
            f_norm = 0
            for j in range(self.dim):
                f_norm += self.f[:, :, :, j]**2
            f_norm = np.sqrt(f_norm)

            # Normalise every component.
            for j in range(self.dim):
                self.f[:, :, :, j] = norm * self.f[:, :, :, j]/f_norm

    def write_oommf_file(self, filename):
        """Write the FD field to the OOMMF (omf, ohf) file.

        This method writes all necessary data to the omf or ohf file,
        so that it can be read by OOMMF.

        Args:
          filename (str): filename including extension

        Example:

        .. code-block:: python

          >>> from oommffield import Field
          >>> field = Field((0, 0, 0), (5, 4, 3), (1, 1, 1))

          >>> field.set((1, 0, 5))
          >>> field.write_oommf_file('fdfield.omf')

        """
        oommf_file = open(filename, 'w')

        # Define header lines.
        header_lines = ['OOMMF OVF 2.0',
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
                        'xbase: {}'.format(self.d[0]),
                        'ybase: {}'.format(self.d[1]),
                        'zbase: {}'.format(self.d[2]),
                        'xnodes: {}'.format(self.n[0]),
                        'ynodes: {}'.format(self.n[1]),
                        'znodes: {}'.format(self.n[2]),
                        'xstepsize: {}'.format(self.d[0]),
                        'ystepsize: {}'.format(self.d[1]),
                        'zstepsize: {}'.format(self.d[2]),
                        'xmin: {}'.format(self.cmin[0]),
                        'ymin: {}'.format(self.cmin[1]),
                        'zmin: {}'.format(self.cmin[2]),
                        'xmax: {}'.format(self.cmax[0]),
                        'ymax: {}'.format(self.cmax[1]),
                        'zmax: {}'.format(self.cmax[2]),
                        'valuedim: {}'.format(self.dim),
                        'valuelabels: Mx My Mz',
                        'valueunits: A/m A/m A/m',
                        '',
                        'End: Header',
                        '',
                        'Begin: Data Text']

        # Define footer lines.
        footer_lines = ['# End: Data Text\n',
                        '# End: Segment']

        # Write header lines to OOMMF file.
        for line in header_lines:
            oommf_file.write('# ' + line + '\n')

        # Write data lines to OOMMF file.
        for iz in range(self.n[2]):
            for iy in range(self.n[1]):
                for ix in range(self.n[0]):
                    v = [str(vi) for vi in self.f[ix, iy, iz, :]]
                    for vi in v:
                        oommf_file.write(' ' + vi)
                    oommf_file.write('\n')

        # Write footer lines to OOMMF file.
        for line in footer_lines:
            oommf_file.write('# ' + line + '\n')

        # Close the file.
        oommf_file.close()


def read_oommf_file(filename, name='unnamed'):
    """Read the OOMMF file and create an Field object.

    Args:
      filename (str): OOMMF file name
      name (str): name of the Field object

    Return:
      Field object.

    Example:

        .. code-block:: python

          from oommffield import read_oommf_file
          oommf_filename = 'vector_field.omf'
          field = read_oommf_file(oommf_filename, name='magnetisation')

    """
    # Open and read the file.
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    # Load metadata.
    dic = {'xmin': None, 'ymin': None, 'zmin': None,
           'xmax': None, 'ymax': None, 'zmax': None,
           'xstepsize': None, 'ystepsize': None, 'zstepsize': None,
           'xbase': None, 'ybase': None, 'zbase': None,
           'xnodes': None, 'ynodes': None, 'znodes': None,
           'valuedim': None}

    for line in lines[0:50]:
        for key in dic.keys():
            if line.find(key) != -1:
                dic[key] = float(line.split()[2])

    cmin = (dic['xmin'], dic['ymin'], dic['zmin'])
    cmax = (dic['xmax'], dic['ymax'], dic['zmax'])
    d = (dic['xstepsize'], dic['ystepsize'], dic['zstepsize'])
    cbase = (dic['xbase'], dic['ybase'], dic['zbase'])
    n = (int(round(dic['xnodes'])),
         int(round(dic['ynodes'])),
         int(round(dic['znodes'])))
    dim = int(dic['valuedim'])

    field = Field(cmin, cmax, d, dim, name=name)

    for j in range(len(lines)):
        if lines[j].find('Begin: Data') != -1:
            data_first_line = j+1

    counter = 0
    for iz in range(n[2]):
        for iy in range(n[1]):
            for ix in range(n[0]):
                i = (ix, iy, iz)
                line_data = lines[data_first_line+counter]
                value = [float(vi) for vi in line_data.split()]
                field.set_at_index(i, value)

                counter += 1

    return field
