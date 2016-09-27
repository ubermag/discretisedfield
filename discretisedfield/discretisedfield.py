"""This module is a Python package that provides:

- Analysing vector fields, such as sampling, averaging, plotting, etc.

It is a member of JOOMMF project - a part of OpenDreamKit
Horizon 2020 European Research Infrastructure project.

"""
import random
import numpy as np
import matplotlib.pyplot as plt
from .mesh import Mesh
import discretisedfield.util.typesystem as ts


@ts.typesystem(dim=ts.UnsignedInt,
               name=ts.String)
class Field(object):
    def __init__(self, mesh, dim=3, value=None, normalisedto=None, name='unnamed'):
        """Class for analysing, manipulating, and writing finite difference fields.

        This class provides the functionality for:
          - Creating FD vector and scalar fields.
          - Plotting FD fields.
          - Computing common values characterising FD fields.

        Args:
          mesh (Mesh): Finite difference mesh.
          dim (Optional[int]): The value dimensionality. Defaults to 3.
            If dim=1, a scalar field is initialised. On the other hand, if
            dim=3, a three dimensional vector field is created.
          value (Optional): Finite difference field values. Defaults to None.
            For the possible types of value argument, refer to set method.
            If no value argument is provided, a zero field is initialised.
          normalisedto (Optional[Real]): vector field norm
          name (Optional[str]): Field name.

        Attributes:
          mesh (Mesh): Finite difference mesh.

          dim (int): The value dimensionality. Defined in Args.

          name (str): Field name.

          f (np.ndarray): A field value four-dimensional numpy array.

        """
        if not isinstance(mesh, Mesh):
            raise TypeError("""mesh must be of type Mesh.""")

        self.mesh = mesh
        self.dim = dim
        self.name = name
        self.normalisedto = normalisedto

        # Create an empty field.
        self.f = np.zeros([self.mesh.n[0],
                           self.mesh.n[1],
                           self.mesh.n[2],
                           dim])

        # Set the Field value if not None.
        if value is not None:
            self.set(value)

    def __call__(self, c):
        """Sample the field at coordinate c.

        Args:
          c (tuple): coordinate at which the field is sampled.

        Returns:
          Field value in cell containing coordinate c.

        """
        return self.sample(c)

    def sample(self, c):
        """Sample the Field at coordinate c.

        Compute the vector field value at the domain's coordinate c.
        Due to the finite difference discretisation, the value this method
        returns is the same for any coorinate in the sell.

        Args:
          c (tuple): A A length 3 tuple of integers/floats.

        Returns:
          The field value at coodinate c.

        """
        i = self.mesh.coord2index(c)
        return self.f[i[0], i[1], i[2]]

    def set(self, value):
        """Set the field value.

        This method sets the field values at all finite difference
        domain cells.

        Args:
          value: This argument can be an integer, float, tuple, list,
            np.ndarray, or Python function.

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
            for ix in range(self.mesh.n[0]):
                for iy in range(self.mesh.n[1]):
                    for iz in range(self.mesh.n[2]):
                        i = (ix, iy, iz)
                        coord = self.mesh.index2coord((ix, iy, iz))
                        self.f[ix, iy, iz, :] = value(coord)

        else:
            raise TypeError("Cannot set field using {}.".format(type(value)))

        # Normalise the vector field if required.
        if self.normalisedto is not None:
            self.normalise()

    def set_at_index(self, i, value):
        """Set the field value at index i.

        This method sets the field value at a single index i.

        Args:
          i (tuple): A length 3 tuple of integers (ix, iy, iz)

        """
        self.f[i[0], i[1], i[2], :] = value

    def average(self):
        """Compute the finite difference field average.

        Returns:
          Finite difference field average.

        """
        # Scalar field.
        if self.dim == 1:
            return np.mean(self.f)

        # Vector field.
        else:
            average = []
            for i in range(self.dim):
                average.append(np.mean(self.f[:, :, :, i]))

        return tuple(average)

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

        if self.mesh.c1[slice_num] <= point <= self.mesh.c2[slice_num]:
            axis1_indices = np.arange(0, self.mesh.n[axes[0]])
            axis2_indices = np.arange(0, self.mesh.n[axes[1]])

            axis1_coords = np.zeros(len(axis1_indices))
            axis2_coords = np.zeros(len(axis2_indices))

            sample_centre = list(self.mesh.domain_centre())
            sample_centre[slice_num] = point
            sample_centre = tuple(sample_centre)

            slice_index = self.mesh.coord2index(sample_centre)[slice_num]

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

                    coord = self.mesh.index2coord(i)

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

        """
        a1, a2, field_slice, coord_system = self.slice_field(axis, point)

        # Vector field.
        if self.dim == 3:
            pm = self._prepare_for_quiver(a1, a2, field_slice, coord_system)

            if np.allclose(pm[:, 2], 0) and np.allclose(pm[:, 3], 0):
                raise ValueError('Vector plane components are zero.')
            else:
                ysize = xsize*(self.mesh.l[coord_system[1]] /
                               self.mesh.l[coord_system[0]])
                fig = plt.figure(figsize=(xsize, ysize))
                plt.quiver(pm[:, 0], pm[:, 1], pm[:, 2], pm[:, 3], pm[:, 4])
                plt.xlim([self.mesh.c1[coord_system[0]],
                          self.mesh.c2[coord_system[0]]])
                plt.ylim([self.mesh.c1[coord_system[1]],
                          self.mesh.c2[coord_system[1]]])
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

    def normalise(self):
        """Normalise the finite difference vector field.

        If the finite difference field is multidimensional (vector),
        its value is normalised so that at all points.

        """
        # Scalar field.
        if self.dim == 1:
            raise NotImplementedError

        # Vector field.
        else:
            # Compute norm.
            f_norm = 0
            for j in range(self.dim):
                f_norm += self.f[:, :, :, j]**2
            f_norm = np.sqrt(f_norm)

            # Normalise every component.
            for j in range(self.dim):
                self.f[:, :, :, j] = self.normalisedto*self.f[:, :, :, j]/f_norm

    def write_oommf_file(self, filename):
        """Write the FD field to the OOMMF (omf, ohf) file.

        This method writes all necessary data to the omf or ohf file,
        so that it can be read by OOMMF.

        Args:
          filename (str): filename including extension

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
                        'xbase: {}'.format(self.mesh.d[0]),
                        'ybase: {}'.format(self.mesh.d[1]),
                        'zbase: {}'.format(self.mesh.d[2]),
                        'xnodes: {}'.format(self.mesh.n[0]),
                        'ynodes: {}'.format(self.mesh.n[1]),
                        'znodes: {}'.format(self.mesh.n[2]),
                        'xstepsize: {}'.format(self.mesh.d[0]),
                        'ystepsize: {}'.format(self.mesh.d[1]),
                        'zstepsize: {}'.format(self.mesh.d[2]),
                        'xmin: {}'.format(self.mesh.c1[0]),
                        'ymin: {}'.format(self.mesh.c1[1]),
                        'zmin: {}'.format(self.mesh.c1[2]),
                        'xmax: {}'.format(self.mesh.c2[0]),
                        'ymax: {}'.format(self.mesh.c2[1]),
                        'zmax: {}'.format(self.mesh.c2[2]),
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
        for iz in range(self.mesh.n[2]):
            for iy in range(self.mesh.n[1]):
                for ix in range(self.mesh.n[0]):
                    v = [str(vi) for vi in self.f[ix, iy, iz, :]]
                    for vi in v:
                        oommf_file.write(' ' + vi)
                    oommf_file.write('\n')

        # Write footer lines to OOMMF file.
        for line in footer_lines:
            oommf_file.write('# ' + line + '\n')

        # Close the file.
        oommf_file.close()


def read_oommf_file(filename, normalisedto=None, name='unnamed'):
    """Read the OOMMF file and create an Field object.

    Args:
      filename (str): OOMMF file name
      name (str): name of the Field object

    Return:
      Field object.

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

    c1 = (dic['xmin'], dic['ymin'], dic['zmin'])
    c2 = (dic['xmax'], dic['ymax'], dic['zmax'])
    d = (dic['xstepsize'], dic['ystepsize'], dic['zstepsize'])
    cbase = (dic['xbase'], dic['ybase'], dic['zbase'])
    n = (int(round(dic['xnodes'])),
         int(round(dic['ynodes'])),
         int(round(dic['znodes'])))
    dim = int(dic['valuedim'])

    mesh = Mesh(c1, c2, d, name=name)
    field = Field(mesh, dim, normalisedto=normalisedto, name=name)

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
