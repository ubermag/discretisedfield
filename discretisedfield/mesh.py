"""This module is a Python package that provides:

- Creating and plotting finite difference mesh.

It is a member of JOOMMF project - a part of OpenDreamKit
Horizon 2020 European Research Infrastructure project.

"""
import random
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import discretisedfield.util.typesystem as ts
import matplotlib.pyplot as plt


def plot_cube(ax, p1, p2, color='blue', linewidth=2):
    """
    Plot a cube on axis that spans between p1 and p2.

    Args:
      p1 (tuple, list, np.ndarray): The minimum coordinate range.
        p1 is of length 3 and defines the minimum x, y, and z
        coordinates of the finite difference domain: (xmin, ymin, zmax)
      p2 (tuple, list, np.ndarray): The maximum coordinate range.
        p2 is of length 3 and defines the maximum x, y, and z
        coordinates of the finite difference domain: (xmax, ymax, zmax)
      color (str): matplotlib color string
      linewidth (Real): matplotlib linewidth parameter

    """
    x1, y1, z1 = p1[0], p1[1], p1[2]
    x2, y2, z2 = p2[0], p2[1], p2[2]

    ax.plot([x1, x2], [y1, y1], [z1, z1], color=color, linewidth=linewidth)
    ax.plot([x1, x2], [y2, y2], [z1, z1], color=color, linewidth=linewidth)
    ax.plot([x1, x2], [y1, y1], [z2, z2], color=color, linewidth=linewidth)
    ax.plot([x1, x2], [y2, y2], [z2, z2], color=color, linewidth=linewidth)

    ax.plot([x1, x1], [y1, y2], [z1, z1], color=color, linewidth=linewidth)
    ax.plot([x2, x2], [y1, y2], [z1, z1], color=color, linewidth=linewidth)
    ax.plot([x1, x1], [y1, y2], [z2, z2], color=color, linewidth=linewidth)
    ax.plot([x2, x2], [y1, y2], [z2, z2], color=color, linewidth=linewidth)

    ax.plot([x1, x1], [y1, y1], [z1, z2], color=color, linewidth=linewidth)
    ax.plot([x2, x2], [y1, y1], [z1, z2], color=color, linewidth=linewidth)
    ax.plot([x1, x1], [y2, y2], [z1, z2], color=color, linewidth=linewidth)
    ax.plot([x2, x2], [y2, y2], [z1, z2], color=color, linewidth=linewidth)

    return ax


@ts.typesystem(p1=ts.RealVector(size=3),
               p2=ts.RealVector(size=3),
               cell=ts.PositiveRealVector(size=3),
               name=ts.ObjectName)
class Mesh(object):
    def __init__(self, p1, p2, cell, name='mesh'):
        """
        Creates a rectangular finite difference mesh.

        Args:
          p1 (tuple, list, np.ndarray): The minimum coordinate range.
            p1 is of length 3 and defines the minimum x, y, and z
            coordinates of the finite difference domain: (xmin, ymin, zmax)
          p2 (tuple, list, np.ndarray): The maximum coordinate range.
            p2 is of length 3 and defines the maximum x, y, and z
            coordinates of the finite difference domain: (xmax, ymax, zmax)
          d (tuple, list, np.ndarray): discretisation
            d is of length 3 and defines the discretisation steps in
            x, y, and z directions: (dx, dy, dz)
          name (Optional[str]): Mesh name.

        Attributes:
          p1 (tuple, list, np.ndarray): The minimum coordinate range

          p2 (tuple, list, np.ndarray): The maximum coordinate range

          d (tuple, list, np.ndarray): Discretisation cell size

          name (str): Mesh name

          l (tuple): length of domain x, y, and z edges (lx, ly, lz):

            lx = xmax - xmin

            ly = ymax - ymin

            lz = zmax - zmin

          n (tuple): The number of cells in all three dimensions (nx, ny, nz):

            nx = lx/dx

            ny = ly/dy

            nz = lz/dz

        """
        tol = 1e-12
        # check whether cell size is greater than or not a multiple of domain
        for i in range(3):
            if cell[i] > abs(p2[i] - p1[i]) or \
               cell[i] - tol > (p2[i] - p1[i]) % cell[i] > tol:
                msg = "Discretisation cell index d[{}]={} ".format(i, cell[i])
                msg += "is greater or not a multiple of simulation domain = "
                msg += "p2[{}] - p1[{}] = {}.".format(i, i, abs(p2[i] - p1[i]))
                raise TypeError(msg)

        self.p1 = p1
        self.p2 = p2
        self.cell = cell
        self.name = name

        # Compute domain edge lengths.
        self.l = (self.p2[0]-self.p1[0],
                  self.p2[1]-self.p1[1],
                  self.p2[2]-self.p1[2])

        # Compute the number of cells in x, y, and z directions.
        self.n = (int(round(self.l[0]/self.cell[0])),
                  int(round(self.l[1]/self.cell[1])),
                  int(round(self.l[2]/self.cell[2])))

    def domain_centre(self):
        """Compute and return the domain centre coordinate.

        Returns:
          A domain centre coordinate tuple.

        """
        c = (self.p1[0] + 0.5*self.l[0],
             self.p1[1] + 0.5*self.l[1],
             self.p1[2] + 0.5*self.l[2])

        return c

    def random_coord(self):
        """Generate a random coordinate in the domain.

        Returns:
          A random domain coordinate.

        """
        c = (self.p1[0] + random.random()*self.l[0],
             self.p1[1] + random.random()*self.l[1],
             self.p1[2] + random.random()*self.l[2])

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

        """
        if i[0] < 0 or i[0] > self.n[0]-1 or \
           i[1] < 0 or i[1] > self.n[1]-1 or \
           i[2] < 0 or i[2] > self.n[2]-1:
            raise ValueError('Index {} out of range.'.format(i))

        else:
            c = (self.p1[0] + (i[0] + 0.5)*self.cell[0],
                 self.p1[1] + (i[1] + 0.5)*self.cell[1],
                 self.p1[2] + (i[2] + 0.5)*self.cell[2])

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

        """
        if c[0] < self.p1[0] or c[0] > self.p2[0] or \
           c[1] < self.p1[1] or c[1] > self.p2[1] or \
           c[2] < self.p1[2] or c[2] > self.p2[2]:
            raise ValueError('Coordinate {} out of domain.'. format(c))

        else:
            i = [int(round(float(c[0]-self.p1[0])/self.cell[0] - 0.5)),
                 int(round(float(c[1]-self.p1[1])/self.cell[1] - 0.5)),
                 int(round(float(c[2]-self.p1[2])/self.cell[2] - 0.5))]

            # If rounded to the out-of-range index.
            for j in range(3):
                if i[j] < 0:  # pragma: no cover
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

        """
        return self.index2coord(self.coord2index(c))

    def plot_mesh(self):
        """Shows a matplotlib figure of sample range and discretisation."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        cd = (self.cell[0] + self.p1[0],
              self.cell[1] + self.p1[1],
              self.cell[2] + self.p1[2])

        plot_cube(ax, self.p1, self.p2)
        plot_cube(ax, self.p1, cd, color='red', linewidth=1)

        ax.set(xlabel=r'$x$', ylabel=r'$y$', zlabel=r'$z$')

        return fig

    def _ipython_display_(self):
        """Shows a matplotlib figure of sample range and discretisation."""
        fig = self.plot_mesh()  # pragma: no cover
        plt.show()  # pragma: no cover

    def script(self):
        """This method should be implemented by a specific
        micromagnetic calculator"""
        raise NotImplementedError
