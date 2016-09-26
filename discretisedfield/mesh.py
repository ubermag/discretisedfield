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


def plot_cube(ax, c1, c2, color='blue', linewidth=2):
    """
    Plot a cube on axis that spans between c1 and c2.

    Args:
      c1 (tuple, list, np.ndarray): The minimum coordinate range.
        c1 is of length 3 and defines the minimum x, y, and z
        coordinates of the finite difference domain: (xmin, ymin, zmax)
      c2 (tuple, list, np.ndarray): The maximum coordinate range.
        c2 is of length 3 and defines the maximum x, y, and z
        coordinates of the finite difference domain: (xmax, ymax, zmax)
      color (str): matplotlib color string
      linewidth (Real): matplotlib linewidth parameter

    """
    x1, y1, z1 = c1[0], c1[1], c1[2]
    x2, y2, z2 = c2[0], c2[1], c2[2]

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


@ts.typesystem(c1=ts.RealVector3D,
               c2=ts.RealVector3D,
               d=ts.PositiveRealVector3D,
               name=ts.String)
class Mesh(object):
    def __init__(self, c1, c2, d, name='mesh'):
        """
        Creates a rectangular finite difference mesh.

        Args:
          c1 (tuple, list, np.ndarray): The minimum coordinate range.
            c1 is of length 3 and defines the minimum x, y, and z
            coordinates of the finite difference domain: (xmin, ymin, zmax)
          c2 (tuple, list, np.ndarray): The maximum coordinate range.
            c2 is of length 3 and defines the maximum x, y, and z
            coordinates of the finite difference domain: (xmax, ymax, zmax)
          d (tuple, list, np.ndarray): discretisation
            d is of length 3 and defines the discretisation steps in
            x, y, and z directions: (dx, dy, dz)
          name (Optional[str]): Mesh name.

        Attributes:
          c1 (tuple, list, np.ndarray): The minimum coordinate range

          c2 (tuple, list, np.ndarray): The maximum coordinate range

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
            if d[i] > abs(c2[i] - c1[i]) or \
               d[i] - tol > (c2[i] - c1[i]) % d[i] > tol:
                msg = "Discretisation cell index d[{}]={} ".format(i, d[i])
                msg += "is greater or not a multiple of simulation domain = "
                msg += "c2[{}] - c1[{}] = {}.".format(i, i, abs(c2[i] - c1[i]))
                raise TypeError(msg)

        self.c1 = c1
        self.c2 = c2
        self.d = d
        self.name = name

        # Compute domain edge lengths.
        self.l = (self.c2[0]-self.c1[0],
                  self.c2[1]-self.c1[1],
                  self.c2[2]-self.c1[2])

        # Compute the number of cells in x, y, and z directions.
        self.n = (int(round(self.l[0]/self.d[0])),
                  int(round(self.l[1]/self.d[1])),
                  int(round(self.l[2]/self.d[2])))

    def domain_centre(self):
        """Compute and return the domain centre coordinate.

        Returns:
          A domain centre coordinate tuple.

        """
        c = (self.c1[0] + 0.5*self.l[0],
             self.c1[1] + 0.5*self.l[1],
             self.c1[2] + 0.5*self.l[2])

        return c

    def random_coord(self):
        """Generate a random coordinate in the domain.

        Returns:
          A random domain coordinate.

        """
        c = (self.c1[0] + random.random()*self.l[0],
             self.c1[1] + random.random()*self.l[1],
             self.c1[2] + random.random()*self.l[2])

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
            c = (self.c1[0] + (i[0] + 0.5)*self.d[0],
                 self.c1[1] + (i[1] + 0.5)*self.d[1],
                 self.c1[2] + (i[2] + 0.5)*self.d[2])

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
        if c[0] < self.c1[0] or c[0] > self.c2[0] or \
           c[1] < self.c1[1] or c[1] > self.c2[1] or \
           c[2] < self.c1[2] or c[2] > self.c2[2]:
            raise ValueError('Coordinate {} out of domain.'. format(c))

        else:
            i = [int(round(float(c[0]-self.c1[0])/self.d[0] - 0.5)),
                 int(round(float(c[1]-self.c1[1])/self.d[1] - 0.5)),
                 int(round(float(c[2]-self.c1[2])/self.d[2] - 0.5))]

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

        cd = (self.d[0] + self.c1[0],
              self.d[1] + self.c1[1],
              self.d[2] + self.c1[2])

        plot_cube(ax, self.c1, self.c2)
        plot_cube(ax, self.c1, cd, color='red', linewidth=1)

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
