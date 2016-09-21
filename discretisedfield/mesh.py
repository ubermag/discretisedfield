import matplotlib
import numpy as np
from numbers import Real
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from .util.typesystem import PositiveRealVector3D, RealVector3D, typesystem


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


@typesystem(c1=RealVector3D,
            c2=RealVector3D,
            d=PositiveRealVector3D)
class Mesh(object):
    def __init__(self, c1, c2, d):
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

        """
        tol = 1e-16
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
        fig = self.plot_mesh()

        plt.show()

    def script(self):
        """This method should be implemented by a specific
        micromagnetic calculator"""
        raise NotImplementedError
