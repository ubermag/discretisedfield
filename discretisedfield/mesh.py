import abc
import six
import matplotlib
import numpy as np
from numbers import Real
matplotlib.use('nbagg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_cube(ax, cmin, cmax, color='blue', linewidth=2):
    """
    Plot a cube on axis that spans between cmin and cmax.

    Args:
      cmin (tuple, list, np.ndarray): The minimum coordinate range.
        cmin is of length 3 and defines the minimum x, y, and z
        coordinates of the finite difference domain: (xmin, ymin, zmax)
      cmax (tuple, list, np.ndarray): The maximum coordinate range.
        cmax is of length 3 and defines the maximum x, y, and z
        coordinates of the finite difference domain: (xmax, ymax, zmax)
      color (str): matplotlib color string
      linewidth (Real): matplotlib linewidth parameter

    """
    x1, y1, z1 = cmin[0], cmin[1], cmin[2]
    x2, y2, z2 = cmax[0], cmax[1], cmax[2]

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


@six.add_metaclass(abc.ABCMeta)
class Mesh(object):
    _name = 'mesh'

    def __init__(self, cmin, cmax, d):
        """
        Creates a rectangular mesh across the space covered by atlas.

        Args:
          cmin (tuple, list, np.ndarray): The minimum coordinate range.
            cmin is of length 3 and defines the minimum x, y, and z
            coordinates of the finite difference domain: (xmin, ymin, zmax)
          cmax (tuple, list, np.ndarray): The maximum coordinate range.
            cmax is of length 3 and defines the maximum x, y, and z
            coordinates of the finite difference domain: (xmax, ymax, zmax)
          d (tuple, list, np.ndarray): discretisation
            d is of length 3 and defines the discretisation steps in
            x, y, and z directions: (dx, dy, dz)

        """
        tol = 1e-12
        
        if not isinstance(cmin, (list, tuple, np.ndarray)) or len(cmin) != 3:
            raise ValueError('cmin must be a 3-element tuple, '
                             'list, or np.ndarray.')

        if not all([isinstance(i, Real) for i in cmin]):
            raise ValueError('All elements of cmin must be real numbers.')

        if not isinstance(cmax, (list, tuple, np.ndarray)) or len(cmax) != 3:
            raise ValueError('cmax must be a 3-element tuple, '
                             'list, or np.ndarray.')
        
        if not all([isinstance(i, Real) for i in cmax]):
            raise ValueError('All elements of cmax must be real numbers.')

        # check that d is sequence and of length 3
        if not isinstance(d, (list, tuple, np.ndarray)) or len(d) != 3:
            raise ValueError('d must be a 3-element tuple, '
                             'list, or np.ndarray.')
        # check that cell size d is number and non-negative
        if not all([isinstance(i, Real) and i >= 0 for i in d]):
            raise ValueError('All d elements must be positive real numbers.')
        # check whether cell size is larger than domain
        for i in range(3):
            if d[i] > abs(cmax[i] - cmin[i]):
                msg = "discretisation cell index d[{}]={} ".format(i, d[i])
                msg += "is greater than simulation domain = cmax[{}] - cmin[{}] ".format(i, i)
                msg += "= {}".format(abs(cmax[i] - cmin[i]))
                raise ValueError(msg)

        # check that simulation domain can be divided into chunks of
        # given cell size
        print((cmax[0]-cmin[0]) % d[0])
        print((cmax[1]-cmin[1]) % d[1])
        print((cmax[2]-cmin[2]) % d[2])
        if d[0] - tol > (cmax[0]-cmin[0]) % d[0] > tol or \
           d[1] - tol > (cmax[1]-cmin[1]) % d[1] > tol or \
           d[2] - tol > (cmax[2]-cmin[2]) % d[2] > tol:
            raise ValueError('Domain is not a multiple of cell with size {}.'.format(d))

        self.cmin = cmin
        self.cmax = cmax
        self.d = d

    def plot_mesh(self):
        """Shows a matplotlib figure of sample range and discretiation."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        cd = (self.d[0] + self.cmin[0],
              self.d[1] + self.cmin[1],
              self.d[2] + self.cmin[2])

        plot_cube(ax, self.cmin, self.cmax)
        plot_cube(ax, self.cmin, cd, color='red', linewidth=1)

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
