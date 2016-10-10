import random
import numpy as np
import joommfutil.typesystem as ts
import discretisedfield.util as dfu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@ts.typesystem(p1=ts.RealVector(size=3),
               p2=ts.RealVector(size=3),
               cell=ts.PositiveRealVector(size=3),
               name=ts.ObjectName,
               l=ts.PositiveRealVector(size=3),
               pmin=ts.RealVector(size=3),
               pmax=ts.RealVector(size=3),
               n=ts.PositiveIntVector(size=3))
class Mesh(object):
    def __init__(self, p1, p2, cell, name="mesh"):
        """
        Creates a rectangular finite difference mesh.

        Args:
          p1 (tuple, list, np.ndarray): First mesh domain point
            p1 is of length 3 (xmin, ymin, zmax).
          p2 (tuple, list, np.ndarray): Second mesh domain point
            p2 is of length 3 (xmin, ymin, zmax).
          cell (tuple, list, np.ndarray): Discretisation cell size
            cell is of length 3 and defines the discretisation steps in
            x, y, and z directions: (dx, dy, dz).
          name (Optional[str]): Mesh name

        Attributes:
          p1 (tuple): First mesh domain point

          p2 (tuple): Second mesh domain point

          cell (tuple): Discretisation cell size

          name (str): Mesh name

          pmin (tuple): Minimum mesh domain point

          pmax (tuple): Maximum mesh domain point

          l (tuple): length of domain x, y, and z edges (lx, ly, lz):

            lx = abs(p2[0] - p1[0])

            ly = abs(p2[1] - p1[2])

            lz = abs(p2[2] - p1[2])

          n (tuple): The number of cells in three dimensions (nx, ny, nz):

            nx = lx/dx

            ny = ly/dy

            nz = lz/dz

        """
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.cell = tuple(cell)
        self.name = name

        # Compute domain edge lengths.
        self.l = (abs(self.p2[0]-self.p1[0]),
                  abs(self.p2[1]-self.p1[1]),
                  abs(self.p2[2]-self.p1[2]))

        # Compute minimum and maximum mesh domain points.
        self.pmin = (min(self.p1[0], self.p2[0]),
                     min(self.p1[1], self.p2[1]),
                     min(self.p1[2], self.p2[2]))
        self.pmax = (max(self.p1[0], self.p2[0]),
                     max(self.p1[1], self.p2[1]),
                     max(self.p1[2], self.p2[2]))

        tol = 1e-12  # picometer tolerance
        # Check if the discretisation cell size is greater than the domain.
        for i in range(3):
            if self.cell[i] > self.l[i]:
                raise ValueError(("Discretisation cell is greater than "
                                  "the domain dimension: cell[{}] > "
                                  "abs(p2[{}]-p1[{}]).").format(i, i, i))

        # Check if the domain is not an aggregate of discretisation cell.
        for i in range(3):
            if tol < self.l[i] % self.cell[i] < self.cell[i] - tol:
                raise ValueError(("Domain is not a multiple (aggregate) of "
                                  "the discretisation cell: "
                                  "abs(p2[{}]-p1[{}]) % "
                                  "cell[{}].").format(i, i, i))

        # Compute the number of cells in all three dimensions.
        self.n = (int(round(self.l[0]/self.cell[0])),
                  int(round(self.l[1]/self.cell[1])),
                  int(round(self.l[2]/self.cell[2])))

    def __repr__(self):
        """Mesh representation method.

        Returns:
          A mesh representation string.

        """
        p1str = "p1=({}, {}, {})".format(self.p1[0], self.p1[1], self.p1[2])
        p2str = "p2=({}, {}, {})".format(self.p2[0], self.p2[1], self.p2[2])
        cellstr = "cell=({}, {}, {})".format(self.cell[0],
                                             self.cell[1],
                                             self.cell[2])
        namestr = "name=\"{}\"".format(self.name)

        return "Mesh({}, {}, {}, {})".format(p1str, p2str, cellstr, namestr)

    def _ipython_display_(self):
        """Shows a matplotlib figure of sample range and discretisation."""
        # TODO: plt.show() works only with nbagg
        fig = self.plot()  # pragma: no cover
        plt.show()  # pragma: no cover

    def centre(self):
        """Compute and return the mesh centre point.

        Returns:
          A mesh centre point tuple of coordinates.

        """
        return (self.pmin[0] + 0.5*self.l[0],
                self.pmin[1] + 0.5*self.l[1],
                self.pmin[2] + 0.5*self.l[2])

    def random_point(self):
        """Generate a random mesh point.

        Returns:
          A random mesh point tuple of coordinates.

        """
        return (self.pmin[0] + random.random()*self.l[0],
                self.pmin[1] + random.random()*self.l[1],
                self.pmin[2] + random.random()*self.l[2])

    def index2point(self, i):
        """Convert the discretisation cell index to its centre point coordinate.

        The finite difference domain is disretised in x, y, and z directions
        in dx, dy, and dz steps, respectively. Accordingly, there are
        nx, ny, and nz discretisation steps. This method converts the cell
        index (ix, iy, iz) to the cell's centre point coordinate.

        This method raises ValueError if the index is out of range.

        Args:
          i (tuple): A length 3 tuple of integers (ix, iy, iz)

        Returns:
          A length 3 tuple of x, y, and z coodinates

        """
        for j in range(3):
            if i[j] < 0 or i[j] > self.n[j] - 1:
                raise ValueError(("Index i[{}]={} out of "
                                  "range.").format(j, i[j]))

        return (self.pmin[0] + (i[0]+0.5)*self.cell[0],
                self.pmin[1] + (i[1]+0.5)*self.cell[1],
                self.pmin[2] + (i[2]+0.5)*self.cell[2])

    def point2index(self, p):
        """Compute the index of a cell containing point p.

        This method is an inverse function of index2point method.
        (For details on index, please refer to the index2point method.)

        It raises ValueError if the point is outside the mesh.

        Args:
          p (tuple): A length 3 tuple of Real numbers (px, py, pz)

        Returns:
          A length 3 cell index tuple (ix, iy, iz).

        """
        for j in range(3):
            if p[j] < self.pmin[j] or p[j] > self.pmax[j]:
                raise ValueError(("Point coordinate p[{}]={} outside "
                                 "the mesh domain."). format(j, p[j]))

        i = []
        for j in range(3):
            ij = int(round((p[j]-self.pmin[j])/self.cell[j] - 0.5))

            # If rounded to the out-of-range mesh index.
            if ij < 0:
                ij = 0  # pragma: no cover
            elif ij > self.n[j] - 1:
                ij = self.n[j] - 1

            i.append(ij)

        return tuple(i)

    def cell_centre(self, p):
        """Computes the centre of cell containing (or nearest) to point p.

        Args:
          p (tuple): A length 3 tuple of point coordinates

        Returns:
          A length 3 tuple of cell's centre coordinates

        """
        return self.index2point(self.point2index(p))

    def plot(self):
        """Creates a figure of a mesh range and discretisation cell."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal")

        cell_point = (self.pmin[0] + self.cell[0],
                      self.pmin[1] + self.cell[1],
                      self.pmin[2] + self.cell[2])

        dfu.plot_box(ax, self.pmin, self.pmax)
        dfu.plot_box(ax, self.pmin, cell_point, props="r-", linewidth=1)

        ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$z$")

        return fig

    def cells(self):
        """Generator iterating through all mesh cells and
        yielding mesh indices and centre coordinates."""
        for k in range(self.n[2]):
            for j in range(self.n[1]):
                for i in range(self.n[0]):
                    yield (i, j, k), self.index2point((i, j, k))

    def line_intersection(self, l, l0, n=100):
        """Generator yielding mesh cell indices and their centre coordinates,
        along the line defined with l and l0 in n points."""
        try:
            p1, p2 = dfu.box_line_intersection(self.pmin, self.pmax, l, l0)
        except TypeError:
            raise ValueError("Line does not intersect mesh in two points.")

        p1, p2 = np.array(p1), np.array(p2)
        dl = (p2-p1) / (n-1)
        for i in range(n):
            point = p1 + i*dl
            yield np.linalg.norm(i*dl), tuple(point)

    def script(self):
        """This method should be implemented by a specific
        micromagnetic calculator."""
        raise NotImplementedError
