import random
import numpy as np
import joommfutil.typesystem as ts
import discretisedfield.util as dfu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@ts.typesystem(p1=ts.RealVector(size=3),
               p2=ts.RealVector(size=3),
               cell=ts.PositiveRealVector(size=3),
               name=ts.ObjectName)
class Mesh(object):
    def __init__(self, p1, p2, cell, name="mesh"):
        """
        Finite Difference Mesh

        Args:
          p1 (tuple): first point of the mesh domain
            p1 = (x1, y1, z1)
          p2 (tuple): second point of the mesh domain
            p2 = (x2, y2, z2)
          cell (tuple): discretisation cell size
            cell = (dx, dy, dz)
          name (Optional[str]): mesh name (defaults to "mesh")

        Attributes:
          p1 (tuple): first point of the mesh domain
          p2 (tuple): second point of the mesh domain
          cell (tuple): discretisation cell size
          name (str): mesh name

        """
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        self.cell = tuple(cell)
        self.name = name

        # Is domain edge length zero?
        for i in range(3):
            if self.l[i] == 0:
                msg = ("Domain edge length is zero (l[{}]==0).").format(i)
                raise ValueError(msg)

        # Is discretisation cell size greater than the domain?
        for i in range(3):
            if self.cell[i] > self.l[i]:
                msg = ("Discretisation cell is greater than the domain: "
                       "cell[{0}] > abs(p2[{0}]-p1[{0}]).").format(i)
                raise ValueError(msg)

        # Is domain not an aggregate of discretisation cells?
        tol = 1e-12  # picometer tolerance
        for i in range(3):
            if tol < self.l[i] % self.cell[i] < self.cell[i] - tol:
                msg = ("Domain is not an aggregate of the discretisation "
                       "cell: abs(p2[{0}]-p1[{0}]) % cell[{0}].").format(i)
                raise ValueError(msg)

    @property
    def pmin(self):
        """Minimum mesh domain coordinate point"""
        return tuple(min(self.p1[i], self.p2[i]) for i in range(3))

    @property
    def pmax(self):
        """Maximum mesh domain coordinate point"""
        return tuple(max(self.p1[i], self.p2[i]) for i in range(3))

    @property
    def l(self):
        """Length of domain edges (lx, ly, lz)
            lx = abs(p2[0] - p1[0])
            ly = abs(p2[1] - p1[2])
            lz = abs(p2[2] - p1[2])

        """
        return tuple(abs(self.p1[i] - self.p2[i]) for i in range(3))

    @property
    def n(self):
        """Number of discretisation cells (nx, ny, nz)
            nx = lx/dx
            ny = ly/dy
            nz = lz/dz

        """
        return tuple(int(round(self.l[i]/self.cell[i])) for i in range(3))

    @property
    def centre(self):
        """Mesh centre point"""
        return tuple(self.pmin[i]+0.5*self.l[i] for i in range(3))

    def __repr__(self):
        """Mesh representation method.

        Returns:
          A mesh representation string.

        """
        return ("Mesh(p1={}, p2={}, cell={}, "
                "name=\"{}\")").format(self.p1, self.p2, self.cell, self.name)

    def _ipython_display_(self):
        """Shows a matplotlib figure of sample range and discretisation."""
        return self.plot()  # pragma: no cover

    def random_point(self):
        """Generate a random mesh point.

        Returns:
          A random mesh point tuple of coordinates (x, y, z).

        """
        return tuple(self.pmin[i]+random.random()*self.l[i] for i in range(3))

    def index2point(self, index):
        """Convert the cell index to its centre point coordinate.

        The finite difference domain is disretised in x, y, and z directions
        in dx, dy, and dz steps, respectively. Consequently, there are
        nx, ny, and nz discretisation steps. This method converts the cell
        index (ix, iy, iz) to the cell's centre point coordinate.

        This method raises ValueError if the index is out of range.

        Args:
          i (tuple): A length 3 tuple of integers (ix, iy, iz)

        Returns:
          A length 3 tuple of x, y, and z coodinates

        """
        # Does index refer to a cell outside the mesh?
        for i in range(3):
            if index[i] < 0 or index[i] > self.n[i] - 1:
                msg = ("index[{}]={} out of range "
                       "[0, n[{}]].").format(i, index[i], self.n[i])
                raise ValueError(msg)

        return tuple(self.pmin[i]+(index[i]+0.5)*self.cell[i]
                     for i in range(3))

    def point2index(self, p):
        """Compute the index of a cell containing point p.

        It raises ValueError if the point is outside the mesh.

        Args:
          p (tuple): A length 3 tuple of Real numbers (px, py, pz)

        Returns:
          A length 3 cell index tuple (ix, iy, iz).

        """
        # Is the point outside the mesh?
        for i in range(3):
            if p[i] < self.pmin[i] or p[i] > self.pmax[i]:
                msg = "Point p[{}]={} outside the mesh.".format(i, p[i])
                raise ValueError(msg)

        index = tuple(int(round((p[i]-self.pmin[i])/self.cell[i] - 0.5))
                      for i in range(3))

        # If rounded to the out-of-range values
        return tuple(max(min(self.n[i]-1, index[i]), 0) for i in range(3))

    def indices(self):
        """Generator iterating through all mesh cells and
        yielding their indices."""
        for k in range(self.n[2]):
            for j in range(self.n[1]):
                for i in range(self.n[0]):
                    yield (i, j, k)

    def coordinates(self):
        """Generator iterating through all mesh cells and
        yielding their centres' coordinates."""
        for i in self.indices():
            yield self.index2point(i)

    def plot(self):
        """Creates a figure of a mesh domain and discretisation cell."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_aspect("equal")

        # domain box
        dfu.plot_box(ax, self.pmin, self.pmax, "b-", linewidth=1.5)

        # cell box
        cell_point = tuple(self.pmin[i]+self.cell[i] for i in range(3))
        dfu.plot_box(ax, self.pmin, cell_point, "r-", linewidth=1)

        ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$z$")

        return fig

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

    @property
    def script(self):
        """This method should be implemented by a specific calculator."""
        raise NotImplementedError
