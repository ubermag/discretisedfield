import random
import numpy as np
import joommfutil.typesystem as ts
import discretisedfield.util as dfu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@ts.typesystem(p1=ts.ConstantRealVector(size=3),
               p2=ts.ConstantRealVector(size=3),
               cell=ts.ConstantPositiveRealVector(size=3),
               name=ts.ConstantObjectName)
class Mesh:
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

        # Is the length of any mesh domain edge zero?
        for i in range(3):
            if self.l[i] == 0:
                msg = "Mesh domain edge length is zero (l[{}]==0).".format(i)
                raise ValueError(msg)

        # Is the discretisation cell greater than the mesh domain?
        for i in range(3):
            if self.cell[i] > self.l[i]:
                msg = ("Discretisation cell is greater than the mesh domain: "
                       "cell[{0}] > abs(p2[{0}]-p1[{0}]).").format(i)
                raise ValueError(msg)

        # Is mesh domain not an aggregate of discretisation cells?
        tol = 1e-12  # picometer tolerance
        for i in range(3):
            if tol < self.l[i] % self.cell[i] < self.cell[i] - tol:
                msg = ("Mesh domain is not an aggregate of the discretisation "
                       "cell: abs(p2[{0}]-p1[{0}]) % cell[{0}].").format(i)
                raise ValueError(msg)

    @property
    def pmin(self):
        """Mesh domain point with minimum coordinate."""
        return tuple(min(self.p1[i], self.p2[i]) for i in range(3))

    @property
    def pmax(self):
        """Mesh domain point with maximum coordinate."""
        return tuple(max(self.p1[i], self.p2[i]) for i in range(3))

    @property
    def l(self):
        """Lengths of domain edges
            l = (abs(p2[0] - p1[0]),
                 abs(p2[1] - p1[1]),
                 abs(p2[2] - p1[2])).

        """
        return tuple(abs(self.p1[i] - self.p2[i]) for i in range(3))

    @property
    def n(self):
        """Number of discretisation cells
            n = (l[0]/cell[0],
                 l[1]/cell[1],
                 l[2]/cell[2])

        """
        return tuple(int(round(self.l[i]/self.cell[i])) for i in range(3))

    @property
    def centre(self):
        """Mesh domain centre"""
        return tuple(self.pmin[i]+0.5*self.l[i] for i in range(3))

    @property
    def indices(self):
        """Generator iterating through all mesh cells and
        yielding their indices."""
        for k in range(self.n[2]):
            for j in range(self.n[1]):
                for i in range(self.n[0]):
                    yield (i, j, k)

    @property
    def coordinates(self):
        """Generator iterating through all mesh cells and
        yielding their centres' coordinates."""
        for i in self.indices:
            yield self.index2point(i)

    def __repr__(self):
        """Mesh representation string."""
        return ("Mesh(p1={}, p2={}, cell={}, "
                "name=\"{}\")").format(self.p1, self.p2, self.cell, self.name)

    def _ipython_display_(self):
        """Figure of mesh domain and discretisation cell."""
        return self.plot()  # pragma: no cover

    def random_point(self):
        """Random point inside the mesh."""
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
    def _script(self):
        """This method should be implemented by a specific calculator."""
        raise NotImplementedError
