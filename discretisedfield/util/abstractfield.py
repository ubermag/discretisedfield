import abc
import matplotlib.pyplot as plt
from .util import plot_line, plot_box


class Field(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self): pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def value(self): pass  # pragma: no cover

    @value.setter
    @abc.abstractmethod
    def value(self, val): pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def norm(self): pass  # pragma: no cover

    @norm.setter
    @abc.abstractmethod
    def norm(self, val): pass  # pragma: no cover

    @property
    @abc.abstractmethod
    def average(self): pass  # pragma: no cover

    @abc.abstractmethod
    def __repr__(self): pass  # pragma: no cover

    @abc.abstractmethod
    def __call__(self, point): pass  # pragma: no cover

    def line_intersection(self, p1, p2, n=100):
        """Slice the field along the line between `p1` and `p2`"""
        ds, points, values = [], [], []
        for parameter, point in self.mesh.line(p1, p2, n=n):
            ds.append(parameter)
            points.append(point)
            values.append(self.__call__(point))

        return ds, values

    def plot_line_intersection(self, p1, p2, n=100):
        # Plot schematic representation of intersection.
        fig = plt.figure()
        ax = fig.add_subplot(211, projection="3d")
        ax.set_aspect("equal")

        plot_box(ax, self.mesh.pmin, self.mesh.pmax, "b-")
        plot_line(ax, p1, p2, "ro-")
        ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$z$")

        # Plot field along line.
        ax = fig.add_subplot(212)
        d, v = self.line_intersection(p1, p2, n=n)
        ax.set(xlabel=r"$d$", ylabel=r"$v$")
        ax.grid()
        ax.plot(d, v)
        plt.close()

        return fig
