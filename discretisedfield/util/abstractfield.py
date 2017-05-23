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

    @abc.abstractmethod
    def __getattr__(self, name): pass  # pragma: no cover

    @abc.abstractmethod
    def __dir__(self): pass  # pragma: no cover

    @abc.abstractmethod
    def plane(self, *args, x=None, y=None, z=None, n=None):
        pass  # pragma: no cover

    @abc.abstractmethod
    def plot_plane(self, *args, x=None, y=None, z=None, n=None):
        pass  # pragma: no cover

    @abc.abstractmethod
    def write(self, filename, **kwargs): pass  # pragma: no cover

    def line(self, p1, p2, n=100):
        """Slice the field along the line between `p1` and `p2`"""
        for point in self.mesh.line(p1, p2, n=n):
            yield point, self.__call__(point)
