import abc


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
