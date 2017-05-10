import abc


class Field(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self): pass  # pragma: no cover
