import numbers
import numpy as np


class Descriptor:
    def __init__(self, name=None, **opts):
        self.name = name
        for key, value in opts.items():
            setattr(self, key, value)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        raise AttributeError("Deleting attribute not allowed.")


class Typed(Descriptor):
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError("Expected " + str(self.expected_type))
        super().__set__(instance, value)


class TypedAttribute(Descriptor):
    def __init__(self, name=None, **opts):
        if "expected_type" not in opts:
            raise TypeError("Missing expected_type option.")
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError("Expected " + str(self.expected_type))
        super().__set__(instance, value)


class MaxSized(Descriptor):
    def __init__(self, name=None, **opts):
        if "size" not in opts:
            raise TypeError("Missing size option")
        super().__init__(name, **opts)

    def __set__(self, instance, value):
        if len(value) != self.size:
            raise ValueError("size must be < " + str(self.size))
        super().__set__(instance, value)


class Unsigned(Descriptor):
    def __set__(self, instance, value):
        if value < 0:
            raise TypeError("Expected >= 0.")
        super().__set__(instance, value)


class Positive(Descriptor):
    def __set__(self, instance, value):
        if value <= 0:
            raise TypeError("Expected > 0.")
        super().__set__(instance, value)


class Real(Typed):
    expected_type = numbers.Real


class Int(Typed):
    expected_type = int


class String(Typed):
    expected_type = str


class UnsignedReal(Real, Unsigned):
    pass


class UnsignedInt(Int, Unsigned):
    pass


class PositiveReal(Real, Positive):
    pass


class Vector(Typed):
    expected_type = (list, tuple, np.ndarray)


class SizedVector(Vector, MaxSized):
    pass


class RealVector(SizedVector):
    def __set__(self, instance, value):
        if not all([isinstance(i, numbers.Real) for i in value]):
            raise TypeError("Expected Real vector components.")
        super().__set__(instance, value)


class PositiveRealVector(RealVector):
    def __set__(self, instance, value):
        if not all([i > 0 for i in value]):
            raise TypeError("Expected Positive vector components.")
        super().__set__(instance, value)


def typesystem(**kwargs):
    def decorate(cls):
        for key, value in kwargs.items():
            if isinstance(value, Descriptor):
                value.name = key
                setattr(cls, key, value)
            else:
                setattr(cls, key, value(key))

        return cls
    return decorate
