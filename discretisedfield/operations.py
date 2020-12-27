import numbers
import discretisedfield as df


class DValue:
    def __init__(self, function, /):
        self._function = function

    def __call__(self, field):
        return self._function(field)

    def __abs__(self):
        return self.__class__(lambda f: abs(self(f)))

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(lambda f: self(f) * other(f))
        elif isinstance(other, df.Field):
            return other * self
        elif isinstance(other, numbers.Real):
            return self.__class__(lambda f: self(f) * other)
        else:
            msg = (f'Unsupported operand type(s) for *: '
                   f'{type(self)} and {type(other)}.')
            raise TypeError(msg)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        if isinstance(other, self.__class__):
            return self.__class__(lambda f: self(f) @ other(f))
        elif isinstance(other, df.Field):
            return other @ self
        elif isinstance(other, (list, tuple, numpy.ndarray)):
            return self.__class__(lambda f: self(f) @ other)
        else:
            msg = (f'Unsupported operand type(s) for *: '
                   f'{type(self)} and {type(other)}.')
            raise TypeError(msg)


dx = DValue(lambda f: f.mesh.dx)
dy = DValue(lambda f: f.mesh.dy)
dz = DValue(lambda f: f.mesh.dz)
dV = DValue(lambda f: f.mesh.dV)
dS = DValue(lambda f: f.mesh.dS)


def integral(field):
    return field.integral
