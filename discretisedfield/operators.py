import numbers

import numpy as np

import discretisedfield as df


class DValue:
    """Infinitesimaly small value (differential).

    This class is used for defining infinitesimaly small values used for
    computing integrals. For instance ``dV``, ``dx``, surface vector field
    ``dS``, etc.

    Parameters
    ----------
    function : Python function

        A function which takes ``discretisedfield.Field`` object as an input
        and extracts the required differential.

    Examples
    --------
    1. Defining ``dV``.

    >>> import discretisedfield as df
    ...
    >>> dV = DValue(lambda f: f.mesh.dV)
    ...
    >>> p1 = (-50, -25, 0)
    >>> p2 = (50, 25, 5)
    >>> cell = (5, 5, 5)
    >>> region = df.Region(p1=p1, p2=p2)
    >>> mesh = df.Mesh(region=region, cell=cell)
    >>> field = df.Field(mesh, dim=1, value=3.14)
    ...
    >>> dV(field)
    125

    """

    def __init__(self, function, /):
        self.function = function

    def __call__(self, field):
        """Call ``self.function`` on ``field``.

        Parameters
        ----------
        field : discretisedfield.Field

            Field object.

        Returns
        -------
        numbers.Real, discretisedfield.Field

            Inifinitesimaly small value.

        Example
        -------
        1. Computing ``dx``.

        >>> import discretisedfield as df
        ...
        >>> dx = DValue(lambda f: f.mesh.dx)
        ...
        >>> p1 = (-50, -25, 0)
        >>> p2 = (50, 25, 5)
        >>> cell = (5, 5, 5)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, cell=cell)
        >>> field = df.Field(mesh, dim=1, value=3.14)
        ...
        >>> dx(field)
        5

        """
        return self.function(field)

    def __abs__(self):
        """Absolute value (norm) operator.

        This method computes norm if differential is a vector field (``dS``).

        Returns
        -------
        DValue

            Result.

        Examples
        --------
        1. Computing absolute value (norm).

        >>> import discretisedfield as df
        ...
        >>> dS = DValue(lambda f: f.mesh.dS)
        ...
        >>> p1 = (-50, -25, 0)
        >>> p2 = (50, 25, 5)
        >>> cell = (5, 5, 5)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, cell=cell)
        >>> field = df.Field(mesh, dim=3, value=(1, -9, 2))
        ...
        >>> abs(dS)(field.plane('z')).average
        25.0

        """
        return self.__class__(lambda f: abs(self(f)))

    def __mul__(self, other):
        """Binary ``*`` operator.

        It can be applied between:

        1. Two ``DValue`` objects.

        2. A field of any dimension and ``DValue`` object,

        3. ``DValue`` object and ``numbers.Real``.

        Parameters
        ----------
        other : DValue, discretisedfield.Field, numbers.Real

            Second operand.

        Returns
        -------
        DValue, discretisedfield.Field

            Result.

        Raises
        ------
        TypeError

            If the operator cannot be applied.

        Examples
        --------
        1. Multiplication examples.

        >>> import discretisedfield as df
        ...
        >>> dx = DValue(lambda f: f.mesh.dx)
        >>> dy = DValue(lambda f: f.mesh.dy)
        >>> dz = DValue(lambda f: f.mesh.dz)
        >>> dV = DValue(lambda f: f.mesh.dV)
        ...
        >>> p1 = (-50, -25, 0)
        >>> p2 = (50, 25, 5)
        >>> cell = (5, 5, 5)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, cell=cell)
        >>> field = df.Field(mesh, dim=3, value=(1, -9, 2))
        ...
        >>> # DValue multiplication
        >>> dxdydz = dx * dy * dz
        >>> # DValue and field multiplication
        >>> field * dV == field * dxdydz
        True
        >>> # Dvalue and int multiplication
        >>> field * dV * 2 == field * dxdydz * 2
        True

        """
        if isinstance(other, self.__class__):
            return self.__class__(lambda f: self(f) * other(f))
        elif isinstance(other, df.Field):
            return other * self
        elif isinstance(other, numbers.Real):
            return self.__class__(lambda f: self(f) * other)
        else:
            msg = (
                f"Unsupported operand type(s) for *: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        """Binary ``@`` operator, defined as a dot product.

        It can be applied between:

        1. Two ``DValue`` objects.

        2. A vector field and ``DValue`` object,

        3. ``DValue`` object and ``array_like`` length-3 object.

        Parameters
        ----------
        other : DValue, discretisedfield.Field, (3,) array_like

            Second operand.

        Returns
        -------
        DValue, discretisedfield.Field

            Result.

        Raises
        ------
        TypeError

            If the operator cannot be applied.

        Example
        -------
        1. Dot product examples.

        >>> import discretisedfield as df
        ...
        >>> dS = DValue(lambda f: f.mesh.dS)
        ...
        >>> p1 = (-50, -25, 0)
        >>> p2 = (50, 25, 5)
        >>> cell = (5, 5, 5)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> mesh = df.Mesh(region=region, cell=cell)
        >>> field = df.Field(mesh, dim=3, value=(1, -9, 2))
        ...
        >>> # DValue dot product
        >>> dS2 = dS * dS
        >>> # DValue and field dot product.
        >>> field.plane('z') @ dS
        Field(...)
        >>> # DValue and array_like object
        >>> field.plane('z') * (dS @ (1, 1, 1))
        Field(...)

        """
        if isinstance(other, self.__class__):
            return self.__class__(lambda f: self(f) @ other(f))
        elif isinstance(other, df.Field):
            return other @ self
        elif isinstance(other, (list, tuple, np.ndarray)):
            return self.__class__(lambda f: self(f) @ other)
        else:
            msg = (
                f"Unsupported operand type(s) for *: {type(self)=} and {type(other)=}."
            )
            raise TypeError(msg)

    def __rmatmul__(self, other):
        return self @ other


def integral(field, direction="xyz", improper=False):
    """Integral.

    This function calls ``integral`` method of the ``discrteisedfield.Field``
    object.

    For details, please refer to
    :py:func:`~discretisedfield.Field.integral`

    """
    return field.integral(direction=direction, improper=improper)


dx = DValue(lambda f: f.mesh.dx)
dy = DValue(lambda f: f.mesh.dy)
dz = DValue(lambda f: f.mesh.dz)
dV = DValue(lambda f: f.mesh.dV)
dS = DValue(lambda f: f.mesh.dS)
