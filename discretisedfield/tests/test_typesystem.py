import pytest
import micromagneticmodel.util.typesystem as ts


def test_typesystem():
    @ts.typesystem(a=ts.Real,
                   b=ts.Int,
                   c=ts.String,
                   d=ts.UnsignedReal,
                   e=ts.PositiveReal,
                   f=ts.Vector,
                   g=ts.SizedVector(size=2),
                   h=ts.RealVector(size=3),
                   i=ts.PositiveRealVector(size=3),
                   j=ts.TypedAttribute(expected_type=dict))
    class DummyClass:
        def __init__(self, a, b, c, d, e, f, g, h, i, j):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
            self.e = e
            self.f = f
            self.g = g
            self.h = h
            self.i = i
            self.j = j

    a = 1.7
    b = 2
    c = 'abc'
    d = 9.5
    e = 11.
    f = (1, 3, -4, 9)
    g = (1, 2)
    h = (-1, 2, 3.1)
    i = (1, 2, 31.1)
    j = {}
    dc = DummyClass(a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h, i=i, j=j)

    # Simple assertions
    assert dc.a == a
    assert dc.b == b
    assert dc.c == c
    assert dc.d == d
    assert dc.e == e
    assert dc.f == f
    assert dc.g == g
    assert dc.h == h
    assert dc.i == i
    assert dc.j == j

    # Valid settings
    dc.a = 77.4
    assert dc.a == 77.4
    dc.b = -77
    assert dc.b == -77
    dc.c = 'dummystring'
    assert dc.c == 'dummystring'
    dc.d = 61.2
    assert dc.d == 61.2
    dc.e = 0.1
    assert dc.e == 0.1
    dc.f = [1, 2, 3, 4, 5, 6.1]
    assert dc.f == [1, 2, 3, 4, 5, 6.1]
    dc.g = (3, 2.1)
    assert dc.g == (3, 2.1)
    dc.h = (-5, 6, 8)
    assert dc.h == (-5, 6, 8)
    dc.i = (1, 2, 3.2)
    assert dc.i == (1, 2, 3.2)
    dc.j = {'a': 1}
    assert dc.j == {'a': 1}

    # Invalid settings
    with pytest.raises(TypeError):
        dc.a = 1+2j
    with pytest.raises(TypeError):
        dc.b = -77.1
    with pytest.raises(TypeError):
        dc.c = 5
    with pytest.raises(TypeError):
        dc.d = -61.2
    with pytest.raises(TypeError):
        dc.e = -0.1
    with pytest.raises(TypeError):
        dc.f = 'abc'
    with pytest.raises(ValueError):
        dc.g = (3, 2.1, -6)
    with pytest.raises(ValueError):
        dc.h = (-5, 6, 8, 9)
    with pytest.raises(TypeError):
        dc.i = (1, -2, 3.2)
    with pytest.raises(TypeError):
        dc.j = 5

    # Attempt deleting attribute
    with pytest.raises(AttributeError):
        del dc.i


def test_missing_size_option():
    with pytest.raises(TypeError):
        @ts.typesystem(a=ts.SizedVector)
        class DummyClass:
            def __init__(self, a):
                self.a = a


def test_missing_expected_type_option():
    with pytest.raises(TypeError):
        @ts.typesystem(a=ts.TypedAttribute)
        class DummyClass:
            def __init__(self, a):
                self.a = a
