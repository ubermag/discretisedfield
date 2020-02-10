import re
import pytest
import numbers
import numpy as np
import discretisedfield as df


def check_line(line):
    assert isinstance(line, df.Line)

    assert isinstance(line.dictionary, dict)

    assert isinstance(line.points, list)
    assert isinstance(line.values, list)
    assert len(line.points) == len(line.values)

    assert isinstance(line.length, numbers.Real)
    assert isinstance(line.n, int)
    assert line.n > 0
    assert isinstance(line.dim, int)
    assert line.dim > 0

    assert isinstance(line(line.points[0]), (tuple, numbers.Real))

    assert isinstance(repr(line), str)
    pattern = r'^Line\(points=..., values=...\)$'
    assert re.search(pattern, repr(line))


class TestLine:
    def test_init(self):
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        values = [-1, 2, -3]
        line = df.Line(points=points, values=values)
        check_line(line)

        assert line.length == 2
        assert line.n == 3
        assert line.dim == 1
        assert line((0, 0, 0)) == -1

        points = [(0, 0, 0), (1, 1, 1)]
        values = [(0, 0, 1), (0, 1, 0)]
        line = df.Line(points=points, values=values)
        check_line(line)

        assert abs(line.length - np.sqrt(3)) < 1e-12
        assert line.n == 2
        assert line.dim == 3
        assert line((1, 1, 1)) == (0, 1, 0)

        # Exceptions
        points = [(0, 0, 0), (1, 0, 0)]
        values = [-1, 2, -3]
        with pytest.raises(ValueError):
            line = df.Line(points=points, values=values)

    def test_mpl(self):
        # Scalar values
        points = [(0, 0, 0), (0, 1e-9, 0), (0, 2e-9, 0)]
        values = [-1e6, 2e6, -3e6]
        line = df.Line(points=points, values=values)
        check_line(line)

        line.mpl()
        line.mpl(figsize=(8, 6))
        line.mpl(multiplier=1e-9)

        # Vector values
        points = [(0, 0, 0), (1, 1, 1)]
        values = [(0, 0, 1), (0, 1, 0)]
        line = df.Line(points=points, values=values)
        check_line(line)

        line.mpl()
        line.mpl(figsize=(8, 6))
        line.mpl(multiplier=1e3)
