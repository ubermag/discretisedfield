import numbers
import os
import tempfile

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import discretisedfield as df


def check_line(line):
    assert isinstance(line, df.Line)
    assert isinstance(line.data, pd.DataFrame)

    assert isinstance(line.n, int)
    assert line.n > 0
    assert isinstance(line.dim, int)
    assert line.dim > 0

    assert isinstance(line.point_columns, list)
    assert len(line.point_columns) == 3
    assert isinstance(line.value_columns, list)
    assert len(line.value_columns) == line.dim

    assert isinstance(line.length, numbers.Real)
    assert line.length > 0

    assert isinstance(repr(line), str)

    assert isinstance(line.slider(), ipywidgets.SelectionRangeSlider)
    assert isinstance(line.selector(), ipywidgets.SelectMultiple)


class TestLine:
    def test_init(self):
        # Scalar values
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        values = [-1, 2, -3]
        line = df.Line(points=points, values=values)
        check_line(line)

        assert line.length == 2
        assert line.n == 3
        assert line.dim == 1

        # Vector values
        points = [(0, 0, 0), (1, 1, 1)]
        values = [(0, 0, 1), (0, 1, 0)]

        line = df.Line(points=points, values=values)
        check_line(line)

        assert abs(line.length - np.sqrt(3)) < 1e-12
        assert line.n == 2
        assert line.dim == 3

        # Setting the point and value columns.
        points = [(0, 0, 0), (1, 1, 1)]
        values = [(0, 0, 1), (0, 1, 0)]

        line = df.Line(
            points=points,
            values=values,
            point_columns=["a", "b", "c"],
            value_columns=["x", "y", "z"],
        )
        check_line(line)
        assert line.point_columns == ["a", "b", "c"]
        assert line.value_columns == ["x", "y", "z"]

        assert abs(line.length - np.sqrt(3)) < 1e-12
        assert line.n == 2
        assert line.dim == 3

        # From field
        p1 = (0, 0, 0)
        p2 = (10e-9, 15e-9, 2e-9)
        n = (10, 15, 2)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        f = df.Field(mesh, dim=3, value=(1, 1, 1))

        line = f.line(p1=p1, p2=(10e-9, 15e-9, 1e-9), n=200)
        check_line(line)

        assert line.n == 200
        assert line.dim == 3

        line = f.y.line(p1=p1, p2=(10e-9, 15e-9, 1e-9), n=100)
        check_line(line)

        assert line.n == 100
        assert line.dim == 1

        # Exceptions
        points = [(0, 0, 0), (1, 0, 0)]
        values = [-1, 2, -3]
        with pytest.raises(ValueError):
            line = df.Line(points=points, values=values)

    def test_point_value_columns(self):
        # Scalar values
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        values = [-1, 2, -3]
        line = df.Line(points=points, values=values)

        assert line.point_columns == ["px", "py", "pz"]
        assert line.value_columns == ["v"]

        line.point_columns = ["a", "b", "c"]
        line.value_columns = ["val"]

        assert line.data.columns.to_list() == ["r", "a", "b", "c", "val"]

        # Vector values.
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        values = [(0, 1, 3), (-1, 0, 0), (-2.13, 0, 0)]
        line = df.Line(points=points, values=values)

        assert line.point_columns == ["px", "py", "pz"]
        assert line.value_columns == ["vx", "vy", "vz"]

        line.point_columns = ["a", "b", "c"]
        line.value_columns = ["v0", "v1", "v2"]

        assert line.data.columns.to_list() == ["r", "a", "b", "c", "v0", "v1", "v2"]

        # Exceptions
        points = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        values = [(0, 1, 3), (-1, 0, 0), (-2.13, 0, 0)]
        line = df.Line(points=points, values=values)

        with pytest.raises(ValueError):
            line.point_columns = ["a", "b"]
        with pytest.raises(ValueError):
            line.value_columns = ["a", "b", "c", "d"]

    def test_mpl(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 15e-9, 2e-9)
        n = (10, 15, 2)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        f = df.Field(mesh, dim=3, value=(1, 1, 1))

        line = f.line(p1=(1e-9, 1e-9, 0.1e-9), p2=(4e-9, 5e-9, 1e-9), n=20)

        # No axis
        line.mpl()

        # Axis
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        line.mpl(ax=ax)

        # figsize
        line.mpl(figsize=(10, 5))

        # multiplier
        line.mpl(multiplier=1e-6)

        # y
        line.mpl(yaxis=["vx", "vz"])

        # xlim
        line.mpl(xlim=(0, 10e-9))

        # kwargs
        line.mpl(marker="o")

        # filename
        filename = "line.pdf"
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpfilename = os.path.join(tmpdir, filename)
            line.mpl(filename=tmpfilename)

        plt.close("all")

    def test_slider_selector(self):
        p1 = (0, 0, 0)
        p2 = (10e-9, 15e-9, 2e-9)
        n = (10, 15, 2)
        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        f = df.Field(mesh, dim=3, value=(1, 1, 1))

        line = f.line(p1=(1e-9, 1e-9, 0.1e-9), p2=(4e-9, 5e-9, 1e-9), n=20)

        assert isinstance(line.slider(), ipywidgets.SelectionRangeSlider)
        assert isinstance(line.selector(), ipywidgets.SelectMultiple)
