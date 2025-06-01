import os
import tempfile

import numpy as np
import pytest
import pyvista as pv

import discretisedfield as df
import discretisedfield.plotting.util as plot_util


def test_inplane_angle():
    field = df.Mesh(p1=(-1, -1), p2=(1, 1), n=(2, 2)).coordinate_field()
    angles = plot_util.inplane_angle(field, "x", "y")
    assert isinstance(angles, df.Field)
    # The field vectors start at (-1, -1) with 180° + 45°
    # then first y increases -> 90° + 45°
    # then x and afterwards y increases -> 270° + 45° and 0° + 45°
    assert np.allclose(
        angles.array.flat, np.array([2, 1, 3, 0]) * np.pi / 2 + np.pi / 4
    )


@pytest.mark.parametrize("extension", ["png", "jpeg", "jpg", "bmp", "tif", "tiff"])
@pytest.mark.pyvista
def test_pyvista_valid_filename_extensions_screenshot(extension):
    pv.OFF_SCREEN = True

    plotter = pv.Plotter()
    plotter.add_mesh(pv.Sphere(), show_edges=True)
    plotter.show()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp_file:
        filename = tmp_file.name

    try:
        plot_util._pyvista_save_to_file(filename, plotter)
        assert os.path.exists(filename), (
            f"File with extension {extension} was not created."
        )
    finally:
        os.remove(filename)


# xfail as need on screen to create graphics. TODO: check how pyvista test this
@pytest.mark.xfail(strict=True)
@pytest.mark.parametrize("extension", ["svg", "eps", "ps", "pdf", "txt"])
@pytest.mark.pyvista
def test_pyvista_valid_filename_extensions_save_graphic(extension):
    pv.OFF_SCREEN = True

    plotter = pv.Plotter()
    plotter.add_mesh(pv.Sphere(), show_edges=True)
    plotter.show()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp_file:
        filename = tmp_file.name

    try:
        plot_util._pyvista_save_to_file(filename, plotter)
        assert os.path.exists(filename), (
            f"File with extension {extension} was not created."
        )
    finally:
        os.remove(filename)
