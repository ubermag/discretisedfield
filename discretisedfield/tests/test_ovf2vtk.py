import subprocess
import sys

import numpy as np

import discretisedfield as df


def test_ovf2vtk(tmp_path, capfd):
    p1 = (0, 0, 0)
    p2 = (10e-9, 7e-9, 2e-9)
    cell = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def value_fun(point):
        x, y, z = point
        c = 1e9
        return c * x, c * y, c * z

    f = df.Field(mesh, dim=3, value=value_fun)

    # Output filename provided.
    omffilename_1 = str(tmp_path / "test-ovf2vtk1.omf")
    vtkfilename_1 = str(tmp_path / "test-ovf2vtk1.vtk")
    f._writeovf(omffilename_1)

    cmd = [
        sys.executable,
        "-m",
        "discretisedfield.ovf2vtk",
        "--input",
        omffilename_1,
        "--output",
        vtkfilename_1,
    ]
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0

    f_read = df.Field.fromfile(vtkfilename_1)
    assert np.allclose(f.array, f_read.array)

    # Output filename not provided.
    omffilename_2 = str(tmp_path / "test-ovf2vtk2.omf")
    vtkfilename_2 = str(tmp_path / "test-ovf2vtk2.vtk")
    f._writeovf(omffilename_2, representation="bin4")

    cmd = [sys.executable, "-m", "discretisedfield.ovf2vtk", "-i", omffilename_2]
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0

    f_read = df.Field.fromfile(vtkfilename_2)
    assert np.allclose(f.array, f_read.array)

    # Number of input and output files do not match.
    cmd = [
        sys.executable,
        "-m",
        "discretisedfield.ovf2vtk",
        "-i",
        omffilename_1,
        omffilename_2,
        "-o",
        "file1.vtk",
    ]
    proc_return = subprocess.run(cmd)
    captured = capfd.readouterr()
    assert proc_return.returncode != 0
    msg = "The number of input files (2) does not match the number of output files (1)."
    assert msg in captured.err

    # Wrong file name.
    cmd = [
        sys.executable,
        "-m",
        "discretisedfield.ovf2vtk",
        "-i",
        "file1.omf",
        "-o",
        "file1.vtk",
    ]
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode != 0
    captured = capfd.readouterr()
    assert "No such file or directory" in captured.err
