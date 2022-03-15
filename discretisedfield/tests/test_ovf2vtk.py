import subprocess
import sys

import numpy as np

import discretisedfield as df


def test_ovf2vtk_explicit(tmp_path):
    p1 = (0, 0, 0)
    p2 = (10e-9, 7e-9, 2e-9)
    cell = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def value_fun(point):
        x, y, z = point
        c = 1e9
        return c*x, c*y, c*z

    f = df.Field(mesh, dim=3, value=value_fun)

    # Output filename provided.
    omffilename = str(tmp_path / 'test-ovf2vtk1.omf')
    vtkfilename = str(tmp_path / 'test-ovf2vtk1.vtk')
    f._writeovf(omffilename)

    cmd = [sys.executable, '-m', 'discretisedfield.ovf2vtk',
           '--input', omffilename, '--output', vtkfilename]
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0

    f_read = df.Field.fromfile(vtkfilename)
    assert np.allclose(f.array, f_read.array)


def test_ovf2vtk_implicit(tmp_path):
    p1 = (0, 0, 0)
    p2 = (10e-9, 7e-9, 2e-9)
    cell = (1e-9, 1e-9, 1e-9)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    def value_fun(point):
        x, y, z = point
        c = 1e9
        return c*x, c*y, c*z

    f = df.Field(mesh, dim=3, value=value_fun)
    # Output filename not provided.
    omffilename = 'test-ovf2vtk2.omf'
    vtkfilename = 'test-ovf2vtk2.vtk'
    f._writeovf(omffilename, representation='bin4')

    cmd = [sys.executable, '-m', 'discretisedfield.ovf2vtk',
           '-i', omffilename]
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0

    f_read = df.Field.fromfile(vtkfilename)
    assert np.allclose(f.array, f_read.array)


def test_ovf2vtk_wrong(capsys):
    # Number of input and output files do not match.
    cmd = [sys.executable, '-m', 'discretisedfield.ovf2vtk',
           '-i', 'file1.omf', 'file2.omf',
           '-o', 'file1.vtk']
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode != 0
    captured = capsys.readouterr()
    msg = ('The number of input files (2) does not '
           'match the number of output files (1).')
    assert msg in captured.out
