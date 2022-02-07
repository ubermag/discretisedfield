import os
import sys
import tempfile
import subprocess
import numpy as np
import discretisedfield as df


def test_ovf2vtk():
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
    omffilename = 'test-ovf2vtk1.omf'
    vtkfilename = 'test-ovf2vtk1.vtk'
    with tempfile.TemporaryDirectory() as tmpdir:
        omftmpfilename = os.path.join(tmpdir, omffilename)
        vtktmpfilename = os.path.join(tmpdir, vtkfilename)
        f._writeovf(omftmpfilename, representation='bin8')

        cmd = [sys.executable, '-m', 'discretisedfield.ovf2vtk',
               '--input', omftmpfilename, '--output', vtktmpfilename]
        proc_return = subprocess.run(cmd)
        assert proc_return.returncode == 0

        f_read = df.Field.fromfile(vtktmpfilename)
        assert np.allclose(f.array, f_read.array)

    # Output filename not provided.
    omffilename = 'test-ovf2vtk2.omf'
    vtkfilename = 'test-ovf2vtk2.vtk'
    with tempfile.TemporaryDirectory() as tmpdir:
        omftmpfilename = os.path.join(tmpdir, omffilename)
        vtktmpfilename = os.path.join(tmpdir, vtkfilename)
        f._writeovf(omftmpfilename, representation='bin4')

        cmd = [sys.executable, '-m', 'discretisedfield.ovf2vtk',
               '-i', omftmpfilename]
        proc_return = subprocess.run(cmd)
        assert proc_return.returncode == 0

        f_read = df.Field.fromfile(vtktmpfilename)
        assert np.allclose(f.array, f_read.array)

    # Number of input and output files do not match.
    cmd = [sys.executable, '-m', 'discretisedfield.ovf2vtk',
           '-i', 'file1.omf', 'file2.omf',
           '-o', 'file1.vtk']
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode != 0
