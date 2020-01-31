import os
import sys
import subprocess
import discretisedfield as df
from discretisedfield.ovf2vtk import convert_files


def check_vtk(vtkfile):
    pattern = ''
    with open(vtkfile, 'r') as f:
        for line in f.readlines():
            if 'CELL_DATA' in line:
                pattern = line.strip()
                break
    assert pattern == 'CELL_DATA 8'


def test_ovf2vtk():
    p1 = (0, 0, 0)
    p2 = (10, 10, 10)
    cell = (5, 5, 5)
    mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

    f = df.Field(mesh, dim=3, value=(1, 2, -3.14))

    omffilename = 'test_ovf2vtk_file.omf'
    vtkfilename = 'test_ovf2vtk_file.vtk'
    f.write(omffilename)

    interpreter = sys.executable.split('/')[-1]

    # No output filename provided.
    cmd = [interpreter, '-m', 'discretisedfield.ovf2vtk',
           '--infile', omffilename]
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0
    check_vtk(vtkfilename)
    os.remove(vtkfilename)

    # Output filename provided.
    cmd = [interpreter, '-m', 'discretisedfield.ovf2vtk',
           '--infile', omffilename, '--outfile', vtkfilename]
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0
    check_vtk(vtkfilename)
    os.remove(vtkfilename)

    # Number of input and output files do not match.
    cmd = [interpreter, '-m', 'discretisedfield.ovf2vtk',
           '--infile', omffilename, '--outfile',
           vtkfilename, 'anotherfile.vtk']
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 1

    os.remove(omffilename)
    os.remove(vtkfilename)
    os.remove('anotherfile.vtk')
