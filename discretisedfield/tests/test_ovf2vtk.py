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
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
    f = df.Field(mesh, dim=3, value=(1, 2, -3.14))

    filename_omf = 'test_ovf2vtk_file.omf'
    filename_vtk = 'test_ovf2vtk_file.vtk'
    f.write(filename_omf)

    interpreter = sys.executable.split('/')[-1]

    # No output filename provided.
    cmd = [interpreter, '-m', 'discretisedfield.ovf2vtk',
           '--infile', f'{filename_omf}']
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0
    check_vtk(filename_vtk)

    # Output filename provided.
    cmd = [interpreter, '-m', 'discretisedfield.ovf2vtk',
           '--infile', f'{filename_omf}', '--outfile',
           f'{filename_vtk}']
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0
    check_vtk(filename_vtk)

    # Number of input and output files do not match.
    cmd = [interpreter, '-m', 'discretisedfield.ovf2vtk',
           '--infile', f'{filename_omf}', '--outfile',
           f'{filename_vtk}', f'anotherfile.vtk']
    proc_return = subprocess.run(cmd)
    assert proc_return.returncode == 0

    os.remove(f'{filename_omf}')
    os.remove(f'{filename_vtk}')
