import os
import sys
import subprocess
import discretisedfield as df
from discretisedfield.ovf2vtk import convert_files


def test_write_vtk_file():
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
    f = df.Field(mesh, dim=3, value=(1, 2, -3.1))

    filename_omf = "test_write_vtk_file.omf"
    filename_vtk = "test_write_vtk_file.vtk"
    f.write(filename_omf)

    convert_files([filename_omf], [filename_vtk])

    interpreter = sys.executable.split('/')[-1]
    cmd = [interpreter, '-m', 'discretisedfield.ovf2vtk',
           '--infile', f'{filename_omf}']
    proc_return = subprocess.run(cmd)

    assert proc_return.returncode == 0, proc_return.stdout + proc_return.stderr

    pattern = ''
    with open(filename_vtk, 'r') as f:
        for line in f.readlines():
            if 'CELL_DATA' in line:
                pattern = line.strip()

    assert pattern == 'CELL_DATA 8'

    os.remove(f'{filename_omf}')
    os.remove(f'{filename_vtk}')
