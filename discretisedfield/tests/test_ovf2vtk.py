import os
import discretisedfield as df


class TestField:

    def test_write_vtk_file(self):
        mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
        f = df.Field(mesh, dim=1, value=-3.1)

        filename_omf = "test_write_vtk_file.omf"
        filename_vtk = "test_write_vtk_file.vtk"
        f.write(filename_omf)

        os.system("python -m discretisedfield.ovf2vtk --infile {}".format(filename_omf))

        pattern = ''
        with open(filename_vtk, 'r') as f:
            for line in f.readlines():
                if 'CELL_DATA' in line:
                    pattern = line.strip()

        assert pattern == 'CELL_DATA 8'
