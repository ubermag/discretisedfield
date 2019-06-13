import os
import discretisedfield as df
from numpy.testing import assert_allclose


def test_write_vtk_file():
    # num dims, default value, relative tolerence
    dims = [(1, -1.23, 1e-7), (3, (1, 2, 3), 1e-14)]
    mesh = df.Mesh(p1=(0, 0, 0), p2=(10, 10, 10), cell=(5, 5, 5))
    filename = "test_write_file.omf"

    for d, v, e in dims:
        f_out = df.Field(mesh, dim=d, value=v)
        for rep in df.field.representations:
            f_out.write(filename, representation=rep)
            f_in = df.read(filename)
            assert_allclose(f_out.array, f_in.array, rtol=e)

    os.remove(filename)
