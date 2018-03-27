import struct
from .mesh import Mesh
from .field import Field


def read(filename, norm=None, name="field"):
    mdatalist = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax",
                 "xstepsize", "ystepsize", "zstepsize", "valuedim"]
    mdatadict = dict()

    try:
        with open(filename, "r") as ovffile:
            f = ovffile.read()
            lines = f.split("\n")

        mdatalines = filter(lambda s: s.startswith("#"), lines)
        datalines = filter(lambda s: not s.startswith("#"), lines)

        for line in mdatalines:
            for mdatum in mdatalist:
                if mdatum in line:
                    mdatadict[mdatum] = float(line.split()[-1])
                    break

        mesh, field = _create_mesh_and_field(mdatadict, name)

        for i, (index, line) in enumerate(zip(mesh.indices, datalines)):
            value = [float(vi) for vi in line.split()]
            if mdatadict["valuedim"] == 1:
                field.array[index] = value[0]
            else:
                field.array[index] = value

        return field

    except UnicodeDecodeError:
        with open(filename, "rb") as ovffile:
                f = ovffile.read()
                lines = f.split(b"\n")

        mdatalines = filter(lambda s: s.startswith(bytes("#", "utf-8")), lines)
        datalines = filter(lambda s: not s.startswith(bytes("#", "utf-8")),
                           lines)

        for line in mdatalines:
            for mdatum in mdatalist:
                if bytes(mdatum, "utf-8") in line:
                    mdatadict[mdatum] = float(line.split()[-1])
                    break

        mesh, field = _create_mesh_and_field(mdatadict, name)

        header = b"# Begin: Data Binary "
        data_start = f.find(header)
        header = f[data_start:data_start + len(header) + 1]

        data_start += len(b"# Begin: Data Binary 8\n")
        data_end = f.find(b"# End: Data Binary ")
        if b"4" in header:
            listdata = list(struct.iter_unpack("@f", f[data_start:data_end]))
            try:
                assert listdata[0] == 1234567.0
            except AssertionError:
                raise AssertionError("Something has gone wrong "
                                     "with reading Binary Data")
        elif b"8" in header:
            listdata = list(struct.iter_unpack("@d", f[data_start:data_end]))
            try:
                assert listdata[0][0] == 123456789012345.0
            except AssertionError:
                raise AssertionError("Something has gone wrong "
                                     "with reading Binary Data")

        counter = 1
        for index in mesh.indices:
            value = (listdata[counter][0],
                     listdata[counter+1][0],
                     listdata[counter+2][0])
            field.array[index] = value

            counter += 3

        return field

    field.norm = norm
    if norm is not None:
        field.norm = norm

    return field


def _create_mesh_and_field(mdatadict, name):
    p1 = (mdatadict[key] for key in ["xmin", "ymin", "zmin"])
    p2 = (mdatadict[key] for key in ["xmax", "ymax", "zmax"])
    cell = (mdatadict[key] for key in ["xstepsize", "ystepsize", "zstepsize"])
    dim = int(mdatadict["valuedim"])

    mesh = Mesh(p1=p1, p2=p2, cell=cell)
    field = Field(mesh, dim=dim, name=name)

    return mesh, field
