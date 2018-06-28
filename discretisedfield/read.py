import struct
import numpy as np
import os
from .mesh import Mesh
from .field import Field


def read(filename, norm=None, name="field"):
    mdatalist = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax",
                 "xstepsize", "ystepsize", "zstepsize", "valuedim"]
    mdatadict = dict()

    try:
        with open(filename, "r", encoding="utf-8") as ovffile:
            f = ovffile.read()
            lines = f.split("\n")

        mdatalines = filter(lambda s: s.startswith("#"), lines)
        datalines = np.loadtxt(filter(lambda s: not s.startswith("#"), lines))

        for line in mdatalines:
            for metadatum in mdatalist:
                if metadatum in line:
                    mdatadict[metadatum] = float(line.split()[-1])
                    break

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

        header = b"# Begin: Data Binary "
        data_start = f.find(header)
        header = f[data_start:data_start + len(header) + 1]

        data_start += len(b"# Begin: Data Binary 8\n")
        data_end = f.find(b"# End: Data Binary ")

        if b"4" in header:
            if os.name == 'nt':                
                listdata = list(struct.iter_unpack("@f", f[data_start+1:data_end]))
            else:
                listdata = list(struct.iter_unpack("@f", f[data_start:data_end])
            try:
                assert listdata[0] == 1234567.0
            except AssertionError:
                raise AssertionError("Something has gone wrong "
                                     "with reading Binary Data")
        elif b"8" in header:
            print((data_end - data_start) / 8 )
            print(f[data_start:data_end])

            if os.name == 'nt':
                listdata = list(struct.iter_unpack("@d", f[data_start+1:data_end]))
            else:
                listdata = list(struct.iter_unpack("@d", f[data_start:data_end]))
            try:
                assert listdata[0][0] == 123456789012345.0
            except AssertionError:
                raise AssertionError("Something has gone wrong "
                                     "with reading Binary Data")

        datalines = np.array(listdata[1:])

    field = _make_field(mdatadict, name)
    r_tuple = tuple(reversed(field.mesh.n)) + (int(mdatadict["valuedim"]),)
    t_tuple = tuple(reversed(range(3))) + (3,)
    field.array = datalines.reshape(r_tuple).transpose(t_tuple)
    field.norm = norm

    return field


def _make_field(mdatadict, name):
    p1 = (mdatadict[key] for key in ["xmin", "ymin", "zmin"])
    p2 = (mdatadict[key] for key in ["xmax", "ymax", "zmax"])
    cell = (mdatadict[key] for key in ["xstepsize", "ystepsize", "zstepsize"])
    dim = int(mdatadict["valuedim"])

    mesh = Mesh(p1=p1, p2=p2, cell=cell)
    field = Field(mesh, dim=dim, name=name)

    return field
