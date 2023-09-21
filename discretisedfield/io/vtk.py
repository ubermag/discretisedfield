import contextlib
import pathlib

import numpy as np
from vtkmodules.util import numpy_support as vns
from vtkmodules.vtkIOLegacy import vtkRectilinearGridReader, vtkRectilinearGridWriter
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader, vtkXMLRectilinearGridWriter

import discretisedfield as df


class _FieldIO_VTK:
    __slots__ = []

    def _to_vtk(self, filename, representation="bin", save_subregions=True):
        filename = pathlib.Path(filename)
        if representation == "xml":
            writer = vtkXMLRectilinearGridWriter()
        elif representation in ["bin", "bin8", "txt"]:
            # Allow bin8 for convenience as this is the default for omf.
            # This does not affect the actual datatype used in vtk files.
            writer = vtkRectilinearGridWriter()
        else:
            raise ValueError(f"Unknown {representation=}.")

        if representation == "txt":
            writer.SetFileTypeToASCII()
        elif representation in ["bin", "bin8"]:
            writer.SetFileTypeToBinary()
        # xml has no distinction between ascii and binary

        writer.SetFileName(str(filename))
        # Convert field to VTK before writing subregion information because
        # to_vtk will fail if ndim is not correct
        writer.SetInputData(self.to_vtk())

        if save_subregions and self.mesh.subregions:
            self.mesh.save_subregions(filename)

        writer.Write()

    @classmethod
    def _from_vtk(cls, filename):
        filename = pathlib.Path(filename)
        with filename.open("rb") as f:
            xml = "xml" in f.readline().decode("utf8")
        if xml:
            reader = vtkXMLRectilinearGridReader()
        else:
            reader = vtkRectilinearGridReader()
            reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
        reader.SetFileName(str(filename))
        reader.Update()

        output = reader.GetOutput()
        p1 = output.GetBounds()[::2]
        p2 = output.GetBounds()[1::2]
        n = [i - 1 for i in output.GetDimensions()]

        cell_data = output.GetCellData()

        if cell_data.GetNumberOfArrays() == 0:
            # Old writing routine did write to points instead of cells.
            return cls._from_vtk_legacy(filename)

        vdims = []
        for i in range(cell_data.GetNumberOfArrays()):
            name = cell_data.GetArrayName(i)
            if name == "field":
                field_idx = i
            elif name.endswith("-component"):
                vdims.append(name[: -len("-component")])
        array = cell_data.GetArray(field_idx)
        dim = array.GetNumberOfComponents()

        if len(vdims) != dim:
            vdims = None

        value = vns.vtk_to_numpy(array).reshape(*reversed(n), dim)
        value = value.transpose((2, 1, 0, 3))

        mesh = df.Mesh(p1=p1, p2=p2, n=n)
        with contextlib.suppress(FileNotFoundError):
            mesh.load_subregions(filename)

        return cls(mesh, nvdim=dim, value=value, vdims=vdims)

    @classmethod
    def _from_vtk_legacy(cls, filename):
        """Read the field from a VTK file (legacy).

        This method reads vtk files written with discretisedfield <= 0.61.0
        in which the data is stored as point data instead of cell data.
        """
        with open(filename, "r") as f:
            content = f.read()
        lines = content.split("\n")

        # Determine the dimension of the field.
        if "VECTORS" in content:
            dim = 3
            data_marker = "VECTORS"
            skip = 0  # after how many lines data starts after marker
        else:
            dim = 1
            data_marker = "SCALARS"
            skip = 1

        # Extract the metadata
        mdatalist = ["X_COORDINATES", "Y_COORDINATES", "Z_COORDINATES"]
        n = []
        cell = []
        origin = []
        for i, line in enumerate(lines):
            for mdatum in mdatalist:
                if mdatum in line:
                    n.append(int(line.split()[1]))
                    coordinates = list(map(float, lines[i + 1].split()))
                    origin.append(coordinates[0])
                    if len(coordinates) > 1:
                        cell.append(coordinates[1] - coordinates[0])
                    else:
                        # If only one cell exists, 1nm cell is used by default.
                        cell.append(1e-9)

        # Create objects from metadata info
        p1 = np.subtract(origin, np.multiply(cell, 0.5))
        p2 = np.add(p1, np.multiply(n, cell))
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        with contextlib.suppress(FileNotFoundError):
            mesh.load_subregions(filename)
        field = df.Field(mesh, nvdim=dim)

        # Find where data starts.
        for i, line in enumerate(lines):
            if line.startswith(data_marker):
                start_index = i
                break

        # Extract data.
        for i, line in zip(mesh.indices, lines[start_index + skip + 1 :]):
            if not line[0].isalpha():
                field.array[i] = list(map(float, line.split()))

        return field
