import contextlib
import pathlib

import numpy as np
from vtkmodules.util import numpy_support as vns
from vtkmodules.vtkIOLegacy import vtkRectilinearGridReader, vtkRectilinearGridWriter
from vtkmodules.vtkIOXML import vtkXMLRectilinearGridReader, vtkXMLRectilinearGridWriter

import discretisedfield as df


def field_to_vtk(field, filename, representation="bin", save_subregions=True):
    """Write the field to a VTK file.

    The data is saved as a ``RECTILINEAR_GRID`` dataset. Scalar field
    (``dim=1``) is saved as ``SCALARS``. On the other hand, vector field
    (``dim=3``) is saved as both ``VECTORS`` as well as ``SCALARS`` for all
    three components to enable easy colouring of vectors in some
    visualisation packages. The data is stored as ``CELL_DATA``.

    The saved VTK file can be opened with `Paraview
    <https://www.paraview.org/>`_ or `Mayavi
    <https://docs.enthought.com/mayavi/mayavi/>`_. To show contour lines in
    Paraview one has to first convert Cell Data to Point Data using a
    filter.

    Parameters
    ----------
    filename : pathlib.Path, str

        File name with an extension.

    representation : str, optional

        Representation; ``'bin'`` [``'bin8'`` as equivalent], ``'txt'``, ``'xml'``;
        defaults to ``'bin'``.

    save_subregions : bool, optional

       If ``True`` and subregions are defined for the mesh the subregions will be saved
       to a json file. Defaults to ``True``.

    Example
    -------
    1. Write field to a VTK file.

    >>> import os
    >>> import discretisedfield as df
    ...
    >>> p1 = (0, 0, 0)
    >>> p2 = (10e-9, 5e-9, 3e-9)
    >>> n = (10, 5, 3)
    >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
    >>> value_fun = lambda point: (point[0], point[1], point[2])
    >>> field = df.Field(mesh, dim=3, value=value_fun)
    ...
    >>> filename = 'mytestfile.vtk'
    >>> field.write(filename)  # write the file
    >>> os.path.isfile(filename)
    True
    >>> os.remove(filename)  # delete the file

    See also
    --------
    ~discretisedfield.Field.write
    field_from_vtk

    """
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

    if save_subregions and field.mesh.subregions:
        field.mesh.save_subregions(filename)

    writer.SetFileName(str(filename))
    writer.SetInputData(field.to_vtk())
    writer.Write()


def field_from_vtk(filename):
    """Read the field from a VTK file.

    This method reads the field from a VTK file defined on RECTILINEAR GRID
    written by ``discretisedfield.io.field_to_vtk``. It expects the data do be
    specified as cell data and one (vector) field with the name ``field``.
    A vector field should also contain data for the individual components.
    The individual component names are used as ``components`` for the new
    field. They must appear in the form ``<componentname>-component``.

    Older versions of discretisedfield did write the data as point data instead of cell
    data. This function can load new and old files and automatically extracts the
    correct data without additional user input.

    Parameters
    ----------
    filename : pathlib.Path, str

        Name of the file to be read.

    Returns
    -------
    discretisedfield.Field

        Field read from the file.

    Example
    -------
    1. Read a field from the VTK file.

    >>> import pathlib
    >>> import discretisedfield as df
    ...
    >>> current_path = pathlib.Path(__file__).absolute().parent
    >>> filepath = current_path / '..' / 'tests' / 'test_sample' / 'vtk-file.vtk'
    >>> field = df.Field.fromfile(filepath)
    >>> field
    Field(...)

    See also
    --------
    ~discretisedfield.Field.fromfile
    field_to_vtk

    """
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
        return fromvtk_legacy(filename)

    components = []
    for i in range(cell_data.GetNumberOfArrays()):
        name = cell_data.GetArrayName(i)
        if name == "field":
            field_idx = i
        elif name.endswith("-component"):
            components.append(name[: -len("-component")])
    array = cell_data.GetArray(field_idx)
    dim = array.GetNumberOfComponents()

    if len(components) != dim:
        components = None

    value = vns.vtk_to_numpy(array).reshape(*reversed(n), dim)
    value = value.transpose((2, 1, 0, 3))

    mesh = df.Mesh(p1=p1, p2=p2, n=n)
    with contextlib.suppress(FileNotFoundError):
        mesh.load_subregions(filename)

    return df.Field(mesh, dim=dim, value=value, components=components)


def fromvtk_legacy(filename):
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
    field = df.Field(mesh, dim=dim)

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
