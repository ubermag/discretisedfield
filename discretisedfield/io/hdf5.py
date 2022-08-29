import contextlib
import pathlib

import h5py
import numpy as np

import discretisedfield as df


def field_to_hdf5(field, filename, save_subregions=True):
    """Write the field to an HDF5 file.

    Parameters
    ----------
    filename : pathlib.Path, str

        Name with an extension of the file written.

    save_subregions : bool, optional

       If ``True`` and subregions are defined for the mesh the subregions will be saved
       to a json file. Defaults to ``True``.

    Example
    -------
    1. Write field to an HDF5 file.

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
    >>> filename = 'mytestfile.h5'
    >>> field.write(filename)  # write the file
    >>> os.path.isfile(filename)
    True
    >>> field_read = df.Field.fromfile(filename)  # read the file
    >>> field_read == field
    True
    >>> os.remove(filename)  # delete the file

    See also
    --------
    ~discretisedfield.Field.write
    field_from_hdf5

    """
    filename = pathlib.Path(filename)
    if save_subregions and field.mesh.subregions:
        field.mesh.save_subregions(filename)

    with h5py.File(filename, "w") as f:
        # Set up the file structure
        gfield = f.create_group("field")
        gmesh = gfield.create_group("mesh")
        gregion = gmesh.create_group("region")

        # Save everything as datasets
        gregion.create_dataset("p1", data=field.mesh.region.p1)
        gregion.create_dataset("p2", data=field.mesh.region.p2)
        gmesh.create_dataset("n", dtype="i4", data=field.mesh.n)
        gfield.create_dataset("dim", dtype="i4", data=field.dim)
        gfield.create_dataset("array", data=field.array)


def field_from_hdf5(filename):
    """Read the field from an HDF5 file.

    This method reads the field from an HDF5 file defined on written by
    ``discretisedfield.io.field_to_hdf5``.

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
    1. Read a field from the HDF5 file.

    >>> import pathlib
    >>> import discretisedfield as df
    ...
    >>> current_path = pathlib.Path(__file__).absolute().parent
    >>> filepath = current_path / '..' / 'tests' / 'test_sample' / 'hdf5-file.hdf5'
    >>> field = df.Field.fromfile(filepath)
    >>> field
    Field(...)

    See also
    --------
    ~discretisedfield.Field.fromfile
    field_to_hdf5

    """
    filename = pathlib.Path(filename)
    with h5py.File(filename, "r") as f:
        # Read data from the file.
        p1 = f["field/mesh/region/p1"]
        p2 = f["field/mesh/region/p2"]
        n = np.array(f["field/mesh/n"]).tolist()
        dim = np.array(f["field/dim"]).tolist()
        array = f["field/array"]

        # Create field.
        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        with contextlib.suppress(FileNotFoundError):
            mesh.load_subregions(filename)

        return df.Field(mesh, dim=dim, value=array[:])
