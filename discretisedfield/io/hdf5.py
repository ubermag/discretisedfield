import contextlib

import h5py
import numpy as np

import discretisedfield as df

from .util import strip_extension


def field_to_hdf5(field, filename, save_subregions=True):
    """Write the field to an HDF5 file.

    Parameters
    ----------
    filename : str

        Name with an extension of the file written.

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
    >>> field._writehdf5(filename)  # write the file
    >>> os.path.isfile(filename)
    True
    >>> field_read = df.Field.fromfile(filename)  # read the file
    >>> field_read == field
    True
    >>> os.remove(filename)  # delete the file

    .. seealso:: :py:func:`~discretisedfield.Field.fromfile`

    """
    if save_subregions and field.mesh.subregions:
        field.mesh.save_subregions(f"{strip_extension(filename)}_subregions.json")

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
    ``discretisedfield._writevtk``.

    This is a ``classmethod`` and should be called as, for instance,
    ``discretisedfield.Field._fromhdf5('myfile.h5')``.

    Parameters
    ----------
    filename : str

        Name of the file to be read.

    Returns
    -------
    discretisedfield.Field

        Field read from the file.

    Example
    -------
    1. Read a field from the HDF5 file.

    >>> import os
    >>> import discretisedfield as df
    ...
    >>> dirname = os.path.join(os.path.dirname(__file__),
    ...                        'tests', 'test_sample')
    >>> filename = os.path.join(dirname, 'hdf5-file.hdf5')
    >>> field = df.Field._fromhdf5(filename)
    >>> field
    Field(...)

    .. seealso:: :py:func:`~discretisedfield.Field._writehdf5`

    """
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
        mesh.load_subregions(f"{strip_extension(filename)}_subregions.json")

    return df.Field(mesh, dim=dim, value=array[:])
