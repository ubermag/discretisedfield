"""Functions to save and load fields.

This module contains functions to save and load ``discretisedfield.Field`` objects.
Generally, their direct use is discouraged. Use :py:func:`discretisedfield.Field.write`
and :py:func:`discretisedfield.Field.fromfile` instead.

"""
import json
import pathlib

import numpy as np

import discretisedfield as df

from .hdf5 import _FieldIOHDF5, _MeshIOHDF5, _RegionIOHDF5
from .ovf import _FieldIOOVF
from .vtk import _FieldIOVTK


class _RegionIO(_RegionIOHDF5):
    class _JSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, df.Region):
                return o.to_dict()
            elif isinstance(o, np.ndarray):
                return tuple(o)
            elif isinstance(o, np.int64):
                return int(o)
            elif isinstance(o, np.float64):
                return float(o)
            else:
                super().default(o)


class _MeshIO(_MeshIOHDF5):
    pass


class _FieldIO(_FieldIOHDF5, _FieldIOOVF, _FieldIOVTK):
    def write(self, *args, **kwargs):
        raise AttributeError("This method has been renamed to 'to_file'.")

    def to_file(
        self, filename, representation="bin8", extend_scalar=False, save_subregions=True
    ):
        """Write the field to OVF, HDF5, or VTK file.

        If the extension of ``filename`` is ``.vtk``, a VTK file is written. The
        representation of the data (``'bin'`` [``'bin8'`` as equivalent], ``'txt'``, or
        ``'xml'``) is passed as ``'representation'``.
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


        For ``.ovf``, ``.omf``, or ``.ohf`` extensions, the field is saved to OVF file.
        In that case, the representation of data (``'bin4'``, ``'bin8'``, or ``'txt'``)
        is passed as ``representation`` and if ``extend_scalar=True``, a scalar field
        will be saved as a vector field. More precisely, if the value at a cell is X,
        that cell will be saved as (X, 0, 0).

        Finally, if the extension of ``filename`` is ``.hdf5``, HDF5 file will be
        written. The format of the HDF5 files was changed in discretisedfield version
        0.70.0.

        Parameters
        ----------
        filename : str

            Name of the file written.

        representation : str, optional

            Only supported for OVF and VTK files. In the case of OVF files
            (``.ovf``, ``.omf``, or ``.ohf``) the representation can be
            ``'bin4'``, ``'bin8'``, or ``'txt'``. For VTK files (``.vtk``) the
            representation can be ``bin``, ``xml``, or ``txt``. Defaults to
            ``'bin8'`` (interpreted as ``bin`` for VTK files).

        extend_scalar : bool, optional

            If ``True``, a scalar field will be saved as a vector field. More
            precisely, if the value at a cell is 3, that cell will be saved as
            (3, 0, 0). This is valid only for the OVF file formats. Defaults to
            ``False``.

        save_subregions : bool, optional

            If ``True`` and subregions are defined for the mesh the subregions will be
            saved to a json file. Defaults to ``True``. This has no effect for hdf5
            files which always contain subregions in the file.

        See also
        --------
        ~discretisedfield.Field.from_file

        Example
        -------
        1. Write field to the OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, -5e-9)
        >>> p2 = (5e-9, 15e-9, 15e-9)
        >>> n = (5, 15, 20)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=3, value=(5, 6, 7))
        ...
        >>> filename = 'mytestfile.omf'
        >>> field.to_file(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.from_file(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        2. Write field to the VTK file.

        >>> filename = 'mytestfile.vtk'
        >>> field.to_file(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> os.remove(filename)  # delete the file

        3. Write field to the HDF5 file.

        >>> filename = 'mytestfile.hdf5'
        >>> field.to_file(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.from_file(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        """
        filename = pathlib.Path(filename)
        if filename.suffix in [".omf", ".ovf", ".ohf"]:
            self._to_ovf(
                filename,
                representation=representation,
                extend_scalar=extend_scalar,
                save_subregions=save_subregions,
            )
        elif filename.suffix in [".hdf5", ".h5"]:
            self._to_hdf5(filename)
        elif filename.suffix == ".vtk":
            self._to_vtk(
                filename,
                representation=representation,
                save_subregions=save_subregions,
            )
        else:
            raise ValueError(
                f"Writing file with extension {filename.suffix} not supported."
            )

    @classmethod
    def fromfile(cls, filename):
        raise AttributeError("This method has been renamed to 'from_file'.")

    @classmethod
    def from_file(cls, filename):
        """Read the field from an OVF (1.0 or 2.0), VTK, or HDF5 file.

        The extension of the ``filename`` should correspond to either:
            - OVF (``.ovf``, ``.omf``, ``.ohf``, ``.oef``)
            - VTK (``.vtk``), or
            - HDF5 (``.hdf5`` or ``.h5``).

        This method automatically determines the file type based on the file name
        extension.

        For OVF the data representation (``txt``, ``bin4``, or ``bin8``) as well as the
        OVF version (OVF1.0 or OVF2.0) are extracted from the file itself.

        For VTK files this method reads the field from a VTK file defined on RECTILINEAR
        GRID written by ``discretisedfield.to_file``. It expects the data do be
        specified as cell data and one (vector) field with the name ``field``. A vector
        field should also contain data for the individual components. The individual
        component names are used as ``vdims`` for the new field. They must appear in the
        form ``<componentname>-component``. Older versions of discretisedfield did write
        the data as point data instead of cell data. This function can load new and old
        files and automatically extracts the correct data without additional user input.

        The file format for hdf5 files was changed in discretisedfield version 0.70.0.
        Older versions of discretisedfield did only did not save all attributes (e.g. no
        subregions). Reading old files is automatically handled internally. Subregions
        can only be read from disk if they were saved in a separate json file.

        Parameters
        ----------
        filename : str

            Name of the file to be read.

        Returns
        -------
        discretisedfield.Field

            Field read from the file.

        See also
        --------
        ~discretisedfield.Field.to_file

        Example
        -------
        1. Read the field from an OVF file.

        >>> import os
        >>> import discretisedfield as df
        ...
        >>> dirname = os.path.join(os.path.dirname(__file__),
        ...                        'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'oommf-ovf2-bin4.omf')
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        2. Read a field from the VTK file.

        >>> filename = os.path.join(dirname, 'vtk-file.vtk')
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        3. Read a field from the HDF5 file.

        >>> filename = os.path.join(dirname, 'hdf5-file.hdf5')
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        """
        filename = pathlib.Path(filename)
        if filename.suffix in [".omf", ".ovf", ".ohf", ".oef"]:
            return cls._from_ovf(filename)
        elif filename.suffix == ".vtk":
            return cls._from_vtk(filename)
        elif filename.suffix in [".hdf5", ".h5"]:
            return cls._from_hdf5(filename)
        else:
            raise ValueError(
                f"Reading file with extension {filename.suffix} not supported."
            )
