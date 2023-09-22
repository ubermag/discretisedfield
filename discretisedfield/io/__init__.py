"""Functions to save and load fields.

This module contains functions to save and load ``discretisedfield.Field`` objects.
Generally, their direct use is discouraged. Use :py:func:`discretisedfield.Field.write`
and :py:func:`discretisedfield.Field.fromfile` instead.

"""
import json
import pathlib

import numpy as np

import discretisedfield as df

from .hdf5 import _FieldIO_HDF5, _MeshIO_HDF5, _RegionIO_HDF5
from .ovf import _FieldIO_OVF
from .vtk import _FieldIO_VTK


class _RegionIO(_RegionIO_HDF5):
    __slots__ = []

    class _JSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, df.Region):
                return o.to_dict()
            elif isinstance(o, np.ndarray):
                return tuple(o)
            elif isinstance(o, (np.int32, np.int64)):
                return int(o)
            elif isinstance(o, (np.float32, np.float64)):
                return float(o)
            else:
                super().default(o)


class _MeshIO(_MeshIO_HDF5):
    __slots__ = []

    def save_subregions(self, field_filename):
        """Save subregions to json file."""
        with pathlib.Path(self._subregion_filename(field_filename)).open(
            mode="wt", encoding="utf-8"
        ) as f:
            json.dump(self.subregions, f, cls=df.Region._JSONEncoder)

    def load_subregions(self, field_filename):
        """Load subregions from json file."""
        with pathlib.Path(self._subregion_filename(field_filename)).open(
            mode="rt", encoding="utf-8"
        ) as f:
            subregions = json.load(f)
        self.subregions = {key: df.Region(**val) for key, val in subregions.items()}

    @staticmethod
    def _subregion_filename(filename):
        return f"{str(filename)}.subregions.json"


class _FieldIO(_FieldIO_HDF5, _FieldIO_OVF, _FieldIO_VTK):
    __slots__ = []

    def to_file(
        self, filename, representation="bin8", extend_scalar=False, save_subregions=True
    ):
        """Write the field to OVF, HDF5, or VTK file.

        For ``.ovf``, ``.omf``, or ``.ohf`` extensions the field is saved in an OVF 2.0
        file. Possible values for `representation` of the data are ``'bin4'``,
        ``'bin8'``, or ``'txt'``. If ``extend_scalar=True``, a scalar field will be
        saved as a vector field. More precisely, if the value at a cell is X, that cell
        will be saved as (X, 0, 0). Subregions are automatically saved in a separate
        json file for ``save_subregions=True``.

        If the extension of `filename` is ``.vtk``, a VTK file is written. Possible
        values for `representation` are ``'bin'`` (``'bin8'`` as an equivalent for
        convenience), ``'txt'``, or ``'xml'``. The data is saved as a
        ``RECTILINEAR_GRID``. A scalar field (``nvdim=1``) is saved as ``SCALARS``. A
        vector field (``nvdim>=1``) is saved as both ``VECTORS`` as well as ``SCALARS``
        for all the components to enable easier colouring of vectors in some
        visualisation packages. The data is stored as ``CELL_DATA``. Subregions are
        automatically saved in a separate json file for ``save_subregions=True``.
        `extend_scalar` has no effect for VTK files. The saved VTK file can be opened
        with `Paraview <https://www.paraview.org/>`_ or `Mayavi
        <https://docs.enthought.com/mayavi/mayavi/>`_. To show contour lines in Paraview
        one has to first convert Cell Data to Point Data using a filter.

        If the extension of `filename` is ``.hdf5`` or ``.h5`` an HDF5 file will be
        written. The parameters `representation`, `extend_scalar` and `save_subregions`
        have no effect for HDF5 files and are silently ignored. Subregions are stored
        inside the HDF5 file, if any are defined for the field.

        Parameters
        ----------
        filename : str

            Name of the file written. The suffix determines the file type.

        representation : str, optional

            Only supported for OVF and VTK files. In the case of OVF files (``.ovf``,
            ``.omf``, or ``.ohf``) the representation can be ``'bin4'``, ``'bin8'``, or
            ``'txt'``. For VTK files (``.vtk``) the representation can be ``bin``
            (``bin8``), ``xml``, or ``txt``. Defaults to ``'bin8'`` (interpreted as
            ``bin`` for VTK files).

        extend_scalar : bool, optional

            If ``True``, a scalar field will be saved as a vector field. More precisely,
            if the value at a cell is 3, that cell will be saved as (3, 0, 0). This is
            valid only for the OVF file formats. Defaults to ``False``.

        save_subregions : bool, optional

            If ``True`` and subregions are defined for the mesh the subregions will be
            saved to a json file. Defaults to ``True``. This has no effect for HDF5
            files which always contain subregions in the file.

        See also
        --------
        ~discretisedfield.Field.from_file

        Example
        -------
        1. Write field to an OVF file.

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

        2. Write field to a VTK file.

        >>> filename = 'mytestfile.vtk'
        >>> field.to_file(filename)  # write the file
        >>> os.path.isfile(filename)
        True
        >>> field_read = df.Field.from_file(filename)  # read the file
        >>> field_read == field
        True
        >>> os.remove(filename)  # delete the file

        3. Write field to an HDF5 file.

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
        elif filename.suffix == ".vtk":
            self._to_vtk(
                filename,
                representation=representation,
                save_subregions=save_subregions,
            )
        elif filename.suffix in [".hdf5", ".h5"]:
            self._to_hdf5(filename)
        else:
            raise ValueError(
                f"Writing file with extension {filename.suffix} not supported."
            )

    @classmethod
    def fromfile(cls, filename):
        raise AttributeError("This method has been renamed to 'from_file'.")

    @classmethod
    def from_file(cls, filename):
        """Read a field from an OVF (1.0 or 2.0), VTK, or HDF5 file.

        The extension of `filename` can be:
            - ``.ovf``, ``.omf``, ``.ohf`` or ``.oef`` for OVF files,
            - ``.vtk`` for VTK files, or
            - ``.hdf5`` or ``.h5`` for HDF5 files.

        This method automatically determines the file type based on the extension of
        `filename`.

        For OVF the data representation (``txt``, ``bin4``, or ``bin8``) as well as the
        OVF version (OVF1.0 or OVF2.0) are extracted from the file itself. Mesh
        subregions are loaded from a separate json file if it exists.

        For VTK files this method reads the field from a VTK file containing a
        ``RECTILINEAR_GRID`` written by ``discretisedfield.to_file``. It expects the
        data do be specified as cell data and one (vector) field with the name
        ``field``. A vector field should also contain data for the individual
        components. The individual component names are used as ``vdims`` for the new
        field. They must appear in the form ``<componentname>-component``. Older
        versions of discretisedfield did write the data as point data instead of cell
        data. This function can load new and old files and automatically extract the
        correct data without additional user input. Mesh subregions are loaded from a
        separate json file if it exists.

        For HDF5 files written with discretisedfield version 0.90.0 or newer all data is
        contained in the file. No separate json file for subregions is read. Older
        versions of discretisedfield did not save all attributes (e.g. no
        subregions). Reading old files is automatically handled internally. For old HDF5
        files subregions can be read from disk if they were saved in a separate json
        file.

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
        ...                        '..', 'tests', 'test_sample')
        >>> filename = os.path.join(dirname, 'oommf-ovf2-bin4.omf')
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        2. Read a field from a VTK file.

        >>> filename = os.path.join(dirname, 'vtk-file.vtk')
        >>> field = df.Field.from_file(filename)
        >>> field
        Field(...)

        3. Read a field from an HDF5 file.

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
