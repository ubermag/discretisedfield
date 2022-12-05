"""Functions to save and load fields.

This module contains functions to save and load ``discretisedfield.Field`` objects.
Generally, their direct use is discouraged. Use :py:func:`discretisedfield.Field.write`
and :py:func:`discretisedfield.Field.fromfile` instead.

"""
import json

import numpy as np

import discretisedfield as df

from .hdf5 import _FieldIOHDF5, _MeshIOHDF5, _RegionIOHDF5
from .ovf import _FieldIOOVF
from .vtk import _FieldIOVTK


class _RegionIO(_RegionIOHDF5):
    class _RegionJSONEncoder(json.JSONEncoder):
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
    pass
