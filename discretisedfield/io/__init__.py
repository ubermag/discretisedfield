"""Functions to save and load fields.

This module contains functions to save and load ``discretisedfield.Field`` objects.
Generally, their direct use is discouraged. Use :py:func:`discretisedfield.Field.write`
and :py:func:`discretisedfield.Field.fromfile` instead.

"""
from .hdf5 import field_from_hdf5, field_to_hdf5
from .ovf import field_from_ovf, field_to_ovf
from .util import _RegionJSONEncoder
from .vtk import field_from_vtk, field_to_vtk
