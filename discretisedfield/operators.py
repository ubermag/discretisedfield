import numpy as np
import discretisedfield as df
import discretisedfield.util as dfu


def dot(f1, f2):
    if dfu.compatible(f1, f2) and f1.dim > 1:
        res_array = np.einsum('ijkl,ijkl->ijk', f1.array, f2.array)
        return df.Field(f1.mesh, dim=1, value=res_array[..., np.newaxis])


def cross(f1, f2):
    if dfu.compatible(f1, f2) and f1.dim == 3:
        res_array = np.cross(f1.array, f2.array)
        return df.Field(f1.mesh, dim=3, value=res_array)
