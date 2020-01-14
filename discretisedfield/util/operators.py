import numpy as np
import discretisedfield as df


def compatible(field1, field2):
    """Check if a binary operator can be applied to two fields.

    A binary operator (`+`, `-`, `*`, `/`) can be applied between two
    fields (`field1` and `field2`) if both fields are:

    1. defined on the same mesh and

    2. have the same dimension.

    """
    if not isinstance(field1, df.Field) or \
       not isinstance(field2, df.Field):
        msg = ('Binary operator can be applied only on '
               'discretisedfield.Field objects.')
        raise TypeError(msg)
    elif field1.mesh != field2.mesh:
        msg = 'Fields must be defined on same meshes.'
        raise ValueError(msg)
    elif field1.dim != field2.dim:
        msg = 'Fields must have the same dimension.'
        raise ValueError(msg)
    else:
        # A binary operator can be applied between two fields.
        return True


def add(field1, field2):
    if compatible(field1, field2):
        res = df.Field(field1.mesh, dim=field1.dim)
        res.value = field1.array + field2.array
        return res


def multiply(field1, field2):
    if isinstance(field1, df.Field) and isinstance(field2, (int, float)):
        res = df.Field(field1.mesh, field1.dim)
        res.value = field1.array * field2
        return res
    else:
        if compatible(field1, field2):
            res = df.Field(field1.mesh, dim=field1.dim)
            res.value = np.multiply(field1.array, field2.array)
            return res
