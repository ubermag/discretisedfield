def integrate(field, direction=None, cumulative=False):
    """Integral.

    This function calls ``integral`` method of the ``discrteisedfield.Field``
    object.

    For details, please refer to :py:func:`~discretisedfield.Field.integral`

    """
    return field.integrate(direction=direction, cumulative=cumulative)
