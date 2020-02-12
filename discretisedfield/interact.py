import ipywidgets


def interact(**kwargs):
    """Decorator for interactive plotting.

    This is a wrapper around ``ipywidgets.interact``. For details, please refer
    to ``interact`` function in ``ipywidgets`` package.

    Example
    -------
    1. Interactive plotting.

    >>> import discretisedfield as df
    ...
    >>> p1 = (-50e-9, -50e-9, -50e-9)
    >>> p2 = (50e-9, 50e-9, 50e-9)
    >>> n = (10, 10, 10)
    >>> mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
    >>> field = df.Field(mesh, dim=3, value=(1, 2, 0))
    >>> @df.interact(x=field.mesh.slider('x'))
    ... def myplot(x):
    ...     field.plane(x=x).mpl()
    interactive(...)

    """
    return ipywidgets.interact(**kwargs)
