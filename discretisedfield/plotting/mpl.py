"""Matplotlib-based plotting."""
import abc

import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1.inset_locator
import numpy as np

import discretisedfield.util as dfu


class Mpl(metaclass=abc.ABCMeta):
    """Matplotlib-based plotting methods."""

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass  # pragma: no cover

    def _setup_axes(self, ax, figsize, **kwargs):
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, **kwargs)

        return ax

    @abc.abstractmethod
    def _setup_multiplier(self, multiplier):
        pass  # pragma: no cover

    @abc.abstractmethod
    def _axis_labels(self, ax, multiplier):
        pass  # pragma: no cover

    def _savefig(self, filename):
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.02)


def add_colorwheel(ax, width=1, height=1, loc="lower right", **kwargs):
    """Colorwheel for hsv plots.

    Creates colorwheel on new inset axis. See
    ``mpl_toolkits.axes_grid1.inset_locator.inset_axes`` for the meaning of the
    arguments and other possible keyword arguments.

    Example
    -------
    .. plot::
        :context: close-figs

        1. Adding a colorwheel to an empty axis
        >>> import discretisedfield.plotting as dfp
        >>> import matplotlib.pyplot as plt
        ...
        >>> fig, ax = plt.subplots()  # doctest: +SKIP
        >>> ins_ax = dfp.add_colorwheel(ax)  # doctest: +SKIP

    """
    n = 200
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    X, Y = np.meshgrid(x, y)

    theta = np.arctan2(Y, X)
    r = np.sqrt(X**2 + Y**2)

    rgb = dfu.hls2rgb(hue=theta, lightness=r, lightness_clim=[0, 1 / np.sqrt(2)])

    theta = theta.reshape((n, n, 1))

    rgba = np.zeros((n, n, 4))
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            if xi**2 + yi**2 <= 1:
                rgba[i, j, :3] = rgb[i, j, :]
                rgba[i, j, 3] = 1

    ax_ins = mpl_toolkits.axes_grid1.inset_locator.inset_axes(
        ax, width=width, height=height, loc=loc, **kwargs
    )
    ax_ins.imshow(rgba[:, ::-1, :])
    ax_ins.axis("off")
    return ax_ins
