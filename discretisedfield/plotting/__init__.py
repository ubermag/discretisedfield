"""Matplotlib based plotting."""
from discretisedfield.plotting.hv import Hv
from discretisedfield.plotting.k3d_field import K3dField
from discretisedfield.plotting.k3d_region import K3dRegion
from discretisedfield.plotting.mpl import add_colorwheel
from discretisedfield.plotting.mpl_field import MplField
from discretisedfield.plotting.mpl_region import MplRegion


class _Defaults:
    """Default settings for plotting."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset values to their defaults."""
        self.norm_filter = True

    def __repr__(self):
        summary = "plotting defaults\n"
        summary += f"  norm_filter: {self.norm_filter}"
        return summary

    @property
    def norm_filter(self):
        """Apply automatic norm-based filtering in convenience plotting methods."""
        return self._norm_filter

    @norm_filter.setter
    def norm_filter(self, norm_filter):
        self._norm_filter = norm_filter
        Hv._norm_filter = norm_filter


"""Default settings for plotting."""
defaults = _Defaults()
