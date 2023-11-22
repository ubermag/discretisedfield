from .field import Field


class VertexField(Field):
    def __call__(self, point):
        raise NotImplementedError

    def diff(self, direction, order=1, restrict2valid=True):
        """Maybe this is slighly wrong and we should ask Claas about this."""
        super().diff(direction, order=order, restrict2valid=restrict2valid)

    def integrate(self, direction=None, cumulative=False):
        """Maybe this is slighly wrong and we should ask Claas about this."""
        super().integrate(direction=direction, cumulative=cumulative)

    def line(self, p1, p2, n):
        pass  # @Martin

    def __getitem__(self, item):
        raise NotImplementedError

    def mpl(self):
        pass  # @Swapneel

    def hv(self):
        pass  # @Swapneel

    # NOTE: We are ignoring all the FFTs for now.

    def to_xarray(self, name="field", unit=None):
        pass  # @Swapneel

    @classmethod
    def from_xarray(cls, xa):
        raise NotImplementedError


# TODO: reimplement the _as_array functions. @Swapneel
