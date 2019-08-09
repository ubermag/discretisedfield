import numpy as np
import ubermagutil.typesystem as ts
import discretisedfield.util as dfu


@ts.typesystem(pmin=ts.Vector(size=3, const=True),
               pmax=ts.Vector(size=3, const=True))
class Region:
    def __init__(self, p1, p2):
        self.pmin = dfu.array2tuple(np.minimum(p1, p2))
        self.pmax = dfu.array2tuple(np.maximum(p1, p2))

        # Is any edge length of the region equal to zero?
        if np.equal(np.abs(np.subtract(self.pmax, self.pmin)), 0).any():
            msg = 'The length of one of the region edges is zero.'
            raise ValueError(msg)

    def __repr__(self):
        """Region representation.

        This method returns the string that can be copied in another
        Python script so that exactly the same region object could be
        defined.

        Returns
        -------
        str
           Region representation string.

        Example
        -------
        1. Getting mesh representation string.x

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> repr(region)
        "Region(p1=(0, 0, 0), p2=(2, 2, 1), cell=(1, 1, 1), pbc={'x'}, name='m')"

        """
        return f'Region(p1={self.pmin}, p2={self.pmax})'
    
    def __contains__(self, item):
        """Determine whether `point` is inside the region. If it is, it returns
        `True`, otherwise `False`.

        Parameters
        ----------
        item : (3,) array_like
            The mesh point coordinate :math:`(p_{x}, p_{y}, p_{z})`.

        Returns
        -------
        True
            If `item` is inside the region.
        False
            If `item` is outside the region.

        Example
        -------
        1. Check whether point is inside the region.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (2, 2, 1)
        >>> region = df.Region(p1=p1, p2=p2)
        >>> point1 = (1, 1, 1)
        >>> point1 in region
        True
        >>> point2 = (1, 3, 1)
        >>> point2 in region
        False

        """
        if np.logical_or(np.less(item, self.pmin),
                         np.greater(item, self.pmax)).any():
            return False
        else:
            return True
