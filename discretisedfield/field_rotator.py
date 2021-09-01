import warnings
import numpy as np
import discretisedfield as df
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation


# TODO support for scalar fields
class FieldRotator:
    r"""Rotate a field.

    Parameters
    ----------
    field : discretisedfield.Field
        Field to rotate.

    Examples
    --------
    >>> import discretisedfield as df
    >>> from math import pi

    Create a ``field`` to rotate.

    >>> region = df.Region(p1=(0, 0, 0), p2=(20, 10, 2))
    >>> mesh = df.Mesh(region=region, cell=(1, 1, 1))
    >>> field = df.Field(mesh, dim=3, value=(0, 0, 1))
    >>> field.mesh.n
    (20, 10, 2)

    Create a ``FieldRotator`` object for the ``field``.

    >>> field_rotator = df.FieldRotator(field)

    Rotate the ``field``.

    >>> field_rotator.rotate(method='from_euler', seq='x', angles=pi/2)

    Access the rotated field.

    >>> field_rotator.field
    Field(...)
    >>> field_rotator.field.mesh.n
    (20, 2, 10)

    Apply a second rotation.

    >>> field_rotator.rotate(method='from_euler', seq='z', angles=pi/2)
    >>> field_rotator.field.mesh.n
    (2, 20, 10)

    """

    def __init__(self, field):
        if field.mesh.bc != '':
            warnings.warn('Boundary conditions are lost when rotating '
                          'the field.')
        if field.dim not in [1, 3]:
            raise ValueError('Rotations are not supported for fields with'
                             f'{field.dim=}.')
        self._orig_field = field
        # set up state without rotations
        self.clear_rotation()

    @property
    def field(self):
        """Rotated field."""
        return self._rotated_field

    def rotate(self, method, *args, n=None, **kwargs):
        """Rotate the field.

        TODO explain the rotation process and remeshing/interpolation to make
        ``n`` understandable.

        Rotates the field using the given ``method``. The definition of the
        rotation is based on ``scipy.spatial.transform.Rotation``. Additional
        parameters required for the different possible rotation methods must be
        given as keyword arguments. These are passed directly to the relevant
        ``scipy`` function. For a detailed explanation and required argements
        of the different methods please refer direcly to the ``scipy``
        documentation.

        The only method that differs from ``scipy`` is ``align_vector``. This
        method expects two keyword arguments ``initial`` and ``final``
        (array-like, 3). This method rotates the vector ``initial`` to the
        vector ``final``, the cross product is kept fixed, i.e. the rotation
        axis is the normal vector of the plane defined by the two vectors
        ``initial`` and ``final``.

        Parameters
        ----------
        method : str
            Rotation method. One of ``'from_quat'``, ``'from_matrix'``,
            ``'from_rotvec'``, ``'from_mpr'``, ``'from_euler'``, or
            ``'align_vector'``.

        args
            Additional positional arguments for the rotation method.

        n : array-like, 3, optional
            Number of cells in the new mesh.

        kwargs
            Additional keyword arguments for the rotation method.

        Examples
        --------
        TODO

        """
        # create rotation object
        if method in ['from_quat', 'from_matrix', 'from_rotvec', 'from_mrp',
                      'from_euler']:
            rotation = getattr(Rotation, method)(*args, **kwargs)
        elif method == 'align_vector':
            initial = kwargs['initial']
            final = kwargs['final']
            fixed = np.cross(initial, final)
            rotation = Rotation.align_vectors([final, fixed],
                                              [initial, fixed])[0]
        else:
            msg = f'Method {method} is unknown.'
            raise ValueError(msg)

        self._rotation = rotation * self._rotation

        # Calculate new region
        new_region = self._calculate_new_region()

        if n is None:
            n = self._calculate_new_n(new_region)

        # Create new mesh
        new_mesh = df.Mesh(region=new_region, n=n)

        # Rotate Field vectors
        if self._orig_field.dim == 1:
            rot_field = self._orig_field.array
        elif self._orig_field.dim == 3:
            rot_field = self._rotation.apply(
                self._orig_field.array.reshape(
                    (-1, self._orig_field.dim))).reshape(
                        (*self._orig_field.mesh.n, self._orig_field.dim))

        # Calculate field at new mesh positions
        new_m = self._map_and_interpolate(new_mesh, rot_field)

        # Construct new field
        self._rotated_field = df.Field(mesh=new_mesh,
                                       dim=self._orig_field.dim,
                                       value=new_m)

    def clear_rotation(self):
        """Remove all rotations."""
        self._rotation = Rotation.from_matrix(np.eye(3))
        self._rotated_field = self._orig_field

    def __repr__(self):
        return (f'FieldRotator(\n* original field:\n{self.field}\n'
                f'* internal rotation matrix:\n{self._rotation.as_matrix()}\n)'
                )

    def _map_and_interpolate(self, new_mesh, rot_field):
        new_mesh_field = df.Field(mesh=new_mesh, dim=3, value=lambda x: x)
        new_mesh_pos = (new_mesh_field.array.reshape((-1, 3))
                        - self._orig_field.mesh.region.centre)

        new_pos_old_mesh = self._rotation.inv().apply(new_mesh_pos)

        # Get values of field at new mesh locations
        result = np.ndarray(shape=[*new_mesh_field.mesh.n,
                                   self._orig_field.dim])
        for i in range(self._orig_field.dim):
            result[..., i] = self._create_interpolation_funcs(
                rot_field[..., i])(new_pos_old_mesh).reshape(new_mesh.n)
        return result

    def _create_interpolation_funcs(self, rot_field_component):
        pmin = np.array(self._orig_field.mesh.region.pmin)
        pmax = np.array(self._orig_field.mesh.region.pmax)
        cell = np.array(self._orig_field.mesh.cell)

        coords = []
        tol = 1e-9  # to avoid numerical errors at the sample boundaries
        for i in range(3):
            coords.append(np.array([pmin[i] - cell[i] * tol,
                                    *np.linspace(pmin[i] + cell[i] / 2,
                                                 pmax[i] - cell[i] / 2,
                                                 self._orig_field.mesh.n[i]),
                                    # *rot_field.mesh.coordinates,
                                    pmax[i] + cell[i] * tol])
                          - self._orig_field.mesh.region.centre[i])

        m = np.pad(rot_field_component,
                   pad_width=[(1, 1), (1, 1), (1, 1)], mode='edge')

        return RegularGridInterpolator(coords, m, fill_value=0,
                                       bounds_error=False)

    def _calculate_new_n(self, new_region):
        cell_edges = np.eye(3) * self._orig_field.mesh.cell
        rotated_cell_edges = abs(self._rotation.apply(cell_edges))
        rotated_edge_lenths = np.sum(rotated_cell_edges, axis=0)

        new_vol = np.prod(rotated_edge_lenths)
        adjust = (self._orig_field.mesh.dV / new_vol)**(1/3)
        return np.round(np.divide(new_region.edges,
                                  rotated_edge_lenths
                                  * adjust)).astype(int).tolist()

    def _calculate_new_region(self):
        edges = np.eye(3) * self._orig_field.mesh.region.edges
        rotated_edges = abs(self._rotation.apply(edges))
        edge_centre_length = np.sum(rotated_edges, axis=0) / 2
        return df.Region(p1=(self._orig_field.mesh.region.centre
                             - edge_centre_length),
                         p2=(self._orig_field.mesh.region.centre
                             + edge_centre_length))
