import numpy as np
import discretisedfield as df
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.transform import Rotation
import functools


# TODO
# - implement special methods that bypass __getattr__ (and __getattribute__)
class RotatedField(df.Field):
    r"""Rotated field class.
    """
    def __init__(self, mesh, dim, value=0, norm=None):
        if mesh.bc != '':
            raise RuntimeError('Rotations are not supported for fields with'
                               'periodic boundary conditions')
        super().__init__(mesh=mesh, dim=dim, value=value, norm=norm)
        # TODO can this cause overriding of a previous rotation matrix?
        self._rotation = None

    # TODO rotate affects the current field and creates a new one
    # which behaviour do we want?
    def rotate(self, method, n=None, **kwargs):
        self.n = n

        if method in ['from_quat', 'from_matrix', 'from_rotvec', 'from_mpr',
                      'from_euler']:
            rotation = getattr(Rotation, method)(**kwargs)
        elif method == 'align_vector':
            from_ = kwargs['from']
            to = kwargs['to']
            fixed = np.cross(from_, to)
            rotation = Rotation.align_vectors([to, fixed], [from_, fixed])[0]
        else:
            msg = f'Method {method} is unknown.'
            raise ValueError(msg)

        if self._rotation is not None:
            self._rotation = rotation * self._rotation
        else:
            self._rotation = rotation
        if hasattr(self, '_rotated_field'):
            del self._rotated_field
        return self

    # TODO
    # - add all df.Field methods/properties
    # - decide where to first rotate the field
    def __getattribute__(self, name):
        if name in ['plane', 'integral', 'mpl', 'div']:
            return self._rotated_field.__getattribute__(name)
        else:
            return object.__getattribute__(self, name)

    # Test implementation
    # TODO
    # - should the result be a RotatedField?
    # - compute addition on rotated fields?
    # - what can be summed, e.g. Field + RotatedField with the later
    #   rotated to have a matching mesh?
    def __add__(self, other):
        if isinstance(other, RotatedField):
            return self._rotated_field + other._rotated_field
        else:
            return self._rotated_field + other

    @functools.cached_property
    def _rotated_field(self):
        # Calculate new region
        new_region = self._calculate_new_region()

        if self.n is None:
            new_n = self._calculate_new_n(new_region)
        else:
            new_n = self.n

        # Create new mesh
        new_mesh = df.Mesh(region=new_region, n=new_n)

        # Rotate Field vectors
        rot_field = self._rotation.apply(
            self.array.reshape((-1, self.dim))).reshape((*self.mesh.n,
                                                        self.dim))

        # Calculate field at new mesh positions
        new_m = self._map_and_interpolate(new_mesh, rot_field)

        # Construct new field
        return df.Field(mesh=new_mesh, dim=self.dim, value=new_m)

    def _map_and_interpolate(self, new_mesh, rot_field):
        new_mesh_field = df.Field(mesh=new_mesh, dim=3, value=lambda x: x)
        new_mesh_pos = (new_mesh_field.array.reshape((-1, 3))
                        - self.mesh.region.centre)

        new_pos_old_mesh = self._rotation.inv().apply(new_mesh_pos)

        # Get values of field at new mesh locations
        result = np.ndarray(shape=new_mesh_field.array.shape)
        for i in range(self.dim):
            result[..., i] = self._create_interpolation_funcs(rot_field[..., i])(
                new_pos_old_mesh).reshape(new_mesh.n)
        return result

    def _create_interpolation_funcs(self, rot_field_component):
        pmin = np.array(self.mesh.region.pmin)
        pmax = np.array(self.mesh.region.pmax)
        cell = np.array(self.mesh.cell)

        coords = []
        tol = 1e-9  # to avoid numerical errors at the sample boundaries
        for i in range(3):
            coords.append(np.array([pmin[i] - cell[i] * tol,
                                    *np.linspace(pmin[i] + cell[i] / 2,
                                                 pmax[i] - cell[i] / 2,
                                                 self.mesh.n[i]),
                                    # *rot_field.mesh.coordinates,
                                    pmax[i] + cell[i] * tol])
                          - self.mesh.region.centre[i])

        m = np.pad(rot_field_component,
                   pad_width=[(1, 1), (1, 1), (1, 1)], mode='edge')

        return RegularGridInterpolator(coords, m, fill_value=0,
                                       bounds_error=False)

    def _calculate_new_n(self, new_region):
        cell_edges = np.eye(3) * self.mesh.cell
        rotated_cell_edges = abs(self._rotation.apply(cell_edges))
        rotated_edge_lenths = np.sum(rotated_cell_edges, axis=0)

        new_vol = np.prod(rotated_edge_lenths)
        adjust = (self.mesh.dV / new_vol)**(1/3)
        return np.round(np.divide(new_region.edges,
                                  rotated_edge_lenths
                                  * adjust)).astype(int).tolist()

    def _calculate_new_region(self):
        edges = np.eye(3) * self.mesh.region.edges
        rotated_edges = abs(self._rotation.apply(edges))
        edge_centre_length = np.sum(rotated_edges, axis=0) / 2
        return df.Region(p1=self.mesh.region.centre - edge_centre_length,
                         p2=self.mesh.region.centre + edge_centre_length)
