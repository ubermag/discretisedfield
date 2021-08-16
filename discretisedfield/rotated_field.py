import numpy as np
import discretisedfield as df
from scipy.interpolate import RegularGridInterpolator
import functools


# TODO
# - derive from df.Field
#   Problem: calls to self.__class__ in df.Field (different __init__ arguments)
# - implement special methods that bypass __getattr__ (and __getattribute__)
class RotatedField:
    r"""Rotated field class.
    """
    def __init__(self, field, method, **kwargs):
        self._field = field
        # super().__init__(field.mesh, field.dim, field.value)
        rotation_matrix = self._compute_rotation_matrix(method, **kwargs)
        if isinstance(field, RotatedField):
            self._rotation_matrix = np.matmul(rotation_matrix,
                                              field._rotation_matrix)
        else:
            self._rotation_matrix = rotation_matrix

        self._inv_rotation_matrix = np.linalg.inv(self._rotation_matrix)

    # TODO
    # - add all df.Field methods/properties
    # - decide where to first rotate the field
    def __getattr__(self, name):
        if name in ['plane', 'integral', 'mpl', 'mesh', 'div']:
            return getattr(self._rotated_field(), name)
        elif name in ['pad']:
            return getattr(self._field, name)
        else:
            raise ValueError(f'No attribute {name}.')

    # TODO this should be a property, i.e. `n` must be passed differently
    @functools.lru_cache
    def _rotated_field(self, n=None):
        # Calculate new region
        new_region = self._calculate_new_region()

        if n is None:
            new_n = self._calculate_new_n(new_region)
        else:
            new_n = n
        # Create new mesh
        new_mesh = df.Mesh(region=new_region, n=new_n)

        # Rotate Field vectors
        rot_field = self._rotate_field_components()

        # Create interpolation
        interp_funcs = self._create_interpolation_funcs(rot_field)

        # Calculate field at new mesh positions
        new_m = self._map_and_interpolate(new_mesh, interp_funcs)

        # Construct new field
        return df.Field(mesh=new_mesh, dim=self._field.dim, value=new_m)

    def _map_and_interpolate(self, new_mesh, interp_funcs):
        new_mesh_field = df.Field(mesh=new_mesh, dim=3, value=lambda x: x)
        nmf0 = (new_mesh_field.array[..., 0].flatten()[..., None]
                - self._field.mesh.region.centre[0])
        nmf1 = (new_mesh_field.array[..., 1].flatten()[..., None]
                - self._field.mesh.region.centre[1])
        nmf2 = (new_mesh_field.array[..., 2].flatten()[..., None]
                - self._field.mesh.region.centre[2])
        new_mesh_pos = np.concatenate([nmf0, nmf1, nmf2], axis=1)

        # Map new mesh onto the old mesh
        new_pos_old_mesh = []
        for mp in new_mesh_pos:
            new_pos_old_mesh.append(self._inv_rotate_vector(mp))
        new_pos_old_mesh = np.array(new_pos_old_mesh)

        # Get values of field at new mesh locations
        interp_func_mx = interp_funcs[0]
        interp_func_my = interp_funcs[1]
        interp_func_mz = interp_funcs[2]
        new_mx = interp_func_mx(new_pos_old_mesh).reshape(new_mesh.n)
        new_my = interp_func_my(new_pos_old_mesh).reshape(new_mesh.n)
        new_mz = interp_func_mz(new_pos_old_mesh).reshape(new_mesh.n)
        return np.stack([new_mx, new_my, new_mz], axis=3)

    def _rotate_field_components(self):
        pre_field = np.concatenate([self._field.x.array.reshape((-1, 1)),
                                    self._field.y.array.reshape((-1, 1)),
                                    self._field.z.array.reshape((-1, 1))],
                                   axis=1)
        rot_field = []
        for v in pre_field:
            rot_field.append(self._inv_rotate_vector(v))
        return np.array(rot_field).reshape(self._field.mesh.n +
                                           (self._field.dim,))

    def _create_interpolation_funcs(self, rot_field):
        mesh_field = df.Field(mesh=self._field.mesh, dim=3, value=lambda x: x)
        x = mesh_field.array[:, 0, 0, 0] - self._field.mesh.region.centre[0]
        y = mesh_field.array[0, :, 0, 1] - self._field.mesh.region.centre[1]
        z = mesh_field.array[0, 0, :, 2] - self._field.mesh.region.centre[2]
        mx = np.pad(rot_field[..., 0],
                    pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')
        my = np.pad(rot_field[..., 1],
                    pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')
        mz = np.pad(rot_field[..., 2],
                    pad_width=((1, 1), (1, 1), (1, 1)), mode='edge')

        tol = 1e-5
        x = np.array([x[0] - self._field.mesh.dx/2*(1+tol), *x,
                      x[-1] + self._field.mesh.dx/2*(1+tol)])
        y = np.array([y[0] - self._field.mesh.dy/2*(1+tol), *y,
                      y[-1] + self._field.mesh.dy/2*(1+tol)])
        z = np.array([z[0] - self._field.mesh.dz/2*(1+tol), *z,
                      z[-1] + self._field.mesh.dz/2*(1+tol)])
        interp_func_mx = RegularGridInterpolator((x, y, z), mx,
                                                 fill_value=0.0,
                                                 bounds_error=False)
        interp_func_my = RegularGridInterpolator((x, y, z), my,
                                                 fill_value=0.0,
                                                 bounds_error=False)
        interp_func_mz = RegularGridInterpolator((x, y, z), mz,
                                                 fill_value=0.0,
                                                 bounds_error=False)
        return (interp_func_mx, interp_func_my, interp_func_mz)

    def _rotate_vector(self, v):
        r""" Rotate vector based on matrix.
        """
        return np.matmul(self._rotation_matrix, v)

    def _inv_rotate_vector(self, v):
        r""" Rotate vector based on matrix.
        """
        return np.matmul(self._inv_rotation_matrix, v)

    def _rot_x(self, theta_x):
        r""" Rotation about the $x$ axis.
        """
        return [[1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)]]

    def _rot_y(self, theta_y):
        r""" Rotation about the $y$ axis.
        """
        return [[np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)]]

    def _rot_z(self, theta_z):
        r""" Rotation about the $z$ axis.
        """
        return [[np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1]]

    def _rot_tot(self, theta_x=0, theta_y=0, theta_z=0):
        r""" Full rotation matrix for basic rotation method.
        """
        return np.matmul(self._rot_z(theta_z),
                         np.matmul(self._rot_y(theta_y),
                                   self._rot_x(theta_x)))

    def _compute_rotation_matrix(self, method, **kwargs):
        r""" Calculate rotation matrix for differnt methods.
        """

        if method == 'basic':
            return self._rot_tot(**kwargs)
        elif method == 'about_axis':
            return self._rot_mat_about_axis(**kwargs)
        elif method == 'to_vector':
            return self._rot_mat_to_axis(**kwargs)
        else:
            msg = f'Method {method} is unknown.'
            raise ValueError(msg)

    def _calculate_new_n(self, new_region):
        d_arr_rot_x = abs(self._rotate_vector([self._field.mesh.dx, 0, 0]))
        d_arr_rot_y = abs(self._rotate_vector([0, self._field.mesh.dy, 0]))
        d_arr_rot_z = abs(self._rotate_vector([0, 0, self._field.mesh.dz]))
        d_arr_rot = d_arr_rot_x + d_arr_rot_y + d_arr_rot_z
        vol_new = np.prod(d_arr_rot)
        vol_old = self._field.mesh.dV
        adjust = (vol_old/vol_new)**(1/3)
        new_n = np.round(np.divide(new_region.edges,
                                   d_arr_rot*adjust)).astype(int).tolist()
        return new_n

    def _calculate_new_region(self):
        edges = np.array(self._field.mesh.region.edges)
        rot_edge_x = abs(self._rotate_vector([edges[0], 0, 0]))
        rot_edge_y = abs(self._rotate_vector([0, edges[1], 0]))
        rot_edge_z = abs(self._rotate_vector([0, 0, edges[2]]))
        tot_edge_len = rot_edge_x + rot_edge_y + rot_edge_z
        p1_new = self._field.mesh.region.centre - tot_edge_len/2
        p2_new = self._field.mesh.region.centre + tot_edge_len/2
        return df.Region(p1=p1_new, p2=p2_new)
