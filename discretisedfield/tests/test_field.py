import os
import random
import pytest
import numpy as np
import discretisedfield as df
import matplotlib


class TestField(object):
    def setup(self):
        self.meshes = self.create_meshes()
        self.scalar_fs = self.create_scalar_fs()
        self.vector_fs = self.create_vector_fs()
        self.constant_values = [0, -5., np.pi,
                                1e-15, 1.2e12, random.random()]
        self.tuple_values = [(0, 0, 1),
                             (0, -5.1, np.pi),
                             [70, 1e15, 2*np.pi],
                             [5, random.random(), np.pi],
                             np.array([4, -1, 3.7]),
                             np.array([2.1, 0.0, -5*random.random()])]
        self.scalar_pyfuncs = self.create_scalar_pyfuncs()
        self.vector_pyfuncs = self.create_vector_pyfuncs()

    def create_meshes(self):
        p1_list = [(0, 0, 0),
                   (-5e-9, -8e-9, -10e-9),
                   (10, -5, -80)]
        p2_list = [(5e-9, 8e-9, 10e-9),
                   (11e-9, 4e-9, 4e-9),
                   (15, 10, 85)]
        cell_list = [(1e-9, 1e-9, 1e-9),
                     (1e-9, 2e-9, 1e-9),
                     (5, 5, 2.5)]

        meshes = []
        for i in range(len(p1_list)):
            mesh = df.Mesh(p1_list[i], p2_list[i], cell_list[i])
            meshes.append(mesh)
        return meshes

    def create_scalar_fs(self):
        scalar_fs = []
        for mesh in self.meshes:
            f = df.Field(mesh, dim=1)
            scalar_fs.append(f)
        return scalar_fs

    def create_vector_fs(self):
        vector_fs = []
        for mesh in self.meshes:
            f = df.Field(mesh, dim=3)
            vector_fs.append(f)
        return vector_fs

    def create_scalar_pyfuncs(self):
        f = [lambda c: 1,
             lambda c: -2.4,
             lambda c: -6.4e-15,
             lambda c: c[0] + c[1] + c[2] + 1,
             lambda c: (c[0]-1)**2 - c[1]+7 + c[2]*0.1,
             lambda c: np.sin(c[0]) + np.cos(c[1]) - np.sin(2*c[2])]

        return f

    def create_vector_pyfuncs(self):
        f = [lambda c: (1, 2, 0),
             lambda c: (-2.4, 1e-3, 9),
             lambda c: (c[0], c[1], c[2] + 100),
             lambda c: (c[0]+c[2]+10, c[1], c[2]+1),
             lambda c: (c[0]-1, c[1]+70, c[2]*0.1),
             lambda c: (np.sin(c[0]), np.cos(c[1]), -np.sin(2*c[2]))]

        return f

    def test_init(self):
        p1 = (0, -4, 11)
        p2 = (15, 10.1, 16.5)
        d = (1, 0.1, 0.5)
        mesh = df.Mesh(p1, p2, d)
        dim = 2
        name = "test_field"
        value = [1, 2]

        f = df.Field(mesh, dim=2, value=value, name=name)

        assert f.name == name

        assert np.all(f.f[:, :, :, 0] == value[0])
        assert np.all(f.f[:, :, :, 1] == value[1])

    def test_wrong_init(self):
        mesh = "wrong_mesh_string"
        with pytest.raises(TypeError):
            f = df.Field(mesh, dim=1, name="wrong_field")

    def test_set_with_constant(self):
        for value in self.constant_values:
            for f in self.scalar_fs + self.vector_fs:
                f.f = value

                # Check all values.
                assert np.all(f.f == value)

                # Check with sampling.
                assert np.all(f(f.mesh.random_point()) == value)

    def test_set_with_tuple_list_ndarray(self):
        for value in self.tuple_values:
            for f in self.vector_fs:
                f.normalisedto = 1
                f.f = value

                norm = (value[0]**2 + value[1]**2 + value[2]**2)**0.5
                for j in range(3):
                    c = f.mesh.random_point()
                    assert np.all(f.f[:, :, :, j] == value[j]/norm)
                    assert np.all(f(c)[j] == value[j]/norm)

    def test_set_from_callable(self):
        # Test scalar fs.
        for f in self.scalar_fs:
            for pyfun in self.scalar_pyfuncs:
                f.f = pyfun

                for j in range(10):
                    c = f.mesh.random_point()
                    expected_value = pyfun(f.mesh.cell_centre(c))
                    assert f(c) == expected_value

        # Test vector fields.
        for f in self.vector_fs:
            for pyfun in self.vector_pyfuncs:
                f.f = pyfun

                for j in range(10):
                    c = f.mesh.random_point()
                    expected_value = pyfun(f.mesh.cell_centre(c))
                    assert np.all(f(c) == expected_value)

    def test_set_exception(self):
        for f in self.vector_fs + self.scalar_fs:
            with pytest.raises(TypeError):
                f.f = "string"

    def test_average_scalar_field(self):
        value = -1e-3 + np.pi
        tol = 1e-12
        for f in self.scalar_fs:
            f.f = value
            average = f.average()

            assert abs(average[0] - value) < tol

    def test_average_vector_field(self):
        value = np.array([1.1, -2e-3, np.pi])
        tol = 1e-12
        for f in self.vector_fs:
            f.f = value
            average = f.average()

            assert abs(average[0] - value[0]) < tol
            assert abs(average[1] - value[1]) < tol
            assert abs(average[2] - value[2]) < tol

    def test_normalise(self):
        for f in self.vector_fs:
            for value in self.vector_pyfuncs + self.tuple_values:
                for norm_value in [1, 2.1, 50, 1e-3, np.pi]:
                    f.f = value
                    f.normalisedto = norm_value
                    f.normalise()

                    # Compute norm.
                    norm = 0
                    for j in range(f.dim):
                        norm += f.f[:, :, :, j]**2
                    norm = np.sqrt(norm)

                    assert norm.shape == (f.mesh.n[0],
                                          f.mesh.n[1],
                                          f.mesh.n[2])
                    diff = norm - norm_value

                    assert np.all(abs(norm - norm_value) < 1e-12)

    def test_normalise_scalar_field_exception(self):
        for f in self.scalar_fs:
            for value in self.scalar_pyfuncs:
                for norm_value in [1, 2.1, 50, 1e-3, np.pi]:
                    with pytest.raises(NotImplementedError):
                        f.normalisedto = norm_value
                        f.f = value

    def test_line_intersection(self):
        mesh = df.Mesh((0, 0, 0), (10e-9, 10e-9, 10e-9), (1e-9, 1e-9, 1e-9))
        field = df.Field(mesh, value=(1, 2, 3))

        d, v = field.line_intersection((0, 0, 1), (5e-9, 5e-9, 5e-9))

        assert len(d) == 100
        assert len(v) == 100

    def test_slice_field(self):
        for s in "xyz":
            for f in self.vector_fs + self.scalar_fs:
                if f.dim == 1:
                    funcs = self.scalar_pyfuncs
                elif f.dim == 3:
                    funcs = self.vector_pyfuncs

                for pyfun in funcs:
                    f.f = pyfun
                    point = f.mesh.centre()["xyz".find(s)]
                    data = f.slice_field(s, point)
                    a1, a2, f_slice, cs = data

                    if s == "x":
                        assert cs == (1, 2, 0)
                    elif s == "y":
                        assert cs == (0, 2, 1)
                    elif s == "z":
                        assert cs == (0, 1, 2)

                    tol = 1e-16
                    assert abs(a1[0] - (f.mesh.p1[cs[0]] +
                                        f.mesh.cell[cs[0]]/2.)) < tol
                    assert abs(a1[-1] - (f.mesh.p2[cs[0]] -
                                         f.mesh.cell[cs[0]]/2.)) < tol
                    assert abs(a2[0] - (f.mesh.p1[cs[1]] +
                                        f.mesh.cell[cs[1]]/2.)) < tol
                    assert abs(a2[-1] - (f.mesh.p2[cs[1]] -
                                         f.mesh.cell[cs[1]]/2.)) < tol
                    assert len(a1) == f.mesh.n[cs[0]]
                    assert len(a2) == f.mesh.n[cs[1]]
                    assert f_slice.shape == (f.mesh.n[cs[0]],
                                             f.mesh.n[cs[1]],
                                             f.dim)

                    for j in range(f.mesh.n[cs[0]]):
                        for k in range(f.mesh.n[cs[1]]):
                            c = list(f.mesh.centre())
                            c[cs[0]] = a1[j]
                            c[cs[1]] = a2[k]
                            c = tuple(c)

                            assert np.all(f_slice[j, k] == f(c))

    def test_slice_field_wrong_axis(self):
        for f in self.vector_fs + self.scalar_fs:
            if f.dim == 1:
                funcs = self.scalar_pyfuncs
            elif f.dim == 3:
                funcs = self.vector_pyfuncs

            for pyfun in funcs:
                f.f = pyfun
                point = f.mesh.centre()[0]
                with pytest.raises(ValueError):
                    data = f.slice_field("xy", point)
                    data = f.slice_field("xyz", point)
                    data = f.slice_field("zy", point)
                    data = f.slice_field("string", point)
                    data = f.slice_field("point", point)

    def test_slice_field_wrong_point(self):
        for s in "xyz":
            for f in self.vector_fs + self.scalar_fs:
                if f.dim == 1:
                    funcs = self.scalar_pyfuncs
                elif f.dim == 3:
                    funcs = self.vector_pyfuncs

                for pyfun in funcs:
                    f.f = pyfun
                    point = f.mesh.centre()["xyz".find(s)] + 100
                    with pytest.raises(ValueError):
                        data = f.slice_field(s, point)

                    point = f.mesh.centre()["xyz".find(s)] - 100
                    with pytest.raises(ValueError):
                        data = f.slice_field(s, point)

    def test_plot_slice_vector_field(self):
        figname = "test_slice_plot_figure.pdf"
        value = (1e-3 + np.pi, -5, 6)
        for f in self.vector_fs:
            f.normalisedto = 1
            f.f = value
            point = f.mesh.centre()["xyz".find("y")]
            fig = f.plot_slice("y", point, axes=True)
            fig = f.plot_slice("y", point, axes=False)

    def test_plot_slice_vector_field_exception(self):
        value = (0, 0, 1)
        for f in self.vector_fs:
            f.normalisedto = 1
            f.f = value
            point = f.mesh.centre()["xyz".find("z")]
            with pytest.raises(ValueError):
                fig = f.plot_slice("z", point)

    def test_write_read_oommf_file(self):
        tol = 1e-12
        filename = "test_write_oommf_file.omf"
        value = (1e-3 + np.pi, -5, 6)
        for f in self.vector_fs:
            f.f = value
            f.write_oommf_file(filename)

            f_loaded = df.read_oommf_file(filename)

            assert f.mesh.p1 == f_loaded.mesh.p1
            assert f.mesh.p2 == f_loaded.mesh.p2
            assert f.mesh.cell == f_loaded.mesh.cell
            assert f.mesh.cell == f_loaded.mesh.cell
            assert np.all(abs(f.f - f_loaded.f) < tol)

            os.system("rm {}".format(filename))
