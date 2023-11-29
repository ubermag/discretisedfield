import contextlib

import pyvista as pv

import discretisedfield as df


class _FieldIO_VTI:
    __slots__ = []

    def _to_vti(self, filename, array_name, save_subregions=True):
        grid = pv.ImageData(
            dimensions=self.mesh.n + 1,
            spacing=self.mesh.cell,
            origin=self.mesh.region.pmin,
        )
        if isinstance(self, df.cell_field.CellField):
            grid.cell_data.set_array(
                self.array.reshape(-1, self.nvdim, order="F"), array_name
            )
        elif isinstance(self, df.vertex_field.VertexField):
            grid.point_data.set_array(
                self.array.reshape(-1, self.nvdim, order="F"), array_name
            )
        else:
            assert False, "This should never happen"

        if save_subregions and self.mesh.subregions:
            self.mesh.save_subregions(filename)

        grid.save(filename)

    @classmethod
    def _from_vti(cls, filename):
        data: pv.core.grid.ImageData = pv.read(filename)

        p1 = data.bounds[::2]
        p2 = data.bounds[1::2]
        cell = data.spacing
        mesh = df.Mesh(p1=p1, p2=p2, cell=cell)

        field_name = data.array_names[0]
        value = data[field_name]
        nvdim = value.shape[-1]

        value = value.reshape((*data.dimensions, nvdim), order="F")
        if mesh.n == value.shape[:-1]:
            data_location = "cell"
        else:
            data_location = "vertex"

        with contextlib.suppress(FileNotFoundError):
            mesh.load_subregions(filename)

        return cls(mesh, nvdim=nvdim, value=value, data_location=data_location)
