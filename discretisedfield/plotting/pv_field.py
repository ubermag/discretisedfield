import pyvista as pv


class PvField:
    """Visualise field using pyvista."""

    def __init__(self, field):
        # TODO
        # either: fix lighting problems for nm-sized meshes
        # or: rescale mesh and update the units (similar to mpl multiplier)
        self.field = field
        # TODO Would it better to create the pv_field in the individual methods?
        self.pv_field = pv.wrap(field.to_vtk())
        # TODO Set the backend only if nothing has been set yet if possible.
        # TODO
        # - What are the advantages of 'client' or 'server'?
        # - Is 'client a good choice?
        pv.set_jupyter_backend("client")

    # TODO Would two separate methods for 'scalar' and 'vector' field be better?
    def __call__(self, plotter=None, **kwargs):  # TODO Do we need extra arguments
        """Plot field values.

        Scalar fields are plotted as voxels, vector fields using cones.
        """
        if plotter is None:
            plotter = pv.Plotter()
        if self.field.nvdim == 1:
            raise NotImplementedError()
        else:
            cone = pv.Cone()  # TODO What are good defaults here
            plotter.add_mesh(
                self.pv_field.glyph(orient="field", scale="norm", geom=cone),
                # TODO How do we set the colour?
                **kwargs,
            )
            # TODO only call plotter.show if plotter was not passed as an arugement
            plotter.show()

    def isosurfaces(self, plotter=None, **kwargs):
        """Show isosurfaces for a specific field component."""
        if plotter is None:
            plotter = pv.Plotter()
        pv_field_points = self.pv_field.cell_data_to_point_data()
        plotter.add_mesh(
            pv_field_points.contour(scalars=..., **kwargs)  # TODO good defaults?
        )
        # TODO only call plotter.show if plotter was not passed as an arugement
        plotter.show()

    def nonzero(self, plotter=None, **kwargs):
        """Show the shape of the non-zero part of the field."""
        raise NotImplementedError()

    # TODO Would any other plot type be useful?
