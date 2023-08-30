import numpy as np
import pyvista as pv
import ubermagutil.units as uu


class PyVistaField:
    def __init__(self, field):
        if field.mesh.region.ndim != 3:
            raise RuntimeError("Only 3d meshes can be plotted.")
        self.field = field * 1

    # TODO: Add geom to args
    def __call__(
        self, plot=None, multiplier=None, scalars=None, cmap="coolwarm", **kwargs
    ):
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.mesh.region.ndim=}."
            )

        if plot is None:
            plotter = pv.Plotter()
        else:
            plotter = plot

        if scalars is None:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        # field_pv = self._vtk_add_point_data(field_pv)

        scale = np.min(self.field.mesh.cell) / np.max(self.field.norm.array)

        vector = pv.Arrow(
            tip_radius=0.18,
            tip_length=0.4,
            scale=scale,
            tip_resolution=80,
            shaft_resolution=80,
            shaft_radius=0.05,
            start=(-0.5 * scale, 0, 0),
        )
        plotter.add_mesh(
            field_pv.glyph(orient="field", scale="norm", geom=vector),
            scalars=scalars,
            cmap=cmap,
            **kwargs,
        )

        self._add_empty_region(plotter, multiplier, self.field.mesh.region)

        if plot is None:
            plotter.show()

    def valid(self, plot=None, multiplier=None, **kwargs):
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.mesh.region.ndim=}."
            )

        if plot is None:
            plotter = pv.Plotter()

        multiplier = self._setup_multiplier(multiplier)

        rescaled_mesh = self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0))

        values = self.field.valid.astype(int)

        grid = pv.RectilinearGrid(*rescaled_mesh.vertices)
        grid.cell_data["values"] = values.flatten(order="F")

        # plotter.add_mesh(grid, scalars='values', opacity='values', **kwargs)
        plotter.add_volume(grid, scalars="values", flip_scalars=True, **kwargs)
        plotter.remove_scalar_bar()

        self._add_empty_region(plotter, multiplier, self.field.mesh.region)

        if plot is None:
            plotter.show()

    def contour(
        self,
        isosurfaces=10,
        scalars=None,
        cmap="RdBu",
        opacity=0.5,
        plot=None,
        multiplier=None,
        **kwargs,
    ):
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.mesh.region.ndim=}."
            )

        if plot is None:
            plotter = pv.Plotter()
        else:
            plotter = plot

        if scalars is None:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk()).cell_data_to_point_data()
        # field_pv = self._vtk_add_point_data(field_pv)

        plotter.add_mesh(
            field_pv.contour(scalars=scalars, isosurfaces=isosurfaces),
            cmap=cmap,
            opacity=opacity,
            smooth_shading=True,
        )

        self._add_empty_region(plotter, multiplier, self.field.mesh.region)
        plotter.enable_eye_dome_lighting()
        if plot is None:
            plotter.show()

    def streamlines(self):
        raise NotImplementedError()

    def _setup_multiplier(self, multiplier):
        return self.field.mesh.region.multiplier if multiplier is None else multiplier

    def _axis_labels(self, multiplier):
        return [
            rf"{dim} ({uu.rsi_prefixes[multiplier]}{unit})"
            for dim, unit in zip(
                self.field.mesh.region.dims, self.field.mesh.region.units
            )
        ]

    def _add_empty_region(self, plotter, multiplier, region):
        label = self._axis_labels(multiplier)
        # Bounds only needed due to axis bug
        bounds = tuple(val for pair in zip(region.pmin, region.pmax) for val in pair)
        box = pv.Box(bounds)
        plotter.add_mesh(box, opacity=0.0)
        plotter.show_grid(xtitle=label[0], ytitle=label[1], ztitle=label[2])
