import numpy as np
import pyvista as pv
import ubermagutil.units as uu


class PyVistaField:
    def __init__(self, field):
        if field.mesh.region.ndim != 3:
            raise RuntimeError("Only 3d meshes can be plotted.")
        self.field = field * 1

    def __call__(self):
        if self.field.nvdim == 3:
            return self.vector()
        elif self.field.nvdim == 1:
            return self.scalar()

    def vector(self, plotter=None, multiplier=None, scalars=None, **kwargs):
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.nvdim=}."
            )

        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        if scalars is None:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())

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
        plot.add_mesh(
            field_pv.glyph(orient="field", scale="norm", geom=vector),
            scalars=scalars,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

    def scalar(self, plotter=None, multiplier=None, cmap="coolwarm", **kwargs):
        if self.field.nvdim != 1:
            raise RuntimeError(
                "Only meshes with scalar dimensions can be plotted not"
                f" {self.field.nvdim=}."
            )

        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())

        plot.add_volume(
            field_pv,
            cmap=cmap,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

    def valid(self, plotter=None, multiplier=None, **kwargs):
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.mesh.region.ndim=}."
            )

        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        multiplier = self._setup_multiplier(multiplier)

        rescaled_mesh = self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0))

        values = self.field.valid.astype(int)

        grid = pv.RectilinearGrid(*rescaled_mesh.vertices)
        grid.cell_data["values"] = values.flatten(order="F")
        threshed = grid.threshold(0.5, scalars="values")

        plot.add_mesh(threshed, **kwargs)
        plot.remove_scalar_bar()

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

    def contour(
        self,
        isosurfaces=10,
        scalars=None,
        cmap="RdBu",
        plotter=None,
        multiplier=None,
        **kwargs,
    ):
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.mesh.region.ndim=}."
            )

        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        if scalars is None:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk()).cell_data_to_point_data()

        plot.add_mesh(
            field_pv.contour(scalars=scalars, isosurfaces=isosurfaces),
            cmap=cmap,
            smooth_shading=True,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)
        plot.enable_eye_dome_lighting()
        if plotter is None:
            plot.show()

    def streamlines(
        self,
        scalars=None,
        cmap="RdBu",
        opacity=0.5,
        plotter=None,
        multiplier=None,
        **kwargs,
    ):
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.mesh.region.ndim=}."
            )

        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        if scalars is None:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk()).cell_data_to_point_data()

        streamlines = field_pv.streamlines(
            "field",
            n_points=20,
            max_time=1000,
        )

        plot.add_mesh(
            streamlines.tube(radius=0.05),
            cmap=cmap,
            opacity=opacity,
            smooth_shading=True,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)
        plot.enable_eye_dome_lighting()
        if plotter is None:
            plot.show()

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
