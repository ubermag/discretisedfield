import pyvista as pv
import ubermagutil.units as uu

import discretisedfield.plotting.util as plot_util


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

    def vector(
        self,
        plotter=None,
        multiplier=None,
        scalars=None,
        vector=plot_util.arrow(),
        filename=None,
        **kwargs,
    ):
        """``pyvista`` vector plot.

        If ``plotter`` is not passed, a new `pyvista` plotter object is created
        automatically.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        Keyword arguments are passed onto ``pyvista.add_mesh``.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        scalars : str, optional

            ``vdims`` on which to colour the glyphs. Defaults to the last ``vdims``.

        vector : pyvista.object

            pyvista object to place at each position. These point in the direction
            of the field and are scaled by the norm of the field.
            Defaults to ``plot_util.arrow()``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.
            The supported formats are png, jpeg, jpg, bmp, tif, tiff, svg,
            eps, ps, pdf, and txt.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`

        """
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
        field_pv = field_pv.extract_cells(field_pv["valid"].astype(bool))

        # scale = np.min(self.field.mesh.cell) / np.max(self.field.norm.array)

        plot.add_mesh(
            field_pv.glyph(orient="field", scale="norm", geom=vector),
            scalars=scalars,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

        if filename is not None:
            self._save_to_file(filename, plot)

    def scalar(
        self, plotter=None, multiplier=None, scalars=None, filename=None, **kwargs
    ):
        """``pyvista`` scalar plot.

        If ``plotter`` is not passed, a new `pyvista` plotter object is created
        automatically.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        Keyword arguments are passed onto ``pyvista.add_mesh``.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        scalars : str, optional

            ``vdims`` on which to colour the cells. Defaults to the last ``vdims``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.
            The supported formats are png, jpeg, jpg, bmp, tif, tiff, svg,
            eps, ps, pdf, and txt.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`

        """
        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        if scalars is None and self.field.nvdim > 1:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        field_pv = field_pv.extract_cells(field_pv["valid"].astype(bool))

        plot.add_mesh_slice(
            field_pv,
            scalars=scalars,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

        if filename is not None:
            self._save_to_file(filename, plot)

    def volume(self, plotter=None, multiplier=None, filename=None, **kwargs):
        """``pyvista`` volume plot.

        If ``plotter`` is not passed, a new `pyvista` plotter object is created
        automatically.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        Keyword arguments are passed onto ``pyvista.add_mesh``.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.
            The supported formats are png, jpeg, jpg, bmp, tif, tiff, svg,
            eps, ps, pdf, and txt.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`

        """

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
        field_pv = field_pv.extract_cells(field_pv["valid"].astype(bool))

        plot.add_volume(
            field_pv,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

        if filename is not None:
            self._save_to_file(filename, plot)

    def valid(self, plotter=None, multiplier=None, filename=None, **kwargs):
        """``pyvista`` valid plot.

        If ``plotter`` is not passed, a new `pyvista` plotter object is created
        automatically.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        Keyword arguments are passed onto ``pyvista.add_mesh``.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.
            The supported formats are png, jpeg, jpg, bmp, tif, tiff, svg,
            eps, ps, pdf, and txt.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`

        """

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

        if filename is not None:
            self._save_to_file(filename, plot)

    def contour(
        self,
        isosurfaces=10,
        plotter=None,
        multiplier=None,
        filename=None,
        contour_kwargs={},
        **kwargs,
    ):
        """``pyvista`` contour plot.

        If ``plotter`` is not passed, a new `pyvista` plotter object is created
        automatically.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        Keyword arguments are passed onto ``pyvista.add_mesh``.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        isosurfaces : int | sequence[float], optional

            Number of isosurfaces to compute across valid data range
            or a sequence of float values to explicitly use as
            the isosurfaces.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.
            The supported formats are png, jpeg, jpg, bmp, tif, tiff, svg,
            eps, ps, pdf, and txt.

        contour_kwargs : dict, optional

            keyword argument to pass to ``pyvista.contour`` function.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`

        """
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

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        field_pv = field_pv.extract_cells(
            field_pv["valid"].astype(bool)
        ).cell_data_to_point_data()

        if "scalars" not in contour_kwargs.keys():
            contour_kwargs["scalars"] = self.field.vdims[-1]

        plot.add_mesh(
            field_pv.contour(isosurfaces=isosurfaces, **contour_kwargs),
            smooth_shading=True,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)
        plot.enable_eye_dome_lighting()

        if plotter is None:
            plot.show()

        if filename is not None:
            self._save_to_file(filename, plot)

    def streamlines(
        self,
        plotter=None,
        multiplier=None,
        filename=None,
        streamlines_kwargs={"max_time": 10, "n_points": 20},
        tube_kwargs={"radius": 0.05},
        **kwargs,
    ):
        """``pyvista`` streamline plot.

        If ``plotter`` is not passed, a new `pyvista` plotter object is created
        automatically.

        For details about ``multiplier``, please refer to
        ``discretisedfield.Region.mpl``.

        Keyword arguments are passed onto ``pyvista.add_mesh``.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        isosurfaces : int | sequence[float], optional

            Number of isosurfaces to compute across valid data range
            or a sequence of float values to explicitly use as
            the isosurfaces.

        multiplier : numbers.Real, optional

            Axes multiplier. Defaults to ``None``.

        filename : str, optional

            If filename is passed, the plot is saved. Defaults to ``None``.
            The supported formats are png, jpeg, jpg, bmp, tif, tiff, svg,
            eps, ps, pdf, and txt.

        streamlines_kwargs : dict, optional

            Keyword argument to pass to ``pyvista.streamlines`` function.

        tube_kwargs : dict, optional

            Keyword argument to pass to ``pyvista.tube`` function.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`

        """
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

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        field_pv = field_pv.extract_cells(
            field_pv["valid"].astype(bool)
        ).cell_data_to_point_data()

        streamlines = field_pv.streamlines("field", **streamlines_kwargs)

        plot.add_mesh(
            streamlines.tube(**tube_kwargs),
            smooth_shading=True,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)
        plot.enable_eye_dome_lighting()

        if plotter is None:
            plot.show()

        if filename is not None:
            self._save_to_file(filename, plot)

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

    def _save_to_file(self, filename, plot):
        extension = filename.split(".")[-1] if "." in filename else None
        if extension in ["png", "jpeg", "jpg", "bmp", "tif", "tiff"]:
            plot.screenshot(filename=filename)
        elif extension in ["svg", "eps", "ps", "pdf", "tex"]:
            plot.save_graphic(filename=filename)
        else:
            raise ValueError(
                f"{extension} extension is not supported. The supported formats are"
                " png, jpeg, jpg, bmp, tif, tiff, svg, eps, ps, pdf, and txt."
            )
