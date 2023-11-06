import pyvista as pv
import ubermagutil.units as uu

import discretisedfield.plotting.util as plot_util


class PyVistaMesh:
    def __init__(self, mesh):
        if mesh.region.ndim != 3:
            raise RuntimeError("Only 3d meshes can be plotted.")
        self.mesh = mesh

    def __call__(
        self,
        *,
        plotter=None,
        color=plot_util.cp_hex,
        multiplier=None,
        wireframe=True,
        filename=None,
        **kwargs,
    ):
        """Generates a ``pyvista`` plot of a mesh.

        This method generates a ``pyvista`` plot of a given mesh by plotting
        the overall region, each subregion of the mesh, and a cell.
        Each subregion can be coloured distinctly, while the discretisation
        cell is always coloured black.

        Keyword arguments are passed onto ``pyvista.add_mesh`` when
        plotting each subregion.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        color : array_like, optional

            Colour of the subregions and the discretisation cell. Defaults to the
            default color palette.

        wireframe : bool, optional

            Show a wireframe to outline the cells. Defaults to ``True``.

        multiplier : numbers.Real, optional

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.

        **kwargs

            Arbitrary keyword arguments passed to `pyvista.add_mesh`, allowing for
            additional customisation of the plot.

        Raises
        ------
        ValueError

            If the mesh associated does not have three spatial dimensions.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.region`

        """
        if self.mesh.region.ndim != 3:
            raise ValueError(
                "Only meshes with 3 spatial dimensions can be plotted not"
                f" {self.data.mesh.region.ndim=}."
            )

        if plotter is None:
            plot = pv.Plotter()
        else:
            plot = plotter

        multiplier = self._setup_multiplier(multiplier)

        rescaled_mesh = self.mesh.scale(1 / multiplier, reference_point=(0, 0, 0))

        for i, (key, subregion) in enumerate(rescaled_mesh.subregions.items()):
            subregion.pyvista(plotter=plotter, color=color[i], label=key, **kwargs)

        grid = pv.RectilinearGrid(*rescaled_mesh.vertices)
        plot.disable_hidden_line_removal()

        # Add single cell
        bounds = tuple(
            val
            for pair in zip(
                rescaled_mesh.region.pmin,
                rescaled_mesh.region.pmin + rescaled_mesh.cell,
            )
            for val in pair
        )
        box = pv.Box(bounds)
        plotter.add_mesh(box, color="black", label="cell")

        label = self._axis_labels(multiplier)
        # Bounds only needed due to axis bug
        bounds = tuple(
            val
            for pair in zip(rescaled_mesh.region.pmin, rescaled_mesh.region.pmax)
            for val in pair
        )
        box = pv.Box(bounds)
        plot.add_mesh(box, opacity=0.0)
        plot.show_grid(xtitle=label[0], ytitle=label[1], ztitle=label[2])

        if wireframe:
            plot.add_mesh(grid, style="wireframe", show_edges=True)
        else:
            edges = box.extract_all_edges()
            plot.add_mesh(edges, color="black")

        plot.add_legend(bcolor=None)

        if plotter is None:
            plot.show()

        if filename is not None:
            self._save_to_file(filename, plot)

    def _setup_multiplier(self, multiplier):
        return self.mesh.region.multiplier if multiplier is None else multiplier

    def _axis_labels(self, multiplier):
        return [
            rf"{dim} ({uu.rsi_prefixes[multiplier]}{unit})"
            for dim, unit in zip(self.mesh.region.dims, self.mesh.region.units)
        ]

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
