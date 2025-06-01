import copy

import numpy as np
import pyvista as pv
import ubermagutil.units as uu

import discretisedfield.plotting.util as plot_util


class PyVistaField:
    def __init__(self, field):
        if field.mesh.region.ndim != 3:
            raise RuntimeError("Only 3d meshes can be plotted.")

        self.field = field.__class__(
            copy.deepcopy(field.mesh),
            nvdim=field.nvdim,
            value=field.array,
            vdims=field.vdims,
            valid=field.valid,
            vdim_mapping=field.vdim_mapping,
        )

    def vector(
        self,
        plotter=None,
        multiplier=None,
        scalars=None,
        vector=None,
        scale=None,
        color_field=None,
        filename=None,
        glyph_kwargs=None,
        **kwargs,
    ):
        """``pyvista`` vector plot.

        This function visualises a vector field where each vector is represented by a
        glyph, by default an arrow, which points in the direction of the vector and
        has a magnitude proportional to the vector's magnitude. Users can specify
        various parameters to customise the plot, including the plotter, scalar values
        for colour mapping, and a file to save the plot.

        Keyword arguments are passed onto ``pyvista.add_mesh``.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        scalars : str, optional

            The name of the field's vector dimension used to determine the colours of
            the glyphs. By default, the last vector dimension of the field is used.

        vector : pyvista.core.pointset.PointSet, optional

            A ``pyvista`` geometric object used as the glyph that represents the vectors
            in the field. The default is set by ``plot_util.arrow()``, which provides
            a simple arrow shape.

        scale : float, optional

            This value scales the vector glyph prior to plotting. The scale defaults
            to the minimum edge length of a cell divided by the maximum norm of the
            field.

        color_field : discretisedfield.field, optional

            A scalar field used for colouring. Defaults to ``None``
            and the colouring is based on ``scalars``. If provided,
            ``scalars`` are ignored.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.

        glyph_kwargs : dict, optional

            Keyword arguments for the `pyvista.glyph` function that generates the
            glyphs from the vector field data.

        **kwargs

            Arbitrary keyword arguments that are passed to `pyvista.add_mesh`,
            allowing for additional customisation.

        Raises
        ------
        RuntimeError

            If the vector field does not have three dimensions.

        Examples
        --------
        1. Visualising the vector field using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=3, value=(0, 0, 1))
        ...
        >>> field.pyvista.vector() # doctest: +SKIP

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`
            :py:func:`~discretisedfield.plotting.pyvista.contour`
            :py:func:`~discretisedfield.plotting.pyvista.valid`

        """
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.nvdim=}."
            )

        if glyph_kwargs is None:
            glyph_kwargs = {}

        if vector is None:
            vector = plot_util.arrow()

        if color_field is not None:
            if color_field.nvdim != 1:
                raise ValueError(f"Cannot use {color_field.nvdim=}.")
            if not self.field.mesh.allclose(color_field.mesh):
                raise ValueError("The color_field has to be defined on the same mesh.")

        plot = pv.Plotter() if plotter is None else plotter

        if scalars is None:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        if color_field is not None:
            field_pv["color_field"] = pv.wrap(color_field.to_vtk())["field"]
            scalars = "color_field"
        field_pv = field_pv.extract_cells(field_pv["valid"].astype(bool))

        if scale is None:
            scale = np.min(self.field.mesh.cell) / np.max(self.field.norm.array)

        scaled_vector = vector.scale(scale, inplace=False)

        plot.add_mesh(
            field_pv.glyph(
                orient="field", scale="norm", geom=scaled_vector, **glyph_kwargs
            ),
            scalars=scalars,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

        if filename is not None:
            plot_util._pyvista_save_to_file(filename, plot)

    def scalar(
        self, plotter=None, multiplier=None, scalars=None, filename=None, **kwargs
    ):
        """``pyvista`` scalar plot.

        This function visualises a scalar field using slices of the mesh which can be
        interactively manipulated. Users can specify various parameters to customise
        the plot, including the plotter, scalar values for colour
        mapping, and a file to save the plot.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        scalars : str, optional

            ``vdims`` on which to colour the cells. Defaults to the last ``vdims``.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.

        **kwargs
            Arbitrary keyword arguments passed to `pyvista.add_mesh_slice` for
            additional customisation of the plot.

        Examples
        --------
        1. Visualising a scalar field using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=1, value=1)
        ...
        >>> field.pyvista.scalar() # doctest: +SKIP

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`
            :py:func:`~discretisedfield.plotting.pyvista.vector`
            :py:func:`~discretisedfield.plotting.pyvista.contour`
            :py:func:`~discretisedfield.plotting.pyvista.valid`
            :py:func:`~discretisedfield.plotting.pyvista.volume`

        """
        plot = pv.Plotter() if plotter is None else plotter

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
            plot_util._pyvista_save_to_file(filename, plot)

    def volume(
        self, plotter=None, multiplier=None, scalars=None, filename=None, **kwargs
    ):
        """``pyvista`` volume plot.

        This method visualises the scalar field within a three-dimensional region
        by rendering a volume. The density and color within the volume
        represent the scalar value at each point.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        scalars : str, optional

            The name of the field's vector dimension used to determine the colours of
            the glyphs. By default, the last vector dimension of the field is used.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.

        **kwargs

            Arbitrary keyword arguments passed directly to `pyvista.add_volume` for
            additional customisation.

        Examples
        --------
        1. Visualising a scalar field using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = df.Field(mesh, nvdim=1, value=1)
        ...
        >>> field.pyvista.volume() # doctest: +SKIP

        Raises
        ------
        RuntimeError

            If the it is not a scalar field.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`
            :py:func:`~discretisedfield.plotting.pyvista.vector`
            :py:func:`~discretisedfield.plotting.pyvista.contour`
            :py:func:`~discretisedfield.plotting.pyvista.valid`

        """

        plot = pv.Plotter() if plotter is None else plotter

        if scalars is None and self.field.nvdim > 1:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        field_pv = field_pv.extract_cells(field_pv["valid"].astype(bool))

        plot.add_volume(
            field_pv,
            scalars=scalars,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

        if filename is not None:
            plot_util._pyvista_save_to_file(filename, plot)

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

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.


        Examples
        --------
        1. Visualising the vector field using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (1, 2, 2)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> valid = [[[True, True], [True, False]]]
        >>> field = df.Field(mesh, nvdim=1, value=1, valid=valid)
        ...
        >>> field.pyvista.valid() # doctest: +SKIP

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`
            :py:func:`~discretisedfield.plotting.pyvista.vector`
            :py:func:`~discretisedfield.plotting.pyvista.contour`

        """

        plot = pv.Plotter() if plotter is None else plotter

        # Default colour
        kwargs.setdefault("color", "blue")

        multiplier = self._setup_multiplier(multiplier)

        rescaled_mesh = self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0))

        values = self.field.valid.astype(int)

        grid = pv.RectilinearGrid(*rescaled_mesh.vertices)
        grid.cell_data["values"] = values.flatten(order="F")
        threshed = grid.threshold(0.5, scalars="values")

        plot.add_mesh(threshed, **kwargs)

        self._add_empty_region(plot, multiplier, self.field.mesh.region)

        if plotter is None:
            plot.show()

        if filename is not None:
            plot_util._pyvista_save_to_file(filename, plot)

    def contour(
        self,
        isosurfaces=10,
        contour_scalars=None,
        plotter=None,
        multiplier=None,
        scalars=None,
        color_field=None,
        filename=None,
        contour_kwargs=None,
        **kwargs,
    ):
        """``pyvista`` contour plot.

        This method computes isosurfaces of the field. Users can specify
        the number of evenly spaced isosurfaces or provide specific
        values for which isosurfaces should be computed.

        Parameters
        ----------
        isosurfaces : int | sequence[float], optional

            Number of isosurfaces to compute across valid data range
            or a sequence of float values to explicitly use as
            the isosurfaces. Defaults to 10.

        contour_scalars : str, optional

            The name of the field's vector dimension used for the isosurfaces.
            By default, the last vector dimension of the field is used.

        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        scalars : str, optional

            The name of the field's vector dimension used to determine the colour.
            By default, the last vector dimension of the field is used.

        color_field : discretisedfield.field, optional

            A scalar field used for colouring. Defaults to ``None``
            and the colouring is based on ``scalars``. If provided,
            ``scalars`` are ignored.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.

        contour_kwargs : dict, optional

            keyword argument to pass to ``pyvista.contour`` function.

        **kwargs

            Arbitrary keyword arguments that are passed to `pyvista.add_mesh`,
            allowing for additional customisation.

        Examples
        --------
        1. Visualising a field using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = mesh.coordinate_field()
        ...
        >>> field.pyvista.contour() # doctest: +SKIP

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`
            :py:func:`~discretisedfield.plotting.pyvista.vector`
            :py:func:`~discretisedfield.plotting.pyvista.contour`
            :py:func:`~discretisedfield.plotting.pyvista.valid`

        """

        if contour_kwargs is None:
            contour_kwargs = {}

        if self.field.nvdim > 1 and "scalars" not in contour_kwargs:
            if contour_scalars is None:
                contour_kwargs["scalars"] = self.field.vdims[-1]
            else:
                contour_kwargs["scalars"] = contour_scalars

        if color_field is not None:
            if color_field.nvdim != 1:
                raise ValueError(f"Cannot use {color_field.nvdim=}.")
            if not self.field.mesh.allclose(color_field.mesh):
                raise ValueError("The color_field has to be defined on the same mesh.")

        if scalars is None and self.field.nvdim > 1:
            scalars = self.field.vdims[-1]

        plot = pv.Plotter() if plotter is None else plotter

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        if color_field is not None:
            field_pv["color_field"] = pv.wrap(color_field.to_vtk())["field"]
            scalars = "color_field"
        field_pv = field_pv.extract_cells(
            field_pv["valid"].astype(bool)
        ).cell_data_to_point_data()

        plot.add_mesh(
            field_pv.contour(isosurfaces=isosurfaces, **contour_kwargs),
            smooth_shading=True,
            scalars=scalars,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)
        plot.enable_eye_dome_lighting()

        if plotter is None:
            plot.show()

        if filename is not None:
            plot_util._pyvista_save_to_file(filename, plot)

    def streamlines(
        self,
        plotter=None,
        multiplier=None,
        scalars=None,
        color_field=None,
        filename=None,
        streamlines_kwargs=None,
        tube_kwargs=None,
        **kwargs,
    ):
        """``pyvista`` streamline plot.

        Generates a plot of streamlines based on ``pyvista.streamlines``
        Users will need to vary the parameters from the default values
        on a case by case basis.

        Parameters
        ----------
        plotter : pyvista.Plotter, optional

            Plotter to which the plotter is added. Defaults to ``None``
            - plot is created internally.

        multiplier : numbers.Real, optional

            A scaling factor applied to the region dimensions. This can be useful for
            adjusting the region size for visualisation purposes. If ``None``, no
            scaling is applied. For more details, see ``discretisedfield.Region.mpl``.

        scalars : str, optional

            The name of the field's vector dimension used to determine the colour.
            By default, the last vector dimension of the field is used.

        color_field : discretisedfield.field, optional

            A scalar field used for colouring. Defaults to ``None``
            and the colouring is based on ``scalars``. If provided,
            ``scalars`` are ignored.

        filename : str, optional

            The path or filename where the plot will be saved. If specified, the plot is
            saved to this file. The file format is inferred from the extension, which
            must be one of: 'png', 'jpeg', 'jpg', 'bmp', 'tif', 'tiff', 'svg', 'eps',
            'ps', 'pdf', or 'txt'. If `None`, the plot is not saved to a file.

        streamlines_kwargs : dict, optional

            Keyword arguments for the `pyvista.streamlines` function that generates the
            streamline geometry from the vector field data. If not provided, the default
            keys are ``max_time=10`` and ``n_points=20``.

        tube_kwargs : dict, optional

            Keyword arguments for the `pyvista.tube` function that creates
            tubes around the streamlines. If not provided, the default keys
            are ``radius=0.05``.

        **kwargs

            Arbitrary keyword arguments passed directly to `pyvista.add_mesh`,
            allowing for further customization of the plot.

        Examples
        --------
        1. Visualising a field using ``pyvista``.

        >>> import discretisedfield as df
        ...
        >>> p1 = (0, 0, 0)
        >>> p2 = (100, 100, 100)
        >>> n = (10, 10, 10)
        >>> mesh = df.Mesh(p1=p1, p2=p2, n=n)
        >>> field = mesh.coordinate_field()
        ...
        >>> field.pyvista.streamlines() # doctest: +SKIP

        Raises
        ------
        RuntimeError

            If the field does not have three value dimensions.

        .. seealso::

            :py:func:`~discretisedfield.plotting.pyvista.scalar`
            :py:func:`~discretisedfield.plotting.pyvista.vector`
            :py:func:`~discretisedfield.plotting.pyvista.contour`
            :py:func:`~discretisedfield.plotting.pyvista.valid`

        """
        if tube_kwargs is None:
            tube_kwargs = {}
        if streamlines_kwargs is None:
            streamlines_kwargs = {}
        if self.field.nvdim != 3:
            raise RuntimeError(
                "Only meshes with 3 vector dimensions can be plotted not"
                f" {self.field.mesh.region.ndim=}."
            )

        streamlines_default_values = {
            "max_length": 10,
            "n_points": 20,
        }

        tube_default_values = {
            "radius": 0.05,
        }

        if streamlines_kwargs is None:
            streamlines_kwargs = streamlines_default_values
        else:
            for key, value in streamlines_default_values.items():
                streamlines_kwargs.setdefault(key, value)

        if tube_kwargs is None:
            tube_kwargs = tube_default_values
        else:
            for key, value in tube_default_values.items():
                tube_kwargs.setdefault(key, value)

        plot = pv.Plotter() if plotter is None else plotter

        if scalars is None and self.field.nvdim > 1:
            scalars = self.field.vdims[-1]

        multiplier = self._setup_multiplier(multiplier)

        self.field.mesh.scale(1 / multiplier, reference_point=(0, 0, 0), inplace=True)

        field_pv = pv.wrap(self.field.to_vtk())
        if color_field is not None:
            field_pv["color_field"] = pv.wrap(color_field.to_vtk())["field"]
            scalars = "color_field"
        field_pv = field_pv.extract_cells(
            field_pv["valid"].astype(bool)
        ).cell_data_to_point_data()

        streamlines = field_pv.streamlines("field", **streamlines_kwargs)

        plot.add_mesh(
            streamlines.tube(**tube_kwargs),
            scalars=scalars,
            **kwargs,
        )

        self._add_empty_region(plot, multiplier, self.field.mesh.region)
        plot.enable_eye_dome_lighting()

        if plotter is None:
            plot.show()

        if filename is not None:
            plot_util._pyvista_save_to_file(filename, plot)

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
        # Bounds only needed due to bug in pyvista.
        # Usually we could just use add_axes but they were not plotted
        # over the full region.
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
