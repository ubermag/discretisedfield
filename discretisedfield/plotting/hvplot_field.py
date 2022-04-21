"""Holoviews-based plotting."""
import holoviews as hv
import hvplot.xarray  # noqa: F401
import numpy as np
import xarray as xr

import discretisedfield as df


class HvplotField:
    """Holoviews-based plotting methods."""

    def __init__(self, field):
        if field.dim > 3:
            raise ValueError(
                f"hvplot does only support fields with dim=1, 2, 3; got {field.dim=}."
            )
        self.field = field
        self.xrfield = field.to_xarray()

    def __call__(self, slider, scalar_kw=None, vector_kw=None):
        """Plot scalar and vector components."""
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")

        scalar_kw = {} if scalar_kw is None else scalar_kw.copy()
        vector_kw = {} if vector_kw is None else vector_kw.copy()
        scalar_kw.setdefault("filter_field", self.field.norm)

        if self.field.dim == 1:
            return self.field.hvplot.scalar(slider, **scalar_kw)
        elif self.field.dim == 2:
            return self.field.hvplot.vector(slider, **vector_kw)
        elif self.field.dim == 3:
            vector_kw.setdefault("use_color", False)
            scalar = getattr(self.field, slider).hvplot.scalar(slider, **scalar_kw)
            vector = self.field.hvplot.vector(slider, **vector_kw)
            return scalar * vector

    def scalar(self, slider, filter_field=None, **kwargs):
        """Plot the scalar field on a plane."""
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))
        groupby = [slider] if self.field.dim == 1 else [slider, "comp"]

        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)
        self._filter_values(filter_field, self.xrfield)
        return self.xrfield.hvplot(x=x, y=y, groupby=groupby, **kwargs)

    def vector(
        self, slider, filter_field=None, use_color=True, color_field=None, **kwargs
    ):
        """Plot the vector field on a plane."""
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        if self.field.dim == 1:
            raise ValueError(f"Cannot plot {self.field.dim=} field.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))

        filter_values = self.field.norm.to_xarray()
        self._filter_values(filter_field, filter_values)
        ip_vector = xr.Dataset(
            {
                "angle": np.arctan2(
                    self.xrfield.sel(comp=y),
                    self.xrfield.sel(comp=x),
                    where=np.logical_and(filter_values != 0, ~np.isnan(filter_values)),
                    out=np.full(self.field.mesh.n, np.nan),
                ),
                "mag": np.sqrt(
                    self.xrfield.sel(comp=x) ** 2 + self.xrfield.sel(comp=y) ** 2
                ),
            }
        )
        vdims = ["angle", "mag"]
        kwargs.setdefault("data_aspect", 1)

        if use_color:
            vdims.append("color_comp")
            kwargs.setdefault("colorbar", True)
            if color_field:
                ip_vector["color_comp"] = color_field.to_xarray()
            else:
                ip_vector["color_comp"] = self.xrfield.sel(comp=slider)

        def _vectorplot(val):
            plot = hv.VectorField(
                data=ip_vector.sel(**{slider: val, "method": "nearest"}),
                kdims=[x, y],
                vdims=vdims,
            )
            plot.opts(magnitude="mag", **kwargs)
            if use_color:
                plot.opts(color="color_comp")
            return plot

        return hv.DynamicMap(_vectorplot, kdims=slider).redim.values(
            **{slider: getattr(self.field.mesh.midpoints, slider)}
        )

    def contour(self, slider, filter_field=None, **kwargs):
        """Plot the scalar field on a plane."""
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))
        groupby = [slider] if self.field.dim == 1 else [slider, "comp"]

        kwargs.setdefault("data_aspect", 1)
        kwargs.setdefault("colorbar", True)
        self._filter_values(filter_field, self.xrfield)
        return self.xrfield.hvplot.contour(x=x, y=y, groupby=groupby, **kwargs)

    def _filter_values(self, filter_field, values):
        if filter_field is None:
            return values

        if filter_field.dim != 1:
            raise ValueError(f"Cannot use {filter_field.dim=}.")

        if self.field.mesh.region not in filter_field.mesh.region:
            raise ValueError(
                "The filter_field region does not contain the field;"
                f" {filter_field.mesh.region=}, {self.field.mesh.region=}."
            )

        if not filter_field.mesh | self.field.mesh:
            filter_field = df.Field(self.field.mesh, dim=1, value=filter_field)
        values.data[filter_field.to_xarray().data == 0] = np.nan
