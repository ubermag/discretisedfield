"""Holoviews-based plotting."""
import hvplot.xarray  # noqa: F401
import numpy as np
import xarray as xr


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
        # vector_kw.setdefault("use_color", False)
        # vector_kw.setdefault("colorbar", False)

        if self.field.dim == 1:
            return self.field.hvplot.scalar(slider, **scalar_kw)
        elif self.field.dim == 2:
            return self.field.hvplot.vector(slider, **vector_kw)
        elif self.field.dim == 3:
            vector_kw.setdefault("use_color", False)
            scalar = getattr(self.field, slider).hvplot.scalar(slider, **scalar_kw)
            vector = self.field.hvplot.vector(slider, **vector_kw)
            return scalar * vector

    def scalar(self, slider, comp=None, **kwargs):
        """Plot the scalar field on a plane."""
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))

        field = self.xrfield if comp is None else self.xrfield.sel(comp=comp)
        return field.hvplot(x=x, y=y, groupby=slider, **kwargs)

    def vector(self, slider, use_color=False, color_field=None, **kwargs):
        """Plot the vector field on a plane."""
        if use_color:
            print("Use_color and color_field are not yet supported.")
            use_color = False
        if slider not in "xyz":
            raise ValueError(f"Unknown value {slider=}; must be 'x', 'y', or 'z'.")
        x = min("xyz".replace(slider, ""))
        y = max("xyz".replace(slider, ""))
        ip_vector = xr.Dataset(
            {
                "angle": np.arctan2(
                    self.xrfield.sel(comp=y),
                    self.xrfield.sel(comp=x),
                    where=self.field.norm.array.squeeze() != 0,
                    out=np.full(self.field.mesh.n, np.nan),
                ),
                "mag": np.sqrt(
                    self.xrfield.sel(comp=x) ** 2 + self.xrfield.sel(comp=y) ** 2
                ),
            }
        )
        plot_kw = dict(x=x, y=y, angle="angle", mag="mag")
        if use_color:
            plot_kw["color"] = "color_comp"
        vectors = ip_vector.hvplot.vectorfield(**plot_kw, **kwargs).opts(
            magnitude="mag"
        )

        if use_color:  # TODO adding the color component does not work
            cfield = (
                color_field.to_xarray()
                if color_field
                else self.xrfield.sel(comp=slider)
            )
            for key, val in vectors.data.items():
                vectors.data[key] = val.add_dimension(
                    "color_comp", -1, cfield, vdim=True
                )

        return vectors
