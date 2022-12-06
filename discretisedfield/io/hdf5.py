import contextlib
import datetime

import h5py
import numpy as np

import discretisedfield as df


class _RegionIOHDF5:
    _h5_attrs = ("pmin", "pmax", "dims", "ndim", "units", "tolerance_factor")

    def _h5_save(self, h5_region):
        for attr in self._h5_attrs:
            h5_region.attrs[attr] = getattr(self, attr)

    @classmethod
    def _h5_load(cls, h5_region):
        return cls(**{attr: h5_region.attrs[attr] for attr in cls._h5_attrs})


class _MeshIOHDF5:
    def _h5_save(self, h5_mesh):
        h5_region = h5_mesh.create_group("region")
        self.region._h5_save(h5_region)

        for attr in ["n", "bc"]:
            h5_mesh.attrs[attr] = getattr(self, attr)

        h5_mesh.create_dataset("subregion_names", data=list(self.subregions.keys()))
        h5_mesh_subregions = h5_mesh.create_dataset(
            "subregions", (len(self.subregions), 2 * self.region.ndim)
        )
        for i, subregion in enumerate(self.subregions.values()):
            h5_mesh_subregions[i] = [*subregion.pmin, *subregion.pmax]

    @classmethod
    def _h5_load(cls, h5_mesh):
        region = df.Region._h5_load(h5_mesh["region"])
        subregions = {
            name.decode("utf-8"): df.Region(
                p1=data[: region.ndim], p2=data[region.ndim :]
            )
            for name, data in zip(h5_mesh["subregion_names"], h5_mesh["subregions"])
        }
        return cls(region=region, n=h5_mesh.attrs["n"], subregions=subregions)


class _FieldIOHDF5:
    def _to_hdf5(self, filename):
        utc_now = datetime.datetime.utcnow().isoformat(timespec="seconds")
        with h5py.File(filename, "w") as f:
            for attribute, value in [
                ("ubermag-hdf5-file-version", "0.1"),
                ("discretisedfield.__version__", df.__version__),
                ("file-creation-time-UTC", utc_now),
                ("type", "discretisedfield.Field"),
            ]:
                f.attrs[attribute] = value

            h5_field = f.create_group("field")
            self._h5_save_structure(h5_field)
            # TODO is 'data' or 'array' a better name for the following dataset?
            h5_field_data = h5_field.create_dataset(
                "data", (*self.mesh.n, self.nvdim), dtype=self.array.dtype
            )
            self._h5_save_data(h5_field_data, slice(None))

    def _h5_save_structure(self, h5_field):
        # TODO better method name?
        h5_mesh = h5_field.create_group("mesh")
        self.mesh._h5_save(h5_mesh)

        h5_field.attrs["nvdim"] = self.nvdim
        if self.nvdim > 1:  # TODO scalar fields have no vdims, consider changing
            h5_field.attrs["vdims"] = self.vdims
        h5_field.attrs["unit"] = str(self.unit)

    def _h5_save_data(self, h5_field_data, location):
        h5_field_data[location] = self.array

    @classmethod
    def _from_hdf5(cls, filename):
        with h5py.File(filename, "r") as f:
            if "ubermag-hdf5-file-version" not in f.attrs:
                return cls._h5_legacy_load_field(f, filename)
            if f.attrs["type"] != "discretisedfield.Field":
                raise ValueError(
                    f"{cls} cannot read hdf5 files with type {f.attrs['type']}."
                )
            assert f.attrs["ubermag-hdf5-file-version"] == "0.1"
            return cls._h5_load_field(f["field"], slice(None))

    @classmethod
    def _h5_load_field(cls, h5_field, data_location):
        vdims = h5_field.attrs["vdims"] if h5_field.attrs["nvdim"] > 1 else None
        return cls(
            mesh=df.Mesh._h5_load(h5_field["mesh"]),
            nvdim=h5_field.attrs["nvdim"],
            value=h5_field["data"][data_location],
            vdims=vdims,
            unit=h5_field.attrs["unit"],
        )

    @classmethod
    def _h5_legacy_load_field(cls, h5_file, filename):
        # reads old hdf5 files written prior to introducing ubermag-hdf5-file-version
        p1 = h5_file["field/mesh/region/p1"]
        p2 = h5_file["field/mesh/region/p2"]
        n = np.array(h5_file["field/mesh/n"]).tolist()
        dim = np.array(h5_file["field/dim"]).tolist()
        array = h5_file["field/array"]

        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        with contextlib.suppress(FileNotFoundError):
            mesh.load_subregions(filename)

        return cls(mesh, dim=dim, value=array[:])
