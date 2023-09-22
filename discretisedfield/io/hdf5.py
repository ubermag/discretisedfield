import contextlib
import datetime

import h5py
import numpy as np

import discretisedfield as df


class _RegionIO_HDF5:
    __slots__ = []

    _h5_attrs = ("pmin", "pmax", "dims", "ndim", "units", "tolerance_factor")

    def _h5_save(self, h5_region: h5py.Group):
        for attr in self._h5_attrs:
            h5_region.attrs[attr] = getattr(self, attr)

    @classmethod
    def _h5_load(cls, h5_region: h5py.Group):
        return cls(**{attr: h5_region.attrs[attr] for attr in cls._h5_attrs})


class _MeshIO_HDF5:
    __slots__ = []

    def _h5_save(self, h5_mesh: h5py.Group):
        """
        Create a group for the underlying region and call the region save method.
        Save mesh attributes ``n`` and ``bc``. If subregions are defined for the mesh,
        these are saved into two datasets ``subregion_names`` and ``subregions``. The
        latter contains pmin and pmax as 2*ndim vectors. The two datasets are related
        via position. If no subregions are defined, the datasets will not be created.
        """
        h5_region = h5_mesh.create_group("region")
        self.region._h5_save(h5_region)

        for attr in ["n", "bc"]:
            h5_mesh.attrs[attr] = getattr(self, attr)

        if len(self.subregions) > 0:
            h5_mesh.create_dataset("subregion_names", data=list(self.subregions.keys()))
            h5_mesh_subregions = h5_mesh.create_dataset(
                "subregions",
                (len(self.subregions), 2 * self.region.ndim),
                dtype=self.region.pmin.dtype,
            )
            for i, subregion in enumerate(self.subregions.values()):
                h5_mesh_subregions[i] = [*subregion.pmin, *subregion.pmax]

    @classmethod
    def _h5_load(cls, h5_mesh: h5py.Group):
        region = df.Region._h5_load(h5_mesh["region"])
        if "subregions" in h5_mesh.keys():
            subregions = {
                name.decode("utf-8"): df.Region(
                    p1=data[: region.ndim], p2=data[region.ndim :]
                )
                for name, data in zip(h5_mesh["subregion_names"], h5_mesh["subregions"])
            }
        else:
            subregions = {}
        return cls(
            region=region,
            n=h5_mesh.attrs["n"],
            bc=h5_mesh.attrs["bc"],
            subregions=subregions,
        )


class _FieldIO_HDF5:
    __slots__ = []

    def _to_hdf5(self, filename):
        """Save a single field in a new hdf5 file."""
        utc_now = datetime.datetime.utcnow().isoformat(timespec="seconds")
        with h5py.File(filename, "w") as f:
            f.attrs["ubermag-hdf5-file-version"] = "0.1"
            f.attrs["discretisedfield.__version__"] = df.__version__
            f.attrs["file-creation-time-UTC"] = utc_now
            f.attrs["type"] = "discretisedfield.Field"

            h5_field = f.create_group("field")
            h5_field_data = self._h5_save_structure(
                h5_field, data_shape=(*self.mesh.n, self.nvdim)
            )

            self._h5_save_data(h5_field_data, slice(None))

    def _h5_save_structure(self, h5_field: h5py.Group, data_shape: tuple):
        """
        Save the 'field structure', that is the mesh, field attributes and valid, into
        an existing hdf5 group and create an ``h5py.Dataset`` for the field data with a
        given ``data_shape``. The shape can have additional dimensions, e.g. an extra
        first dimension for a time series that should be stored in the hdf5 file. Valid
        is always static and does not support extra dimensions. The field data is NOT
        saved.

        The ``h5py.Dataset`` that will store the field values is returned.
        """
        h5_mesh = h5_field.create_group("mesh")
        self.mesh._h5_save(h5_mesh)

        h5_field.attrs["nvdim"] = self.nvdim
        h5_field.attrs["vdims"] = self.vdims if self.vdims is not None else "None"
        h5_field.attrs["unit"] = str(self.unit)

        # empty dataset that can later contain field.array
        h5_field_data = h5_field.create_dataset(
            "array", data_shape, dtype=self.array.dtype
        )

        h5_field.create_dataset("valid", data=self.valid, dtype=np.bool_)

        return h5_field_data

    def _h5_save_data(self, h5_field_data: h5py.Dataset, location):
        """
        Save field data into an existing hdf5 dataset at a given ``location`` inside the
        dataset. For a single field in that dataset the ``location``` refers to the
        whole dataset (``slice(None)``). Other values for ``location`` are useful to
        save a single field into a bigger dataset, e.g. a dataset meant to contain a
        time series.
        """
        h5_field_data[location] = self.array

    @classmethod
    def _from_hdf5(cls, filename):
        """Read an hdf5 file containing a single Field object."""
        with h5py.File(filename, "r") as f:
            if "ubermag-hdf5-file-version" not in f.attrs:
                return cls._h5_legacy_load_field(f, filename)
            if f.attrs["type"] != "discretisedfield.Field":
                raise ValueError(
                    f"{cls} cannot read hdf5 files with type {f.attrs['type']}."
                )
            # check for the correct version; in the future multiple code paths may
            # be required to handle different versions
            assert f.attrs["ubermag-hdf5-file-version"] in ["0.1"]
            return cls._h5_load_field(f["field"], slice(None))

    @classmethod
    def _h5_load_field(cls, h5_field: h5py.Group, data_location):
        """
        Load a Field from an hdf5 group containing a single field. The hdf5 dataset
        ``array`` containing the field data can contain data for multiple fields on the
        same mesh (e.g. in a time series). The correct part of the array can be selected
        with ``data_location``.
        """
        vdims = h5_field.attrs["vdims"]
        if isinstance(vdims, str) and vdims == "None":
            vdims = None
        return cls(
            mesh=df.Mesh._h5_load(h5_field["mesh"]),
            nvdim=h5_field.attrs["nvdim"],
            value=h5_field["array"][data_location],
            vdims=vdims,
            unit=h5_field.attrs["unit"],
            valid=h5_field["valid"],
        )

    @classmethod
    def _h5_legacy_load_field(cls, h5_file: h5py.Group, filename):
        """
        Reads old hdf5 files written prior to introducing ubermag-hdf5-file-version.
        These lack most of the additional metadata defined in the new standard.
        """
        p1 = tuple(h5_file["field/mesh/region/p1"])
        p2 = tuple(h5_file["field/mesh/region/p2"])
        n = np.array(h5_file["field/mesh/n"]).tolist()
        dim = np.array(h5_file["field/dim"]).tolist()
        array = h5_file["field/array"]

        mesh = df.Mesh(region=df.Region(p1=p1, p2=p2), n=n)
        with contextlib.suppress(FileNotFoundError):
            mesh.load_subregions(filename)

        return cls(mesh, dim=dim, value=array[:])
