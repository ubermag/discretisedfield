from .field import Field


class CellField(Field):
    def __call__(self, point):
        return self.array[self.mesh.point2index(point)]

    # diff, integrate depending on how we calculate those for the VertexField

    def __getitem__(self, item):
        submesh = self.mesh[item]

        index_min = self.mesh.point2index(
            submesh.index2point((0,) * submesh.region.ndim)
        )
        index_max = np.add(index_min, submesh.n)
        slices = [slice(i, j) for i, j in zip(index_min, index_max)]
        return self.__class__(
            submesh,
            nvdim=self.nvdim,
            value=self.array[tuple(slices)],
            vdims=self.vdims,
            unit=self.unit,
            valid=self.valid[tuple(slices)],
            vdim_mapping=self.vdim_mapping,
        )
