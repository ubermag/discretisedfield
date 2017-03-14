import numpy as np
import pyvtk

a = [0, 1, 2]
grid = (a, a, a)

structure = pyvtk.RectilinearGrid(*grid)
vtk_data = pyvtk.VtkData(structure)

vtk_data.cell_data.append(pyvtk.Vectors([[0, 1, 3], [1,1,1], [1,1,1],[1,1,1],[1,1,1],[1,1,1],[1,1,1],[10,10,100]], "data"))
vtk_data.cell_data.append(pyvtk.Scalars([0, 1, 2, 3, 4, 5, 6, 7], "data"))

vtk_data.tofile("test_file.vtk")
