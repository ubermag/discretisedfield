import numpy as np

cimport cython
cimport numpy as cnp

#from cython.parallel import prange

# From the cython documentation:
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. Therefore we recommend
# always calling "import_array" whenever you "cimport numpy"
cnp.import_array()

# output from: findiff.coefficients(1, acc=2)
# {'center': {'coefficients': array([-0.5,  0. ,  0.5]),
#             'offsets': array([-1,  0,  1]),
#             'accuracy': 2},
#  'forward': {'coefficients': array([-1.5,  2. , -0.5]),
#              'offsets': array([0, 1, 2]),
#              'accuracy': 2},
#  'backward': {'coefficients': array([ 0.5, -2. ,  1.5]),
#               'offsets': array([-2, -1,  0]),
#               'accuracy': 2}}

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def fast_diff_order_1(double[:,:,::1] array, cnp.npy_bool[:,::1] valid, double dx):
    loop_max = array.shape[0]
    vdim_max = array.shape[1]
    cdef Py_ssize_t diff_max = array.shape[2]
    cdef cnp.ndarray[cnp.float_t, ndim=3] res = np.zeros([loop_max, vdim_max, diff_max], dtype=np.double)
    cdef double[:, :, ::1] res_view = res

    last = diff_max - 1
    cdef Py_ssize_t i, d, j
    # for j in prange(loop_max, nogil=True):
    for j in range(loop_max):
        for d in range(vdim_max):
            for i in range(diff_max):
                if i == 0:
                    if valid[j, i] and valid[j, i+1] and valid[j, i+2]:
                        res_view[j, d, i] = (-1.5*array[j, d, i] + 2.*array[j, d, i+1] - 0.5*array[j, d, i+2]) / dx
                    else:
                        res_view[j, d, i] = 0.  # TODO How do we treat missing data?
                elif i == last:
                    if valid[j, i] and valid[j, i-1] and valid[j, i-2]:
                        res_view[j, d, i] = (0.5*array[j, d, i-2] - 2.*array[j, d, i-1] + 1.5*array[j, d, i]) / dx
                    else:
                        res_view[j, d, i] = 0.  # TODO How do we treat missing data?
                else:
                    if valid[j, i-1] and valid[j, i] and valid[j, i+1]:
                        res_view[j, d, i] = (-0.5*array[j, d, i-1] + 0.5*array[j, d, i+1]) / dx
                    elif i < diff_max - 2 and valid[j, i] and valid[j, i+1] and valid[j, i+2]:
                        res_view[j, d, i] = (-1.5*array[j, d, i] + 2.*array[j, d, i+1] - 0.5*array[j, d, i+2]) / dx
                    elif i > 1 and valid[j, i] and valid[j, i-1] and valid[j, i-2]:
                        res_view[j, d, i] = (0.5*array[j, d, i-2] - 2.*array[j, d, i-1] + 1.5*array[j, d, i]) / dx
                    else:
                        res_view[j, d, i] = 0.  # TODO How do we treat missing data?
    return res


