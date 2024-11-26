import numpy as np
cimport numpy as np

import cython
cimport cython

from numpy.math cimport INFINITY

np.import_array()

DTYPE_FLOAT = np.float32
DTYPE_UINT = np.uint32

ctypedef np.float32_t DTYPE_FLOAT_t
ctypedef np.uint32_t DTYPE_UINT_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void do_k_steps_2d(DTYPE_FLOAT_t[:,:] centroids, DTYPE_FLOAT_t[:,:] data_2d, int iterations):
    cdef DTYPE_FLOAT_t[:,:] totals = np.zeros_like(centroids)
    cdef DTYPE_UINT_t[:] counts = np.zeros((centroids.shape[0]), dtype=DTYPE_UINT)

    cdef int i, j, idx_centroid
    cdef DTYPE_FLOAT_t[:] sample
    cdef DTYPE_FLOAT_t delta, best_delta
    cdef int idx_best_delta

    for i in range(iterations):
        totals[:] = 0
        counts[:] = 0
        
        for sample in data_2d:
            
            best_delta = INFINITY

            for idx_centroid in range(centroids.shape[0]):

                delta = 0
                
                for j in range(data_2d.shape[1]):
                    delta += (centroids[idx_centroid,j] - sample[j]) ** 2
                
                if delta < best_delta:
                    best_delta = delta
                    idx_best_delta = idx_centroid
            
            counts[idx_best_delta] += 1

            for j in range(data_2d.shape[1]):
                totals[idx_best_delta,j] += sample[j]
    
        for idx_centroid in range(centroids.shape[0]):
            if counts[idx_centroid] > 0:
                for j in range(centroids.shape[1]):
                    centroids[idx_centroid,j] = totals[idx_centroid,j] / counts[idx_centroid]