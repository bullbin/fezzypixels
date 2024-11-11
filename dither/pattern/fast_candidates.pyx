import numpy as np
cimport numpy as np

import cython
cimport cython

from cython.parallel import prange, parallel
from numpy.math cimport INFINITY
from libc.math cimport cbrt

np.import_array()

DTYPE_FLOAT = np.float32
DTYPE_UINT = np.uint8

ctypedef np.float32_t DTYPE_FLOAT_t
ctypedef np.uint8_t DTYPE_UINT_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void rgb_to_oklab(DTYPE_FLOAT_t r, DTYPE_FLOAT_t g, DTYPE_FLOAT_t b, DTYPE_FLOAT_t* ok_l, DTYPE_FLOAT_t* ok_a, DTYPE_FLOAT_t* ok_b) noexcept nogil:
	cdef DTYPE_FLOAT_t l, m, s, l_prime, m_prime, s_prime;

	l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
	m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
	s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;

	l_prime = cbrt(l);
	m_prime = cbrt(m);
	s_prime = cbrt(s);

	ok_l[0] = 0.2104542553 * l_prime + 0.7936177850 * m_prime - 0.0040720468 * s_prime
	ok_a[0] = 1.9779984951 * l_prime - 2.4285922050 * m_prime + 0.4505937099 * s_prime
	ok_b[0] = 0.0259040371 * l_prime + 0.7827717662 * m_prime - 0.8086757660 * s_prime

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void candidate_internal(DTYPE_UINT_t[:,:,:] candidate,
							DTYPE_FLOAT_t[:,:,:] target,
							DTYPE_FLOAT_t[:,:] palette_lin,
							DTYPE_FLOAT_t[:,:] palette_lab,
							float q) noexcept nogil:

	cdef int x, y, d, n, i, best_idx, idx_palette
	cdef DTYPE_FLOAT_t rgb_r, rgb_g, rgb_b, pl, pa, pb, dl, da, db, sim, l, a, b, best_similarity, er, eg, eb

	# Convert palette to OKLAB
	for i in range(palette_lin.shape[0]):
		rgb_to_oklab(palette_lin[i,0], palette_lin[i,1], palette_lin[i,2], &l, &a, &b)
		palette_lab[i,0] = l
		palette_lab[i,1] = a
		palette_lab[i,2] = b

	# Solve candidates across image
	for y in prange(candidate.shape[0]):
		for x in range(candidate.shape[1]):
			er = 0
			eg = 0
			eb = 0

			for n in range(candidate.shape[2]):
				# HACK - Required or LAB breaks when using prange...
				l = 0
				a = 0
				b = 0

				# Compute t value in linear RGB (same as before)
				rgb_r = target[y,x,0] + (er * q)
				rgb_g = target[y,x,1] + (eg * q)
				rgb_b = target[y,x,2] + (eb * q)

				# Convert to LAB for color comparison (same as before)
				rgb_to_oklab(rgb_r, rgb_g, rgb_b, &l, &a, &b)

				best_similarity = INFINITY
				best_idx = 0

				# Brute force palette to find closest color
				for idx_palette in range(palette_lab.shape[0]):
					pl = palette_lab[idx_palette,0]
					pa = palette_lab[idx_palette,1]
					pb = palette_lab[idx_palette,2]

					dl = (pl - l) ** 2
					da = (pa - a) ** 2
					db = (pb - b) ** 2
					sim = dl + da + db
					if sim < best_similarity:
						best_idx = idx_palette
						best_similarity = sim

				candidate[y,x,n] = best_idx

				# Update error accumulation
				er = er + (target[y,x,0] - palette_lin[best_idx,0])
				eg = eg + (target[y,x,1] - palette_lin[best_idx,1])
				eb = eb + (target[y,x,2] - palette_lin[best_idx,2])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[DTYPE_UINT_t, ndim=3] get_candidates(np.ndarray[DTYPE_FLOAT_t, ndim=3] lin_srgb,
													  np.ndarray[DTYPE_FLOAT_t, ndim=2] palette_lin,
													  unsigned int n,
													  float q):

	cdef np.ndarray[DTYPE_UINT_t, ndim=3] candidate
	cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] quantization_err
	cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] palette_lab

	candidate           = np.zeros((lin_srgb.shape[0], lin_srgb.shape[1], n), dtype=DTYPE_UINT)
	palette_lab         = np.zeros([palette_lin.shape[0], palette_lin.shape[1]], dtype=DTYPE_FLOAT)

	candidate_internal(candidate, lin_srgb, palette_lin, palette_lab, q)

	return candidate