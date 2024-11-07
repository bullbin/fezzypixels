import numpy as np
cimport numpy as np

import cython
cimport cython

from cython.parallel import prange, parallel
from numpy.math cimport INFINITY
from libc.math cimport cbrt

np.import_array()

DTYPE_FLOAT = np.float32
DTYPE_UINT = np.uint32

ctypedef np.float32_t DTYPE_FLOAT_t
ctypedef np.uint32_t DTYPE_UINT_t

cdef DTYPE_FLOAT_t RGB2XYZ_X_R = 0.4124564
cdef DTYPE_FLOAT_t RGB2XYZ_X_G = 0.3575761
cdef DTYPE_FLOAT_t RGB2XYZ_X_B = 0.1804375
cdef DTYPE_FLOAT_t RGB2XYZ_Y_R = 0.2126729
cdef DTYPE_FLOAT_t RGB2XYZ_Y_G = 0.7151522
cdef DTYPE_FLOAT_t RGB2XYZ_Y_B = 0.0721750
cdef DTYPE_FLOAT_t RGB2XYZ_Z_R = 0.0193339
cdef DTYPE_FLOAT_t RGB2XYZ_Z_G = 0.1191920
cdef DTYPE_FLOAT_t RGB2XYZ_Z_B = 0.9503041

cdef DTYPE_FLOAT_t X_D65 = 0.9504492182750991
cdef DTYPE_FLOAT_t Z_D65 = 1.0889166484304715

cdef DTYPE_FLOAT_t XYZ_EPSILON = 216.0/24389.0
cdef DTYPE_FLOAT_t XYZ_KAPPA = 24389.0/27.0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef DTYPE_FLOAT_t rgb_to_lab_trans_f(DTYPE_FLOAT_t c) noexcept nogil:
	if c <= XYZ_EPSILON:
		return (XYZ_KAPPA * c + 16.0) / 116.0
	return cbrt(c)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void rgb_to_lab(DTYPE_FLOAT_t r, DTYPE_FLOAT_t g, DTYPE_FLOAT_t b, DTYPE_FLOAT_t* l, DTYPE_FLOAT_t* a, DTYPE_FLOAT_t* b2) noexcept nogil:
	cdef DTYPE_FLOAT_t x, y, z;
	x = RGB2XYZ_X_R * r + RGB2XYZ_X_G * g + RGB2XYZ_X_B * b
	y = RGB2XYZ_Y_R * r + RGB2XYZ_Y_G * g + RGB2XYZ_Y_B * b
	z = RGB2XYZ_Z_R * r + RGB2XYZ_Z_G * g + RGB2XYZ_Z_B * b

	x = rgb_to_lab_trans_f(x / X_D65)
	y = rgb_to_lab_trans_f(y)
	z = rgb_to_lab_trans_f(z / Z_D65)

	l[0] = 116.0 * y - 16.0
	a[0] = 500.0 * (x - y)
	b2[0] = 200.0 * (y - z)

@cython.boundscheck(False)
@cython.wraparound(False)
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
cdef void candidate_internal(DTYPE_UINT_t[:,:] candidate,
						DTYPE_FLOAT_t[:,:,:] target,
						DTYPE_FLOAT_t[:,:] palette_lin,
						DTYPE_FLOAT_t[:,:] palette_lab,
						DTYPE_FLOAT_t[:,:,:] e,
						float q,
						bint use_oklab) noexcept nogil:

	cdef DTYPE_FLOAT_t rgb_r,rgb_g,rgb_b,l,a,b,pl, pa, pb, dl, da, db
	cdef DTYPE_FLOAT_t best_similarity = INFINITY
	cdef DTYPE_FLOAT_t sim
	cdef int x, y, i, d, best_idx, idx_palette

	# Convert palette to either CIELAB or OKLAB
	for i in range(palette_lin.shape[0]):
		rgb_r = palette_lin[i,0]
		rgb_g = palette_lin[i,1]
		rgb_b = palette_lin[i,2]

		if use_oklab:
			rgb_to_oklab(rgb_r,rgb_g,rgb_b,&l,&a,&b)
		else:
			rgb_to_lab(rgb_r,rgb_g,rgb_b,&l,&a,&b)

		palette_lab[i,0] = l
		palette_lab[i,1] = a
		palette_lab[i,2] = b

	# Solve candidates across image
	for y in range(target.shape[0]):
		for x in range(target.shape[1]):

			# Compute t value in linear RGB
			rgb_r = target[y,x,0] + (e[y,x,0] * q)
			rgb_g = target[y,x,1] + (e[y,x,1] * q)
			rgb_b = target[y,x,2] + (e[y,x,2] * q)

			# Convert to LAB for color comparison
			if use_oklab:
				rgb_to_oklab(rgb_r,rgb_g,rgb_b,&l,&a,&b)
			else:
				rgb_to_lab(rgb_r,rgb_g,rgb_b,&l,&a,&b)

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
			
			candidate[y,x] = best_idx

			# Update error accumulation
			for d in range(3):
				e[y,x,d] += (target[y,x,d] - palette_lin[best_idx,d])

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
	quantization_err    = np.zeros([lin_srgb.shape[0], lin_srgb.shape[1], lin_srgb.shape[2]], dtype=DTYPE_FLOAT)
	palette_lab         = np.zeros([palette_lin.shape[0], palette_lin.shape[1]], dtype=DTYPE_FLOAT)

	for i in range(n):
		candidate_internal(candidate[:,:,i], lin_srgb, palette_lin, palette_lab, quantization_err, q, True)

	return candidate