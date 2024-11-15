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
cdef DTYPE_UINT_t get_closest_shade_lab(DTYPE_FLOAT_t[:,:] pal_lab, DTYPE_FLOAT_t l, DTYPE_FLOAT_t a, DTYPE_FLOAT_t b) noexcept nogil:
	cdef DTYPE_UINT_t best_idx, idx
	cdef float best_err = INFINITY
	cdef float curr_err

	best_err = INFINITY

	for idx in range(pal_lab.shape[0]):
		curr_err = (pal_lab[idx][0] - l) ** 2 + (pal_lab[idx][1] - a) ** 2 + (pal_lab[idx][2] - b) ** 2

		if curr_err < best_err:
			best_idx = idx
			best_err = curr_err
	
	return best_idx

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void dither_internal(DTYPE_UINT_t[:,:] out_pal_idx,
						  DTYPE_FLOAT_t[:,:,:] img_lin_rgb,
						  DTYPE_FLOAT_t[:,:] pal_lin_rgb,
						  DTYPE_FLOAT_t[:,:] pal_lab,
						  DTYPE_UINT_t[:,:] mask,
						  DTYPE_UINT_t[:,:] offsets,
						  DTYPE_FLOAT_t[:] err_weights,
						  bint serpentine,
						  bint skip_serp,
						  float weight) noexcept nogil:
	
	cdef bint do_serp
	cdef int y, x, idx_mat, dest_y, dest_x
	cdef DTYPE_UINT_t idx_closest
	cdef float err_r,err_g,err_b,l,a,b,pix_err

	for y in range(out_pal_idx.shape[0]):
		do_serp = serpentine and y % 2 == 1

		for x in range(out_pal_idx.shape[1]):

			if do_serp:
				x = out_pal_idx.shape[1] - x - 1
			
			rgb_to_oklab(img_lin_rgb[y,x,0],img_lin_rgb[y,x,1],img_lin_rgb[y,x,2],&l,&a,&b)
			idx_closest = get_closest_shade_lab(pal_lab,l,a,b)

			out_pal_idx[y,x] = idx_closest
			
			err_r = img_lin_rgb[y,x,0] - pal_lin_rgb[idx_closest,0]
			err_g = img_lin_rgb[y,x,1] - pal_lin_rgb[idx_closest,1]
			err_b = img_lin_rgb[y,x,2] - pal_lin_rgb[idx_closest,2]

			if skip_serp and do_serp:
				continue
			
			for idx_mat in range(offsets.shape[0]):
				dest_y = y + offsets[idx_mat][0]

				if do_serp:
					dest_x = x - offsets[idx_mat][1]
				else:
					dest_x = x + offsets[idx_mat][1]
				
				if 0 <= dest_y < out_pal_idx.shape[0] and 0 <= dest_x < out_pal_idx.shape[1]:
					pix_err = weight * err_weights[idx_mat] * mask[dest_y,dest_x]
					img_lin_rgb[dest_y,dest_x,0] += err_r * pix_err
					img_lin_rgb[dest_y,dest_x,1] += err_g * pix_err
					img_lin_rgb[dest_y,dest_x,2] += err_b * pix_err

cpdef np.ndarray[DTYPE_UINT_t, ndim=2] dither_to_palette(np.ndarray[DTYPE_FLOAT_t, ndim=3] image_lin,
													   np.ndarray[DTYPE_FLOAT_t, ndim=2] palette_lin,
													   np.ndarray[DTYPE_FLOAT_t, ndim=2] palette_lab,
													   np.ndarray[DTYPE_UINT_t, ndim=2] mask,
													   np.ndarray[DTYPE_UINT_t, ndim=2] offsets,
													   np.ndarray[DTYPE_FLOAT_t, ndim=1] err_weights,
													   bint serpentine,
													   bint skip_serp,
													   float weight):

	cdef np.ndarray[DTYPE_FLOAT_t, ndim=3] image_copy = np.copy(image_lin)
	cdef np.ndarray[DTYPE_UINT_t, ndim=2] output = np.zeros([image_copy.shape[0], image_copy.shape[1]], dtype=DTYPE_UINT)

	dither_internal(output, image_lin, palette_lin, palette_lab, mask, offsets, err_weights, serpentine, skip_serp, weight)

	return output