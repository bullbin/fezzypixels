import numpy as np

def lin_srgb_to_srgb(lin_srgb : np.ndarray) -> np.ndarray:
	"""Convert a linearized sRGB image to a true sRGB image by applying the gamma curve.

	Args:
		rgb (np.ndarray): Linearized sRGB image. No clamping is applied.

	Returns:
		np.ndarray: sRGB image.
	"""
	# Credit - https://entropymine.com/imageworsener/srgbformula/
	return np.where(lin_srgb <= 0.0031308, lin_srgb * 12.92, (1.055 * (lin_srgb ** (1 / 2.4))) - 0.055)

def srgb_to_lin_srgb(srgb : np.ndarray) -> np.ndarray:
	"""Convert an sRGB image to a linearized sRGB image by removing the gamma curve.

	Args:
		srgb (np.ndarray): sRGB image. No clamping is applied.

	Returns:
		np.ndarray: Linearized sRGB image.
	"""
	# Credit - https://entropymine.com/imageworsener/srgbformula/
	return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)

def lin_srgb_to_oklab(lin_srgb : np.ndarray) -> np.ndarray:
	"""Convert a linearized sRGB image to Oklab.

	Args:
		lin_srgb (np.ndarray): Linearized sRGB image.

	Returns:
		np.ndarray: Oklab image.
	"""
	# Credit - https://bottosson.github.io/posts/oklab/
	r,g,b = lin_srgb[...,0], lin_srgb[...,1], lin_srgb[...,2]

	l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
	m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
	s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

	l_prime = np.cbrt(l)
	m_prime = np.cbrt(m)
	s_prime = np.cbrt(s)

	ok_l = 0.2104542553 * l_prime + 0.7936177850 * m_prime - 0.0040720468 * s_prime
	ok_a = 1.9779984951 * l_prime - 2.4285922050 * m_prime + 0.4505937099 * s_prime
	ok_b = 0.0259040371 * l_prime + 0.7827717662 * m_prime - 0.8086757660 * s_prime
	return np.dstack((ok_l,ok_a,ok_b))

def oklab_to_lin_srgb(oklab : np.ndarray) -> np.ndarray:
	"""Convert an Oklab image to linear sRGB.

	Args:
		oklab (np.ndarray): Oklab image. No clamping is applied.

	Returns:
		np.ndarray: Linearized sRGB image.
	"""
	# Credit - https://bottosson.github.io/posts/oklab/
	ok_l,ok_a,ok_b = oklab[...,0], oklab[...,1], oklab[...,2]
    
	l_prime = ok_l + 0.3963377774 * ok_a + 0.2158037573 * ok_b
	m_prime = ok_l - 0.1055613458 * ok_a - 0.0638541728 * ok_b
	s_prime = ok_l - 0.0894841775 * ok_a - 1.2914855480 * ok_b

	l_prime = l_prime ** 3
	m_prime = m_prime ** 3
	s_prime = s_prime ** 3

	r =  4.0767416621 * l_prime - 3.3077115913 * m_prime + 0.2309699292 * s_prime
	g = -1.2684380046 * l_prime + 2.6097574011 * m_prime - 0.3413193965 * s_prime
	b = -0.0041960863 * l_prime - 0.7034186147 * m_prime + 1.7076147010 * s_prime
	return np.dstack((r,g,b))

def srgb_to_luminance(srgb : np.ndarray) -> np.ndarray:
	"""Convert an sRGB image to a relative luminance map.

	Args:
		srgb (np.ndarray): sRGB image. No clamping is applied.

	Returns:
		np.ndarray: Relative luminance map.
	"""

	# Credit - https://en.wikipedia.org/wiki/Relative_luminance
	lin = srgb_to_lin_srgb(srgb)
	return 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]