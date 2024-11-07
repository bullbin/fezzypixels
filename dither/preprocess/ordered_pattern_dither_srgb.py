import numpy as np
from dither.shift import rgb888_to_rgb555_scale, rgb555_to_norm
from dither.color import srgb_to_luminance, srgb_to_lin_srgb, lin_srgb_to_srgb
from typing import Optional
from os.path import join, dirname
from dither.helper import load_precompute_bn_l_image

# Blue noise texture. CC0, credit - https://momentsingraphics.de/BlueNoise.html
PATH_HDR_TEMPLATE : str = join(dirname(__file__), "HDR_L_1.png")
CACHE_LIN_BLUE_NOISE : Optional[np.ndarray] = None

def pattern_dither_to_srgb555(image_srgb_norm : np.ndarray, n : int = 32, q : float = 0.5) -> np.ndarray:
    """Apply Thomas Knoll's 'Pattern Dithering' algorithm to quantize an image down to sRGB555.

    This is a modification of dither.pattern.pattern_dither_srgb and works entirely in linearized
    sRGB. Instead of comparing to any palette, at each step the algorithm rounds the colors to the
    closest sRGB555 counterpart and recomputes error. The intent of this method is to produce variations
    of shades for use in palette generation so the thresholding pattern is fixed to blue noise. This
    is a recommended alternative to error-diffusion dithering for preprocessing because it can be
    entirely vectorized so is very fast.

    Args:
        image_srgb_norm (np.ndarray): Image in normalized sRGB color.
        n (int, optional): Number of candidates to find for each pixel. Larger costs more but increases depth of grain. Defaults to 32.
        q (float, optional): Threshold mode for final dithering step. Changes texture of output. Defaults to 0.5.

    Raises:
        FileNotFoundError: Raised if noise pattern image could not be loaded for any reason.

    Returns:
        np.ndarray: Image in normalized sRGB555 color.
    """
    global CACHE_LIN_BLUE_NOISE, PATH_HDR_TEMPLATE
    # For each pixel, compute a list of possible candidates
    # Each candidate down the list is increasingly noisy (because of accumulated error) 
    #     but still relevant to original, like it has been dithered

    if CACHE_LIN_BLUE_NOISE is None:
        CACHE_LIN_BLUE_NOISE = load_precompute_bn_l_image(PATH_HDR_TEMPLATE)
    
    if CACHE_LIN_BLUE_NOISE is None:
        raise FileNotFoundError("Grayscale blue noise texture not found at '%s'" % PATH_HDR_TEMPLATE)

    candidate_colors = np.zeros((image_srgb_norm.shape[0], image_srgb_norm.shape[1], n, 3), dtype=np.float32)

    c = srgb_to_lin_srgb(image_srgb_norm)
    e = np.zeros_like(image_srgb_norm)
    for i in range(n):
        t = c + (e * q)
        current_assigned_srgb = rgb555_to_norm(rgb888_to_rgb555_scale(lin_srgb_to_srgb(t)))
        current_assigned_lin = srgb_to_lin_srgb(current_assigned_srgb)
        candidate_colors[..., i, :] = current_assigned_srgb
        e += (c - current_assigned_lin)
    
    candidate_colors_lumi = srgb_to_luminance(candidate_colors)
    idx_sorted = candidate_colors_lumi.argsort(axis=2)

    # Sort color array along luminance, indexing isn't nice. I don't get this but it works
    idx_y, idx_x = np.meshgrid(np.arange(image_srgb_norm.shape[0]), np.arange(image_srgb_norm.shape[1]), indexing='ij')
    candidate_colors = candidate_colors[idx_y[:, :, np.newaxis],
                                        idx_x[:, :, np.newaxis],
                                        idx_sorted,
                                        :]
    
    # Tile blue noise to meet input size
    repeat_y = int(np.ceil(candidate_colors.shape[0] / CACHE_LIN_BLUE_NOISE.shape[0]))
    repeat_x = int(np.ceil(candidate_colors.shape[1] / CACHE_LIN_BLUE_NOISE.shape[1]))
    shift = np.tile(CACHE_LIN_BLUE_NOISE, (repeat_y, repeat_x))[:candidate_colors.shape[0], :candidate_colors.shape[1]]

    # Vectorize candidate and output
    candidate = np.clip(np.floor(shift * n).astype(np.uint32), 0, n - 1)
    output = candidate_colors[idx_y,idx_x,candidate]
    return output