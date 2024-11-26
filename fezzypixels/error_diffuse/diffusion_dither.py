import numpy as np
from fezzypixels.color import srgb_to_lin_srgb, lin_srgb_to_oklab
from .diffusion_weights import get_dither_weighting, DitheringWeightingMode
from .lab_err_diff import dither_to_palette
from typing import Optional

def error_diffusion_dither_srgb(image_srgb_norm : np.ndarray, palette_srgb : np.ndarray,
                                diffuse_mode : DitheringWeightingMode = DitheringWeightingMode.FLOYD_STEINBERG,
                                serpentine : bool = True, skip_mask : Optional[np.ndarray] = None,
                                error_weight : float = 1.0, serp_skip : bool = False) -> np.ndarray:
    """Dither an image using error-diffusion dithering.

    Args:
        image_srgb_norm (np.ndarray): Image in normalized sRGB color.
        palette_srgb (np.ndarray): Palette in normalized sRGB color.
        diffuse_mode (DitheringWeightingMode, optional): Diffusion matrix to use. Defaults to DitheringWeightingMode.FLOYD_STEINBERG.
        serpentine (bool, optional): Whether to swap diffusion direction per-row. Reduces diffusion artifacts. Defaults to True.
        skip_mask (np.ndarray, optional): Mask of where to skip error diffusion. Must be same size as input image. True (any value != 0) will skip diffusion on image. None to diffuse everywhere. Defaults to None.
        error_weight (float, optional): Scale factor of weighting. 1.0 is normal. Must be [0,1]. Defaults to 1.
        serp_skip (bool, optional): Skip diffusing on serpentine lines. May produce more pleasing images. Defaults to False.

    Returns:
        np.ndarray: Image as indices into palette.
    """

    assert 0 <= error_weight <= 1

    if not(skip_mask is None):
        assert skip_mask.shape[0] == image_srgb_norm.shape[0] and skip_mask.shape[1] == image_srgb_norm.shape[1]

        skip_mask = skip_mask.astype(np.float32)
        skip_mask[skip_mask != 0] = 1
        skip_mask = 1 - skip_mask.astype(np.uint8)
    else:
        skip_mask = np.ones((image_srgb_norm.shape[0], image_srgb_norm.shape[1]), dtype=np.uint8)
    
    palette_linear = srgb_to_lin_srgb(palette_srgb)
    image_linear = srgb_to_lin_srgb(image_srgb_norm)
    
    offsets, weights = get_dither_weighting(diffuse_mode)
    return dither_to_palette(image_linear.astype(np.float32), palette_linear.astype(np.float32), lin_srgb_to_oklab(palette_linear)[0].astype(np.float32),
                             skip_mask, offsets.astype(np.uint8), weights.astype(np.float32), serpentine, serp_skip, error_weight)