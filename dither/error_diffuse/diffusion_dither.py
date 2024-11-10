import numpy as np
from dither.color import lin_srgb_to_srgb, srgb_to_lin_srgb, lin_srgb_to_oklab
from dither.palette import ColorSimilarityMetric, get_most_similar_shade_index_lab, get_most_similar_shade_index_srgb_binary, get_most_similar_shade_index_srgb_blend
from .diffusion_weights import get_dither_weighting, DitheringWeightingMode
from typing import Optional

def error_diffusion_dither_srgb(image_srgb_norm : np.ndarray, palette_srgb : np.ndarray, metric : ColorSimilarityMetric = ColorSimilarityMetric.DELTA_E_FAST,
                                diffuse_mode : DitheringWeightingMode = DitheringWeightingMode.ATKINSON,
                                serpentine : bool = True, skip_mask : Optional[np.ndarray] = None) -> np.ndarray:
    """Dither an image using error-diffusion dithering.

    Args:
        image_srgb_norm (np.ndarray): Image in normalized sRGB color.
        palette_srgb (np.ndarray): Palette in normalized sRGB color.
        metric (ColorSimilarityMetric, optional): Color similarity metric. RGB is faster, DELTA_E is more perceptual. . Defaults to ColorSimilarityMetric.DELTA_E_FAST.
        diffuse_mode (DitheringWeightingMode, optional): Diffusion matrix to use. Defaults to DitheringWeightingMode.ATKINSON.
        serpentine (bool, optional): Whether to swap diffusion direction per-row. Reduces diffusion artifacts. Defaults to True.
        skip_mask (np.ndarray, optional): Mask of where to skip error diffusion. Must be same size as input image. True (any value != 0) will skip diffusion on image. None to diffuse everywhere. Defaults to None.

    Returns:
        np.ndarray: Quantized image in normalized sRGB color.
    """

    if not(skip_mask is None):
        assert skip_mask.shape[0] == image_srgb_norm.shape[0] and skip_mask.shape[1] == image_srgb_norm.shape[1]

        skip_mask = skip_mask.astype(np.float32)
        skip_mask[skip_mask != 0] = 1
        skip_mask = 1 - skip_mask.astype(np.uint8)
    else:
        skip_mask = np.ones((image_srgb_norm.shape[0], image_srgb_norm.shape[1]))

    output_srgb = np.zeros(shape=(image_srgb_norm.shape[0], image_srgb_norm.shape[1], 3), dtype=np.float32)
    
    palette_linear = srgb_to_lin_srgb(palette_srgb)
    image_linear = srgb_to_lin_srgb(image_srgb_norm)

    if metric == ColorSimilarityMetric.RGB_BINARY:
        def get_most_similar_shade_index(pixel_lin_rgb : np.ndarray) -> int:
            pixel_srgb = lin_srgb_to_srgb(pixel_lin_rgb)[0,0]
            return get_most_similar_shade_index_srgb_binary(palette_srgb, pixel_srgb)

    elif metric == ColorSimilarityMetric.RGB_BLEND:
        def get_most_similar_shade_index(pixel_lin_rgb : np.ndarray) -> int:
            pixel_srgb = lin_srgb_to_srgb(pixel_lin_rgb)[0,0]
            return get_most_similar_shade_index_srgb_blend(palette_srgb, pixel_srgb)
        
    else:
        palette_lab = lin_srgb_to_oklab(palette_linear)[0]

        def get_most_similar_shade_index(pixel_lin_srgb : np.ndarray) -> int:
            pixel_lab = lin_srgb_to_oklab(pixel_lin_srgb)[0,0]
            return get_most_similar_shade_index_lab(palette_lab, pixel_lab)
    
    offsets, weights = get_dither_weighting(diffuse_mode)
        
    for y in range(image_linear.shape[0]):
        do_serp = serpentine and y % 2 == 1

        for x in range(image_linear.shape[1]):
            
            if do_serp:
                x = image_linear.shape[1] - x - 1

            pixel_old = image_linear[y,x]

            idx_closest = get_most_similar_shade_index(pixel_old)
            
            closest_color = palette_linear[idx_closest]
            output_srgb[y,x] = palette_srgb[idx_closest]
            error_linear = pixel_old - closest_color

            for coord, weight in zip(offsets, weights):
                dest_y = coord[0] + y

                if do_serp:
                    dest_x = coord[1] - x
                else:
                    dest_x = coord[1] + x

                if 0 <= dest_y < image_linear.shape[0] and 0 <= dest_x < image_linear.shape[1]:
                    image_linear[dest_y,dest_x] += error_linear * weight * skip_mask[dest_y,dest_x]

    return output_srgb.astype(np.float32)