import numpy as np
from dither.shift import rgb555_to_norm, rgb888_to_rgb555_scale

def error_diffuse_dither_to_srgb555(image_srgb_norm : np.ndarray) -> np.ndarray:
    """Dither an image from sRGB888 to sRGB555 (padded to sRGB888).

    Args:
        image_srgb_norm (np.ndarray): Image in normalized sRGB color.

    Returns:
        np.ndarray: Image in normalized sRGB555 color.
    """

    image_srgb_norm = image_srgb_norm.copy()
    for y in range(image_srgb_norm.shape[0]):
        for x in range(image_srgb_norm.shape[1]):
            closest = rgb555_to_norm(rgb888_to_rgb555_scale(image_srgb_norm[y,x]))
            error = (image_srgb_norm[y,x] - closest)
            image_srgb_norm[y,x] = closest

            diffusion_grid = [[y,x+1],[y+1,x-1],[y+1,x],[y+1,x+2]]
            diffusion_weights = [7/16,3/16,5/16,1/16]
            
            for coord, weight in zip(diffusion_grid, diffusion_weights):
                dest_y, dest_x = coord[0], coord[1]
                if 0 <= dest_y < image_srgb_norm.shape[0] and 0 <= dest_x < image_srgb_norm.shape[1]:
                    image_srgb_norm[dest_y,dest_x] += error * weight

    return image_srgb_norm.astype(np.float32)