# Written by github.com/bullbin

# RGB888 to RGB555 with Restricted Palette Image Converter
#     Designed primarily to help with Layton-like image conversions

# This imaging algorithm is designed to be easier to read and understand
#     than the madhatter algorithm but primarily uses the same concepts
#     (pre-dithering, k-means paletting and gamma-correct rendering) to
#     produce its output.
# They're not the same though. Which generates the nicer image will be
#     down to the image and personal preference. madhatter additionally
#     supports alpha channels and custom palette mixing - this is just
#     to make the color parts and pipeline more readable.
# madhatter is here: https://github.com/bullbin/madhatter

# Requirements:
# numpy
# opencv-python
# colour-science

# License:
# MIT 2024 (same as madhatter)

import numpy as np
from enum import Enum, auto
from dither.color import lin_srgb_to_srgb, srgb_to_lin_srgb, lin_srgb_to_oklab

class ColorSimilarityMetric(Enum):
    DELTA_E_FAST = auto()
    RGB_BLEND = auto()
    RGB_BINARY = auto()

class PreprocessMode(Enum):
    QUALITY_DITHER = auto()
    FAST_SCALE = auto()

def atkinson_dither_srgb_555(image_srgb_norm : np.ndarray, palette_srgb : np.ndarray, metric : ColorSimilarityMetric = ColorSimilarityMetric.DELTA_E_FAST) -> np.ndarray:
    """Dither an image to sRGB 555.

    Args:
        image_srgb_rgb888_norm (np.ndarray): Image in normalized sRGB color.
        palette_srgb (np.ndarray): Palette in normalized sRGB color.
        metric (ColorSimilarityMetric, optional): Color similarity metric. Names explain tradeoffs. Defaults to ColorSimilarityMetric.RGB_BLEND.

    Returns:
        np.ndarray: Quantized image in normalized sRGB color.
    """
    output_srgb = np.zeros(shape=(image_srgb_norm.shape[0], image_srgb_norm.shape[1], 3), dtype=np.float32)
    
    palette_linear = srgb_to_lin_srgb(palette_srgb)
    palette_lab = lin_srgb_to_oklab(palette_linear)[0]
    image_linear = srgb_to_lin_srgb(image_srgb_norm)
    
    # Match against palette using either fast RGB function (good quality, smooth gradiation) or
    #     CIELAB function (more faithful color, can be very slow)
    # Credit (RGB) - https://en.wikipedia.org/wiki/Color_difference

    COEFF_DIFF_LOW = np.array([2,4,3])
    COEFF_DIFF_HIGH = np.array([3,4,2])

    if metric == ColorSimilarityMetric.RGB_BINARY:
        def get_most_similar_shade_index(y : int, x : int) -> np.ndarray:
            val = lin_srgb_to_srgb(image_linear[y,x])
            deltas = palette_srgb - val
            deltas_sq = np.square(deltas)
            average_red = (palette_srgb + val)[:, 0]
            average_red = average_red / 2
            distances = (2 + average_red) * deltas_sq[:,0] + 4 * deltas_sq[:,1] + (3 - average_red) * deltas_sq[:,2]
            idx_smallest = np.where(distances==np.amin(distances))
            return np.amin(idx_smallest[0])
        
    elif metric == ColorSimilarityMetric.RGB_BLEND:
        def get_most_similar_shade_index(y : int, x : int) -> np.ndarray:
            val = lin_srgb_to_srgb(image_linear[y,x])
            deltas = palette_srgb - val
            deltas_sq = np.square(deltas)
            percept_low = np.sum(deltas_sq * COEFF_DIFF_LOW, axis=1)
            percept_high = np.sum(deltas_sq * COEFF_DIFF_HIGH, axis=1)
            average_red = (palette_srgb + val)[:, 0]
            average_red = average_red / 2
            distances = np.where(average_red < 0.5, percept_low, percept_high)
            idx_smallest = np.where(distances==np.amin(distances))
            return np.amin(idx_smallest[0])
        
    else:
        def get_most_similar_shade_index(y : int, x : int):
            pixel = lin_srgb_to_oklab(image_linear[y,x])[0,0]
            delta_l = (palette_lab[..., 0] - pixel[0]) ** 2
            delta_a = (palette_lab[..., 1] - pixel[1]) ** 2
            delta_b = (palette_lab[..., 2] - pixel[2]) ** 2
            delta_squared = delta_l + delta_a + delta_b
            return np.argmin(delta_squared)
        
    for y in range(image_linear.shape[0]):
        for x in range(image_linear.shape[1]):
            pixel_old = image_linear[y,x]

            idx_closest = get_most_similar_shade_index(y,x)
            
            closest_color = palette_linear[idx_closest]
            output_srgb[y,x] = palette_srgb[idx_closest]
            error_linear = pixel_old - closest_color

            # This is just the Atkinson dither weights
            diffusion_grid = [                [y,     x + 1], [y,     x + 2],
                              [y + 1, x - 1], [y + 1, x],     [y + 1, x + 1],
                                              [y + 2, x]]
            diffusion_weights = [1/8,1/8,1/8,1/8,1/8,1/8]

            #diffusion_grid = [[y,x+1],[y+1,x-1],[y+1,x],[y+1,x+2]]
            #diffusion_weights = [7/16,3/16,5/16,1/16]

            for coord, weight in zip(diffusion_grid, diffusion_weights):
                dest_y, dest_x = coord[0], coord[1]
                if 0 <= dest_y < image_linear.shape[0] and 0 <= dest_x < image_linear.shape[1]:
                    image_linear[dest_y,dest_x] += error_linear * weight
    
    return output_srgb.astype(np.float32)