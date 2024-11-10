import numpy as np
from enum import Enum, auto

COEFF_DIFF_LOW = np.array([2,4,3])
COEFF_DIFF_HIGH = np.array([3,4,2])

class ColorSimilarityMetric(Enum):
    DELTA_E_FAST = auto()
    RGB_BLEND = auto()
    RGB_BINARY = auto()

def get_most_similar_shade_index_srgb_blend(palette_srgb : np.ndarray, pixel : np.ndarray) -> int:
    """Get the index of the color in the palette closest to the reference pixel.

    This uses blended sRGB weighting to model perception of both dark and lighter colors. For more perceptual results,
    using a LAB space may improve results. Alternatively, binary RGB may be used.

    Args:
        palette_srgb (np.ndarray): Palette in sRGB space. Must be in same space as image.
        pixel (np.ndarray): Pixel in sRGB space. Must be in same space as palette.

    Returns:
        int: Index of most similar palette color.
    """
    deltas = palette_srgb - pixel
    deltas_sq = np.square(deltas)
    average_red = (palette_srgb + pixel)[:, 0]
    average_red = average_red / 2
    distances = (COEFF_DIFF_LOW[0] + average_red) * deltas_sq[:,0] + COEFF_DIFF_LOW[1] * deltas_sq[:,1] + (COEFF_DIFF_LOW[2] - average_red) * deltas_sq[:,2]
    idx_smallest = np.where(distances==np.amin(distances))
    return np.amin(idx_smallest[0])
        
def get_most_similar_shade_index_srgb_binary(palette_srgb : np.ndarray, pixel : np.ndarray) -> int:
    """Get the index of the color in the palette closest to the reference pixel.

    This uses blended sRGB weighting to model perception of both dark and lighter colors. For more perceptual results,
    using a LAB space may improve results. Alternatively, binary RGB may be used.

    Args:
        palette_srgb (np.ndarray): Palette in sRGB space. Must be in same space as image.
        pixel (np.ndarray): Pixel in sRGB space. Must be in same space as palette.

    Returns:
        int: Index of most similar palette color.
    """
    deltas = palette_srgb - pixel
    deltas_sq = np.square(deltas)
    percept_low = np.sum(deltas_sq * COEFF_DIFF_LOW, axis=1)
    percept_high = np.sum(deltas_sq * COEFF_DIFF_HIGH, axis=1)
    average_red = (palette_srgb + pixel)[:, 0]
    average_red = average_red / 2
    distances = np.where(average_red < 0.5, percept_low, percept_high)
    idx_smallest = np.where(distances==np.amin(distances))
    return np.amin(idx_smallest[0])

def get_most_similar_shade_index_lab(palette_lab : np.ndarray, pixel : np.ndarray) -> int:
    """Get the index of the color in the palette closest to the reference pixel.

    This uses Delta E 1976 for the metric which is fast to compute but has weaknesses depending
    on the colorspace. It is strongly suggested to use an upgraded LAB space (like Oklab) to avoid
    issues with darker colors, saturated colors, neutrals and blues.

    Args:
        palette_lab (np.ndarray): Palette in LAB space. Must be in same space as image.
        pixel (np.ndarray): Pixel in LAB space. Must be in same space as palette.

    Returns:
        int: Index of most similar palette color.
    """
    delta_l = (palette_lab[..., 0] - pixel[0]) ** 2
    delta_a = (palette_lab[..., 1] - pixel[1]) ** 2
    delta_b = (palette_lab[..., 2] - pixel[2]) ** 2
    delta_squared = delta_l + delta_a + delta_b
    return np.argmin(delta_squared)