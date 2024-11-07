import numpy as np

def rgb888_to_norm(rgb888 : np.ndarray) -> np.ndarray:
    """Normalize RGB888 to floating point.

    No clamping is applied.

    Args:
        rgb888 (np.ndarray): RGB888 image.

    Returns:
        np.ndarray: Floating point image, range [0,1].
    """
    return rgb888 / 255

def rgb888_to_rgb555_shift(rgb888_norm : np.ndarray) -> np.ndarray:
    """Quantize RGB888 to RGB555 by numerically shifting the colors.

    No clamping is applied.

    Args:
        rgb888_norm (np.ndarray): Floating point image, range [0,1].

    Returns:
        np.ndarray: RGB555 image, range [0,31].
    """
    return (rgb888_norm * 255).astype(np.uint8) >> 3

def rgb888_to_rgb555_scale(rgb888_norm : np.ndarray) -> np.ndarray:
    """Quantize RGB888 to RGB555 by scaling the colors.

    No clamping is applied.

    Args:
        rgb888_norm (np.ndarray): Floating point image, range [0,1].

    Returns:
        np.ndarray: RGB555 image, range [0,31].
    """
    return np.round((rgb888_norm * 31)).astype(np.uint8)

def rgb555_to_norm(rgb555 : np.ndarray) -> np.ndarray:
    """Normalize RGB555 to floating point.

    No clamping is applied.

    Args:
        rgb555 (np.ndarray): RGB555 image, range [0,31].

    Returns:
        np.ndarray: Floating point image, range [0,1].
    """
    return rgb555 / 31