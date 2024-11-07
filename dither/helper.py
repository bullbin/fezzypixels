from typing import Optional
import cv2
import numpy as np

def load_precompute_bn_l_image(path_l : str) -> Optional[np.ndarray]:
    """Load and normalize a precomputed sRGB grayscale blue noise pattern.

    Args:
        path_l (str): Path to single-channel blue noise pattern.

    Returns:
        Optional[np.ndarray]: Normalized blue noise if valid, None otherwise.
    """
    try:
        image = cv2.imread(path_l, -1)
    except FileNotFoundError:
        return None
    
    if len(image.shape) != 2:
        return None
    if image.dtype != np.uint16 and image.dtype != np.uint8:
        return None
    
    # Normalize irregardless of whether 8 or 16 bit
    image = image.astype(np.float32) / np.iinfo(image.dtype).max

    # Linearize the sRGB
    return np.where(image <= 0.04045, image / 12.92, ((image + 0.055) / 1.055) ** 2.4)