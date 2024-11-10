from typing import Optional

import numpy as np
from dither.blend import get_flatness_and_edges

def add_flat_regions_to_k_means_input(image_srgb : np.ndarray, image_ref : Optional[np.ndarray] = None, texture_ratio : float = 0.4, texture_threshold : float = 0.05, v_max : float = 0.005, k : int = 5) -> np.ndarray:
    """Add additional importance to slow gradations in an image by repeating their pixels.

    Args:
        image_srgb (np.ndarray): Input image, normalized sRGB. Algorithm finds pixels based on this image.
        image_ref (Optional[np.ndarray], optional): Image to extract pixels from. None to use input image. Defaults to None.
        texture_ratio (float, optional): Ratio between texture and edges to find slow gradations. 0 leans towards flat texture, 1 leans towards areas far from edges. Must be in [0,1]. Defaults to 0.4.
        texture_threshold (float, optional): Values higher than this threshold are considered unimportant. Defaults to 0.05.
        v_max (float, optional): Maximum variance before area considered unimportant. Higher accepts more detail (and noise). Defaults to 0.005.
        k (int, optional): Size of kernel for feature extraction. Larger is less noisy but less detailed. Defaults to 5.

    Returns:
        np.ndarray: List of flattened pixels for use with k-means algorithm. This will not have the shape of the image anymore so viewing is not recommended.
    """

    assert 0 <= texture_ratio <= 1
    if not(image_ref is None):
        assert image_srgb.shape[0] == image_ref.shape[0] and image_srgb.shape[1] == image_ref.shape[1]

    flatness, edge = get_flatness_and_edges(image_srgb, k, v_max)
    texture_map = (flatness * texture_ratio + 
                    edge * (1 - texture_ratio)) < texture_threshold

    mask_flattened = texture_map.flatten()
    
    if image_ref is None:
        image_ref = image_srgb

    return np.concatenate((image_ref.reshape(-1, 3), image_ref.reshape(-1, 3)[mask_flattened]), axis=0) 