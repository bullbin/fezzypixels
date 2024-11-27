from fezzypixels.color import lin_srgb_to_oklab, srgb_to_lin_srgb

from typing import Tuple

import numpy as np
import cv2

def get_2d_variance(brightness : np.ndarray, domain_k : int) -> np.ndarray:
    """Get brightness variance across sliding window. Edges are reflected to match input dimension.

    Args:
        brightness (np.ndarray): 2D image representing original image brightness.
        domain_k (int): Window size.

    Returns:
        np.ndarray: Brightness variance array, un-normalized.
    """

    # Credit - https://stackoverflow.com/questions/28931265/calculating-variance-of-an-image-python-efficiently/36266187#36266187
    w_mean, w_mean_sqr = (cv2.boxFilter(x, -1, (domain_k, domain_k), borderType=cv2.BORDER_REFLECT) for x in (brightness, brightness * brightness))
    return w_mean_sqr - w_mean * w_mean

def get_flatness_and_edges(image_srgb : np.ndarray, window_size : int = 3, v_clip : float = 0.0075) -> Tuple[np.ndarray, np.ndarray]:
    """Get flatness (variance) and edge maps.

    Args:
        image_srgb (np.ndarray): Input image, normalized sRGB.
        window_size (int, optional): Window size for operators. Smaller kernels catch smaller detail but are noisier. Defaults to 3.
        v_clip (float, optional): Variance clipping maximum. Higher values flatten texture response more. Defaults to 0.0075.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Flatness map, Edge map) where smaller values are less textured.
    """

    brightness = lin_srgb_to_oklab(srgb_to_lin_srgb(image_srgb))[...,0].astype(np.float32)

    # Calculate local variance
    variance_map = get_2d_variance(brightness, window_size)
    
    # Clip variance to useful range, too much variance is just noise which won't matter for blending
    variance_map = variance_map / v_clip
    variance_map = np.clip(variance_map, 0, 1)
    
    # Detect edges using Sobel, normalize magnitude
    edge_h = cv2.Sobel(brightness, cv2.CV_32F, 1, 0, ksize=3)
    edge_v = cv2.Sobel(brightness, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(edge_h**2 + edge_v**2)
    edge_magnitude = edge_magnitude / edge_magnitude.max()
    
    return variance_map, edge_magnitude

def grow_mask(image_srgb : np.ndarray, mask : np.ndarray, texture_ratio : float = 0.2, texture_threshold : float = 0.066) -> np.ndarray:
    """Grow a binary image mask into areas of similar flatness.

    Args:
        image_srgb (np.ndarray): Input image, normalized sRGB.
        mask (np.ndarray): Input mask where x,y lengths are equal to input image. Converted to binary map where any mask > 0 will be grown.
        texture_ratio (float, optional): Balance between texture and edge sensitivity. Lower values prefer flatness, higher values prefer edges. Must be in [0,1]. Defaults to 0.2.
        texture_threshold (float, optional): Flatness threshold. Values below this threshold are considered flat (and growable). Lower is stricter. Defaults to 0.066.

    Returns:
        np.ndarray: Binary mask of shape (x,y,1) grown from (and including) the input mask.
    """

    assert 0 <= texture_ratio <= 1
    assert len(image_srgb.shape) == 3 and len(mask.shape) >= 2 and image_srgb.shape[0] == mask.shape[0] and image_srgb.shape[1] == mask.shape[1]
    variance_map, edge_map = get_flatness_and_edges(image_srgb)
    
    # Create binary map where 0 is flat, 1 is textured
    texture_map = (variance_map * texture_ratio + 
                   edge_map * (1 - texture_ratio)) < texture_threshold

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))

    # Copy mask and force it into uint8 regardless of input dtype
    mask_working = np.copy(mask).astype(np.uint8)
    mask_working[mask > 0] = 1
    current = np.sum(mask_working)

    while True:
        # Grow mask with dilation
        dilated = cv2.dilate(mask_working, element)
        
        # Cut mask back where texture map is too textured
        optimized_mask = dilated * texture_map

        # Restore any mask which was trimmed away
        mask_working[mask_working == 0] = optimized_mask[mask_working == 0]
        sum_working = np.sum(mask_working)
        if sum_working == current:
            break

        current = sum_working
    
    return (mask_working * 255).astype(np.bool_)