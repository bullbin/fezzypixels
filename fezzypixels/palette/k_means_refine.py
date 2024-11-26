import numpy as np
from fezzypixels.color import srgb_to_lin_srgb, lin_srgb_to_oklab, oklab_to_lin_srgb, lin_srgb_to_srgb
from fezzypixels.shift import rgb888_to_rgb555_scale, rgb555_to_norm
from .k_means import do_k_steps_2d

def refine_palette(image_srgb : np.ndarray, palette_srgb : np.ndarray, iterations : int = 3) -> np.ndarray:
    """Improve the fit of a palette by running a few iterations of a k-means solver. This is faster than
    a full k-means solve (when iterations are set low) and can retain more colors but isn't likely to be
    as optimal. Duplicates encountered during the process (due to a lower quality initial palette)
    will be retained but skipped over during iterations.

    Args:
        image_srgb (np.ndarray): Normalized sRGB image. Downsampling is recommended to reduce cost of k-means fitting steps.
        palette_srgb (np.ndarray): Normalized sRGB palette.
        iterations (int, optional): Iterations of k-means solver. Diminishing returns are hit fast when using quantized palettes. Defaults to 3.

    Returns:
        np.ndarray: Normalized sRGB palette. Only unique colors are retained.
    """

    pixels_lab      = lin_srgb_to_oklab(srgb_to_lin_srgb(image_srgb)).reshape(-1, 3).astype(np.float32)
    centroids_lab   = lin_srgb_to_oklab(srgb_to_lin_srgb(palette_srgb))[0].astype(np.float32)
    do_k_steps_2d(centroids_lab, pixels_lab, iterations)
    centroids_modified = rgb888_to_rgb555_scale(lin_srgb_to_srgb(oklab_to_lin_srgb(centroids_lab)[0]))
    return np.unique(rgb555_to_norm(centroids_modified), axis=0)