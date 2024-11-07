import numpy as np
from dither.color import lin_srgb_to_oklab, srgb_to_lin_srgb, lin_srgb_to_srgb, oklab_to_lin_srgb
from dither.shift import rgb555_to_norm, rgb888_to_rgb555_scale

from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import shuffle

def k_means_get_srgb_palette(image_srgb_norm : np.ndarray, count_colors : int = 199, seed = 1, k_means_batch_size : int = 4096, k_means_sample_size : int = 24576) -> np.ndarray:
    """Get an sRGB555 palette using a k-means solver.

    Args:
        image_srgb_norm (np.ndarray): Normalized sRGB image.
        count_colors (int, optional): Maximum amount of colors to generate. Must be greater than 0 and less than 200. Defaults to 199.
        seed (int, optional): Seed for random number generator. Defaults to 1.
        k_means_batch_size (int, optional): Amount of pixels considered in each k-means step. Larger may improve quality but will be slower. Must be greater than 0. Defaults to 4096.
        k_means_sample_size (int, optional): Amount of pixels to use for solving. Larger may improve quality but will be slower. Must be greater than 0. Defaults to 24576.

    Returns:
        np.ndarray: sRGB555 palette in shape (count, 3). Colors may be less than count_colors.
    """

    assert 1 <= count_colors < 200
    assert k_means_batch_size > 0
    assert k_means_sample_size > 0
    
    kmeans_flattened = lin_srgb_to_oklab(srgb_to_lin_srgb(image_srgb_norm)).reshape(-1, 3)
    shuffled_input = shuffle(kmeans_flattened, random_state=seed, n_samples=min(k_means_sample_size, kmeans_flattened.shape[0]))
    
    kmeans = MiniBatchKMeans(n_clusters=count_colors, random_state=seed, batch_size=k_means_batch_size).fit(shuffled_input)
    centroids_lab = np.copy(kmeans.cluster_centers_)

    centroids_srgb = lin_srgb_to_srgb(oklab_to_lin_srgb(centroids_lab)[0])
    centroids_srgb = rgb888_to_rgb555_scale(centroids_srgb)
    return rgb555_to_norm(centroids_srgb)