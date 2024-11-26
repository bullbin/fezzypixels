import numpy as np
from typing import Tuple, Optional
from fezzypixels.color import srgb_to_luminance, srgb_to_lin_srgb, lin_srgb_to_oklab
from fezzypixels.pattern.bayer import get_bayer_map_8, get_bayer_map_2, get_bayer_map_4
from fezzypixels.palette import get_most_similar_shade_index_lab
from fezzypixels.helper import load_precompute_bn_l_image
from os.path import dirname, join
from enum import Enum, auto

try:
    from .fast_candidates import get_candidates
    HAS_ACCELERATION : bool = True
except:
    HAS_ACCELERATION : bool = False

class ThresholdMode(Enum):
    BAYER_2 = auto()
    BAYER_4 = auto()
    BAYER_8 = auto()
    BLUE_256 = auto()

# Blue noise texture. CC0, credit - https://momentsingraphics.de/BlueNoise.html
PATH_HDR_TEMPLATE : str = join(dirname(__file__), "HDR_L_0.png")
CACHE_LIN_BLUE_NOISE : Optional[np.ndarray] = None

def pattern_dither_srgb(image_srgb : np.ndarray, palette_srgb : np.ndarray, n : int = 32, q : float = 0.05, threshold_mode : ThresholdMode = ThresholdMode.BAYER_4) -> np.ndarray:
    """Apply Thomas Knoll's 'Pattern Dithering' algorithm to quantize an image down to an arbitrary palette.

    This algorithm is a variant of ordered dithering but is particularly effective at preserving detail and
    is able to produce grain which can mimic error diffusion without actually performing error diffusion. It
    does this by finding similar palette colors for each pixel then propogating per-pixel to adjust the next
    colors. At output, thresholded dithering is used to pick between candidates which produces variations
    in texture.

    This implementation follows the patented algorithm (US6606166 - expired!) faithfully down to sorting colors
    by luminance and switching between linear and LAB colorspaces to improve similarity. It is recommended to
    build the Cython extensions for better performance but speed is generally acceptable with smaller n. The
    following threshold maps are provided:
    - BAYER_2: Bayer matrix, 2x2 size. Stylized. Low texture. 
    - BAYER_4: Bayer matrix, 4x4 size. Stylized. Medium texture.
    - BAYER_8: Bayer matrix, 8x8 size. Stylized. High texture.
    - BLUE_256: Blue noise, 256x256 size. Diffusion-like. Consistent but unstructured texture.

    Args:
        image_srgb (np.ndarray): Image in normalized sRGB color.
        palette_srgb (np.ndarray): Normalized sRGB palette.
        n (int, optional): Number of candidates to find for each pixel. Larger costs more but increases depth of grain. Defaults to 32.
        q (float, optional): Dithering factor. Higher values reintroduce more error into the image but can improve smoothness of gradients. Defaults to 0.05.
        threshold_mode (ThresholdMode, optional): Threshold mode for final dithering step. Changes texture of output. Defaults to ThresholdMode.BAYER_4.

    Returns:
        np.ndarray: Image as indices into palette.
    """
    
    # Use some globals so we only have to load the blue noise texture once (it's okay...)
    global CACHE_LIN_BLUE_NOISE, PATH_HDR_TEMPLATE
    
    # Sort palette by luminance, helps reduce cost of sorting candidates later
    # Optimization credit - https://www.shadertoy.com/view/dlcGzN
    idx_sorted = np.argsort(srgb_to_luminance(palette_srgb))
    palette_srgb_sorted = np.copy(palette_srgb)
    for idx in range(idx_sorted.shape[0]):
        palette_srgb_sorted[idx] = palette_srgb[idx_sorted[idx]]
    
    palette_srgb = palette_srgb_sorted
    palette_lin = srgb_to_lin_srgb(palette_srgb)

    # For each pixel, compute a list of possible candidates
    # Each candidate down the list is increasingly noisy (because of accumulated error) 
    #     but still relevant to original, like it has been dithered
    c = srgb_to_lin_srgb(image_srgb)
    
    if HAS_ACCELERATION:
        # Accelerated version is same as below, uses Oklab and parallel compute on per-line basis
        # Some precision differences though - non-accelerated is slightly smoother (sometimes) but
        #     much, much slower
        candidate_array = get_candidates(c.astype(np.float32), palette_lin.astype(np.float32), n, q)
    else:
        palette_lab = lin_srgb_to_oklab(srgb_to_lin_srgb(palette_srgb))[0]

        def get_paletted_by_closest(draft_lin_srgb : np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            draft_lab = lin_srgb_to_oklab(draft_lin_srgb)
            output_candidate = np.zeros((image_srgb.shape[0], image_srgb.shape[1]), dtype=np.uint32)
            output_color = np.zeros_like(draft_lin_srgb)

            for y in range(output_candidate.shape[0]):
                for x in range(output_candidate.shape[1]):
                    shade = get_most_similar_shade_index_lab(palette_lab, draft_lab[y,x])
                    output_candidate[y,x] = shade
                    output_color[y,x] = palette_lin[shade]
            
            return (output_candidate, output_color)
    
        candidate_array = np.zeros((image_srgb.shape[0], image_srgb.shape[1], n), dtype=np.uint32)

        e = np.zeros_like(image_srgb)
        print("Extension missing, generating candidates in pure Python...")
        for i in range(n):
            print("\tCandidate", i)
            t = c + (e * q)
            candidate_array[..., i], current_assigned_lin = get_paletted_by_closest(t)
            e += (c - current_assigned_lin)

    # Paper recommends sorting candidates by luminance
    # Because our palette is pre-sorted by luminance, sorting the indices does the same thing
    candidate_array = np.sort(candidate_array, axis=2)
    
    if threshold_mode == ThresholdMode.BLUE_256:
        if CACHE_LIN_BLUE_NOISE is None:
            CACHE_LIN_BLUE_NOISE = load_precompute_bn_l_image(PATH_HDR_TEMPLATE)
        
        if CACHE_LIN_BLUE_NOISE is None:
            print("Failed to load blue noise map, falling back to Bayer4!")
            threshold_map = get_bayer_map_4()
        else:
            threshold_map = CACHE_LIN_BLUE_NOISE

    elif threshold_mode == ThresholdMode.BAYER_8:
        threshold_map = get_bayer_map_8()
    elif threshold_mode == ThresholdMode.BAYER_4:
        threshold_map = get_bayer_map_4()
    else:
        threshold_map = get_bayer_map_2()

    output = np.zeros_like(image_srgb)

    # Tile blue noise to meet input size
    idx_y, idx_x = np.meshgrid(np.arange(candidate_array.shape[0]), np.arange(candidate_array.shape[1]), indexing='ij')
    repeat_y = int(np.ceil(candidate_array.shape[0] / threshold_map.shape[0]))
    repeat_x = int(np.ceil(candidate_array.shape[1] / threshold_map.shape[1]))
    shift = np.tile(threshold_map, (repeat_y, repeat_x))[:candidate_array.shape[0], :candidate_array.shape[1]]

    # Vectorize candidate and output
    candidate = np.clip(np.floor(shift * n).astype(np.uint32), 0, n - 1)
    return candidate_array[idx_y,idx_x,candidate]