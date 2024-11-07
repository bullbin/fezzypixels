import numpy as np

def get_bayer_map(n : int) -> np.ndarray:
    """Get the Bayer map for a given axis dimension n.

    n should be a power of 2; an assertion will be raised if n is smaller than 2.

    Args:
        n (int): Length of axis. Will be rounded to closest power of 2.

    Returns:
        np.ndarray: Normalized square Bayer thresholding map.
    """
    assert n >= 2
    n = np.ceil(np.log2(n))
    current = 1
    last = np.array([[0,2],
                     [3,1]])
    
    while current < n:
        edge = 2 ** current
        last = np.tile(last, (2,2)) * 4
        last[edge:,edge:] += 1
        last[:edge,edge:] += 2
        last[edge:,:edge] += 3
        current += 1
    
    return last / (2 ** (n * 2))

def get_bayer_map_2() -> np.ndarray:
    """Returns the normalized 2x2 Bayer thresholding map.

    Returns:
        np.ndarray: Normalized 2x2 Bayer thresholding map.
    """
    return get_bayer_map(2)

def get_bayer_map_4() -> np.ndarray:
    """Returns the normalized 4x4 Bayer thresholding map.

    Returns:
        np.ndarray: Normalized 4x4 Bayer thresholding map.
    """
    return get_bayer_map(4)

def get_bayer_map_8() -> np.ndarray:
    """Returns the normalized 8x8 Bayer thresholding map.

    Returns:
        np.ndarray: Normalized 8x8 Bayer thresholding map.
    """
    return get_bayer_map(8)