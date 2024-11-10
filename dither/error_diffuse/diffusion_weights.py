import numpy as np
from enum import Enum, auto
from typing import Tuple

ATKINSON_OFFSETS : np.ndarray = np.array([[0,1], [0,2],[1,-1],[1,0],[1,1],[2,0]])
ATKINSON_WEIGHTS : np.ndarray = np.array([1/8,1/8,1/8,1/8,1/8,1/8])

FLOYD_STEINBERG_OFFSETS : np.ndarray = np.array([[0,1],[1,-1],[1,0],[1,1]])
FLOYD_STEINBERG_WEIGHTS : np.ndarray = np.array([7/16,3/16,5/16,1/16])

class DitheringWeightingMode(Enum):
    ATKINSON = auto()
    FLOYD_STEINBERG = auto()

def get_dither_weighting(mode : DitheringWeightingMode) -> Tuple[np.ndarray, np.ndarray]:
    """Get diffusion offsets and weighting for a given mode.

    Args:
        mode (DitheringWeightingMode): Diffusion mode.

    Raises:
        IndexError: Raised if mode is not in DitheringWeightingMode.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (Offsets, Weights)
    """
    if mode == DitheringWeightingMode.ATKINSON:
        return (np.copy(ATKINSON_OFFSETS), np.copy(ATKINSON_WEIGHTS))
    elif mode == DitheringWeightingMode.FLOYD_STEINBERG:
        return (np.copy(FLOYD_STEINBERG_OFFSETS), np.copy(FLOYD_STEINBERG_WEIGHTS))
    raise IndexError("Dither mode undefined for " + str(mode))