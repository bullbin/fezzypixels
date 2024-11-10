import numpy as np
from enum import Enum, auto
from typing import Tuple

ATKINSON_OFFSETS : np.ndarray = np.array([[0,1], [0,2],[1,-1],[1,0],[1,1],[2,0]])
ATKINSON_WEIGHTS : np.ndarray = np.array([1,1,1,1,1,1]) /8

FLOYD_STEINBERG_OFFSETS : np.ndarray = np.array([[0,1],[1,-1],[1,0],[1,1]])
FLOYD_STEINBERG_WEIGHTS : np.ndarray = np.array([7,3,5,1]) /16

JJN_MMA_OFFSETS : np.ndarray = np.array([[0,1],[0,2],[1,-2],[1,-1],[1,0],[1,1],[1,2],[2,-2],[2,-1],[2,0],[2,1],[2,2]])
JJN_MMA_WEIGHTS : np.ndarray = np.array([7,5,3,5,7,5,3,1,3,5,3,1]) / 48

class DitheringWeightingMode(Enum):
    ATKINSON = auto()
    FLOYD_STEINBERG = auto()
    JJN_MMA = auto()

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
    elif mode == DitheringWeightingMode.JJN_MMA:
        return (np.copy(JJN_MMA_OFFSETS), np.copy(JJN_MMA_WEIGHTS))
    raise IndexError("Dither mode undefined for " + str(mode))