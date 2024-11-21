from __future__ import annotations
import numpy as np
from typing import List
from dither.color import lin_srgb_to_srgb, oklab_to_lin_srgb, srgb_to_lin_srgb, lin_srgb_to_oklab
from dither.shift import rgb555_to_norm, rgb888_to_rgb555_scale

class Cube3d():
    def __init__(self, pixels : np.ndarray):
        self.__pixels : np.ndarray = pixels

        self.__err_l : float = 0
        self.__err_a : float = 0
        self.__err_b : float = 0

        self.__error : float = 0
        self.__update_err()

    def get_ranges(self) -> np.ndarray:
        min = np.min(self.__pixels, axis=0)
        max = np.max(self.__pixels, axis=0)
        return np.abs(max - min)
    
    def __update_err(self) -> float:
        color = self.get_color()
        self.__err_l = np.sum((self.__pixels[:,0] - color[0]) ** 2)
        self.__err_a = np.sum((self.__pixels[:,1] - color[1]) ** 2)
        self.__err_b = np.sum((self.__pixels[:,2] - color[2]) ** 2)
        self.__error = (self.__err_l + self.__err_a + self.__err_b) / 3
    
    def get_error(self):
        return self.__error

    def split(self, axis : int) -> Cube3d:
        self.__pixels = self.__pixels[self.__pixels[:, axis].argsort()]
        idx_median = self.__pixels.shape[0] // 2
        output = Cube3d(self.__pixels[:idx_median])
        self.__pixels = self.__pixels[idx_median:]
        self.__update_err()
        return output
    
    def split_noaxis(self) -> Cube3d:
        axis = np.argmax([self.__err_l, self.__err_a, self.__err_b])
        return self.split(axis)
    
    def get_color(self) -> np.ndarray:
        return np.median(self.__pixels, axis=0)

def median_cut_srgb_palette(image_srgb_norm : np.ndarray, count_colors : int = 199, max_iters_mult : float = 5, mse_split : bool = True) -> np.ndarray:
    """Get an sRGB555 palette using a median-cut solver. At larger color counts (e.g., 64+), quality is comparable to
    k-means while being significantly faster with an exact solve. Gradation quality is reduced but colors closer to
    sRGB555 are given exact matches, producing flatter gradations during dithering. Not recommended for small
    palettes (expect saturation loss and major shifting). Preprocessing is not as necessary to achieve good results.

    Args:
        image_srgb_norm (np.ndarray): Normalized sRGB image.
        count_colors (int, optional): Maximum amount of colors to generate. Must be greater than 0 and less than 200. Defaults to 199.
        max_iters_mult (float, optional): Maximum iterations across the palette. Higher is slower but produces more colors. Will terminate early once palette fills. Must be greater than 1. Defaults to 5.
        mse_split (bool, optional): If True, uses within-split error instead of box size to choose which color volume to split. May be slower but improves quality especially at lower color counts. Recommended. Defaults to True.

    Returns:
        np.ndarray: sRGB555 palette in shape (count, 3). Colors may be less than count_colors.
    """
    assert max_iters_mult > 1
    assert count_colors > 0

    max_iters = round(count_colors * max_iters_mult)
    
    pixels = lin_srgb_to_oklab(srgb_to_lin_srgb(image_srgb_norm)).reshape(-1, 3)

    cubes : List[Cube3d] = [Cube3d(pixels)]

    working_colors : np.ndarray = np.ones((max_iters, 3), dtype=np.float64) * np.inf
    
    if mse_split:
        cube_errors : np.ndarray = np.zeros((max_iters), dtype=np.float64)
        cube_errors[0] = cubes[0].get_error()
    else:
        cube_errors : np.ndarray = np.ones((max_iters, 3), dtype=np.float64) * -1
        cube_errors[0] = cubes[0].get_ranges()

    unique_cols = np.array([rgb888_to_rgb555_scale(cubes[0].get_color())])

    if count_colors > 1:

        current = 1
        
        if mse_split:
            while current < max_iters:
                # Credit - https://blog.pkh.me/p/39-improving-color-quantization-heuristics.html
                #          Try to get the speed of median cut with smoothness of k-means
                # TODO - I'm dumb I don't know what the weight is meant to be
                
                idx_worst = np.argmax(cube_errors)
                cubes.append(cubes[idx_worst].split_noaxis())

                cube_errors[idx_worst] = cubes[idx_worst].get_error()
                cube_errors[current] = cubes[current].get_error()
                working_colors[idx_worst] = rgb888_to_rgb555_scale(lin_srgb_to_srgb(oklab_to_lin_srgb(cubes[idx_worst].get_color())[0,0]))
                working_colors[current] = rgb888_to_rgb555_scale(lin_srgb_to_srgb(oklab_to_lin_srgb(cubes[current].get_color())[0,0]))

                current += 1
                unique_cols = np.unique(working_colors[:current], axis=0)

                if unique_cols.shape[0] >= count_colors:
                    break
        else:
            while current < max_iters:
                idx_worst, axis = np.unravel_index(np.argmax(cube_errors), cube_errors.shape)
                cubes.append(cubes[idx_worst].split(axis))

                cube_errors[idx_worst] = cubes[idx_worst].get_ranges()
                cube_errors[current] = cubes[current].get_ranges()
                working_colors[idx_worst] = rgb888_to_rgb555_scale(lin_srgb_to_srgb(oklab_to_lin_srgb(cubes[idx_worst].get_color())[0,0]))
                working_colors[current] = rgb888_to_rgb555_scale(lin_srgb_to_srgb(oklab_to_lin_srgb(cubes[current].get_color())[0,0]))
                
                current += 1
                unique_cols = np.unique(working_colors[:current], axis=0)

                if unique_cols.shape[0] >= count_colors:
                    break
    
    return rgb555_to_norm(unique_cols)