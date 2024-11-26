# fezzypixels

fezzypixels is a library for producing high-quality quantized (and paletted) images in sRGB555. It's primary design goal was a better replacement for [madhatter's](https://www.github.com/bullbin/madhatter) aging imaging pipeline but that's not its only use case!

## what makes fezzypixels good?

fezzypixels is written from the ground up to produce great looking images while using little color. It has the following features:

- Color space optimized rendering with gamma-correction and seamless transitioning between sRGB, Oklab and linear sRGB depending on the operation
- Aggressive k-means paletting with heuristics for improving gradations
- Support for both static and animation-safe dithering
	- Static dithering uses error-diffusion dithering with the choice of Floyd-Steinberg, Atkinson or JJN diffusion patterns
	- Animation-safe dithering uses Pattern dithering with the choice of either Bayer or Blue Noise thresholding
		- **As far as I know, [the patent has expired](https://patents.google.com/patent/US6606166B1).** I'm not a lawyer!
- Texture-aware masking for transitioning between dithering approaches while hiding any seams

## how do I install this?

 1. Clone the repo
 2. Install requirements with `pip install -r requirements.txt`
 3. (optional) Build pattern dithering acceleration module with `py setup.py build_ext --inplace`
 4. Try some examples 👇

## how do I use this?

The library is still a work-in-progress so subject to change. A submodule branch is planned in future for easier deployment, but as an example, this is how to quantize an image from sRGB888 to sRGB555 using the faster (but less structured) Pattern dithering:

```
import cv2
import numpy as np

from fezzypixels.palette.k_means_minibatch_lab import k_means_get_srgb_palette
from fezzypixels.palette.preprocess import add_flat_regions_to_k_means_input
from fezzypixels.pattern.pattern_dither import ThresholdMode, pattern_dither_srgb
from fezzypixels.preprocess.ordered_pattern_dither_srgb import pattern_dither_to_srgb555
from fezzypixels.shift import rgb888_to_norm

PATH_INPUT_IMAGE : str = ...
PATH_OUTPUT_IMAGE : str = ...

# Load the image, resize it to NDS dimensions
input_image_srgb = cv2.imread(PATH_INPUT_IMAGE)
input_image_srgb = cv2.resize(input_image_srgb, (256,192), interpolation=cv2.INTER_AREA)

# Convert to RGB then normalize to [0,1]
input_image_srgb = cv2.cvtColor(input_image_srgb, cv2.COLOR_BGR2RGB)
input_image_srgb = rgb888_to_norm(input_image_srgb).astype(np.float32)

# Dither to generate good candidates for palette
palette_input = pattern_dither_to_srgb555(input_image_srgb)

# Generate palette with additional helper to improve quality in flatter regions
palette_srgb = k_means_get_srgb_palette(add_flat_regions_to_k_means_input(input_image_srgb, palette_input))

# Quantize image - in this case, using pattern dithering
quantized = pattern_dither_srgb(input_image_srgb, palette_srgb, q=0.045, threshold_mode=ThresholdMode.BLUE_256).astype(np.float32)

cv2.imwrite(PATH_OUTPUT_IMAGE, cv2.cvtColor(quantized, cv2.COLOR_RGB2BGR) * 255)
```

## credits ❤️

- [Christoph Peters](https://momentsingraphics.de/BlueNoise.html) for his free Blue noise textures (included in repo)
- [matejloub](https://www.shadertoy.com/view/dlcGzN) for their implementation of Pattern dithering, we also pre-sort early as an optimization
- Everything about [Oklab](https://bottosson.github.io/posts/oklab/), it's significantly better than LAB and fixed so many color issues
- [Surma](https://surma.dev/) for their excellent post on [dithering](https://surma.dev/things/ditherpunk/)
