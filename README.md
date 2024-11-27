# fezzypixels
fezzypixels is a library for producing high-quality quantized (and paletted) images in sRGB555.

## why use fezzypixels?
fezzypixels was written from the ground up to make generating smooth RGB555 images easy. fezzypixels includes...

 - Color space optimized rendering with internal gamma-correction and perceptual matching as needed
 - Quality-first approaches to maximise palette quality even on complex gradations
 - Support for both static (error-diffusion) and animation-safe ([pattern](https://patents.google.com/patent/US6606166B1)) dithering techniques
 - A variety of artistic kernels for both approaches, including Floyd-Steinberg, Atkinson and JJN-MMA for error diffusion and Bayer and blue noise for pattern dithering
 - Texture-aware masking for blending static and animated components in the same image while hiding any seams

## how do I install this?

### (easiest) PyPI via pip
fezzypixels and (limited) pre-build wheels are available on PyPI, so the easiest way to get the latest release candidate is using `pip install fezzypixels`

### Build from source
fezzypixels uses a standard Python build system and has few requirements so should be fairly easy to build. You'll need a functional C compiler and Python 3 - development used Python 3.12 but other versions may work. To install the project, follow these steps:

 1. Clone the repo
 2. In the main folder, run `pip install .`

## how do I use this?

### Importing images
fezzypixels works on images in the form of floating-point numpy arrays with shape (width, height, 3). Channel order must be RGB so assuming you are using OpenCV, images can be loaded like the following:
```
input_image_srgb = cv2.imread(...)
input_image_srgb = cv2.cvtColor(input_image_srgb, cv2.COLOR_BGR2RGB)
```
All images must be normalized to [0,1] regardless of input bit depth. For 8-bit images, fezzypixels contains convenience methods for this:
```
from fezzypixels.shift import rgb888_to_norm
input_image_norm = rgb888_to_norm(input_image_srgb)
```

### Paletting images
fezzypixels provides 3 approaches for palette generation, all of which run in LAB space. This includes a full k-means solver (`k_means_get_srgb_palette`), a modified median cut solver (`median_cut_srgb_palette`) and a hybrid median cut solver (`median_cut_srgb_palette` with iterations of 	`refine_palette`). Before paletting, choose which solver is best for your images.

#### What paletting approach is best for my application?
For quick reference, here's my subjective summary for each method, scale 🟥🟧🟨🟩✅. ✅ is my top preference.

||speed (colorful)|speed (flat)|artifacts vivid @ 199|artifacts dark @ 199|quality @ 24|quality @ 48|quality @ 96|quality @ 199
|--|--|--|--|--|--|--|--|--|
|k-means|🟥|🟧|✅|🟨|🟨|🟨|✅|✅
|median cut|✅|🟧|🟩|🟩|🟥|🟧|🟩|✅
|hybrid median cut|🟩|🟥|✅|✅|🟥|🟨|✅|✅

In general, k-means offers consistent performance across every category but is almost always the slowest method. For darker and flatter images, it may not provide a dithering-friendly palette and will produce more artifacts than other methods. Hybrid median cut is typically the preferred method because it is faster in most situations, has stronger performance in flatter images and, unless using small palettes, has a similar quality profile to k-means.

<details>
  <summary>Detailed breakdown of pros and cons of each method</summary>
  
  ##### k-means paletting
  ###### Pros
  - Generally finds optimal palette for an image, i.e., a palette that minimizes deltaE given the palette length restriction
  - Retains saturation even at lower color counts
  ###### Cons
  - Not sRGB555 aware so generates less colors in flat images which are very dark or very bright (LAB compression)
  - Requires sRGB555 preprocessing to generate dithering-friendly shades
  - Slower on systems with low core counts or slow cores
  ##### median cut with sRGB555 trick
  ###### Pros
  - Capable of filling palette to arbitrary amount of colors by quantizing during computation
  - Quality holds even without preprocessing
  - Much faster than k-means on colorful images
  - Much faster than k-means on systems with low core counts or slow cores
  - Much better gradations than k-means on flatter (and darker) images
  ###### Cons
  - Cost approaches k-means on flatter images (but generates more colors)
  - Saturation loss at lower color counts
  ##### hybrid median cut with low k-means iterations
  ###### Pros
  - Same as median cut
  - Optimality within distance of k-means on colorful images while retaining performance advantages of median cut
  - Less saturation loss at lower color counts compared to median cut
  ###### Cons
  - Mostly same as median cut
  - Expensive at higher pixel counts (downsampling recommended)
 </details>

#### Maximising quality with preprocessing
All solvers benefit from some preprocessing to drop the image to sRGB555 prior to paletting. For k-means, the improvement is drastic and always preferrable; for median cut, the difference is more subtle and may not be worth the cost.

```
from fezzypixels.preprocess import pattern_dither_to_srgb555

palette_input = pattern_dither_to_srgb555(input_image_norm)
```

fezzypixels also provides methods to improve the quality of flatter areas of the image. This works by duplicating these areas in the palette input so more colors will be created for smoother diffusion later. The default values are typically sufficient for good performance but this enhancement can be tweaked if it is duplicating unwanted textures.

```
from fezzypixels.palette import flatten_with_flat_roi_enhancement

palette_input = flatten_with_flat_roi_enhancement(input_image_norm, palette_input)
```

#### Running the palette quantizer
All quantizers in `fezzypixels.palette` generate up to 256 colors. Only unique colors are returned so if a solver generates duplicate colors, the length of the palette may be less than the target count. All solvers may be sped up by running them on a smaller subset of the image at the cost of palette quality.

##### k-means
`palette = k_means_get_srgb_palette(palette_input, count_colors=...)`

##### median cut
`palette = median_cut_srgb_palette(palette_input, count_colors=...)`

##### hybrid median cut
Hybrid median cut extends the standard median cut by running limited steps of a full k-means solver. This may reduce color count but typically boosts saturation in low quality areas. Low iterations are recommended because median cut already provides a good quality palette so the k-means solver will typically converge quickly. It is strongly recommended to downsample the input to the k-means solver because cost grows rapidly with increased samples.
```
palette = median_cut_srgb_palette(palette_input, count_colors=...)

# It is recommended to downsample palette_input to reduce cost
palette = refine_palette(palette_input[::8], palette, iterations=...)
```

### Quantizing images
fezzypixels provides 2 quantizers: `error_diffusion_dither_srgb` from `fezzypixels.error_diffuse` and `pattern_dither_srgb` from `fezzypixels.pattern`. Both quantizers produce high-quality results but error diffusion will typically provide a more perceptually-close image to the original. Pattern dithering is useful when blending animated and static content because it does not diffuse error so will leave static areas untouched when dithering, producing no visible seam. It is recommended to experiment with both before settling on one method.

All quantizers produce indexed images of shape `(width, height)`. The RGB equivalent is `palette[quantized]`.

#### Error diffusion dithering
`quantized = error_diffusion_dither_srgb(input_image_srgb, palette_srgb)`

fezzypixels provides 3 kernels to experiment with in `DitheringWeightingMode`. Floyd-Steinberg produces the smoothest but grainiest result; Atkinson trades more banding, and JJN-MMA sits between the two. `serpentine` controls whether serpentine dithering is used which produces less distracting dithering patterns. `error_weight` can be used to globally adjust error propogation, reducing both dithering and noise simultaneously.

#### Pattern dithering
`quantized = pattern_dither_srgb(input_image_srgb, palette_srgb)`

fezzypixels provides 2 thresholds to experiment with in `ThresholdMode`. Bayer produces more structured patterns whereas blue noise is uniformly grainy, producing images more like error-diffusion. `n` and `q` tune the algorithm; `n` refers to sample count and `q` the amount of error pushed forwards per sample. Keep `n` as low as possible for your application to optimize runtime.

### On RGBA5551 support
fezzypixels was not written to support alpha but permits skipping of pixels during palette generation and error-diffusion dithering. 1-bit alpha support can be achieved in this way.

During palette generation, `skip_mask` can be set as a mask in `flatten_with_flat_roi_enhancement` to hide alpha pixels from the solver. This mask can be reused with `error_diffusion_dither_srgb` to prevent error from alpha pixels being diffused into the output. For preprocessing or when using pattern dithering, alpha may be ignored because it doesn't change the value of opaque pixels.

## credits ❤️
- [Christoph Peters](https://momentsingraphics.de/BlueNoise.html) for his free Blue noise textures (included in repo)
- [matejloub](https://www.shadertoy.com/view/dlcGzN) for their implementation of Pattern dithering, we also pre-sort early as an optimization
- Everything about [Oklab](https://bottosson.github.io/posts/oklab/), it's significantly better than LAB and fixed so many color issues
- [Surma](https://surma.dev/) for their excellent post on [dithering](https://surma.dev/things/ditherpunk/)