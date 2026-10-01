# Scene2 / Doss Trento: source and redistribution notice

Source dataset: **NeRFBK**, Doss Trento scene, created by the **3D Optical
Metrology unit (3DOM), Fondazione Bruno Kessler (FBK)**.

- Source: https://github.com/3DOM-FBK/NeRFBK
- Dataset licence: **Creative Commons Attribution-NonCommercial-ShareAlike 4.0
  International (CC BY-NC-SA 4.0)**.
- Licence: https://creativecommons.org/licenses/by-nc-sa/4.0/
- Legal terms: https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

The upstream dataset permits sharing and adaptation subject to attribution,
non-commercial use and ShareAlike. Preserve attribution, the licence link,
notices of modifications and applicable upstream notices. No endorsement by
NeRFBK or FBK is implied. The EGSA code's Apache-2.0 licence does not replace
the dataset licence.

## Dataset preprocessing

Upstream lists 761 Huawei P20 Pro images at 540 x 960 and terrestrial laser
scanner reference data. The selected EGSA input contains **380 RGB images at
384 x 384** and SDFStudio camera metadata. Omnidata v2 depth/normal predictions
used in the experiments are not distributed in this release.
SDFStudio's official [conversion script](https://github.com/autonomousvision/sdfstudio/blob/master/scripts/datasets/process_nerfstudio_to_sdfstudio.py)
provides this input resolution with `--mono-prior --crop-mult 1`: images are
centre-cropped to a square, resized to 384 x 384 with bilinear interpolation,
and camera intrinsics are adjusted accordingly. Image resizing and frame
selection are separate steps; the released archive contains the 380-view
processed subset, not all 761 upstream images.

The original dataset and adaptations retain the applicable CC BY-NC-SA terms.
NeRFBK's Doss reference geometry is also third-party data, not self-collected
EGSA ground truth. The source geometry package contains no monocular priors.
The released images are a selected, cropped/resized subset, and the supplied
GT is a processed reference; neither is presented as an unmodified upstream
dataset. In the RGB archive, metadata references to the omitted priors have
been removed and `has_mono_prior` is set to `false`.

## Citation

Ziyang Yan, Gabriele Mazzacca, Simone Rigon, Elisa Mariarosaria Farella,
Pawel Trybala, and Fabio Remondino (2023). *NeRFBK: A Holistic Dataset for
Benchmarking NeRF-Based 3D Reconstruction*. The International Archives of the
Photogrammetry, Remote Sensing and Spatial Information Sciences,
XLVIII-1/W3-2023, 219–226.
https://doi.org/10.5194/isprs-archives-XLVIII-1-W3-2023-219-2023

Please also acknowledge the source repository and its relevant publications,
as requested by the dataset authors. The paper's own licence is not a
substitute for the dataset licence above.
