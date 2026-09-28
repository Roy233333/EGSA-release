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
384 x 384**, SDFStudio camera metadata, and generated depth/normal arrays.
The original-to-processed image mapping and crop/resize recipe are not
included; no specific frame-selection rule is claimed.
These are processed experimental inputs, not an unmodified mirror of NeRFBK.

The original dataset and adaptations retain the applicable CC BY-NC-SA terms.
This source licence does not establish redistribution rights for the added
monocular priors. Access to the combined training archive is restricted;
public redistribution of those priors is not authorised by this notice.
NeRFBK's Doss reference geometry is also third-party data, not self-collected
EGSA ground truth. The source geometry package contains no monocular priors.

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
